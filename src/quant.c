#include "quant.h"

#include <string.h>

#include "gguf.h"

float
fp16_to_f32(uint16_t h)
{
	uint32_t s = (uint32_t)(h & 0x8000) << 16;
	uint32_t e = (h >> 10) & 0x1f;
	uint32_t m = h & 0x3ff;
	uint32_t u;
	float f;

	if (e == 0x1f)
		u = s | 0x7f800000u | (m << 13);
	else if (e != 0)
		u = s | ((e + 112) << 23) | (m << 13);
	else if (m == 0)
		u = s;
	else {
		/* subnormal: normalise */
		e = 113;
		while (!(m & 0x400)) {
			m <<= 1;
			e--;
		}
		u = s | (e << 23) | ((m & 0x3ff) << 13);
	}
	memcpy(&f, &u, 4);
	return f;
}

static float
h16(const uint8_t *p)
{
	return fp16_to_f32((uint16_t)(p[0] | p[1] << 8));
}

/* block layouts, as byte offsets (see ggml-common.h) */

/* Q2_K: scales[16] qs[64] d dmin = 84 */
static void
dq_q2_k(const uint8_t *b, float *y)
{
	const uint8_t *sc = b, *q = b + 16;
	float d = h16(b + 80), dmin = h16(b + 82);
	int n, j, l, is = 0;

	for (n = 0; n < 256; n += 128, q += 32) {
		int shift = 0;
		for (j = 0; j < 4; j++, shift += 2) {
			float dl = d * (sc[is] & 15), ml = dmin * (sc[is] >> 4);
			is++;
			for (l = 0; l < 16; l++)
				*y++ = dl * ((q[l] >> shift) & 3) - ml;
			dl = d * (sc[is] & 15);
			ml = dmin * (sc[is] >> 4);
			is++;
			for (l = 0; l < 16; l++)
				*y++ = dl * ((q[l + 16] >> shift) & 3) - ml;
		}
	}
}

/* unpack the twelve 6-bit Q3_K scales to sixteen signed values */
static void
q3_scales(const uint8_t *s12, int8_t *out)
{
	int i;
	for (i = 0; i < 16; i++) {
		int lo = (i < 8) ? (s12[i] & 15) : (s12[i - 8] >> 4);
		int hi = (s12[8 + (i & 3)] >> (2 * (i >> 2))) & 3;
		out[i] = (int8_t)((lo | hi << 4) - 32);
	}
}

/* Q3_K: hmask[32] qs[64] scales[12] d = 110 */
static void
dq_q3_k(const uint8_t *b, float *y)
{
	const uint8_t *hm = b, *q = b + 32;
	float d = h16(b + 108);
	int8_t sc[16];
	int n, j, l, is = 0;
	uint8_t m = 1;

	q3_scales(b + 96, sc);
	for (n = 0; n < 256; n += 128, q += 32) {
		int shift = 0;
		for (j = 0; j < 4; j++, shift += 2, m <<= 1) {
			float dl = d * sc[is++];
			for (l = 0; l < 16; l++)
				*y++ = dl * ((int)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
			dl = d * sc[is++];
			for (l = 0; l < 16; l++)
				*y++ = dl * ((int)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
		}
	}
}

static void
scale_min_k4(int j, const uint8_t *q, int *d, int *m)
{
	if (j < 4) {
		*d = q[j] & 63;
		*m = q[j + 4] & 63;
	} else {
		*d = (q[j + 4] & 15) | ((q[j - 4] >> 6) << 4);
		*m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
	}
}

/* Q4_K: d dmin scales[12] qs[128] = 144 */
static void
dq_q4_k(const uint8_t *b, float *y)
{
	const uint8_t *q = b + 16;
	float d = h16(b), dmin = h16(b + 2);
	int j, l, is = 0, sc, m;

	for (j = 0; j < 256; j += 64, q += 32, is += 2) {
		float d1, m1, d2, m2;
		scale_min_k4(is, b + 4, &sc, &m);
		d1 = d * sc;
		m1 = dmin * m;
		scale_min_k4(is + 1, b + 4, &sc, &m);
		d2 = d * sc;
		m2 = dmin * m;
		for (l = 0; l < 32; l++)
			*y++ = d1 * (q[l] & 15) - m1;
		for (l = 0; l < 32; l++)
			*y++ = d2 * (q[l] >> 4) - m2;
	}
}

/* Q5_K: d dmin scales[12] qh[32] qs[128] = 176 */
static void
dq_q5_k(const uint8_t *b, float *y)
{
	const uint8_t *qh = b + 16, *ql = b + 48;
	float d = h16(b), dmin = h16(b + 2);
	int j, l, is = 0, sc, m;
	uint8_t u1 = 1, u2 = 2;

	for (j = 0; j < 256; j += 64, ql += 32, is += 2, u1 <<= 2, u2 <<= 2) {
		float d1, m1, d2, m2;
		scale_min_k4(is, b + 4, &sc, &m);
		d1 = d * sc;
		m1 = dmin * m;
		scale_min_k4(is + 1, b + 4, &sc, &m);
		d2 = d * sc;
		m2 = dmin * m;
		for (l = 0; l < 32; l++)
			*y++ = d1 * ((ql[l] & 15) + (qh[l] & u1 ? 16 : 0)) - m1;
		for (l = 0; l < 32; l++)
			*y++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
	}
}

/* Q6_K: ql[128] qh[64] scales[16] d = 210 */
static void
dq_q6_k(const uint8_t *b, float *y)
{
	const uint8_t *ql = b, *qh = b + 128;
	const int8_t *sc = (const int8_t *)(b + 192);
	float d = h16(b + 208);
	int n, l;

	for (n = 0; n < 256; n += 128, y += 128, ql += 64, qh += 32, sc += 8) {
		for (l = 0; l < 32; l++) {
			int is = l / 16;
			int q1 = ((ql[l] & 15) | ((qh[l] >> 0) & 3) << 4) - 32;
			int q2 = ((ql[l + 32] & 15) | ((qh[l] >> 2) & 3) << 4) - 32;
			int q3 = ((ql[l] >> 4) | ((qh[l] >> 4) & 3) << 4) - 32;
			int q4 = ((ql[l + 32] >> 4) | ((qh[l] >> 6) & 3) << 4) - 32;
			y[l] = d * sc[is] * q1;
			y[l + 32] = d * sc[is + 2] * q2;
			y[l + 64] = d * sc[is + 4] * q3;
			y[l + 96] = d * sc[is + 6] * q4;
		}
	}
}

/* Q8_0: d qs[32] = 34 */
static void
dq_q8_0(const uint8_t *b, float *y)
{
	float d = h16(b);
	int l;
	for (l = 0; l < 32; l++)
		y[l] = d * (int8_t)b[2 + l];
}

void
dequant_row(uint32_t t, const void *src, float *dst, size_t n)
{
	const uint8_t *b = src;
	size_t i, bs = ggml_type_block(t), bb = ggml_type_bytes(t);

	if (t == GT_F32) {
		memcpy(dst, src, n * 4);
		return;
	}
	if (t == GT_F16) {
		for (i = 0; i < n; i++)
			dst[i] = h16(b + 2 * i);
		return;
	}
	for (i = 0; i < n; i += bs, b += bb, dst += bs) {
		switch (t) {
		case GT_Q2_K: dq_q2_k(b, dst); break;
		case GT_Q3_K: dq_q3_k(b, dst); break;
		case GT_Q4_K: dq_q4_k(b, dst); break;
		case GT_Q5_K: dq_q5_k(b, dst); break;
		case GT_Q6_K: dq_q6_k(b, dst); break;
		case GT_Q8_0: dq_q8_0(b, dst); break;
		default: memset(dst, 0, bs * sizeof(float)); break;
		}
	}
}

float
dot_row(uint32_t t, const void *row, const float *x, size_t n)
{
	float tmp[256];
	const uint8_t *b = row;
	size_t i, j, bs, bb;
	float s = 0;

	if (t == GT_F32) {
		const float *w = row;
		for (i = 0; i < n; i++)
			s += w[i] * x[i];
		return s;
	}
	bs = ggml_type_block(t);
	bb = ggml_type_bytes(t);
	for (i = 0; i < n; i += bs, b += bb) {
		dequant_row(t, b, tmp, bs);
		for (j = 0; j < bs; j++)
			s += tmp[j] * x[i + j];
	}
	return s;
}
