#include "quant.h"

#include <string.h>

#include "gguf.h"
#include "iqtab.h"

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

/* ---- i-quants (layouts and math as ggml's dequantize_row_iq*) ---- */

static uint32_t
u32le(const uint8_t *p)
{
	return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

static float
gsign(uint8_t s, int j)
{
	return s & kmask_iq2xs[j] ? -1.f : 1.f;
}

/* IQ2_XXS: d qs[32 x u16] = 66 */
static void
dq_iq2_xxs(const uint8_t *b, float *y)
{
	float d = h16(b);
	int ib, l, j;
	for (ib = 0; ib < 8; ib++) {
		const uint8_t *q = b + 2 + 8 * ib;
		uint32_t a1 = u32le(q + 4);
		float db = d * (0.5f + (a1 >> 28)) * 0.25f;
		for (l = 0; l < 4; l++) {
			const uint8_t *g = (const uint8_t *)(iq2xxs_grid + q[l]);
			uint8_t sg = ksigns_iq2xs[(a1 >> 7 * l) & 127];
			for (j = 0; j < 8; j++)
				*y++ = db * g[j] * gsign(sg, j);
		}
	}
}

/* IQ2_XS: d qs[32 x u16] scales[8] = 74 */
static void
dq_iq2_xs(const uint8_t *b, float *y)
{
	float d = h16(b);
	int ib, l, j;
	for (ib = 0; ib < 8; ib++) {
		uint8_t sc = b[66 + ib];
		float db[2];
		db[0] = d * (0.5f + (sc & 15)) * 0.25f;
		db[1] = d * (0.5f + (sc >> 4)) * 0.25f;
		for (l = 0; l < 4; l++) {
			uint16_t q = (uint16_t)(b[2 + 8 * ib + 2 * l] | b[3 + 8 * ib + 2 * l] << 8);
			const uint8_t *g = (const uint8_t *)(iq2xs_grid + (q & 511));
			uint8_t sg = ksigns_iq2xs[q >> 9];
			for (j = 0; j < 8; j++)
				*y++ = db[l / 2] * g[j] * gsign(sg, j);
		}
	}
}

/* IQ2_S: d qs[64] signs[32] qh[8] scales[8] = 82 (qs[32] then signs[32]) */
static void
dq_iq2_s(const uint8_t *b, float *y)
{
	float d = h16(b);
	const uint8_t *qs = b + 2, *sg = b + 34, *qh = b + 66, *sc = b + 74;
	int ib, l, j;
	for (ib = 0; ib < 8; ib++, qs += 4, sg += 4) {
		float db[2];
		db[0] = d * (0.5f + (sc[ib] & 15)) * 0.25f;
		db[1] = d * (0.5f + (sc[ib] >> 4)) * 0.25f;
		for (l = 0; l < 4; l++) {
			const uint8_t *g = (const uint8_t *)(iq2s_grid + (qs[l] | ((qh[ib] << (8 - 2 * l)) & 0x300)));
			for (j = 0; j < 8; j++)
				*y++ = db[l / 2] * g[j] * gsign(sg[l], j);
		}
	}
}

/* IQ3_XXS: d qs[64] scales_and_signs[32] = 98 */
static void
dq_iq3_xxs(const uint8_t *b, float *y)
{
	float d = h16(b);
	const uint8_t *qs = b + 2;
	int ib, l, j;
	for (ib = 0; ib < 8; ib++, qs += 8) {
		uint32_t a = u32le(b + 66 + 4 * ib);
		float db = d * (0.5f + (a >> 28)) * 0.5f;
		for (l = 0; l < 4; l++) {
			uint8_t sg = ksigns_iq2xs[(a >> 7 * l) & 127];
			const uint8_t *g1 = (const uint8_t *)(iq3xxs_grid + qs[2 * l]);
			const uint8_t *g2 = (const uint8_t *)(iq3xxs_grid + qs[2 * l + 1]);
			for (j = 0; j < 4; j++) {
				y[j] = db * g1[j] * gsign(sg, j);
				y[j + 4] = db * g2[j] * gsign(sg, j + 4);
			}
			y += 8;
		}
	}
}

/* IQ3_S: d qs[64] qh[8] signs[32] scales[4] = 110 */
static void
dq_iq3_s(const uint8_t *b, float *y)
{
	float d = h16(b);
	const uint8_t *qs = b + 2, *qh = b + 66, *sg = b + 74, *sc = b + 106;
	int ib, l, j, h;
	for (ib = 0; ib < 8; ib += 2) {
		for (h = 0; h < 2; h++, qs += 8, sg += 4) {
			float db = d * (1 + 2 * (h ? sc[ib / 2] >> 4 : sc[ib / 2] & 15));
			for (l = 0; l < 4; l++) {
				const uint8_t *g1 = (const uint8_t *)(iq3s_grid + (qs[2 * l] | ((qh[ib + h] << (8 - 2 * l)) & 256)));
				const uint8_t *g2 = (const uint8_t *)(iq3s_grid + (qs[2 * l + 1] | ((qh[ib + h] << (7 - 2 * l)) & 256)));
				for (j = 0; j < 4; j++) {
					y[j] = db * g1[j] * gsign(sg[l], j);
					y[j + 4] = db * g2[j] * gsign(sg[l], j + 4);
				}
				y += 8;
			}
		}
	}
}

/* IQ1_S: d qs[32] qh[8 x u16] = 50 */
static void
dq_iq1_s(const uint8_t *b, float *y)
{
	float d = h16(b);
	const uint8_t *qs = b + 2;
	int ib, l, j;
	for (ib = 0; ib < 8; ib++, qs += 4) {
		uint16_t qh = (uint16_t)(b[34 + 2 * ib] | b[35 + 2 * ib] << 8);
		float dl = d * (2 * ((qh >> 12) & 7) + 1);
		float delta = qh & 0x8000 ? -IQ1S_DELTA : IQ1S_DELTA;
		for (l = 0; l < 4; l++) {
			const int8_t *g = (const int8_t *)(iq1s_grid + (qs[l] | (((qh >> 3 * l) & 7) << 8)));
			for (j = 0; j < 8; j++)
				*y++ = dl * (g[j] + delta);
		}
	}
}

/* IQ4_XS: d scales_h(u16) scales_l[4] qs[128] = 136 */
static void
dq_iq4_xs(const uint8_t *b, float *y)
{
	float d = h16(b);
	uint16_t sh = (uint16_t)(b[2] | b[3] << 8);
	const uint8_t *sl = b + 4, *qs = b + 8;
	int ib, j;
	for (ib = 0; ib < 8; ib++, qs += 16, y += 32) {
		int ls = ((sl[ib / 2] >> 4 * (ib % 2)) & 15) | (((sh >> 2 * ib) & 3) << 4);
		float dl = d * (ls - 32);
		for (j = 0; j < 16; j++) {
			y[j] = dl * kvalues_iq4nl[qs[j] & 15];
			y[j + 16] = dl * kvalues_iq4nl[qs[j] >> 4];
		}
	}
}

/* ---- q2k's transcoded types (gguf.h) ---- */

static const float x2mag[4] = { 8, 25, 43, 0 };

/* 2-bit value of element e in Q2_K order */
static int
get2(const uint8_t *p, int e)
{
	return (p[32 * (e / 128) + e % 32] >> (2 * ((e % 128) / 32))) & 3;
}

static void
dq_x2(const uint8_t *b, float *y)
{
	const float d = h16(b) * 0.125f;
	int e;
	for (e = 0; e < 256; e++) {
		int sg = (b[82 + e % 32] >> (4 * (e / 128) + (e % 128) / 32)) & 1;
		y[e] = d * b[2 + e / 16] * x2mag[get2(b + 18, e)] * (sg ? -1.f : 1.f);
	}
}

static void
dq_x3(const uint8_t *b, float *y, int s)
{
	const float d = h16(b) * (s ? 1.0f : 0.25f);
	int ib, j;
	for (ib = 0; ib < 8; ib++)
		for (j = 0; j < 32; j++) {
			int q = b[10 + 16 * ib + j % 16] >> (j < 16 ? 0 : 4) & 15;
			float v = s ? (float)(2 * (q & 7) + 1) : (float)(8 * (q & 7) + 4 + ((q & 7) == 7 ? 2 : 0));
			y[32 * ib + j] = d * b[2 + ib] * (q & 8 ? -v : v);
		}
}

static void
dq_x1(const uint8_t *b, float *y)
{
	const float d = h16(b);
	int e;
	for (e = 0; e < 256; e++) {
		uint8_t sd = b[2 + e / 32];
		y[e] = d * (sd & 127) * ((float)(get2(b + 10, e) - 1) + (sd & 128 ? -0.125f : 0.125f));
	}
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
		case GT_IQ2_XXS: dq_iq2_xxs(b, dst); break;
		case GT_IQ2_XS: dq_iq2_xs(b, dst); break;
		case GT_IQ2_S: dq_iq2_s(b, dst); break;
		case GT_IQ3_XXS: dq_iq3_xxs(b, dst); break;
		case GT_IQ3_S: dq_iq3_s(b, dst); break;
		case GT_IQ1_S: dq_iq1_s(b, dst); break;
		case GT_IQ4_XS: dq_iq4_xs(b, dst); break;
		case GT_X2: dq_x2(b, dst); break;
		case GT_X3: dq_x3(b, dst, 0); break;
		case GT_X3S: dq_x3(b, dst, 1); break;
		case GT_X1: dq_x1(b, dst); break;
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
