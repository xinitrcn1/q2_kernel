#include "kq.h"

#include <math.h>
#include <string.h>

#include "gguf.h"
#include "quant.h"
#include "iqtab.h"

#if defined(__AVX2__)
#include <immintrin.h>
#endif

static float
h16(const uint8_t *p)
{
	return fp16_to_f32((uint16_t)(p[0] | p[1] << 8));
}

void
q8k_quantize(const float *x, q8k *y, size_t n)
{
	size_t b;
	int i, j;

	for (b = 0; b < n / 256; b++, x += 256, y++) {
		float amax = 0, mx = 0, is;
#if defined(__AVX2__)
		/* same result as the scalar loop below: lrintf and cvtps both
		 * round to nearest even, and |is*x| <= 127 */
		__m256 vmax = _mm256_loadu_ps(x), vmin = vmax;
		for (i = 8; i < 256; i += 8) {
			__m256 v = _mm256_loadu_ps(x + i);
			vmax = _mm256_max_ps(vmax, v);
			vmin = _mm256_min_ps(vmin, v);
		}
		{
			float mxs[8], mns[8], hi, lo;
			_mm256_storeu_ps(mxs, vmax);
			_mm256_storeu_ps(mns, vmin);
			hi = mxs[0];
			lo = mns[0];
			for (i = 1; i < 8; i++) {
				hi = mxs[i] > hi ? mxs[i] : hi;
				lo = mns[i] < lo ? mns[i] : lo;
			}
			/* the signed extreme; on a tie ggml takes the first
			 * index, but either sign gives the same products */
			mx = hi >= -lo ? hi : lo;
			amax = fabsf(mx);
		}
		if (amax == 0) {
			memset(y, 0, sizeof(*y));
			continue;
		}
		is = -127.0f / mx;
		y->d = 1 / is;
		{
			__m256 vs = _mm256_set1_ps(is);
			for (i = 0; i < 256; i += 32) {
				__m256i a = _mm256_cvtps_epi32(_mm256_mul_ps(vs, _mm256_loadu_ps(x + i)));
				__m256i c = _mm256_cvtps_epi32(_mm256_mul_ps(vs, _mm256_loadu_ps(x + i + 8)));
				__m256i d = _mm256_cvtps_epi32(_mm256_mul_ps(vs, _mm256_loadu_ps(x + i + 16)));
				__m256i e = _mm256_cvtps_epi32(_mm256_mul_ps(vs, _mm256_loadu_ps(x + i + 24)));
				__m256i p = _mm256_packs_epi16(_mm256_packs_epi32(a, c), _mm256_packs_epi32(d, e));
				/* packs interleave 128-bit lanes: restore order */
				p = _mm256_permutevar8x32_epi32(p, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
				_mm256_storeu_si256((__m256i *)(y->qs + i), p);
			}
		}
		for (j = 0; j < 16; j++) {
			int s = 0;
			for (i = 0; i < 16; i++)
				s += y->qs[16 * j + i];
			y->bsums[j] = (int16_t)s;
		}
		continue;
#endif
		/* same rounding as ggml's quantize_row_q8_K: scale by the
		 * signed extreme, round to nearest even */
		for (i = 0; i < 256; i++) {
			float a = fabsf(x[i]);
			if (a > amax) {
				amax = a;
				mx = x[i];
			}
		}
		if (amax == 0) {
			memset(y, 0, sizeof(*y));
			continue;
		}
		is = -127.0f / mx;
		y->d = 1 / is;
		for (i = 0; i < 256; i++) {
			long v = lrintf(is * x[i]);
			y->qs[i] = (int8_t)(v > 127 ? 127 : v);
		}
		for (j = 0; j < 16; j++) {
			int s = 0;
			for (i = 0; i < 16; i++)
				s += y->qs[16 * j + i];
			y->bsums[j] = (int16_t)s;
		}
	}
}

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

/* the eight 6-bit scale/min pairs of Q4_K and Q5_K */
static void
k4_scales(const uint8_t *q, uint8_t *sc, uint8_t *mn)
{
	int j;
	for (j = 0; j < 4; j++) {
		sc[j] = q[j] & 63;
		mn[j] = q[j + 4] & 63;
	}
	for (j = 4; j < 8; j++) {
		sc[j] = (uint8_t)((q[j + 4] & 15) | ((q[j - 4] >> 6) << 4));
		mn[j] = (uint8_t)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
	}
}

/* ---------------------------------------------------------------- */
/* scalar integer path                                              */

float
kq_dot_scalar(uint32_t t, const void *row, const q8k *x, size_t n)
{
	const uint8_t *b = row;
	size_t nb, i;
	float sum = 0;
	int k, l;

	switch (t) {
	case GT_Q2_K:
		for (nb = 0; nb < n / 256; nb++, b += 84, x++) {
			const uint8_t *sc = b, *q = b + 16;
			int isum = 0, summ = 0, j, h, c;
			for (k = 0; k < 16; k++)
				summ += (sc[k] >> 4) * x->bsums[k];
			for (c = 0; c < 2; c++)
				for (j = 0; j < 4; j++)
					for (h = 0; h < 2; h++) {
						int s = sc[8 * c + 2 * j + h] & 15, p = 0;
						const uint8_t *qq = q + 32 * c + 16 * h;
						const int8_t *xq = x->qs + 128 * c + 32 * j + 16 * h;
						for (l = 0; l < 16; l++)
							p += ((qq[l] >> (2 * j)) & 3) * xq[l];
						isum += s * p;
					}
			sum += x->d * (h16(b + 80) * isum - h16(b + 82) * summ);
		}
		return sum;
	case GT_Q3_K:
		for (nb = 0; nb < n / 256; nb++, b += 110, x++) {
			const uint8_t *hm = b, *q = b + 32;
			int8_t sc[16];
			int isum = 0, offs = 0, j, h, c;
			q3_scales(b + 96, sc);
			for (k = 0; k < 16; k++)
				offs += sc[k] * x->bsums[k];
			for (c = 0; c < 2; c++)
				for (j = 0; j < 4; j++)
					for (h = 0; h < 2; h++) {
						int p = 0;
						const uint8_t *qq = q + 32 * c + 16 * h;
						const uint8_t *hh = hm + 16 * h;
						const int8_t *xq = x->qs + 128 * c + 32 * j + 16 * h;
						for (l = 0; l < 16; l++) {
							int u = ((qq[l] >> (2 * j)) & 3) |
							    (((hh[l] >> (4 * c + j)) & 1) << 2);
							p += u * xq[l];
						}
						isum += sc[8 * c + 2 * j + h] * p;
					}
			sum += x->d * h16(b + 108) * (float)(isum - 4 * offs);
		}
		return sum;
	case GT_Q4_K:
	case GT_Q5_K:
		for (nb = 0; nb < n / 256; nb++, x++) {
			uint8_t sc[8], mn[8];
			const uint8_t *qh = b + 16;
			const uint8_t *q = b + (t == GT_Q4_K ? 16 : 48);
			int isum = 0, summ = 0, c;
			k4_scales(b + 4, sc, mn);
			for (k = 0; k < 8; k++)
				summ += mn[k] * (x->bsums[2 * k] + x->bsums[2 * k + 1]);
			for (c = 0; c < 4; c++) {
				int p0 = 0, p1 = 0;
				const int8_t *xq = x->qs + 64 * c;
				for (l = 0; l < 32; l++) {
					int u0 = q[32 * c + l] & 15, u1 = q[32 * c + l] >> 4;
					if (t == GT_Q5_K) {
						u0 |= ((qh[l] >> (2 * c)) & 1) << 4;
						u1 |= ((qh[l] >> (2 * c + 1)) & 1) << 4;
					}
					p0 += u0 * xq[l];
					p1 += u1 * xq[32 + l];
				}
				isum += sc[2 * c] * p0 + sc[2 * c + 1] * p1;
			}
			sum += x->d * (h16(b) * isum - h16(b + 2) * summ);
			b += (t == GT_Q4_K ? 144 : 176);
		}
		return sum;
	case GT_Q6_K:
		for (nb = 0; nb < n / 256; nb++, b += 210, x++) {
			const int8_t *sc = (const int8_t *)(b + 192);
			int isum = 0, offs = 0, c;
			for (k = 0; k < 16; k++)
				offs += sc[k] * x->bsums[k];
			for (c = 0; c < 2; c++) {
				const uint8_t *ql = b + 64 * c, *qh = b + 128 + 32 * c;
				const int8_t *xq = x->qs + 128 * c;
				for (k = 0; k < 8; k++) {
					/* 16-group k of this 128 chunk: quarter k/2 */
					int qtr = k >> 1, p = 0;
					for (l = 16 * (k & 1); l < 16 * (k & 1) + 16; l++) {
						int lo = (qtr & 1) ? ql[32 + l] : ql[l];
						int u = ((qtr & 2) ? lo >> 4 : lo & 15) |
						    (((qh[l] >> (2 * qtr)) & 3) << 4);
						p += u * xq[32 * qtr + l];
					}
					isum += sc[8 * c + k] * p;
				}
			}
			sum += x->d * h16(b + 208) * (float)(isum - 32 * offs);
		}
		return sum;
	case GT_Q8_0:
		for (nb = 0; nb < n / 256; nb++, x++) {
			float s = 0;
			for (k = 0; k < 8; k++, b += 34) {
				int p = 0;
				for (l = 0; l < 32; l++)
					p += (int8_t)b[2 + l] * x->qs[32 * k + l];
				s += h16(b) * (float)p;
			}
			sum += x->d * s;
		}
		return sum;
	default:
		/* i-quants (until they have integer kernels): dequantize */
		if (ggml_type_block(t) == 256) {
			float w[256];
			size_t bb = ggml_type_bytes(t);
			for (nb = 0; nb < n / 256; nb++, b += bb, x++) {
				float s = 0;
				dequant_row(t, b, w, 256);
				for (l = 0; l < 256; l++)
					s += w[l] * x->qs[l];
				sum += x->d * s;
			}
			return sum;
		}
		break;
	}
	(void)i;
	return 0;
}

/* ---------------------------------------------------------------- */
/* AVX2                                                             */

#if defined(__AVX2__)

/*
 * Each kernel accumulates per-block integer sums, converts them once
 * and folds them into a float vector with the block's scales; there is
 * one horizontal sum per row.  The 6-bit scale unpacking works on whole
 * 32-bit words (four scales per op); each word form was proved equal to
 * the byte-wise definition (kept in kq_dot_scalar) by SAT with ucas.
 */

#if defined(__F16C__)
static float
h16f(const uint8_t *p)
{
	uint16_t v;
	memcpy(&v, p, 2);
	return _cvtsh_ss(v);
}
#else
#define h16f h16
#endif

static uint32_t
rd32(const uint8_t *p)
{
	uint32_t v;
	memcpy(&v, p, 4);
	return v;
}

static float
hsumf(__m256 v)
{
	__m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
	s = _mm_add_ps(s, _mm_movehl_ps(s, s));
	s = _mm_add_ss(s, _mm_movehdup_ps(s));
	return _mm_cvtss_f32(s);
}

/* Shuffle masks are built from an opaque zero: with constant masks
 * clang turns half of the vpshufb into vpshuflw+vpermd pairs, two
 * port-5 uops instead of one, and port 5 bounds these loops. */
/* (an external table: the compiler cannot prove it constant) */
#define PM(a, b) { a, a + 1, a, a + 1, a, a + 1, a, a + 1, a, a + 1, a, a + 1, \
	a, a + 1, a, a + 1, b, b + 1, b, b + 1, b, b + 1, b, b + 1, b, b + 1, \
	b, b + 1, b, b + 1, b, b + 1 }
#define LM(s) { 2 * s, 2 * s + 1, 2 * s, 2 * s + 1, 2 * s, 2 * s + 1, 2 * s, 2 * s + 1, \
	2 * s, 2 * s + 1, 2 * s, 2 * s + 1, 2 * s, 2 * s + 1, 2 * s, 2 * s + 1, \
	2 * s, 2 * s + 1, 2 * s, 2 * s + 1, 2 * s, 2 * s + 1, 2 * s, 2 * s + 1, \
	2 * s, 2 * s + 1, 2 * s, 2 * s + 1, 2 * s, 2 * s + 1, 2 * s, 2 * s + 1 }
uint8_t q2k_shuf_masks[12][32] = {
	PM(0, 2), PM(4, 6), PM(8, 10), PM(12, 14),
	LM(0), LM(1), LM(2), LM(3), LM(4), LM(5), LM(6), LM(7)
};

/* byte masks picking i16 lanes (2j, 2j+1) of a broadcast 8 x i16 into
 * [lane 2j x8 | lane 2j+1 x8] */
static __m256i
pair_mask(int j)
{
	return _mm256_loadu_si256((const __m256i *)q2k_shuf_masks[j]);
}

static float
dot_q2_k(const uint8_t *b, const q8k *x, size_t nb)
{
	const __m256i m3 = _mm256_set1_epi8(3);
	const __m128i m4 = _mm_set1_epi8(15);
	__m256i pm[4];
	__m256 acc = _mm256_setzero_ps();
	size_t i;
	int j, c;

	for (j = 0; j < 4; j++)
		pm[j] = pair_mask(j);
	for (i = 0; i < nb; i++, b += 84, x++) {
		__m128i sc8 = _mm_loadu_si128((const __m128i *)b);
		__m256i mins = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(sc8, 4), m4));
		__m256i s16 = _mm256_cvtepu8_epi16(_mm_and_si128(sc8, m4));
		__m256i msum = _mm256_madd_epi16(mins,
		    _mm256_loadu_si256((const __m256i *)x->bsums));
		__m256i isum = _mm256_setzero_si256();
		const int8_t *xq = x->qs;

		for (c = 0; c < 2; c++) {
			__m256i qv = _mm256_loadu_si256((const __m256i *)(b + 16 + 32 * c));
			__m256i sb = _mm256_broadcastsi128_si256(c ? _mm256_extracti128_si256(s16, 1)
			    : _mm256_castsi256_si128(s16));
			for (j = 0; j < 4; j++, xq += 32) {
				__m256i u = _mm256_and_si256(_mm256_srli_epi16(qv, 2 * j), m3);
				__m256i p = _mm256_maddubs_epi16(u, _mm256_loadu_si256((const __m256i *)xq));
				isum = _mm256_add_epi32(isum, _mm256_madd_epi16(p, _mm256_shuffle_epi8(sb, pm[j])));
			}
		}
		acc = _mm256_fmadd_ps(_mm256_set1_ps(h16f(b + 80) * x->d),
		    _mm256_cvtepi32_ps(isum), acc);
		acc = _mm256_fmadd_ps(_mm256_set1_ps(-h16f(b + 82) * x->d),
		    _mm256_cvtepi32_ps(msum), acc);
	}
	return hsumf(acc);
}

/* sixteen signed Q3_K scales from the 12 packed bytes */
static __m256i
q3_scales16(const uint8_t *s)
{
	const uint32_t m4 = 0x0f0f0f0fu, m2 = 0x03030303u;
	uint32_t a0 = rd32(s), a1 = rd32(s + 4), a2 = rd32(s + 8);
	__m128i w = _mm_setr_epi32(
	    (int)((a0 & m4) | ((a2 & m2) << 4)),
	    (int)((a1 & m4) | (((a2 >> 2) & m2) << 4)),
	    (int)(((a0 >> 4) & m4) | (((a2 >> 4) & m2) << 4)),
	    (int)(((a1 >> 4) & m4) | (((a2 >> 6) & m2) << 4)));
	return _mm256_sub_epi16(_mm256_cvtepu8_epi16(w), _mm256_set1_epi16(32));
}

static float
dot_q3_k(const uint8_t *b, const q8k *x, size_t nb)
{
	const __m256i m3 = _mm256_set1_epi8(3), m1 = _mm256_set1_epi8(1);
	__m256i pm[4];
	__m256 acc = _mm256_setzero_ps();
	size_t i;
	int j, c;

	for (j = 0; j < 4; j++)
		pm[j] = pair_mask(j);
	for (i = 0; i < nb; i++, b += 110, x++) {
		__m256i s16 = q3_scales16(b + 96), isum = _mm256_setzero_si256();
		__m256i hv = _mm256_loadu_si256((const __m256i *)b);
		__m256i offs = _mm256_madd_epi16(s16,
		    _mm256_loadu_si256((const __m256i *)x->bsums));
		const int8_t *xq = x->qs;

		for (c = 0; c < 2; c++) {
			__m256i qv = _mm256_loadu_si256((const __m256i *)(b + 32 + 32 * c));
			__m256i sb = _mm256_broadcastsi128_si256(c ? _mm256_extracti128_si256(s16, 1)
			    : _mm256_castsi256_si128(s16));
			for (j = 0; j < 4; j++, xq += 32) {
				__m256i u = _mm256_and_si256(_mm256_srli_epi16(qv, 2 * j), m3);
				__m256i h = _mm256_and_si256(_mm256_srli_epi16(hv, 4 * c + j), m1);
				__m256i p;
				u = _mm256_or_si256(u, _mm256_slli_epi16(h, 2));
				p = _mm256_maddubs_epi16(u, _mm256_loadu_si256((const __m256i *)xq));
				isum = _mm256_add_epi32(isum, _mm256_madd_epi16(p, _mm256_shuffle_epi8(sb, pm[j])));
			}
		}
		isum = _mm256_sub_epi32(isum, _mm256_slli_epi32(offs, 2));
		acc = _mm256_fmadd_ps(_mm256_set1_ps(h16f(b + 108) * x->d),
		    _mm256_cvtepi32_ps(isum), acc);
	}
	return hsumf(acc);
}

/* byte mask broadcasting i16 lane s of a 128-bit half to all 16 lanes */
static __m256i
lane_mask(int s)
{
	return _mm256_loadu_si256((const __m256i *)q2k_shuf_masks[4 + s]);
}

static float
dot_q45_k(const uint8_t *b, const q8k *x, size_t nb, int five)
{
	__m256i lm[8];
	const __m256i m4 = _mm256_set1_epi8(15), m1 = _mm256_set1_epi8(1);
	const uint32_t k6 = 0x3f3f3f3fu, k4 = 0x0f0f0f0fu, k2 = 0x03030303u;
	const size_t bsz = five ? 176 : 144;
	__m256 acc = _mm256_setzero_ps();
	size_t i;
	int c;

	for (c = 0; c < 8; c++)
		lm[c] = lane_mask(c);
	for (i = 0; i < nb; i++, b += bsz, x++) {
		const uint8_t *q = b + (five ? 48 : 16);
		uint32_t a0 = rd32(b + 4), a1 = rd32(b + 8), a2 = rd32(b + 12);
		/* 8 scales and 8 mins, 6 bits each */
		__m128i sm = _mm_setr_epi32((int)(a0 & k6),
		    (int)((a2 & k4) | (((a0 >> 6) & k2) << 4)),
		    (int)(a1 & k6),
		    (int)(((a2 >> 4) & k4) | (((a1 >> 6) & k2) << 4)));
		__m128i mn8 = _mm_unpackhi_epi64(sm, sm);
		__m256i mins = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(mn8, mn8));
		/* lanes 0-7 scales, 8-15 mins; scales broadcast to both halves */
		__m256i sb = _mm256_broadcastsi128_si256(_mm256_castsi256_si128(
		    _mm256_cvtepu8_epi16(sm)));
		__m256i msum = _mm256_madd_epi16(mins,
		    _mm256_loadu_si256((const __m256i *)x->bsums));
		__m256i isum = _mm256_setzero_si256();
		__m256i hv = five ? _mm256_loadu_si256((const __m256i *)(b + 16)) : isum;

		for (c = 0; c < 4; c++) {
			__m256i qv = _mm256_loadu_si256((const __m256i *)(q + 32 * c));
			__m256i u0 = _mm256_and_si256(qv, m4);
			__m256i u1 = _mm256_and_si256(_mm256_srli_epi16(qv, 4), m4);
			__m256i p0, p1;
			if (five) {
				u0 = _mm256_or_si256(u0, _mm256_slli_epi16(
				    _mm256_and_si256(_mm256_srli_epi16(hv, 2 * c), m1), 4));
				u1 = _mm256_or_si256(u1, _mm256_slli_epi16(
				    _mm256_and_si256(_mm256_srli_epi16(hv, 2 * c + 1), m1), 4));
			}
			p0 = _mm256_maddubs_epi16(u0, _mm256_loadu_si256((const __m256i *)(x->qs + 64 * c)));
			p1 = _mm256_maddubs_epi16(u1, _mm256_loadu_si256((const __m256i *)(x->qs + 64 * c + 32)));
			/* scales 2c and 2c+1, each over 32 weights */
			isum = _mm256_add_epi32(isum, _mm256_madd_epi16(p0,
			    _mm256_shuffle_epi8(sb, lm[2 * c])));
			isum = _mm256_add_epi32(isum, _mm256_madd_epi16(p1,
			    _mm256_shuffle_epi8(sb, lm[2 * c + 1])));
		}
		acc = _mm256_fmadd_ps(_mm256_set1_ps(h16f(b) * x->d),
		    _mm256_cvtepi32_ps(isum), acc);
		acc = _mm256_fmadd_ps(_mm256_set1_ps(-h16f(b + 2) * x->d),
		    _mm256_cvtepi32_ps(msum), acc);
	}
	return hsumf(acc);
}

static float
dot_q6_k(const uint8_t *b, const q8k *x, size_t nb)
{
	const __m256i m4 = _mm256_set1_epi8(15), m3 = _mm256_set1_epi8(3);
	__m256i pm[4];
	__m256 acc = _mm256_setzero_ps();
	size_t i;
	int j, c;

	for (j = 0; j < 4; j++)
		pm[j] = pair_mask(j);
	for (i = 0; i < nb; i++, b += 210, x++) {
		__m256i s16 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(b + 192)));
		__m256i offs = _mm256_madd_epi16(s16,
		    _mm256_loadu_si256((const __m256i *)x->bsums));
		__m256i isum = _mm256_setzero_si256();
		const int8_t *xq = x->qs;

		for (c = 0; c < 2; c++) {
			__m256i l0 = _mm256_loadu_si256((const __m256i *)(b + 64 * c));
			__m256i l1 = _mm256_loadu_si256((const __m256i *)(b + 64 * c + 32));
			__m256i qh = _mm256_loadu_si256((const __m256i *)(b + 128 + 32 * c));
			__m256i sb = _mm256_broadcastsi128_si256(c ? _mm256_extracti128_si256(s16, 1)
			    : _mm256_castsi256_si128(s16));
			__m256i u[4];
			u[0] = _mm256_and_si256(l0, m4);
			u[1] = _mm256_and_si256(l1, m4);
			u[2] = _mm256_and_si256(_mm256_srli_epi16(l0, 4), m4);
			u[3] = _mm256_and_si256(_mm256_srli_epi16(l1, 4), m4);
			for (j = 0; j < 4; j++, xq += 32) {
				__m256i h = _mm256_and_si256(_mm256_srli_epi16(qh, 2 * j), m3);
				__m256i v = _mm256_or_si256(u[j], _mm256_slli_epi16(h, 4));
				__m256i p = _mm256_maddubs_epi16(v, _mm256_loadu_si256((const __m256i *)xq));
				isum = _mm256_add_epi32(isum, _mm256_madd_epi16(p, _mm256_shuffle_epi8(sb, pm[j])));
			}
		}
		isum = _mm256_sub_epi32(isum, _mm256_slli_epi32(offs, 5));
		acc = _mm256_fmadd_ps(_mm256_set1_ps(h16f(b + 208) * x->d),
		    _mm256_cvtepi32_ps(isum), acc);
	}
	return hsumf(acc);
}

static float
dot_q8_0(const uint8_t *b, const q8k *x, size_t nb)
{
	const __m256i ones = _mm256_set1_epi16(1);
	__m256 acc = _mm256_setzero_ps();
	size_t i;
	int k;

	for (i = 0; i < nb; i++, x++) {
		for (k = 0; k < 8; k++, b += 34) {
			__m256i w = _mm256_loadu_si256((const __m256i *)(b + 2));
			__m256i xv = _mm256_loadu_si256((const __m256i *)(x->qs + 32 * k));
			__m256i p = _mm256_maddubs_epi16(_mm256_sign_epi8(w, w),
			    _mm256_sign_epi8(xv, w));
			__m256 pf = _mm256_cvtepi32_ps(_mm256_madd_epi16(p, ones));
			acc = _mm256_fmadd_ps(_mm256_set1_ps(h16f(b) * x->d), pf, acc);
		}
	}
	return hsumf(acc);
}

/* ---------------------------------------------------------------- */
/* i-quants, AVX2.  Each 8 weights come from a codebook entry (unsigned
 * magnitudes) and an 8-bit sign mask; the signs go onto the
 * activations (vpsignb) so maddubs takes the magnitudes as they are. */

/* +-1 bytes for every 8-bit sign mask (bit j set: byte j = -1) */
static uint64_t sgn8[256];
/* the same for the 7-bit codes of ksigns_iq2xs (parity sign bit) */
static uint64_t sgn7[128];

void
kq_init(void)
{
	static int done;
	int i, j;
	if (done)
		return;
	for (i = 0; i < 256; i++) {
		uint64_t v = 0;
		for (j = 0; j < 8; j++)
			v |= (uint64_t)((i >> j) & 1 ? 0xff : 0x01) << (8 * j);
		sgn8[i] = v;
	}
	for (i = 0; i < 128; i++)
		sgn7[i] = sgn8[ksigns_iq2xs[i]];
	done = 1;
}

static __m256i
grid4(uint64_t a, uint64_t b, uint64_t c, uint64_t d)
{
	return _mm256_set_epi64x((long long)d, (long long)c, (long long)b, (long long)a);
}

/* maddubs(u, x*sign) with its i16 pairs summed and scaled by sc
 * (|maddubs| <= 2*62*127, sc <= 63: no overflow) -> 8 x i32 */
static __m256i
dot32s(__m256i u, __m256i x, __m256i sg, int sc)
{
	return _mm256_madd_epi16(_mm256_maddubs_epi16(u, _mm256_sign_epi8(x, sg)),
	    _mm256_set1_epi16((short)sc));
}

static float
dot_iq2_xxs(const uint8_t *b, const q8k *x, size_t nb)
{
	__m256 acc = _mm256_setzero_ps();
	size_t i;
	int ib;
	for (i = 0; i < nb; i++, b += 66, x++) {
		__m256i s = _mm256_setzero_si256();
		for (ib = 0; ib < 8; ib++) {
			const uint8_t *q = b + 2 + 8 * ib;
			uint32_t a1 = rd32(q + 4);
			__m256i u = grid4(iq2xxs_grid[q[0]], iq2xxs_grid[q[1]], iq2xxs_grid[q[2]],
			    iq2xxs_grid[q[3]]);
			__m256i sg = grid4(sgn7[a1 & 127], sgn7[(a1 >> 7) & 127], sgn7[(a1 >> 14) & 127],
			    sgn7[(a1 >> 21) & 127]);
			s = _mm256_add_epi32(s, dot32s(u, _mm256_loadu_si256((const __m256i *)(x->qs + 32 * ib)),
			    sg, 2 * (int)(a1 >> 28) + 1));
		}
		acc = _mm256_fmadd_ps(_mm256_set1_ps(h16f(b) * x->d * 0.125f), _mm256_cvtepi32_ps(s), acc);
	}
	return hsumf(acc);
}

static float
dot_iq2_xs(const uint8_t *b, const q8k *x, size_t nb)
{
	__m256 acc = _mm256_setzero_ps();
	size_t i;
	int ib;
	for (i = 0; i < nb; i++, b += 74, x++) {
		__m256i s = _mm256_setzero_si256();
		for (ib = 0; ib < 8; ib++) {
			const uint8_t *q = b + 2 + 8 * ib;
			uint16_t q0 = (uint16_t)(q[0] | q[1] << 8), q1 = (uint16_t)(q[2] | q[3] << 8);
			uint16_t q2 = (uint16_t)(q[4] | q[5] << 8), q3 = (uint16_t)(q[6] | q[7] << 8);
			uint8_t sc = b[66 + ib];
			__m256i u = grid4(iq2xs_grid[q0 & 511], iq2xs_grid[q1 & 511], iq2xs_grid[q2 & 511],
			    iq2xs_grid[q3 & 511]);
			__m256i sg = grid4(sgn7[q0 >> 9], sgn7[q1 >> 9], sgn7[q2 >> 9], sgn7[q3 >> 9]);
			__m256i p = _mm256_maddubs_epi16(u, _mm256_sign_epi8(
			    _mm256_loadu_si256((const __m256i *)(x->qs + 32 * ib)), sg));
			/* scales per 16: low half (sc & 15), high half (sc >> 4) */
			__m256i w = _mm256_setr_m128i(_mm_set1_epi16((short)(2 * (sc & 15) + 1)),
			    _mm_set1_epi16((short)(2 * (sc >> 4) + 1)));
			s = _mm256_add_epi32(s, _mm256_madd_epi16(p, w));
		}
		acc = _mm256_fmadd_ps(_mm256_set1_ps(h16f(b) * x->d * 0.125f), _mm256_cvtepi32_ps(s), acc);
	}
	return hsumf(acc);
}

static float
dot_iq2_s(const uint8_t *b, const q8k *x, size_t nb)
{
	__m256 acc = _mm256_setzero_ps();
	size_t i;
	int ib;
	for (i = 0; i < nb; i++, b += 82, x++) {
		const uint8_t *qs = b + 2, *sgb = b + 34, *qh = b + 66, *scs = b + 74;
		__m256i s = _mm256_setzero_si256();
		for (ib = 0; ib < 8; ib++, qs += 4, sgb += 4) {
			uint8_t h = qh[ib], sc = scs[ib];
			__m256i u = grid4(iq2s_grid[qs[0] | ((h << 8) & 0x300)],
			    iq2s_grid[qs[1] | ((h << 6) & 0x300)], iq2s_grid[qs[2] | ((h << 4) & 0x300)],
			    iq2s_grid[qs[3] | ((h << 2) & 0x300)]);
			__m256i sg = grid4(sgn8[sgb[0]], sgn8[sgb[1]], sgn8[sgb[2]], sgn8[sgb[3]]);
			__m256i p = _mm256_maddubs_epi16(u, _mm256_sign_epi8(
			    _mm256_loadu_si256((const __m256i *)(x->qs + 32 * ib)), sg));
			__m256i w = _mm256_setr_m128i(_mm_set1_epi16((short)(2 * (sc & 15) + 1)),
			    _mm_set1_epi16((short)(2 * (sc >> 4) + 1)));
			s = _mm256_add_epi32(s, _mm256_madd_epi16(p, w));
		}
		acc = _mm256_fmadd_ps(_mm256_set1_ps(h16f(b) * x->d * 0.125f), _mm256_cvtepi32_ps(s), acc);
	}
	return hsumf(acc);
}

static float
dot_iq3_xxs(const uint8_t *b, const q8k *x, size_t nb)
{
	__m256 acc = _mm256_setzero_ps();
	size_t i;
	int ib;
	for (i = 0; i < nb; i++, b += 98, x++) {
		const uint8_t *qs = b + 2;
		__m256i s = _mm256_setzero_si256();
		for (ib = 0; ib < 8; ib++, qs += 8) {
			uint32_t a = rd32(b + 66 + 4 * ib);
			__m256i u = _mm256_set_epi32((int)iq3xxs_grid[qs[7]], (int)iq3xxs_grid[qs[6]],
			    (int)iq3xxs_grid[qs[5]], (int)iq3xxs_grid[qs[4]], (int)iq3xxs_grid[qs[3]],
			    (int)iq3xxs_grid[qs[2]], (int)iq3xxs_grid[qs[1]], (int)iq3xxs_grid[qs[0]]);
			__m256i sg = grid4(sgn7[a & 127], sgn7[(a >> 7) & 127], sgn7[(a >> 14) & 127],
			    sgn7[(a >> 21) & 127]);
			s = _mm256_add_epi32(s, dot32s(u, _mm256_loadu_si256((const __m256i *)(x->qs + 32 * ib)),
			    sg, 2 * (int)(a >> 28) + 1));
		}
		acc = _mm256_fmadd_ps(_mm256_set1_ps(h16f(b) * x->d * 0.25f), _mm256_cvtepi32_ps(s), acc);
	}
	return hsumf(acc);
}

static float
dot_iq3_s(const uint8_t *b, const q8k *x, size_t nb)
{
	__m256 acc = _mm256_setzero_ps();
	size_t i;
	int ib;
	for (i = 0; i < nb; i++, b += 110, x++) {
		const uint8_t *qs = b + 2, *qh = b + 66, *sgb = b + 74, *scs = b + 106;
		__m256i s = _mm256_setzero_si256();
		for (ib = 0; ib < 8; ib++, qs += 8, sgb += 4) {
			const uint32_t h = qh[ib];
			const int sc = ib & 1 ? scs[ib / 2] >> 4 : scs[ib / 2] & 15;
			__m256i u = _mm256_set_epi32(
			    (int)iq3s_grid[qs[7] | ((h << 1) & 256)], (int)iq3s_grid[qs[6] | ((h << 2) & 256)],
			    (int)iq3s_grid[qs[5] | ((h << 3) & 256)], (int)iq3s_grid[qs[4] | ((h << 4) & 256)],
			    (int)iq3s_grid[qs[3] | ((h << 5) & 256)], (int)iq3s_grid[qs[2] | ((h << 6) & 256)],
			    (int)iq3s_grid[qs[1] | ((h << 7) & 256)], (int)iq3s_grid[qs[0] | ((h << 8) & 256)]);
			__m256i sg = grid4(sgn8[sgb[0]], sgn8[sgb[1]], sgn8[sgb[2]], sgn8[sgb[3]]);
			s = _mm256_add_epi32(s, dot32s(u, _mm256_loadu_si256((const __m256i *)(x->qs + 32 * ib)),
			    sg, 2 * sc + 1));
		}
		acc = _mm256_fmadd_ps(_mm256_set1_ps(h16f(b) * x->d), _mm256_cvtepi32_ps(s), acc);
	}
	return hsumf(acc);
}

static float
dot_iq1_s(const uint8_t *b, const q8k *x, size_t nb)
{
	const __m256i one = _mm256_set1_epi8(1);
	__m256 acc = _mm256_setzero_ps();
	float sum = 0;
	size_t i;
	int ib;
	for (i = 0; i < nb; i++, b += 50, x++) {
		const uint8_t *qs = b + 2;
		const float dd = h16f(b) * x->d;
		__m256i s = _mm256_setzero_si256();
		float bs = 0;
		for (ib = 0; ib < 8; ib++, qs += 4) {
			uint16_t qh = (uint16_t)(b[34 + 2 * ib] | b[35 + 2 * ib] << 8);
			int sc = 2 * ((qh >> 12) & 7) + 1;
			/* grid values in {-1, 0, 1}: (g + 1) is unsigned; the -1
			 * comes back with the delta through sum(x) */
			__m256i u = _mm256_add_epi8(grid4(iq1s_grid[qs[0] | ((qh << 8) & 0x700)],
			    iq1s_grid[qs[1] | ((qh << 5) & 0x700)], iq1s_grid[qs[2] | ((qh << 2) & 0x700)],
			    iq1s_grid[qs[3] | ((qh >> 1) & 0x700)]), one);
			s = _mm256_add_epi32(s, _mm256_madd_epi16(_mm256_maddubs_epi16(u,
			    _mm256_loadu_si256((const __m256i *)(x->qs + 32 * ib))), _mm256_set1_epi16((short)sc)));
			bs += (float)sc * ((qh & 0x8000 ? -0.125f : 0.125f) - 1.0f) *
			    (float)(x->bsums[2 * ib] + x->bsums[2 * ib + 1]);
		}
		acc = _mm256_fmadd_ps(_mm256_set1_ps(dd), _mm256_cvtepi32_ps(s), acc);
		sum += dd * bs;
	}
	return hsumf(acc) + sum;
}

static float
dot_iq4_xs(const uint8_t *b, const q8k *x, size_t nb)
{
	const __m128i tab = _mm_loadu_si128((const __m128i *)kvalues_iq4nl);
	const __m256i tbl = _mm256_broadcastsi128_si256(tab), m4 = _mm256_set1_epi8(15);
	__m256 acc = _mm256_setzero_ps();
	size_t i;
	int ib;
	for (i = 0; i < nb; i++, b += 136, x++) {
		uint16_t sh = (uint16_t)(b[2] | b[3] << 8);
		const uint8_t *sl = b + 4, *qs = b + 8;
		__m256i s = _mm256_setzero_si256();
		for (ib = 0; ib < 8; ib++, qs += 16) {
			int ls = ((sl[ib / 2] >> 4 * (ib % 2)) & 15) | (((sh >> 2 * ib) & 3) << 4);
			__m128i q = _mm_loadu_si128((const __m128i *)qs);
			/* elements 0-15 low nibbles, 16-31 high nibbles */
			__m256i n = _mm256_and_si256(_mm256_setr_m128i(q, _mm_srli_epi16(q, 4)), m4);
			__m256i w = _mm256_shuffle_epi8(tbl, n);
			__m256i xv = _mm256_loadu_si256((const __m256i *)(x->qs + 32 * ib));
			s = _mm256_add_epi32(s, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_sign_epi8(w, w),
			    _mm256_sign_epi8(xv, w)), _mm256_set1_epi16((short)(ls - 32))));
		}
		acc = _mm256_fmadd_ps(_mm256_set1_ps(h16f(b) * x->d), _mm256_cvtepi32_ps(s), acc);
	}
	return hsumf(acc);
}

float
kq_dot(uint32_t t, const void *row, const q8k *x, size_t n)
{
	switch (t) {
	case GT_Q2_K: return dot_q2_k(row, x, n / 256);
	case GT_Q3_K: return dot_q3_k(row, x, n / 256);
	case GT_Q4_K: return dot_q45_k(row, x, n / 256, 0);
	case GT_Q5_K: return dot_q45_k(row, x, n / 256, 1);
	case GT_Q6_K: return dot_q6_k(row, x, n / 256);
	case GT_Q8_0: return dot_q8_0(row, x, n / 256);
	case GT_IQ2_XXS: return dot_iq2_xxs(row, x, n / 256);
	case GT_IQ2_XS: return dot_iq2_xs(row, x, n / 256);
	case GT_IQ2_S: return dot_iq2_s(row, x, n / 256);
	case GT_IQ3_XXS: return dot_iq3_xxs(row, x, n / 256);
	case GT_IQ3_S: return dot_iq3_s(row, x, n / 256);
	case GT_IQ1_S: return dot_iq1_s(row, x, n / 256);
	case GT_IQ4_XS: return dot_iq4_xs(row, x, n / 256);
	default: return kq_dot_scalar(t, row, x, n);
	}
}

/*
 * Many tokens against one row: each 256-weight block is decoded once
 * into unsigned weights u_j (8 x 32 bytes), per-32 scale vectors s_j
 * (i16, already arranged per 16-group as madd expects) and a min /
 * offset term m (16 x i16 against bsums), so for every K-quant type
 *     dot = dx * (A * sum_j madd(maddubs(u_j, x_j), s_j) + C * m.bsums)
 */
typedef struct {
	__m256i u[8], s[8], m;
	__m256i sg[8];          /* i-quants: +-1 per weight, applied to x */
	int sgn;
	float A, C;
} kblk;

static __m256i
s16pair(int lo, int hi)
{
	return _mm256_setr_m128i(_mm_set1_epi16((short)lo), _mm_set1_epi16((short)hi));
}

/* i-quant blocks for kq_dotT: magnitudes u_j, signs sg_j, scales s_j
 * (per 16), the block scale A, and for IQ1_S the bsums term C*m */
static int
dec_block_iq(uint32_t t, const uint8_t *b, kblk *k)
{
	int ib, l;
	k->sgn = 1;
	k->m = _mm256_setzero_si256();
	k->C = 0;
	switch (t) {
	case GT_IQ2_XXS:
		for (ib = 0; ib < 8; ib++) {
			const uint8_t *q = b + 2 + 8 * ib;
			uint32_t a1 = rd32(q + 4);
			int sc = 2 * (int)(a1 >> 28) + 1;
			k->u[ib] = grid4(iq2xxs_grid[q[0]], iq2xxs_grid[q[1]], iq2xxs_grid[q[2]], iq2xxs_grid[q[3]]);
			k->sg[ib] = grid4(sgn7[a1 & 127], sgn7[(a1 >> 7) & 127], sgn7[(a1 >> 14) & 127],
			    sgn7[(a1 >> 21) & 127]);
			k->s[ib] = s16pair(sc, sc);
		}
		k->A = h16f(b) * 0.125f;
		return 1;
	case GT_IQ2_XS:
		for (ib = 0; ib < 8; ib++) {
			const uint8_t *q = b + 2 + 8 * ib;
			uint16_t q0 = (uint16_t)(q[0] | q[1] << 8), q1 = (uint16_t)(q[2] | q[3] << 8);
			uint16_t q2 = (uint16_t)(q[4] | q[5] << 8), q3 = (uint16_t)(q[6] | q[7] << 8);
			uint8_t sc = b[66 + ib];
			k->u[ib] = grid4(iq2xs_grid[q0 & 511], iq2xs_grid[q1 & 511], iq2xs_grid[q2 & 511],
			    iq2xs_grid[q3 & 511]);
			k->sg[ib] = grid4(sgn7[q0 >> 9], sgn7[q1 >> 9], sgn7[q2 >> 9], sgn7[q3 >> 9]);
			k->s[ib] = s16pair(2 * (sc & 15) + 1, 2 * (sc >> 4) + 1);
		}
		k->A = h16f(b) * 0.125f;
		return 1;
	case GT_IQ2_S: {
		const uint8_t *qs = b + 2, *sgb = b + 34, *qh = b + 66, *scs = b + 74;
		for (ib = 0; ib < 8; ib++, qs += 4, sgb += 4) {
			uint8_t h = qh[ib], sc = scs[ib];
			k->u[ib] = grid4(iq2s_grid[qs[0] | ((h << 8) & 0x300)], iq2s_grid[qs[1] | ((h << 6) & 0x300)],
			    iq2s_grid[qs[2] | ((h << 4) & 0x300)], iq2s_grid[qs[3] | ((h << 2) & 0x300)]);
			k->sg[ib] = grid4(sgn8[sgb[0]], sgn8[sgb[1]], sgn8[sgb[2]], sgn8[sgb[3]]);
			k->s[ib] = s16pair(2 * (sc & 15) + 1, 2 * (sc >> 4) + 1);
		}
		k->A = h16f(b) * 0.125f;
		return 1;
	}
	case GT_IQ3_XXS: {
		const uint8_t *qs = b + 2;
		for (ib = 0; ib < 8; ib++, qs += 8) {
			uint32_t a = rd32(b + 66 + 4 * ib);
			int sc = 2 * (int)(a >> 28) + 1;
			k->u[ib] = _mm256_set_epi32((int)iq3xxs_grid[qs[7]], (int)iq3xxs_grid[qs[6]],
			    (int)iq3xxs_grid[qs[5]], (int)iq3xxs_grid[qs[4]], (int)iq3xxs_grid[qs[3]],
			    (int)iq3xxs_grid[qs[2]], (int)iq3xxs_grid[qs[1]], (int)iq3xxs_grid[qs[0]]);
			k->sg[ib] = grid4(sgn7[a & 127], sgn7[(a >> 7) & 127], sgn7[(a >> 14) & 127],
			    sgn7[(a >> 21) & 127]);
			k->s[ib] = s16pair(sc, sc);
		}
		k->A = h16f(b) * 0.25f;
		return 1;
	}
	case GT_IQ3_S: {
		const uint8_t *qs = b + 2, *qh = b + 66, *sgb = b + 74, *scs = b + 106;
		for (ib = 0; ib < 8; ib++, qs += 8, sgb += 4) {
			const uint32_t h = qh[ib];
			const int sc = 2 * (ib & 1 ? scs[ib / 2] >> 4 : scs[ib / 2] & 15) + 1;
			k->u[ib] = _mm256_set_epi32(
			    (int)iq3s_grid[qs[7] | ((h << 1) & 256)], (int)iq3s_grid[qs[6] | ((h << 2) & 256)],
			    (int)iq3s_grid[qs[5] | ((h << 3) & 256)], (int)iq3s_grid[qs[4] | ((h << 4) & 256)],
			    (int)iq3s_grid[qs[3] | ((h << 5) & 256)], (int)iq3s_grid[qs[2] | ((h << 6) & 256)],
			    (int)iq3s_grid[qs[1] | ((h << 7) & 256)], (int)iq3s_grid[qs[0] | ((h << 8) & 256)]);
			k->sg[ib] = grid4(sgn8[sgb[0]], sgn8[sgb[1]], sgn8[sgb[2]], sgn8[sgb[3]]);
			k->s[ib] = s16pair(sc, sc);
		}
		k->A = h16f(b);
		return 1;
	}
	case GT_IQ1_S: {
		/* w = dl*(g + delta), g in {-1,0,1}: dl*(g+1) + dl*(delta-1);
		 * (delta - 1)*8 is -7 or -9 */
		const uint8_t *qs = b + 2;
		short m16[16];
		k->sgn = 0;
		for (ib = 0; ib < 8; ib++, qs += 4) {
			uint16_t qh = (uint16_t)(b[34 + 2 * ib] | b[35 + 2 * ib] << 8);
			int sc = 2 * ((qh >> 12) & 7) + 1;
			k->u[ib] = _mm256_add_epi8(grid4(iq1s_grid[qs[0] | ((qh << 8) & 0x700)],
			    iq1s_grid[qs[1] | ((qh << 5) & 0x700)], iq1s_grid[qs[2] | ((qh << 2) & 0x700)],
			    iq1s_grid[qs[3] | ((qh >> 1) & 0x700)]), _mm256_set1_epi8(1));
			k->s[ib] = s16pair(sc, sc);
			m16[2 * ib] = m16[2 * ib + 1] = (short)(sc * (qh & 0x8000 ? -9 : -7));
		}
		k->m = _mm256_loadu_si256((const __m256i *)m16);
		k->A = h16f(b);
		k->C = h16f(b) * 0.125f;
		return 1;
	}
	case GT_IQ4_XS: {
		const __m256i tbl = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)kvalues_iq4nl));
		uint16_t sh = (uint16_t)(b[2] | b[3] << 8);
		const uint8_t *sl = b + 4, *qs = b + 8;
		for (ib = 0; ib < 8; ib++, qs += 16) {
			int ls = ((sl[ib / 2] >> 4 * (ib % 2)) & 15) | (((sh >> 2 * ib) & 3) << 4);
			__m128i q = _mm_loadu_si128((const __m128i *)qs);
			__m256i n = _mm256_and_si256(_mm256_setr_m128i(q, _mm_srli_epi16(q, 4)), _mm256_set1_epi8(15));
			__m256i w = _mm256_shuffle_epi8(tbl, n);
			k->u[ib] = _mm256_sign_epi8(w, w);
			k->sg[ib] = w;
			k->s[ib] = s16pair(ls - 32, ls - 32);
		}
		k->A = h16f(b);
		return 1;
	}
	}
	(void)l;
	return 0;
}

static void
dec_block(uint32_t t, const uint8_t *b, kblk *k)
{
	const __m256i m3 = _mm256_set1_epi8(3), m1 = _mm256_set1_epi8(1), m4 = _mm256_set1_epi8(15);
	int j, c;

	switch (t) {
	case GT_Q2_K: {
		__m128i sc8 = _mm_loadu_si128((const __m128i *)b);
		__m256i s16 = _mm256_cvtepu8_epi16(_mm_and_si128(sc8, _mm_set1_epi8(15)));
		k->m = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(sc8, 4), _mm_set1_epi8(15)));
		for (c = 0; c < 2; c++) {
			__m256i qv = _mm256_loadu_si256((const __m256i *)(b + 16 + 32 * c));
			__m256i sb = _mm256_broadcastsi128_si256(c ? _mm256_extracti128_si256(s16, 1)
			    : _mm256_castsi256_si128(s16));
			for (j = 0; j < 4; j++) {
				k->u[4 * c + j] = _mm256_and_si256(_mm256_srli_epi16(qv, 2 * j), m3);
				k->s[4 * c + j] = _mm256_shuffle_epi8(sb, pair_mask(j));
			}
		}
		k->A = h16f(b + 80);
		k->C = -h16f(b + 82);
		break;
	}
	case GT_Q3_K: {
		__m256i s16 = q3_scales16(b + 96);
		__m256i hv = _mm256_loadu_si256((const __m256i *)b);
		for (c = 0; c < 2; c++) {
			__m256i qv = _mm256_loadu_si256((const __m256i *)(b + 32 + 32 * c));
			__m256i sb = _mm256_broadcastsi128_si256(c ? _mm256_extracti128_si256(s16, 1)
			    : _mm256_castsi256_si128(s16));
			for (j = 0; j < 4; j++) {
				__m256i u = _mm256_and_si256(_mm256_srli_epi16(qv, 2 * j), m3);
				__m256i h = _mm256_and_si256(_mm256_srli_epi16(hv, 4 * c + j), m1);
				k->u[4 * c + j] = _mm256_or_si256(u, _mm256_slli_epi16(h, 2));
				k->s[4 * c + j] = _mm256_shuffle_epi8(sb, pair_mask(j));
			}
		}
		k->m = s16;
		k->A = h16f(b + 108);
		k->C = -4 * k->A;
		break;
	}
	case GT_Q4_K:
	case GT_Q5_K: {
		const int five = t == GT_Q5_K;
		const uint8_t *q = b + (five ? 48 : 16);
		const uint32_t k6 = 0x3f3f3f3fu, k4 = 0x0f0f0f0fu, k2 = 0x03030303u;
		uint32_t a0 = rd32(b + 4), a1 = rd32(b + 8), a2 = rd32(b + 12);
		__m128i sm = _mm_setr_epi32((int)(a0 & k6),
		    (int)((a2 & k4) | (((a0 >> 6) & k2) << 4)),
		    (int)(a1 & k6),
		    (int)(((a2 >> 4) & k4) | (((a1 >> 6) & k2) << 4)));
		__m128i mn8 = _mm_unpackhi_epi64(sm, sm);
		__m256i sb = _mm256_broadcastsi128_si256(_mm256_castsi256_si128(_mm256_cvtepu8_epi16(sm)));
		__m256i hv = five ? _mm256_loadu_si256((const __m256i *)(b + 16)) : _mm256_setzero_si256();
		k->m = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(mn8, mn8));
		for (c = 0; c < 4; c++) {
			__m256i qv = _mm256_loadu_si256((const __m256i *)(q + 32 * c));
			__m256i u0 = _mm256_and_si256(qv, m4);
			__m256i u1 = _mm256_and_si256(_mm256_srli_epi16(qv, 4), m4);
			if (five) {
				u0 = _mm256_or_si256(u0, _mm256_slli_epi16(
				    _mm256_and_si256(_mm256_srli_epi16(hv, 2 * c), m1), 4));
				u1 = _mm256_or_si256(u1, _mm256_slli_epi16(
				    _mm256_and_si256(_mm256_srli_epi16(hv, 2 * c + 1), m1), 4));
			}
			k->u[2 * c] = u0;
			k->u[2 * c + 1] = u1;
			k->s[2 * c] = _mm256_shuffle_epi8(sb, lane_mask(2 * c));
			k->s[2 * c + 1] = _mm256_shuffle_epi8(sb, lane_mask(2 * c + 1));
		}
		k->A = h16f(b);
		k->C = -h16f(b + 2);
		break;
	}
	case GT_Q6_K: {
		__m256i s16 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(b + 192)));
		for (c = 0; c < 2; c++) {
			__m256i l0 = _mm256_loadu_si256((const __m256i *)(b + 64 * c));
			__m256i l1 = _mm256_loadu_si256((const __m256i *)(b + 64 * c + 32));
			__m256i qh = _mm256_loadu_si256((const __m256i *)(b + 128 + 32 * c));
			__m256i sb = _mm256_broadcastsi128_si256(c ? _mm256_extracti128_si256(s16, 1)
			    : _mm256_castsi256_si128(s16));
			__m256i u[4];
			u[0] = _mm256_and_si256(l0, m4);
			u[1] = _mm256_and_si256(l1, m4);
			u[2] = _mm256_and_si256(_mm256_srli_epi16(l0, 4), m4);
			u[3] = _mm256_and_si256(_mm256_srli_epi16(l1, 4), m4);
			for (j = 0; j < 4; j++) {
				__m256i h = _mm256_and_si256(_mm256_srli_epi16(qh, 2 * j), m3);
				k->u[4 * c + j] = _mm256_or_si256(u[j], _mm256_slli_epi16(h, 4));
				k->s[4 * c + j] = _mm256_shuffle_epi8(sb, pair_mask(j));
			}
		}
		k->m = s16;
		k->A = h16f(b + 208);
		k->C = -32 * k->A;
		break;
	}
	}
}

int
kq_dotT(uint32_t t, const void *row, const q8k *x, size_t xs, int T, size_t n,
    float *out, size_t ostride)
{
	__m256 acc[64];
	kblk k;
	size_t bb = ggml_type_bytes(t), i;
	const uint8_t *b = row;
	int tt, j;

	const int iq = t == GT_IQ2_XXS || t == GT_IQ2_XS || t == GT_IQ2_S || t == GT_IQ3_XXS ||
	    t == GT_IQ3_S || t == GT_IQ1_S || t == GT_IQ4_XS;

	if (T > 64 || (!iq && t != GT_Q2_K && t != GT_Q3_K && t != GT_Q4_K && t != GT_Q5_K &&
	    t != GT_Q6_K))
		return 0;
	k.sgn = 0;
	for (tt = 0; tt < T; tt++)
		acc[tt] = _mm256_setzero_ps();
	for (i = 0; i < n / 256; i++, b += bb) {
		if (iq)
			dec_block_iq(t, b, &k);
		else
			dec_block(t, b, &k);
		for (tt = 0; tt < T; tt++) {
			const q8k *xb = x + (size_t)tt * xs + i;
			__m256i isum = _mm256_setzero_si256();
			__m256i msum = _mm256_madd_epi16(k.m, _mm256_loadu_si256((const __m256i *)xb->bsums));
			for (j = 0; j < 8; j++) {
				__m256i xv = _mm256_loadu_si256((const __m256i *)(xb->qs + 32 * j));
				__m256i p = _mm256_maddubs_epi16(k.u[j], k.sgn ? _mm256_sign_epi8(xv, k.sg[j]) : xv);
				isum = _mm256_add_epi32(isum, _mm256_madd_epi16(p, k.s[j]));
			}
			acc[tt] = _mm256_fmadd_ps(_mm256_set1_ps(k.A * xb->d), _mm256_cvtepi32_ps(isum), acc[tt]);
			acc[tt] = _mm256_fmadd_ps(_mm256_set1_ps(k.C * xb->d), _mm256_cvtepi32_ps(msum), acc[tt]);
		}
	}
	for (tt = 0; tt < T; tt++)
		out[(size_t)tt * ostride] = hsumf(acc[tt]);
	return 1;
}

const char *
kq_isa(void)
{
#if defined(__F16C__)
	return "avx2+f16c";
#else
	return "avx2";
#endif
}

#else

float
kq_dot(uint32_t t, const void *row, const q8k *x, size_t n)
{
	return kq_dot_scalar(t, row, x, n);
}

const char *
kq_isa(void)
{
	return "scalar";
}

#endif
