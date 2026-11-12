#include "kq.h"

#include <math.h>
#include <string.h>

#include "gguf.h"
#include "quant.h"

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
		break;
	}
	(void)i;
	return 0;
}

/* ---------------------------------------------------------------- */
/* AVX2                                                             */

#if defined(__AVX2__)

static int
hsum8(__m256i v)
{
	__m128i s = _mm_add_epi32(_mm256_castsi256_si128(v),
	    _mm256_extracti128_si256(v, 1));
	s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4e));
	s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xb1));
	return _mm_cvtsi128_si32(s);
}

/* byte masks picking i16 lanes (2j, 2j+1) of a broadcast 8 x i16 into
 * [lane 2j x8 | lane 2j+1 x8] */
static __m256i
pair_mask(int j)
{
	int a = 4 * j, b = 4 * j + 2;
	return _mm256_setr_epi8(
	    a, a + 1, a, a + 1, a, a + 1, a, a + 1,
	    a, a + 1, a, a + 1, a, a + 1, a, a + 1,
	    b, b + 1, b, b + 1, b, b + 1, b, b + 1,
	    b, b + 1, b, b + 1, b, b + 1, b, b + 1);
}

static float
dot_q2_k(const uint8_t *b, const q8k *x, size_t nb)
{
	const __m256i m3 = _mm256_set1_epi8(3);
	const __m128i m4 = _mm_set1_epi8(15);
	__m256i pm[4];
	float sum = 0;
	size_t i;
	int j, c;

	for (j = 0; j < 4; j++)
		pm[j] = pair_mask(j);
	for (i = 0; i < nb; i++, b += 84, x++) {
		__m128i sc8 = _mm_loadu_si128((const __m128i *)b);
		__m256i mins = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(sc8, 4), m4));
		__m256i s16 = _mm256_cvtepu8_epi16(_mm_and_si128(sc8, m4));
		__m256i bs = _mm256_loadu_si256((const __m256i *)x->bsums);
		int summ = hsum8(_mm256_madd_epi16(mins, bs));
		__m256i acc = _mm256_setzero_si256();
		const int8_t *xq = x->qs;

		for (c = 0; c < 2; c++) {
			__m256i qv = _mm256_loadu_si256((const __m256i *)(b + 16 + 32 * c));
			__m256i sb = _mm256_broadcastsi128_si256(c ? _mm256_extracti128_si256(s16, 1)
			    : _mm256_castsi256_si128(s16));
			for (j = 0; j < 4; j++, xq += 32) {
				__m256i u = _mm256_and_si256(_mm256_srli_epi16(qv, 2 * j), m3);
				__m256i p = _mm256_maddubs_epi16(u, _mm256_loadu_si256((const __m256i *)xq));
				acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p, _mm256_shuffle_epi8(sb, pm[j])));
			}
		}
		sum += x->d * (h16(b + 80) * (float)hsum8(acc) - h16(b + 82) * (float)summ);
	}
	return sum;
}

static float
dot_q3_k(const uint8_t *b, const q8k *x, size_t nb)
{
	const __m256i m3 = _mm256_set1_epi8(3), m1 = _mm256_set1_epi8(1);
	__m256i pm[4];
	float sum = 0;
	size_t i;
	int j, c;

	for (j = 0; j < 4; j++)
		pm[j] = pair_mask(j);
	for (i = 0; i < nb; i++, b += 110, x++) {
		int8_t sc[16];
		__m256i s16, acc = _mm256_setzero_si256();
		__m256i hv = _mm256_loadu_si256((const __m256i *)b);
		const int8_t *xq = x->qs;
		int offs;

		q3_scales(b + 96, sc);
		s16 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)sc));
		offs = hsum8(_mm256_madd_epi16(s16,
		    _mm256_loadu_si256((const __m256i *)x->bsums)));
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
				acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p, _mm256_shuffle_epi8(sb, pm[j])));
			}
		}
		sum += x->d * h16(b + 108) * (float)(hsum8(acc) - 4 * offs);
	}
	return sum;
}

static float
dot_q45_k(const uint8_t *b, const q8k *x, size_t nb, int five)
{
	const __m256i m4 = _mm256_set1_epi8(15), m1 = _mm256_set1_epi8(1);
	const size_t bsz = five ? 176 : 144;
	float sum = 0;
	size_t i;
	int c, k;

	for (i = 0; i < nb; i++, b += bsz, x++) {
		uint8_t sc[8], mn[8];
		const uint8_t *q = b + (five ? 48 : 16);
		__m256i acc = _mm256_setzero_si256();
		__m256i hv = five ? _mm256_loadu_si256((const __m256i *)(b + 16)) : acc;
		int summ = 0;

		k4_scales(b + 4, sc, mn);
		for (k = 0; k < 8; k++)
			summ += mn[k] * (x->bsums[2 * k] + x->bsums[2 * k + 1]);
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
			acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p0, _mm256_set1_epi16(sc[2 * c])));
			acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p1, _mm256_set1_epi16(sc[2 * c + 1])));
		}
		sum += x->d * (h16(b) * (float)hsum8(acc) - h16(b + 2) * (float)summ);
	}
	return sum;
}

static float
dot_q6_k(const uint8_t *b, const q8k *x, size_t nb)
{
	const __m256i m4 = _mm256_set1_epi8(15), m3 = _mm256_set1_epi8(3);
	__m256i pm[4];
	float sum = 0;
	size_t i;
	int j, c;

	for (j = 0; j < 4; j++)
		pm[j] = pair_mask(j);
	for (i = 0; i < nb; i++, b += 210, x++) {
		__m256i s16 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(b + 192)));
		int offs = hsum8(_mm256_madd_epi16(s16,
		    _mm256_loadu_si256((const __m256i *)x->bsums)));
		__m256i acc = _mm256_setzero_si256();
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
				acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p, _mm256_shuffle_epi8(sb, pm[j])));
			}
		}
		sum += x->d * h16(b + 208) * (float)(hsum8(acc) - 32 * offs);
	}
	return sum;
}

static float
dot_q8_0(const uint8_t *b, const q8k *x, size_t nb)
{
	const __m256i ones = _mm256_set1_epi16(1);
	float sum = 0;
	size_t i;
	int k;

	for (i = 0; i < nb; i++, x++) {
		__m256 fs = _mm256_setzero_ps();
		for (k = 0; k < 8; k++, b += 34) {
			__m256i w = _mm256_loadu_si256((const __m256i *)(b + 2));
			__m256i xv = _mm256_loadu_si256((const __m256i *)(x->qs + 32 * k));
			__m256i p = _mm256_maddubs_epi16(_mm256_sign_epi8(w, w),
			    _mm256_sign_epi8(xv, w));
			__m256 pf = _mm256_cvtepi32_ps(_mm256_madd_epi16(p, ones));
			fs = _mm256_fmadd_ps(_mm256_set1_ps(h16(b)), pf, fs);
		}
		{
			__m128 s = _mm_add_ps(_mm256_castps256_ps128(fs), _mm256_extractf128_ps(fs, 1));
			s = _mm_add_ps(s, _mm_movehl_ps(s, s));
			s = _mm_add_ss(s, _mm_movehdup_ps(s));
			sum += x->d * _mm_cvtss_f32(s);
		}
	}
	return sum;
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
	default: return kq_dot_scalar(t, row, x, n);
	}
}

const char *
kq_isa(void)
{
	return "avx2";
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
