#include "vmath.h"

#include <math.h>

#include <immintrin.h>

/*
 * exp(x) = 2^n * P(r), n = round(x / ln2), r = x - n*ln2 (Cody-Waite,
 * two-part ln2) in [-ln2/2, ln2/2].  P is the degree-6 minimax fit
 * from ucas ("approx exp(r) in -0.3466..0.3466 degree 6"): 0.69 ulp in
 * f32 Horner/FMA over the reduced range.
 */
static __m256
vexp8(__m256 x)
{
	const __m256 hi = _mm256_set1_ps(88.7f), lo = _mm256_set1_ps(-87.3f);
	__m256 n, r, p;
	__m256i e;

	x = _mm256_min_ps(_mm256_max_ps(x, lo), hi);
	n = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504f)),
	    _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693145751953125f), x);
	r = _mm256_fnmadd_ps(n, _mm256_set1_ps(1.428606765330187e-06f), r);
	p = _mm256_set1_ps(0.00138345419f);
	p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.00836876035f));
	p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.0416681878f));
	p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.166665196f));
	p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.49999994f));
	p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
	p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
	/* 2^n; n in [-126, 128] after the clamp: split to stay normal */
	e = _mm256_cvtps_epi32(n);
	{
		__m256i h = _mm256_srai_epi32(e, 1);
		__m256 s1 = _mm256_castsi256_ps(_mm256_slli_epi32(
		    _mm256_add_epi32(h, _mm256_set1_epi32(127)), 23));
		__m256 s2 = _mm256_castsi256_ps(_mm256_slli_epi32(
		    _mm256_add_epi32(_mm256_sub_epi32(e, h), _mm256_set1_epi32(127)), 23));
		return _mm256_mul_ps(_mm256_mul_ps(p, s1), s2);
	}
}

static __m256
vsilu8(__m256 x)
{
	__m256 d = _mm256_add_ps(_mm256_set1_ps(1.0f),
	    vexp8(_mm256_sub_ps(_mm256_setzero_ps(), x)));
	return _mm256_div_ps(x, d);
}

static float
hsum(__m256 v)
{
	__m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
	s = _mm_add_ps(s, _mm_movehl_ps(s, s));
	s = _mm_add_ss(s, _mm_movehdup_ps(s));
	return _mm_cvtss_f32(s);
}


static float
silu1(float x)
{
	return x / (1.0f + expf(-x));
}

void
v_exp(float *y, const float *x, size_t n)
{
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
		_mm256_storeu_ps(y + i, vexp8(_mm256_loadu_ps(x + i)));
	if (i < n) {
		float a[8] = { 0 }, b[8];
		size_t k;
		for (k = 0; k < n - i; k++)
			a[k] = x[i + k];
		_mm256_storeu_ps(b, vexp8(_mm256_loadu_ps(a)));
		for (k = 0; k < n - i; k++)
			y[i + k] = b[k];
		return;
	}
	for (; i < n; i++)
		y[i] = expf(x[i]);
}

void
v_silu(float *x, size_t n)
{
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
		_mm256_storeu_ps(x + i, vsilu8(_mm256_loadu_ps(x + i)));
	for (; i < n; i++)
		x[i] = silu1(x[i]);
}

void
v_swiglu(float *g, const float *u, size_t n)
{
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
		_mm256_storeu_ps(g + i, _mm256_mul_ps(vsilu8(_mm256_loadu_ps(g + i)),
		    _mm256_loadu_ps(u + i)));
	for (; i < n; i++)
		g[i] = silu1(g[i]) * u[i];
}

void
v_mul_silu(float *y, const float *z, size_t n)
{
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
		_mm256_storeu_ps(y + i, _mm256_mul_ps(_mm256_loadu_ps(y + i),
		    vsilu8(_mm256_loadu_ps(z + i))));
	for (; i < n; i++)
		y[i] *= silu1(z[i]);
}

float
v_dot(const float *a, const float *b, size_t n)
{
	size_t i = 0;
	float s = 0;
	__m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
	for (; i + 32 <= n; i += 32) {
		a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), a0);
		a1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), a1);
		a2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), a2);
		a3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), a3);
	}
	for (; i + 8 <= n; i += 8)
		a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), a0);
	s = hsum(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
	for (; i < n; i++)
		s += a[i] * b[i];
	return s;
}

void
v_rmsnorm(float *o, const float *x, const float *w, size_t n, float eps)
{
	float sc = 1.0f / sqrtf(v_dot(x, x, n) / (float)n + eps);
	size_t i = 0;
	__m256 s = _mm256_set1_ps(sc);
	for (; i + 8 <= n; i += 8)
		_mm256_storeu_ps(o + i, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x + i), s),
		    _mm256_loadu_ps(w + i)));
	for (; i < n; i++)
		o[i] = x[i] * sc * w[i];
}

void
v_axpy(float *y, float a, const float *x, size_t n)
{
	size_t i = 0;
	__m256 av = _mm256_set1_ps(a);
	for (; i + 8 <= n; i += 8)
		_mm256_storeu_ps(y + i, _mm256_fmadd_ps(av, _mm256_loadu_ps(x + i),
		    _mm256_loadu_ps(y + i)));
	for (; i < n; i++)
		y[i] += a * x[i];
}

void
v_to_f16(uint16_t *d, const float *s, size_t n)
{
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
		_mm_storeu_si128((__m128i *)(d + i),
		    _mm256_cvtps_ph(_mm256_loadu_ps(s + i), _MM_FROUND_TO_NEAREST_INT));
	for (; i < n; i++)
		d[i] = _cvtss_sh(s[i], _MM_FROUND_TO_NEAREST_INT);
}

static void
f16_row(float *d, const uint16_t *s, size_t n)
{
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
		_mm256_storeu_ps(d + i, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(s + i))));
	for (; i < n; i++)
		d[i] = _cvtsh_ss(s[i]);
}

#define LDH(p) _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(p)))

/* out[h] = dot(q + h*qs, k) for h < nq, k f16: k is loaded once for all
 * the heads; 8 heads reduce together (one hadd tree, no per-dot hsum) */
void
v_dotn(float *out, const float *q, size_t qs, int nq, const uint16_t *k, size_t n)
{
	size_t i;
	int h;
	if (nq != 8 || n % 8 || n > 1024) {
		float kf[1024];
		size_t c;
		for (c = 0; c < n; c += 1024) {
			size_t m = n - c < 1024 ? n - c : 1024;
			f16_row(kf, k + c, m);
			for (h = 0; h < nq; h++)
				out[h] = (c ? out[h] : 0) + v_dot(q + (size_t)h * qs + c, kf, m);
		}
		return;
	}
	{
		__m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
		__m256 a4 = a0, a5 = a0, a6 = a0, a7 = a0, t0, t1, t2, t3;
		for (i = 0; i < n; i += 8) {
			const __m256 kv = LDH(k + i);
			a0 = _mm256_fmadd_ps(_mm256_loadu_ps(q + i), kv, a0);
			a1 = _mm256_fmadd_ps(_mm256_loadu_ps(q + qs + i), kv, a1);
			a2 = _mm256_fmadd_ps(_mm256_loadu_ps(q + 2 * qs + i), kv, a2);
			a3 = _mm256_fmadd_ps(_mm256_loadu_ps(q + 3 * qs + i), kv, a3);
			a4 = _mm256_fmadd_ps(_mm256_loadu_ps(q + 4 * qs + i), kv, a4);
			a5 = _mm256_fmadd_ps(_mm256_loadu_ps(q + 5 * qs + i), kv, a5);
			a6 = _mm256_fmadd_ps(_mm256_loadu_ps(q + 6 * qs + i), kv, a6);
			a7 = _mm256_fmadd_ps(_mm256_loadu_ps(q + 7 * qs + i), kv, a7);
		}
		t0 = _mm256_hadd_ps(a0, a1);
		t1 = _mm256_hadd_ps(a2, a3);
		t2 = _mm256_hadd_ps(a4, a5);
		t3 = _mm256_hadd_ps(a6, a7);
		t0 = _mm256_hadd_ps(t0, t1);    /* lanes: h0..h3 | h0..h3 */
		t2 = _mm256_hadd_ps(t2, t3);    /* lanes: h4..h7 | h4..h7 */
		_mm256_storeu_ps(out, _mm256_add_ps(_mm256_permute2f128_ps(t0, t2, 0x20),
		    _mm256_permute2f128_ps(t0, t2, 0x31)));
	}
}

/* o + h*os += sum_t w[h*ws + t] * v[t*vs] for h < nq, t < nt (f16 rows
 * of n): per block of 8 columns the nq outputs stay in registers over
 * all the rows */
void
v_wsumn(float *o, size_t os, const float *w, size_t ws, int nq, const uint16_t *v, size_t vs,
    int nt, size_t n)
{
	size_t d;
	int t, h;
	if (nq != 8 || n % 8 || n > 1024) {
		float vf[1024];
		size_t c;
		for (t = 0; t < nt; t++)
			for (c = 0; c < n; c += 1024) {
				size_t m = n - c < 1024 ? n - c : 1024;
				f16_row(vf, v + (size_t)t * vs + c, m);
				for (h = 0; h < nq; h++)
					v_axpy(o + (size_t)h * os + c, w[(size_t)h * ws + t], vf, m);
			}
		return;
	}
	for (d = 0; d < n; d += 8) {
		__m256 a0 = _mm256_loadu_ps(o + d), a1 = _mm256_loadu_ps(o + os + d);
		__m256 a2 = _mm256_loadu_ps(o + 2 * os + d), a3 = _mm256_loadu_ps(o + 3 * os + d);
		__m256 a4 = _mm256_loadu_ps(o + 4 * os + d), a5 = _mm256_loadu_ps(o + 5 * os + d);
		__m256 a6 = _mm256_loadu_ps(o + 6 * os + d), a7 = _mm256_loadu_ps(o + 7 * os + d);
		const uint16_t *vp = v + d;
		for (t = 0; t < nt; t++, vp += vs) {
			const __m256 vv = LDH(vp);
			a0 = _mm256_fmadd_ps(_mm256_broadcast_ss(w + t), vv, a0);
			a1 = _mm256_fmadd_ps(_mm256_broadcast_ss(w + ws + t), vv, a1);
			a2 = _mm256_fmadd_ps(_mm256_broadcast_ss(w + 2 * ws + t), vv, a2);
			a3 = _mm256_fmadd_ps(_mm256_broadcast_ss(w + 3 * ws + t), vv, a3);
			a4 = _mm256_fmadd_ps(_mm256_broadcast_ss(w + 4 * ws + t), vv, a4);
			a5 = _mm256_fmadd_ps(_mm256_broadcast_ss(w + 5 * ws + t), vv, a5);
			a6 = _mm256_fmadd_ps(_mm256_broadcast_ss(w + 6 * ws + t), vv, a6);
			a7 = _mm256_fmadd_ps(_mm256_broadcast_ss(w + 7 * ws + t), vv, a7);
		}
		_mm256_storeu_ps(o + d, a0);
		_mm256_storeu_ps(o + os + d, a1);
		_mm256_storeu_ps(o + 2 * os + d, a2);
		_mm256_storeu_ps(o + 3 * os + d, a3);
		_mm256_storeu_ps(o + 4 * os + d, a4);
		_mm256_storeu_ps(o + 5 * os + d, a5);
		_mm256_storeu_ps(o + 6 * os + d, a6);
		_mm256_storeu_ps(o + 7 * os + d, a7);
	}
}

void
v_copy_nt(float *dst, const float *src, size_t n)
{
	size_t i;
	for (i = 0; i < n; i += 8)
		_mm256_stream_ps(dst + i, _mm256_loadu_ps(src + i));
	_mm_sfence();
}
