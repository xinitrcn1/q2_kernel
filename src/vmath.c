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
