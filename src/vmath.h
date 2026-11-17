/* Vectorized elementwise helpers (AVX2+FMA with scalar fallbacks). */
#ifndef Q2K_VMATH_H
#define Q2K_VMATH_H

#include <stddef.h>
#include <stdint.h>

/* exp with |error| <= ~2 ulp on [-87.3, 88.7]; see vmath.c */
void v_exp(float *y, const float *x, size_t n);
/* x[i] = x[i] * sigmoid(x[i]) */
void v_silu(float *x, size_t n);
/* g[i] = silu(g[i]) * u[i] */
void v_swiglu(float *g, const float *u, size_t n);
/* y[i] *= silu(z[i]) */
void v_mul_silu(float *y, const float *z, size_t n);
float v_dot(const float *a, const float *b, size_t n);
/* o = x / rms(x) * w */
void v_rmsnorm(float *o, const float *x, const float *w, size_t n, float eps);
/* y[i] += a * x[i] */
void v_axpy(float *y, float a, const float *x, size_t n);
/* the KV cache of the dense archs is f16 (as llama.cpp's default) */
void v_to_f16(uint16_t *d, const float *s, size_t n);
/* out[h] = dot(q + h*qs, k) for h < nq, k f16 (fast for nq == 8, n % 8 == 0) */
void v_dotn(float *out, const float *q, size_t qs, int nq, const uint16_t *k, size_t n);
/* o + h*os += sum_t w[h*ws + t] * v[t*vs] for h < nq, t < nt; f16 rows of n */
void v_wsumn(float *o, size_t os, const float *w, size_t ws, int nq, const uint16_t *v, size_t vs,
    int nt, size_t n);
/* dst = src with streaming stores (dst 32-byte aligned, n % 8 == 0):
 * no read-for-ownership of dst; ends with a store fence */
void v_copy_nt(float *dst, const float *src, size_t n);

#endif
