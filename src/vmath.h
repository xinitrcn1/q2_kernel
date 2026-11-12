/* Vectorized elementwise helpers (AVX2+FMA with scalar fallbacks). */
#ifndef Q2K_VMATH_H
#define Q2K_VMATH_H

#include <stddef.h>

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
/* dst = src with streaming stores (dst 32-byte aligned, n % 8 == 0):
 * no read-for-ownership of dst; ends with a store fence */
void v_copy_nt(float *dst, const float *src, size_t n);

#endif
