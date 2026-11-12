/* Optimized CPU dot products: the activation is quantized once per
 * matvec to 8-bit blocks of 256 (q8k), then each weight row is dotted
 * with integer arithmetic.  Every K-quant weight w = a*u - b with u an
 * unsigned small integer, so
 *     sum(w*x) = dx * (a * sum(u*xq) - b * sum(xq))
 * and sum(xq) per 16-group is precomputed (bsums).  AVX2 when built
 * with it, a scalar integer path otherwise. */
#ifndef Q2K_KQ_H
#define Q2K_KQ_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
	float d;
	int8_t qs[256];
	int16_t bsums[16];
} q8k;

/* n % 256 == 0 */
void q8k_quantize(const float *x, q8k *y, size_t n);

/* dot of one weight row of type t (n elements) with quantized x */
float kq_dot(uint32_t t, const void *row, const q8k *x, size_t n);

/* scalar integer path, always built (for self-tests) */
float kq_dot_scalar(uint32_t t, const void *row, const q8k *x, size_t n);

const char *kq_isa(void);

#endif
