/* Reference (scalar, bit-exact with ggml) dequantization of the block
 * formats present in the model, and the f32 row dot built on it. */
#ifndef Q2K_QUANT_H
#define Q2K_QUANT_H

#include <stddef.h>
#include <stdint.h>

float fp16_to_f32(uint16_t h);

/* dequantize n elements (n % block == 0) of a row of type t */
void dequant_row(uint32_t t, const void *src, float *dst, size_t n);

/* dot(row, x) over n elements: dequantize block by block, then f32 dot */
float dot_row(uint32_t t, const void *row, const float *x, size_t n);

#endif
