/*
 * Lossless transcode of the codebook i-quants into fixed bit-plane
 * layouts (types X2, X3, X3S, X1 in gguf.h) that decode with shifts and
 * table shuffles instead of codebook gathers.  Each block is exploded
 * into its exact parts -- the f16 scale d, an integer multiplier per
 * 16/32 weights, a level and a sign per weight -- and repacked; the
 * other tensors and all metadata are copied as they are.
 *
 *   q2k -m model.gguf --transcode out.gguf
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gguf.h"
#include "iqtab.h"
#include "quant.h"

static uint32_t
u32le(const uint8_t *p)
{
	return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

/* a block, exploded */
typedef struct {
	uint8_t d[2];
	int mult[16];           /* per 16 weights */
	uint8_t lv[256], sg[256];
	int dneg[8];            /* IQ1_S: delta negative, per 32 */
} xb;

static int
lv2(int v)
{
	return v == 8 ? 0 : v == 25 ? 1 : 2;    /* {8, 25, 43} */
}

static void
grid8(uint64_t g, uint8_t *lv, uint8_t sgns, uint8_t *sg)
{
	int j;
	for (j = 0; j < 8; j++) {
		lv[j] = (uint8_t)lv2((int)((g >> (8 * j)) & 255));
		sg[j] = (uint8_t)((sgns >> j) & 1);
	}
}

static void
explode(uint32_t t, const uint8_t *b, xb *x)
{
	int ib, l, j;
	memset(x, 0, sizeof(*x));
	x->d[0] = b[0];
	x->d[1] = b[1];
	switch (t) {
	case GT_IQ2_XXS:
		for (ib = 0; ib < 8; ib++) {
			const uint8_t *q = b + 2 + 8 * ib;
			uint32_t a1 = u32le(q + 4);
			x->mult[2 * ib] = x->mult[2 * ib + 1] = 2 * (int)(a1 >> 28) + 1;
			for (l = 0; l < 4; l++)
				grid8(iq2xxs_grid[q[l]], x->lv + 32 * ib + 8 * l,
				    ksigns_iq2xs[(a1 >> 7 * l) & 127], x->sg + 32 * ib + 8 * l);
		}
		break;
	case GT_IQ2_XS:
		for (ib = 0; ib < 8; ib++) {
			uint8_t sc = b[66 + ib];
			x->mult[2 * ib] = 2 * (sc & 15) + 1;
			x->mult[2 * ib + 1] = 2 * (sc >> 4) + 1;
			for (l = 0; l < 4; l++) {
				uint16_t q = (uint16_t)(b[2 + 8 * ib + 2 * l] | b[3 + 8 * ib + 2 * l] << 8);
				grid8(iq2xs_grid[q & 511], x->lv + 32 * ib + 8 * l, ksigns_iq2xs[q >> 9],
				    x->sg + 32 * ib + 8 * l);
			}
		}
		break;
	case GT_IQ2_S: {
		const uint8_t *qs = b + 2, *sgb = b + 34, *qh = b + 66, *sc = b + 74;
		for (ib = 0; ib < 8; ib++, qs += 4, sgb += 4) {
			x->mult[2 * ib] = 2 * (sc[ib] & 15) + 1;
			x->mult[2 * ib + 1] = 2 * (sc[ib] >> 4) + 1;
			for (l = 0; l < 4; l++)
				grid8(iq2s_grid[qs[l] | ((qh[ib] << (8 - 2 * l)) & 0x300)],
				    x->lv + 32 * ib + 8 * l, sgb[l], x->sg + 32 * ib + 8 * l);
		}
		break;
	}
	case GT_IQ3_XXS: {
		const uint8_t *qs = b + 2;
		for (ib = 0; ib < 8; ib++, qs += 8) {
			uint32_t a = u32le(b + 66 + 4 * ib);
			x->mult[2 * ib] = x->mult[2 * ib + 1] = 2 * (int)(a >> 28) + 1;
			for (l = 0; l < 4; l++) {
				uint8_t sgns = ksigns_iq2xs[(a >> 7 * l) & 127];
				for (j = 0; j < 8; j++) {
					uint32_t g = iq3xxs_grid[qs[2 * l + j / 4]];
					int v = (int)((g >> (8 * (j % 4))) & 255);
					x->lv[32 * ib + 8 * l + j] = (uint8_t)((v - 4) / 8);
					x->sg[32 * ib + 8 * l + j] = (uint8_t)((sgns >> j) & 1);
				}
			}
		}
		break;
	}
	case GT_IQ3_S: {
		const uint8_t *qs = b + 2, *qh = b + 66, *sgb = b + 74, *sc = b + 106;
		for (ib = 0; ib < 8; ib++, qs += 8, sgb += 4) {
			int s = ib & 1 ? sc[ib / 2] >> 4 : sc[ib / 2] & 15;
			x->mult[2 * ib] = x->mult[2 * ib + 1] = 2 * s + 1;
			for (l = 0; l < 4; l++)
				for (j = 0; j < 8; j++) {
					int k = 2 * l + j / 4;
					uint32_t g = iq3s_grid[qs[k] | ((qh[ib] << (8 - k)) & 256)];
					int v = (int)((g >> (8 * (j % 4))) & 255);
					x->lv[32 * ib + 8 * l + j] = (uint8_t)((v - 1) / 2);
					x->sg[32 * ib + 8 * l + j] = (uint8_t)((sgb[l] >> j) & 1);
				}
		}
		break;
	}
	case GT_IQ1_S: {
		const uint8_t *qs = b + 2;
		for (ib = 0; ib < 8; ib++, qs += 4) {
			uint16_t qh = (uint16_t)(b[34 + 2 * ib] | b[35 + 2 * ib] << 8);
			x->mult[2 * ib] = x->mult[2 * ib + 1] = 2 * ((qh >> 12) & 7) + 1;
			x->dneg[ib] = (qh & 0x8000) != 0;
			for (l = 0; l < 4; l++) {
				uint64_t g = iq1s_grid[qs[l] | (((qh >> 3 * l) & 7) << 8)];
				for (j = 0; j < 8; j++)
					x->lv[32 * ib + 8 * l + j] = (uint8_t)((int8_t)((g >> (8 * j)) & 255) + 1);
			}
		}
		break;
	}
	}
}

/* 2-bit values in Q2_K order: byte 32c + l, bits 2j: element 128c + 32j + l */
static void
pack2(const uint8_t *v, uint8_t *o)
{
	int c, j, l;
	memset(o, 0, 64);
	for (c = 0; c < 2; c++)
		for (j = 0; j < 4; j++)
			for (l = 0; l < 32; l++)
				o[32 * c + l] |= (uint8_t)((v[128 * c + 32 * j + l] & 3) << (2 * j));
}

static int
xtype(uint32_t t)
{
	switch (t) {
	case GT_IQ2_XXS: case GT_IQ2_XS: case GT_IQ2_S: return GT_X2;
	case GT_IQ3_XXS: return GT_X3;
	case GT_IQ3_S: return GT_X3S;
	case GT_IQ1_S: return GT_X1;
	default: return -1;
	}
}

static void
pack(int xt, const xb *x, uint8_t *o)
{
	int i, c, j, l, ib;
	o[0] = x->d[0];
	o[1] = x->d[1];
	switch (xt) {
	case GT_X2:
		for (i = 0; i < 16; i++)
			o[2 + i] = (uint8_t)x->mult[i];
		pack2(x->lv, o + 18);
		memset(o + 82, 0, 32);
		for (c = 0; c < 2; c++)
			for (j = 0; j < 4; j++)
				for (l = 0; l < 32; l++)
					o[82 + l] |= (uint8_t)(x->sg[128 * c + 32 * j + l] << (4 * c + j));
		break;
	case GT_X3:
	case GT_X3S:
		for (ib = 0; ib < 8; ib++) {
			o[2 + ib] = (uint8_t)x->mult[2 * ib];
			for (j = 0; j < 16; j++) {
				int e0 = 32 * ib + j, e1 = e0 + 16;
				o[10 + 16 * ib + j] = (uint8_t)((x->lv[e0] | x->sg[e0] << 3) |
				    (x->lv[e1] | x->sg[e1] << 3) << 4);
			}
		}
		break;
	case GT_X1:
		for (ib = 0; ib < 8; ib++)
			o[2 + ib] = (uint8_t)(x->mult[2 * ib] | x->dneg[ib] << 7);
		pack2(x->lv, o + 10);
		break;
	}
}

static void
wr(FILE *f, const void *p, size_t n)
{
	if (n && fwrite(p, 1, n, f) != n) {
		perror("transcode: write");
		exit(1);
	}
}

static void
wr32(FILE *f, uint32_t v)
{
	wr(f, &v, 4);
}

static void
wr64(FILE *f, uint64_t v)
{
	wr(f, &v, 8);
}

static void
pad(FILE *f, size_t *pos, size_t al)
{
	static const uint8_t z[256];
	size_t n = (al - *pos % al) % al;
	wr(f, z, n);
	*pos += n;
}

int
gguf_transcode(const gguf *g, const char *out, int verbose)
{
	FILE *f = fopen(out, "wb");
	uint64_t i, off = 0, *noff;
	size_t pos, nconv = 0, before = 0, after = 0;
	uint8_t *buf;

	if (!f) {
		perror(out);
		return -1;
	}
	noff = calloc(g->n_tensors ? g->n_tensors : 1, sizeof(*noff));
	buf = malloc(1u << 20);
	if (!noff || !buf)
		return -1;
	wr(f, "GGUF", 4);
	wr32(f, 3);
	wr64(f, g->n_tensors);
	wr64(f, g->n_kv);
	wr(f, g->map + 24, (size_t)(g->kv_end - (g->map + 24)));
	for (i = 0; i < g->n_tensors; i++) {
		const gtensor *t = &g->t[i];
		int xt = xtype(t->type);
		uint32_t nt = xt >= 0 ? (uint32_t)xt : t->type, d;
		size_t nb = ggml_row_bytes(nt, t->ne[0]) * (size_t)(t->ne[1] * t->ne[2] * t->ne[3]);
		wr64(f, t->name.len);
		wr(f, t->name.p, t->name.len);
		wr32(f, t->ndim);
		for (d = 0; d < t->ndim; d++)
			wr64(f, t->ne[d]);
		wr32(f, nt);
		noff[i] = off;
		wr64(f, off);
		off += (nb + g->alignment - 1) / g->alignment * g->alignment;
	}
	pos = (size_t)ftell(f);
	pad(f, &pos, g->alignment);
	for (i = 0; i < g->n_tensors; i++) {
		const gtensor *t = &g->t[i];
		int xt = xtype(t->type);
		size_t start = pos;
		if (xt < 0) {
			wr(f, t->data, t->nbytes);
			pos += t->nbytes;
		} else {
			size_t nblk = t->nbytes / ggml_type_bytes(t->type), k, fill = 0;
			size_t ob = ggml_type_bytes((uint32_t)xt);
			const uint8_t *src = t->data;
			xb x;
			for (k = 0; k < nblk; k++, src += ggml_type_bytes(t->type)) {
				if (fill + ob > (1u << 20)) {
					wr(f, buf, fill);
					fill = 0;
				}
				explode(t->type, src, &x);
				pack(xt, &x, buf + fill);
				fill += ob;
			}
			wr(f, buf, fill);
			pos += nblk * ob;
			nconv++;
			before += t->nbytes;
			after += nblk * ob;
		}
		(void)start;
		pad(f, &pos, g->alignment);
	}
	if (fclose(f) != 0) {
		perror(out);
		return -1;
	}
	if (verbose || 1)
		fprintf(stderr, "transcode: %zu tensors converted, %.1f -> %.1f MiB; wrote %s\n",
		    nconv, before / 1048576.0, after / 1048576.0, out);
	free(noff);
	free(buf);
	return 0;
}
