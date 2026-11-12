/* granitehybrid (Granite 4.0-H) forward pass, one token at a time. */
#ifndef Q2K_MODEL_H
#define Q2K_MODEL_H

#include <stddef.h>
#include <stdint.h>

#include "gguf.h"

/* a (possibly 3-D, expert-stacked) quantized weight: N rows of K */
typedef struct {
	char name[48];
	uint32_t type;
	int K, N, E;            /* E = experts (1 for dense) */
	const uint8_t *data;    /* mmap */
	size_t row_bytes;
	size_t nbytes;
	int gpu;                /* resident on the GPU */
	uint64_t gpu_off;       /* byte offset in the GPU weight buffer */
	size_t gpu_row_bytes;   /* rows are repacked to 4-byte-aligned blocks */
	double heat;            /* bytes read per token / bytes stored */
} wt;

typedef struct {
	int attn;
	const float *attn_norm, *ffn_norm;
	/* mamba2 */
	wt ssm_in, ssm_out;
	const float *conv_w, *conv_b, *dt_b, *A, *D, *ssm_norm;
	/* attention */
	wt q, k, v, o;
	/* moe */
	const float *router;    /* [n_expert][n_embd] f32 */
	wt gate_e, up_e, down_e;
	wt gate_s, up_s, down_s;
} layer;

typedef struct {
	int n_embd, n_layer, n_vocab, n_head, n_head_kv, head_dim;
	int n_expert, n_expert_used, n_ff, n_ff_sh;
	int d_conv, d_inner, d_state, ssm_heads, ssm_head_dim, conv_ch, d_in_proj;
	float eps, emb_scale, res_scale, logit_scale, attn_scale;
} hparams;

typedef struct {
	gguf g;
	hparams hp;
	layer *L;
	wt tok_embd;            /* also the (tied) output projection */
	const float *out_norm;
	wt *all[512];           /* every quantized matrix, for planning */
	int n_all;
} model;

typedef struct {
	int ctx;                /* KV capacity */
	int pos;                /* tokens consumed */
	float *x, *xn, *tmp;
	float *zxbcdt, *xbc, *y;
	float **conv;           /* per layer [d_conv-1][conv_ch] */
	float **ssm;            /* per layer [heads][head_dim][d_state] */
	float **kc, **vc;       /* per layer [ctx][n_head_kv*head_dim] */
	float *q, *k, *v, *att, *ao;
	float *rl;              /* router logits */
	float *hs_g, *hs_u, *he_g, *he_u, *eo, *so;
	float *logits;
} state;

int model_load(model *m, const char *path, char *err, size_t errlen);
void model_free(model *m);

int state_init(state *s, const model *m, int ctx);
void state_reset(state *s, const model *m);
void state_free(state *s, const model *m);

/* consume one token at s->pos; returns logits if want_logits */
const float *model_forward(const model *m, state *s, int tok, int want_logits);

/* matvec batch: CPU pool and GPU run concurrently */
typedef struct {
	const wt *w;
	int e;                  /* expert index */
	const float *x;
	float *y;
} mv_item;

void mv_run(const mv_item *it, int n);

/* time spent per phase, seconds */
enum { PROF_CPU_MV, PROF_CPU_MV_OVERLAP, PROF_GPU_WAIT, PROF_SCAN, PROF_ATTN, PROF_N };
extern double prof_t[PROF_N];

/* compute settings */
extern int g_ref;           /* reference float kernels on CPU */

#endif
