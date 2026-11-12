/* Forward passes, one token at a time (prefill.c batches prompts):
 * granitehybrid (Granite 4.0-H) in model.c, qwen35 (Qwen3.5-style
 * Gated DeltaNet + gated attention) in qwen35.c. */
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
	int *gslot;             /* per expert: slot in the GPU copy, -1 = CPU */
	int gpu_n;              /* experts resident on the GPU */
	uint64_t gpu_off;       /* byte offset in the GPU weight buffer */
	size_t gpu_row_bytes;   /* rows are repacked to 4-byte-aligned blocks */
	double heat;            /* bytes read per token / bytes stored */
} wt;

typedef struct {
	int attn;
	const float *attn_norm, *ffn_norm;
	/* mamba2 */
	wt ssm_in, ssm_out;
	wt ssm_out_cpu;         /* the CPU's column slice of ssm_out, contiguous */
	const float *conv_w, *conv_b, *dt_b, *A, *D, *ssm_norm;
	/* attention */
	wt q, k, v, o;
	/* moe */
	const float *router;    /* [n_expert][n_embd] f32 */
	wt gate_e, up_e, down_e;
	wt gate_s, up_s, down_s;
	/* qwen35: post-attention norm, dense FFN, gated delta net, q/k norms */
	const float *post_norm, *q_norm, *k_norm, *dt_bias, *ssm_a;
	wt ffn_gate, ffn_up, ffn_down;
	wt wqkv, wz, wbeta, walpha;
} layer;

enum { ARCH_GRANITE, ARCH_QWEN35 };

typedef struct {
	int n_embd, n_layer, n_vocab, n_head, n_head_kv, head_dim;
	int n_expert, n_expert_used, n_ff, n_ff_sh;
	int d_conv, d_inner, d_state, ssm_heads, ssm_head_dim, conv_ch, d_in_proj;
	float eps, emb_scale, res_scale, logit_scale, attn_scale;
	/* qwen35 */
	int rope_dim;           /* rotated dims per head (NeoX pairs) */
	float rope_base;
	int dn_kh, dn_vh, dn_dim;   /* delta net: key heads, value heads, head dim */
	int dn_qkv;             /* conv channels: q, k (dn_kh*dn_dim each), v */
} hparams;

typedef struct {
	gguf g;
	int arch;
	hparams hp;
	layer *L;
	wt tok_embd;            /* also the (tied) output projection */
	wt output;              /* untied output projection (qwen35) */
	const float *out_norm;
	wt *all[1024];          /* every quantized matrix, for planning */
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
	void *xq;               /* q8k scratch for the MoE input */
	float *og;              /* GPU half of the mixer output */
	/* qwen35 */
	float *qkv, *z, *beta, *alpha, *ffg, *ffu;
} state;

int model_load(model *m, const char *path, char *err, size_t errlen);
/* after GPU placement: private copies for the CPU's share of split work */
int model_prepare_split(model *m);
void model_free(model *m);

int state_init(state *s, const model *m, int ctx);
void state_reset(state *s, const model *m);
void state_free(state *s, const model *m);

/* device memory the fused Mamba2 path needs for recurrent state */
size_t model_gpu_state_bytes(const model *m);

/* qwen35.c */
int qwen35_load(model *m, char *err, size_t errlen);
int qwen35_state_init(state *s, const model *m, int ctx);
void qwen35_state_reset(state *s, const model *m);
const float *qwen35_forward(const model *m, state *s, int tok, int want_logits);
/* shared helpers from model.c */
int model_get_wt(model *m, wt *w, const char *name, int need, char *err, size_t errlen);
const float *model_get_f32(model *m, const char *name, uint64_t n, char *err, size_t errlen);
float *model_falloc(size_t n);

/* consume n prompt tokens (batched on the CPU); last token's logits */
const float *model_prefill(const model *m, state *s, const int *tok, int n);

/* consume one token at s->pos; returns logits if want_logits */
const float *model_forward(const model *m, state *s, int tok, int want_logits);

/* matvec batch: CPU pool and GPU run concurrently */
typedef struct {
	const wt *w;
	int e;                  /* expert index */
	const float *x;
	float *y;               /* y[r] for r in [r0, r1) */
	int r0, r1;             /* row range (r1 = 0: all rows) */
	int k0;                 /* first column (multiple of 256): partial dot */
} mv_item;

mv_item mv(const wt *w, int e, const float *x, float *y);
mv_item mv_rows(const wt *w, int e, const float *x, float *y, int r0, int r1, int k0);
void mv_run(const mv_item *it, int n);
/* CPU only (when a GPU sequence of ours is already in flight) */
void mv_cpu(const mv_item *it, int n);

/* batched matmul (prefill.c): y[t*ldy + r] = dot(W_e[r], x_t), rows
 * [r0, r1), for T token vectors of xs q8k blocks each */
struct q8k_s;
typedef struct {
	const wt *w;
	int e;
	const struct q8k_s *xq;
	int xs, T;
	float *y;
	int ldy, r0, r1;
} mm_item;
void mm_run(const mm_item *it, int n);
mm_item mmi(const wt *w, int e, const struct q8k_s *xq, int xs, int T, float *y, int ldy);
const float *qwen35_prefill(const model *m, state *s, const int *tok, int n);

/* time spent per phase, seconds */
enum { PROF_CPU_MV, PROF_CPU_MV_OVERLAP, PROF_GPU_WAIT, PROF_SCAN, PROF_ATTN,
	PROF_GPU_MIXER, PROF_MIX_IN, PROF_MIX_OUT, PROF_MOE_GU, PROF_MOE_D, PROF_N };
extern double prof_b[PROF_N];   /* weight bytes per phase */
extern double prof_t[PROF_N];

/* compute settings */
extern int g_ref;           /* reference float kernels on CPU */
extern double g_split;      /* GPU share of split work (mixer heads, output rows) */

#endif
