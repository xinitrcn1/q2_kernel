/*
 * qwen35 (Qwen3.5-style hybrid) forward pass, one token at a time.
 *
 * Layers (i+1) % interval != 0 are Gated DeltaNet ("linear attention"),
 * the others gated full attention; every layer has a dense SwiGLU FFN:
 *   x += mixer(rms(x) * attn_norm)
 *   x += ffn(rms(x) * post_attention_norm)
 *
 * Gated DeltaNet, per value head h (key head h % n_key_heads), with
 * qkv = conv1d+SiLU(W_qkv x), q and k l2-normalized, beta =
 * sigmoid(W_beta x), g = softplus(W_alpha x + dt_bias) * ssm_a:
 *   S <- S * exp(g);  d = (v - S^T k) * beta;  S <- S + k d^T
 *   o = S^T q / sqrt(dim);  y = rms(o) * ssm_norm * silu(z_h)
 *   out = ssm_out y
 * The state is kept transposed (M[j][i] = S[i][j]) as in ggml.
 *
 * Gated attention: W_q gives per head [query | gate]; q and k get an
 * RMS norm per head and RoPE (NeoX pairs) on the first rope_dim dims;
 * out = W_o (softmax(q k^T / sqrt(d)) v * sigmoid(gate)).
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model.h"
#include "pool.h"
#include "quant.h"
#include "vmath.h"

#define KEY(s) "qwen35." s

int
qwen35_load(model *m, char *err, size_t errlen)
{
	hparams *h = &m->hp;
	char nm[96];
	uint32_t u, interval = 4, kv = 0;
	int i;

#define U32(k, dst) do { if (gguf_get_u32(&m->g, KEY(k), &u)) { \
	snprintf(err, errlen, "missing key %s", KEY(k)); return -1; } (dst) = (int)u; } while (0)
	U32("embedding_length", h->n_embd);
	U32("block_count", h->n_layer);
	U32("attention.head_count", h->n_head);
	U32("feed_forward_length", h->n_ff);
	U32("attention.key_length", h->head_dim);
	U32("rope.dimension_count", h->rope_dim);
	U32("ssm.conv_kernel", h->d_conv);
	U32("ssm.state_size", h->dn_dim);
	U32("ssm.group_count", h->dn_kh);
	U32("ssm.time_step_rank", h->dn_vh);
	U32("ssm.inner_size", h->d_inner);
#undef U32
	gguf_get_arr_u32(&m->g, KEY("attention.head_count_kv"), 0, &kv);
	h->n_head_kv = (int)kv;
	gguf_get_u32(&m->g, KEY("full_attention_interval"), &interval);
	/* the trailing next-token-prediction block(s) are not run */
	if (gguf_get_u32(&m->g, KEY("nextn_predict_layers"), &u) == 0)
		h->n_layer -= (int)u;
	h->eps = 1e-6f;
	h->rope_base = 10000.0f;
	gguf_get_f32(&m->g, KEY("attention.layer_norm_rms_epsilon"), &h->eps);
	gguf_get_f32(&m->g, KEY("rope.freq_base"), &h->rope_base);
	h->attn_scale = 1.0f / sqrtf((float)h->head_dim);
	h->dn_qkv = 2 * h->dn_kh * h->dn_dim + h->dn_vh * h->dn_dim;
	if (h->d_inner != h->dn_vh * h->dn_dim || h->dn_vh % h->dn_kh || h->d_conv != 4 ||
	    h->n_head % h->n_head_kv || h->rope_dim > h->head_dim || h->dn_dim % 8) {
		snprintf(err, errlen, "unsupported qwen35 shape");
		return -1;
	}

	if (model_get_wt(m, &m->tok_embd, "token_embd.weight", 1, err, errlen))
		return -1;
	h->n_vocab = m->tok_embd.N;
	if (!(m->out_norm = model_get_f32(m, "output_norm.weight", (uint64_t)h->n_embd, err, errlen)))
		return -1;
	if (model_get_wt(m, &m->output, "output.weight", 0, err, errlen))
		return -1;
	if (!m->output.data)
		m->output = m->tok_embd;
	m->output.heat = 1;

	m->L = calloc((size_t)h->n_layer, sizeof(*m->L));
	if (!m->L) {
		snprintf(err, errlen, "out of memory");
		return -1;
	}
	for (i = 0; i < h->n_layer; i++) {
		layer *L = &m->L[i];
		const int E = h->n_embd;
#define T(field, s) (snprintf(nm, sizeof(nm), "blk.%d." s, i), \
	model_get_wt(m, &L->field, nm, 1, err, errlen))
#define F(field, s, n) do { snprintf(nm, sizeof(nm), "blk.%d." s, i); \
	if (!(L->field = model_get_f32(m, nm, (uint64_t)(n), err, errlen))) return -1; } while (0)
		L->attn = (i + 1) % (int)interval == 0;
		F(attn_norm, "attn_norm.weight", E);
		F(post_norm, "post_attention_norm.weight", E);
		if (L->attn) {
			if (T(q, "attn_q.weight") || T(k, "attn_k.weight") || T(v, "attn_v.weight") ||
			    T(o, "attn_output.weight"))
				return -1;
			F(q_norm, "attn_q_norm.weight", h->head_dim);
			F(k_norm, "attn_k_norm.weight", h->head_dim);
			if (L->q.N != 2 * h->n_head * h->head_dim ||
			    L->k.N != h->n_head_kv * h->head_dim) {
				snprintf(err, errlen, "attention shape mismatch");
				return -1;
			}
			L->q.heat = L->k.heat = L->v.heat = L->o.heat = 1;
		} else {
			if (T(wqkv, "attn_qkv.weight") || T(wz, "attn_gate.weight") ||
			    T(wbeta, "ssm_beta.weight") || T(walpha, "ssm_alpha.weight") ||
			    T(ssm_out, "ssm_out.weight"))
				return -1;
			F(conv_w, "ssm_conv1d.weight", h->d_conv * h->dn_qkv);
			F(dt_bias, "ssm_dt.bias", h->dn_vh);
			F(ssm_a, "ssm_a", h->dn_vh);
			F(ssm_norm, "ssm_norm.weight", h->dn_dim);
			if (L->wqkv.N != h->dn_qkv || L->wz.N != h->d_inner) {
				snprintf(err, errlen, "delta net shape mismatch");
				return -1;
			}
			L->wqkv.heat = L->wz.heat = L->wbeta.heat = L->walpha.heat = L->ssm_out.heat = 1;
		}
		if (T(ffn_gate, "ffn_gate.weight") || T(ffn_up, "ffn_up.weight") ||
		    T(ffn_down, "ffn_down.weight"))
			return -1;
		L->ffn_gate.heat = L->ffn_up.heat = L->ffn_down.heat = 1;
#undef T
#undef F
	}
	if (m->n_all < (int)(sizeof(m->all) / sizeof(m->all[0])) && m->output.data != m->tok_embd.data)
		m->all[m->n_all++] = &m->output;
	return 0;
}

/* ---------------------------------------------------------------- */

int
qwen35_state_init(state *s, const model *m, int ctx)
{
	const hparams *h = &m->hp;
	const size_t kvd = (size_t)h->n_head_kv * (size_t)h->head_dim;
	int i;

	memset(s, 0, sizeof(*s));
	s->ctx = ctx;
	s->x = model_falloc((size_t)h->n_embd);
	s->xn = model_falloc((size_t)h->n_embd);
	s->tmp = model_falloc((size_t)h->n_embd);
	s->qkv = model_falloc((size_t)h->dn_qkv);
	s->xbc = model_falloc((size_t)h->dn_qkv);
	s->z = model_falloc((size_t)h->d_inner);
	s->y = model_falloc((size_t)(h->d_inner > h->n_head * h->head_dim ? h->d_inner :
	    h->n_head * h->head_dim));
	s->beta = model_falloc((size_t)h->dn_vh);
	s->alpha = model_falloc((size_t)h->dn_vh);
	s->q = model_falloc((size_t)(2 * h->n_head * h->head_dim));
	s->k = model_falloc(kvd);
	s->v = model_falloc(kvd);
	s->att = model_falloc((size_t)h->n_head * (size_t)ctx);
	s->ao = model_falloc((size_t)(h->n_head * h->head_dim));
	s->ffg = model_falloc((size_t)h->n_ff);
	s->ffu = model_falloc((size_t)h->n_ff);
	s->logits = model_falloc((size_t)h->n_vocab);
	s->conv = calloc((size_t)h->n_layer, sizeof(float *));
	s->ssm = calloc((size_t)h->n_layer, sizeof(float *));
	s->kc = calloc((size_t)h->n_layer, sizeof(float *));
	s->vc = calloc((size_t)h->n_layer, sizeof(float *));
	if (!s->conv || !s->ssm || !s->kc || !s->vc || !s->logits || !s->att)
		return -1;
	for (i = 0; i < h->n_layer; i++) {
		if (m->L[i].attn) {
			s->kc[i] = model_falloc((size_t)ctx * kvd);
			s->vc[i] = model_falloc((size_t)ctx * kvd);
			if (!s->kc[i] || !s->vc[i])
				return -1;
		} else {
			s->conv[i] = model_falloc((size_t)(h->d_conv - 1) * (size_t)h->dn_qkv);
			s->ssm[i] = model_falloc((size_t)h->dn_vh * (size_t)h->dn_dim * (size_t)h->dn_dim);
			if (!s->conv[i] || !s->ssm[i])
				return -1;
		}
	}
	return 0;
}

void
qwen35_state_reset(state *s, const model *m)
{
	const hparams *h = &m->hp;
	int i;
	s->pos = 0;
	for (i = 0; i < h->n_layer; i++) {
		if (s->conv[i])
			memset(s->conv[i], 0, sizeof(float) * (size_t)(h->d_conv - 1) * (size_t)h->dn_qkv);
		if (s->ssm[i])
			memset(s->ssm[i], 0, sizeof(float) * (size_t)h->dn_vh * (size_t)h->dn_dim *
			    (size_t)h->dn_dim);
	}
}

/* ---------------------------------------------------------------- */
/* Gated DeltaNet                                                    */

static float
sigmoidf(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

static float
softplusf(float x)
{
	return x > 20.0f ? x : log1pf(expf(x));
}

/* x / sqrt(sum x^2 + eps), as ggml's rms_norm(eps/n) * 1/sqrt(n) */
static void
l2norm(float *x, int n, float eps)
{
	float r = 1.0f / sqrtf(v_dot(x, x, (size_t)n) + eps);
	int i;
	for (i = 0; i < n; i++)
		x[i] *= r;
}

typedef struct {
	const model *m;
	const layer *L;
	state *s;
	int il;
	long next;
} dn_job;

static void
dn_worker(void *arg, int ith, int nth)
{
	dn_job *j = arg;
	const hparams *h = &j->m->hp;
	const int D = h->dn_dim, KH = h->dn_kh;
	const float *q0 = j->s->xbc, *k0 = q0 + KH * D, *v0 = k0 + KH * D;
	const float scale = 1.0f / sqrtf((float)D);
	float delta[256];
	long hv;

	(void)ith;
	(void)nth;
	while ((hv = pool_take(&j->next, 1, h->dn_vh)) >= 0) {
		const float *q = q0 + (hv % KH) * D, *k = k0 + (hv % KH) * D, *v = v0 + hv * D;
		float *M = j->s->ssm[j->il] + (size_t)hv * D * D;
		const float decay = expf(softplusf(j->s->alpha[hv] + j->L->dt_bias[hv]) * j->L->ssm_a[hv]);
		const float beta = sigmoidf(j->s->beta[hv]);
		float *o = j->s->y + hv * D;
		int r;

		/* M[j][:] *= decay, then delta[j] = (v[j] - M[j].k) * beta */
		for (r = 0; r < D; r++) {
			float *row = M + (size_t)r * D;
			int c;
			for (c = 0; c < D; c++)
				row[c] *= decay;
			delta[r] = (v[r] - v_dot(row, k, (size_t)D)) * beta;
		}
		/* M[j][:] += delta[j] * k; o[j] = M[j].q * scale */
		for (r = 0; r < D; r++) {
			float *row = M + (size_t)r * D;
			v_axpy(row, delta[r], k, (size_t)D);
			o[r] = v_dot(row, q, (size_t)D) * scale;
		}
	}
}

static void
delta_net(const model *m, const layer *L, state *s, int il)
{
	const hparams *h = &m->hp;
	const int CH = h->dn_qkv, D = h->dn_dim;
	float *cs = s->conv[il];
	mv_item it[4];
	dn_job dj;
	int c, i;

	it[0] = mv(&L->wqkv, 0, s->xn, s->qkv);
	it[1] = mv(&L->wz, 0, s->xn, s->z);
	it[2] = mv(&L->wbeta, 0, s->xn, s->beta);
	it[3] = mv(&L->walpha, 0, s->xn, s->alpha);
	mv_run(it, 4);

	/* causal depthwise conv (no bias) over [cs0, cs1, cs2, new], SiLU */
	for (c = 0; c < CH; c++) {
		const float *w = L->conv_w + 4 * c;
		float v = cs[c] * w[0] + cs[CH + c] * w[1] + cs[2 * CH + c] * w[2] + s->qkv[c] * w[3];
		cs[c] = cs[CH + c];
		cs[CH + c] = cs[2 * CH + c];
		cs[2 * CH + c] = s->qkv[c];
		s->xbc[c] = v;
	}
	v_silu(s->xbc, (size_t)CH);
	for (i = 0; i < 2 * h->dn_kh; i++)
		l2norm(s->xbc + i * D, D, h->eps);

	dj.m = m;
	dj.L = L;
	dj.s = s;
	dj.il = il;
	dj.next = 0;
	pool_run(dn_worker, &dj);

	/* per head: rms norm with ssm_norm, times silu(z) */
	for (i = 0; i < h->dn_vh; i++)
		v_rmsnorm(s->y + i * D, s->y + i * D, L->ssm_norm, (size_t)D, h->eps);
	v_mul_silu(s->y, s->z, (size_t)h->d_inner);

	it[0] = mv(&L->ssm_out, 0, s->y, s->tmp);
	mv_run(it, 1);
}

/* ---------------------------------------------------------------- */
/* gated attention                                                   */

/* NeoX RoPE on the first rd dims: pairs (i, i + rd/2) */
static void
rope(float *x, int rd, int pos, float base)
{
	int i, half = rd / 2;
	for (i = 0; i < half; i++) {
		float th = (float)pos * powf(base, -2.0f * (float)i / (float)rd);
		float c = cosf(th), sn = sinf(th), a = x[i], b = x[i + half];
		x[i] = a * c - b * sn;
		x[i + half] = a * sn + b * c;
	}
}

typedef struct {
	const model *m;
	state *s;
	int il, n;
	long next;
} at_job;

static void
at_worker(void *arg, int ith, int nth)
{
	at_job *j = arg;
	const hparams *h = &j->m->hp;
	const int hd = h->head_dim, kvd = h->n_head_kv * hd, grp = h->n_head / h->n_head_kv;
	const float *kc = j->s->kc[j->il], *vc = j->s->vc[j->il];
	long hh;

	(void)ith;
	(void)nth;
	while ((hh = pool_take(&j->next, 1, h->n_head)) >= 0) {
		const float *q = j->s->q + hh * 2 * hd, *gate = q + hd;
		const int off = (int)(hh / grp) * hd;
		float *a = j->s->att + (size_t)hh * (size_t)j->s->ctx;
		float *o = j->s->ao + hh * hd, mx = -INFINITY, sum = 0;
		int t, d;
		for (t = 0; t < j->n; t++) {
			a[t] = v_dot(q, kc + (size_t)t * kvd + off, (size_t)hd) * h->attn_scale;
			if (a[t] > mx)
				mx = a[t];
		}
		for (t = 0; t < j->n; t++)
			a[t] -= mx;
		v_exp(a, a, (size_t)j->n);
		for (t = 0; t < j->n; t++)
			sum += a[t];
		memset(o, 0, sizeof(float) * (size_t)hd);
		for (t = 0; t < j->n; t++)
			v_axpy(o, a[t] / sum, vc + (size_t)t * kvd + off, (size_t)hd);
		for (d = 0; d < hd; d++)
			o[d] *= sigmoidf(gate[d]);
	}
}

static void
attention(const model *m, const layer *L, state *s, int il)
{
	const hparams *h = &m->hp;
	const int hd = h->head_dim, kvd = h->n_head_kv * hd;
	float *kd = s->kc[il] + (size_t)s->pos * kvd, *vd = s->vc[il] + (size_t)s->pos * kvd;
	mv_item it[3];
	at_job aj;
	int i;

	it[0] = mv(&L->q, 0, s->xn, s->q);
	it[1] = mv(&L->k, 0, s->xn, kd);
	it[2] = mv(&L->v, 0, s->xn, vd);
	mv_run(it, 3);
	for (i = 0; i < h->n_head; i++) {
		float *q = s->q + i * 2 * hd;
		v_rmsnorm(q, q, L->q_norm, (size_t)hd, h->eps);
		rope(q, h->rope_dim, s->pos, h->rope_base);
	}
	for (i = 0; i < h->n_head_kv; i++) {
		v_rmsnorm(kd + i * hd, kd + i * hd, L->k_norm, (size_t)hd, h->eps);
		rope(kd + i * hd, h->rope_dim, s->pos, h->rope_base);
	}
	aj.m = m;
	aj.s = s;
	aj.il = il;
	aj.n = s->pos + 1;
	aj.next = 0;
	pool_run(at_worker, &aj);

	it[0] = mv(&L->o, 0, s->ao, s->tmp);
	mv_run(it, 1);
}

/* ---------------------------------------------------------------- */

static void
ffn(const model *m, const layer *L, state *s)
{
	mv_item it[2];
	it[0] = mv(&L->ffn_gate, 0, s->xn, s->ffg);
	it[1] = mv(&L->ffn_up, 0, s->xn, s->ffu);
	mv_run(it, 2);
	v_swiglu(s->ffg, s->ffu, (size_t)m->hp.n_ff);
	it[0] = mv(&L->ffn_down, 0, s->ffg, s->tmp);
	mv_run(it, 1);
}

const float *
qwen35_forward(const model *m, state *s, int tok, int want_logits)
{
	const hparams *h = &m->hp;
	const int E = h->n_embd;
	mv_item it;
	int il;

	if (tok < 0 || tok >= h->n_vocab)
		tok = 0;
	dequant_row(m->tok_embd.type, m->tok_embd.data + (size_t)tok * m->tok_embd.row_bytes,
	    s->x, (size_t)E);
	for (il = 0; il < h->n_layer; il++) {
		const layer *L = &m->L[il];
		v_rmsnorm(s->xn, s->x, L->attn_norm, (size_t)E, h->eps);
		if (L->attn)
			attention(m, L, s, il);
		else
			delta_net(m, L, s, il);
		v_axpy(s->x, 1.0f, s->tmp, (size_t)E);
		v_rmsnorm(s->xn, s->x, L->post_norm, (size_t)E, h->eps);
		ffn(m, L, s);
		v_axpy(s->x, 1.0f, s->tmp, (size_t)E);
	}
	s->pos++;
	if (!want_logits)
		return NULL;
	v_rmsnorm(s->xn, s->x, m->out_norm, (size_t)E, h->eps);
	it = mv(&m->output, 0, s->xn, s->logits);
	mv_run(&it, 1);
	return s->logits;
}

/* ---------------------------------------------------------------- */
/* batched prompts (CPU): chunks of up to QB tokens per layer        */

#include "kq.h"
#include <time.h>

#define QB 128

static struct {
	int ok;
	float *x, *xn, *qkv, *z, *beta, *alpha, *y, *tmp, *fg, *fu, *q, *ao;
	q8k *xq, *yq, *fq;
} P;

static int
pbufs(const hparams *h)
{
	const size_t T = QB, E = (size_t)h->n_embd;
	if (P.ok)
		return 0;
#define A(p, n) if (!((p) = calloc((n), sizeof(*(p))))) return -1
	A(P.x, T * E); A(P.xn, T * E); A(P.tmp, T * E);
	A(P.qkv, T * (size_t)h->dn_qkv); A(P.z, T * (size_t)h->d_inner);
	A(P.beta, T * (size_t)h->dn_vh); A(P.alpha, T * (size_t)h->dn_vh);
	A(P.y, T * (size_t)(h->d_inner > h->n_head * h->head_dim ? h->d_inner : h->n_head * h->head_dim));
	A(P.fg, T * (size_t)h->n_ff); A(P.fu, T * (size_t)h->n_ff);
	A(P.q, T * (size_t)(2 * h->n_head * h->head_dim)); A(P.ao, T * (size_t)(h->n_head * h->head_dim));
	A(P.xq, T * E / 256); A(P.yq, T * (size_t)(h->d_inner > h->n_head * h->head_dim ?
	    h->d_inner : h->n_head * h->head_dim) / 256);
	A(P.fq, T * (size_t)h->n_ff / 256);
#undef A
	P.ok = 1;
	return 0;
}

static void
quant_rows(const float *x, q8k *q, int T, int n)
{
	int t;
	for (t = 0; t < T; t++)
		q8k_quantize(x + (size_t)t * (size_t)n, q + (size_t)t * (size_t)(n / 256), (size_t)n);
}

typedef struct {
	const model *m;
	const layer *L;
	state *s;
	int il, T;
	long next;
} pj;

/* conv + SiLU per channel over the chunk, in token order */
static void
pconv(void *arg, int ith, int nth)
{
	pj *j = arg;
	const int CH = j->m->hp.dn_qkv;
	float *cs = j->s->conv[j->il];
	long c0;
	(void)ith;
	(void)nth;
	while ((c0 = pool_take(&j->next, 256, CH)) >= 0) {
		int c, t, c1 = (int)c0 + 256 > CH ? CH : (int)c0 + 256;
		for (c = (int)c0; c < c1; c++) {
			const float *w = j->L->conv_w + 4 * c;
			float s0 = cs[c], s1 = cs[CH + c], s2 = cs[2 * CH + c];
			for (t = 0; t < j->T; t++) {
				float in = P.qkv[(size_t)t * CH + c];
				P.qkv[(size_t)t * CH + c] = s0 * w[0] + s1 * w[1] + s2 * w[2] + in * w[3];
				s0 = s1;
				s1 = s2;
				s2 = in;
			}
			cs[c] = s0;
			cs[CH + c] = s1;
			cs[2 * CH + c] = s2;
		}
	}
}

/* the delta rule per value head over the chunk, in token order */
static void
pdelta(void *arg, int ith, int nth)
{
	pj *j = arg;
	const hparams *h = &j->m->hp;
	const int D = h->dn_dim, KH = h->dn_kh, CH = h->dn_qkv, DI = h->d_inner;
	const float scale = 1.0f / sqrtf((float)D);
	float delta[256];
	long hv;
	(void)ith;
	(void)nth;
	while ((hv = pool_take(&j->next, 1, h->dn_vh)) >= 0) {
		float *M = j->s->ssm[j->il] + (size_t)hv * D * D;
		int t, r;
		for (t = 0; t < j->T; t++) {
			const float *xb = P.qkv + (size_t)t * CH;
			const float *q = xb + (hv % KH) * D, *k = xb + KH * D + (hv % KH) * D;
			const float *v = xb + 2 * KH * D + hv * D;
			const float decay = expf(softplusf(P.alpha[(size_t)t * h->dn_vh + hv] +
			    j->L->dt_bias[hv]) * j->L->ssm_a[hv]);
			const float beta = sigmoidf(P.beta[(size_t)t * h->dn_vh + hv]);
			float *o = P.y + (size_t)t * DI + hv * D;
			for (r = 0; r < D; r++) {
				float *row = M + (size_t)r * D;
				int c;
				for (c = 0; c < D; c++)
					row[c] *= decay;
				delta[r] = (v[r] - v_dot(row, k, (size_t)D)) * beta;
			}
			for (r = 0; r < D; r++) {
				float *row = M + (size_t)r * D;
				v_axpy(row, delta[r], k, (size_t)D);
				o[r] = v_dot(row, q, (size_t)D) * scale;
			}
		}
	}
}

static void
pdeltanet(const model *m, const layer *L, state *s, int il, int T)
{
	const hparams *h = &m->hp;
	const int E = h->n_embd, D = h->dn_dim, CH = h->dn_qkv, DI = h->d_inner;
	mm_item it[4];
	pj j;
	int t, i;

	it[0] = mmi(&L->wqkv, 0, P.xq, E / 256, T, P.qkv, CH);
	it[1] = mmi(&L->wz, 0, P.xq, E / 256, T, P.z, DI);
	it[2] = mmi(&L->wbeta, 0, P.xq, E / 256, T, P.beta, h->dn_vh);
	it[3] = mmi(&L->walpha, 0, P.xq, E / 256, T, P.alpha, h->dn_vh);
	mm_run(it, 4);
	j.m = m;
	j.L = L;
	j.s = s;
	j.il = il;
	j.T = T;
	j.next = 0;
	pool_run(pconv, &j);
	for (t = 0; t < T; t++) {
		float *xb = P.qkv + (size_t)t * CH;
		v_silu(xb, (size_t)CH);
		for (i = 0; i < 2 * h->dn_kh; i++)
			l2norm(xb + i * D, D, h->eps);
	}
	j.next = 0;
	pool_run(pdelta, &j);
	for (t = 0; t < T; t++) {
		float *y = P.y + (size_t)t * DI;
		for (i = 0; i < h->dn_vh; i++)
			v_rmsnorm(y + i * D, y + i * D, L->ssm_norm, (size_t)D, h->eps);
		v_mul_silu(y, P.z + (size_t)t * DI, (size_t)DI);
	}
	quant_rows(P.y, P.yq, T, DI);
	it[0] = mmi(&L->ssm_out, 0, P.yq, DI / 256, T, P.tmp, E);
	mm_run(it, 1);
}

typedef struct {
	const model *m;
	state *s;
	int il, T, pos;
	long next;
} paj;

static void
patt(void *arg, int ith, int nth)
{
	paj *j = arg;
	const hparams *h = &j->m->hp;
	const int hd = h->head_dim, kvd = h->n_head_kv * hd, grp = h->n_head / h->n_head_kv;
	const float *kc = j->s->kc[j->il], *vc = j->s->vc[j->il];
	float *a = malloc(sizeof(float) * (size_t)(j->pos + j->T));
	long w;
	(void)ith;
	(void)nth;
	if (!a)
		return;
	while ((w = pool_take(&j->next, 1, (long)j->T * h->n_head)) >= 0) {
		int t = (int)(w / h->n_head), hh = (int)(w % h->n_head), n = j->pos + t + 1, k, d;
		const float *q = P.q + (size_t)t * 2 * h->n_head * hd + (size_t)hh * 2 * hd, *gate = q + hd;
		const int off = (hh / grp) * hd;
		float *o = P.ao + (size_t)t * h->n_head * hd + (size_t)hh * hd, mx = -INFINITY, sum = 0;
		for (k = 0; k < n; k++) {
			a[k] = v_dot(q, kc + (size_t)k * kvd + off, (size_t)hd) * h->attn_scale;
			if (a[k] > mx)
				mx = a[k];
		}
		for (k = 0; k < n; k++)
			a[k] -= mx;
		v_exp(a, a, (size_t)n);
		for (k = 0; k < n; k++)
			sum += a[k];
		memset(o, 0, sizeof(float) * (size_t)hd);
		for (k = 0; k < n; k++)
			v_axpy(o, a[k] / sum, vc + (size_t)k * kvd + off, (size_t)hd);
		for (d = 0; d < hd; d++)
			o[d] *= sigmoidf(gate[d]);
	}
	free(a);
}

static void
pattention(const model *m, const layer *L, state *s, int il, int T)
{
	const hparams *h = &m->hp;
	const int E = h->n_embd, hd = h->head_dim, kvd = h->n_head_kv * hd, QD = 2 * h->n_head * hd;
	const int AD = h->n_head * hd;
	float *kd = s->kc[il] + (size_t)s->pos * kvd, *vd = s->vc[il] + (size_t)s->pos * kvd;
	mm_item it[3];
	paj j;
	int t, i;

	it[0] = mmi(&L->q, 0, P.xq, E / 256, T, P.q, QD);
	it[1] = mmi(&L->k, 0, P.xq, E / 256, T, kd, kvd);
	it[2] = mmi(&L->v, 0, P.xq, E / 256, T, vd, kvd);
	mm_run(it, 3);
	for (t = 0; t < T; t++) {
		for (i = 0; i < h->n_head; i++) {
			float *q = P.q + (size_t)t * QD + i * 2 * hd;
			v_rmsnorm(q, q, L->q_norm, (size_t)hd, h->eps);
			rope(q, h->rope_dim, s->pos + t, h->rope_base);
		}
		for (i = 0; i < h->n_head_kv; i++) {
			float *k = kd + (size_t)t * kvd + i * hd;
			v_rmsnorm(k, k, L->k_norm, (size_t)hd, h->eps);
			rope(k, h->rope_dim, s->pos + t, h->rope_base);
		}
	}
	j.m = m;
	j.s = s;
	j.il = il;
	j.T = T;
	j.pos = s->pos;
	j.next = 0;
	pool_run(patt, &j);
	quant_rows(P.ao, P.yq, T, AD);
	it[0] = mmi(&L->o, 0, P.yq, AD / 256, T, P.tmp, E);
	mm_run(it, 1);
}

static void
pffn(const model *m, const layer *L, int T)
{
	const hparams *h = &m->hp;
	const int E = h->n_embd, FF = h->n_ff;
	mm_item it[2];
	it[0] = mmi(&L->ffn_gate, 0, P.xq, E / 256, T, P.fg, FF);
	it[1] = mmi(&L->ffn_up, 0, P.xq, E / 256, T, P.fu, FF);
	mm_run(it, 2);
	v_swiglu(P.fg, P.fu, (size_t)T * FF);
	quant_rows(P.fg, P.fq, T, FF);
	it[0] = mmi(&L->ffn_down, 0, P.fq, FF / 256, T, P.tmp, E);
	mm_run(it, 1);
}

static void
pchunk(const model *m, state *s, const int *tok, int T)
{
	const hparams *h = &m->hp;
	const int E = h->n_embd;
	int il, t;

	for (t = 0; t < T; t++) {
		int id = tok[t] < 0 || tok[t] >= h->n_vocab ? 0 : tok[t];
		dequant_row(m->tok_embd.type, m->tok_embd.data + (size_t)id * m->tok_embd.row_bytes,
		    P.x + (size_t)t * E, (size_t)E);
	}
	for (il = 0; il < h->n_layer; il++) {
		const layer *L = &m->L[il];
		for (t = 0; t < T; t++)
			v_rmsnorm(P.xn + (size_t)t * E, P.x + (size_t)t * E, L->attn_norm, (size_t)E, h->eps);
		quant_rows(P.xn, P.xq, T, E);
		if (L->attn)
			pattention(m, L, s, il, T);
		else
			pdeltanet(m, L, s, il, T);
		v_axpy(P.x, 1.0f, P.tmp, (size_t)T * E);
		for (t = 0; t < T; t++)
			v_rmsnorm(P.xn + (size_t)t * E, P.x + (size_t)t * E, L->post_norm, (size_t)E, h->eps);
		quant_rows(P.xn, P.xq, T, E);
		pffn(m, L, T);
		v_axpy(P.x, 1.0f, P.tmp, (size_t)T * E);
	}
	s->pos += T;
}

const float *
qwen35_prefill(const model *m, state *s, const int *tok, int n)
{
	int done = 0;
	if (pbufs(&m->hp) != 0) {
		const float *lg = NULL;
		int i;
		for (i = 0; i < n; i++)
			lg = qwen35_forward(m, s, tok[i], i == n - 1);
		return lg;
	}
	/* all but the last token in chunks; the last one through the
	 * single-token path, which also produces the logits */
	{
		extern double pf_t[8], vk_t_stage;
		struct timespec a, b;
		clock_gettime(CLOCK_MONOTONIC, &a);
		while (done < n - 1) {
			int T = n - 1 - done > QB ? QB : n - 1 - done;
			pchunk(m, s, tok + done, T);
			done += T;
		}
		clock_gettime(CLOCK_MONOTONIC, &b);
		if (getenv("Q2K_PREFILL_PROF"))
			fprintf(stderr, "prefill %d tok %.2f s: cpu matmul %.2f s, gpu wait after %.2f s, "
			    "staging %.2f s\n", n - 1, (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) * 1e-9,
			    pf_t[0], pf_t[7], vk_t_stage);
	}
	return qwen35_forward(m, s, tok[n - 1], 1);
}
