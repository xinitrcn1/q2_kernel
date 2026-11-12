#include "model.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kq.h"
#include "pool.h"
#include "quant.h"
#include "vk.h"

int g_ref;

/* ---------------------------------------------------------------- */
/* loading                                                          */

static int
get_wt(model *m, wt *w, const char *name, int need, char *err, size_t errlen)
{
	const gtensor *t = gguf_tensor(&m->g, name);
	memset(w, 0, sizeof(*w));
	if (!t) {
		if (need)
			snprintf(err, errlen, "missing tensor %s", name);
		return need ? -1 : 0;
	}
	snprintf(w->name, sizeof(w->name), "%s", name);
	w->type = t->type;
	w->K = (int)t->ne[0];
	w->N = (int)t->ne[1];
	w->E = (int)t->ne[2];
	w->data = t->data;
	w->row_bytes = ggml_row_bytes(t->type, t->ne[0]);
	w->nbytes = t->nbytes;
	if (w->K % 256) {
		snprintf(err, errlen, "%s: row length %d not a multiple of 256", name, w->K);
		return -1;
	}
	if (m->n_all < (int)(sizeof(m->all) / sizeof(m->all[0])))
		m->all[m->n_all++] = w;
	return 0;
}

static const float *
get_f32(model *m, const char *name, uint64_t n, char *err, size_t errlen)
{
	const gtensor *t = gguf_tensor(&m->g, name);
	if (!t || t->type != GT_F32 || t->ne[0] * t->ne[1] * t->ne[2] != n) {
		snprintf(err, errlen, "missing or bad f32 tensor %s", name);
		return NULL;
	}
	return t->data;
}

#define KEY(s) "granitehybrid." s

int
model_load(model *m, const char *path, char *err, size_t errlen)
{
	hparams *h = &m->hp;
	gstr arch;
	char nm[96];
	uint32_t u;
	int i;

	memset(m, 0, sizeof(*m));
	if (gguf_open(&m->g, path, err, errlen) != 0)
		return -1;
	if (gguf_get_str(&m->g, "general.architecture", &arch) || !gstr_eq(arch, "granitehybrid")) {
		snprintf(err, errlen, "architecture is not granitehybrid");
		return -1;
	}
#define U32(k, dst) do { if (gguf_get_u32(&m->g, KEY(k), &u)) { \
	snprintf(err, errlen, "missing key %s", KEY(k)); return -1; } (dst) = (int)u; } while (0)
	U32("embedding_length", h->n_embd);
	U32("block_count", h->n_layer);
	U32("attention.head_count", h->n_head);
	U32("expert_count", h->n_expert);
	U32("expert_used_count", h->n_expert_used);
	U32("feed_forward_length", h->n_ff);
	U32("expert_shared_feed_forward_length", h->n_ff_sh);
	U32("ssm.conv_kernel", h->d_conv);
	U32("ssm.inner_size", h->d_inner);
	U32("ssm.state_size", h->d_state);
	U32("ssm.time_step_rank", h->ssm_heads);
	U32("ssm.group_count", u);
#undef U32
	if (u != 1) {
		snprintf(err, errlen, "ssm.group_count %u unsupported", u);
		return -1;
	}
	h->eps = 1e-5f;
	h->emb_scale = h->res_scale = h->logit_scale = 1;
	h->attn_scale = 0;
	gguf_get_f32(&m->g, KEY("attention.layer_norm_rms_epsilon"), &h->eps);
	gguf_get_f32(&m->g, KEY("embedding_scale"), &h->emb_scale);
	gguf_get_f32(&m->g, KEY("residual_scale"), &h->res_scale);
	gguf_get_f32(&m->g, KEY("logit_scale"), &h->logit_scale);
	gguf_get_f32(&m->g, KEY("attention.scale"), &h->attn_scale);
	{
		uint32_t rf = 0;
		if (gguf_get_u32(&m->g, KEY("rope.scaling.finetuned"), &rf) == 0 && rf) {
			snprintf(err, errlen, "RoPE-enabled granitehybrid not supported");
			return -1;
		}
	}
	h->head_dim = h->n_embd / h->n_head;
	if (h->attn_scale == 0)
		h->attn_scale = 1.0f / sqrtf((float)h->head_dim);
	h->ssm_head_dim = h->d_inner / h->ssm_heads;
	h->conv_ch = h->d_inner + 2 * h->d_state;
	h->d_in_proj = 2 * h->d_inner + 2 * h->d_state + h->ssm_heads;
	if (h->d_state % 8 || h->n_embd % 256 || h->d_conv != 4) {
		snprintf(err, errlen, "unsupported shape");
		return -1;
	}

	if (get_wt(m, &m->tok_embd, "token_embd.weight", 1, err, errlen))
		return -1;
	h->n_vocab = m->tok_embd.N;
	if (!(m->out_norm = get_f32(m, "output_norm.weight", (uint64_t)h->n_embd, err, errlen)))
		return -1;
	if (gguf_tensor(&m->g, "output.weight")) {
		snprintf(err, errlen, "untied output.weight not supported");
		return -1;
	}
	m->tok_embd.heat = 1;

	m->L = calloc((size_t)h->n_layer, sizeof(*m->L));
	if (!m->L) {
		snprintf(err, errlen, "out of memory");
		return -1;
	}
	for (i = 0; i < h->n_layer; i++) {
		layer *L = &m->L[i];
		uint32_t kv = 0;
		const int E = h->n_embd;
#define T(field, s) (snprintf(nm, sizeof(nm), "blk.%d." s, i), \
	get_wt(m, &L->field, nm, 1, err, errlen))
#define F(field, s, n) do { snprintf(nm, sizeof(nm), "blk.%d." s, i); \
	if (!(L->field = get_f32(m, nm, (uint64_t)(n), err, errlen))) return -1; } while (0)
		gguf_get_arr_u32(&m->g, KEY("attention.head_count_kv"), (uint64_t)i, &kv);
		L->attn = kv != 0;
		F(attn_norm, "attn_norm.weight", E);
		F(ffn_norm, "ffn_norm.weight", E);
		if (L->attn) {
			if (h->n_head_kv && h->n_head_kv != (int)kv) {
				snprintf(err, errlen, "varying n_head_kv unsupported");
				return -1;
			}
			h->n_head_kv = (int)kv;
			if (T(q, "attn_q.weight") || T(k, "attn_k.weight") ||
			    T(v, "attn_v.weight") || T(o, "attn_output.weight"))
				return -1;
			L->q.heat = L->k.heat = L->v.heat = L->o.heat = 1;
		} else {
			if (T(ssm_in, "ssm_in.weight") || T(ssm_out, "ssm_out.weight"))
				return -1;
			if (L->ssm_in.N != h->d_in_proj) {
				snprintf(err, errlen, "ssm_in shape mismatch");
				return -1;
			}
			L->ssm_in.heat = L->ssm_out.heat = 1;
			F(conv_w, "ssm_conv1d.weight", h->d_conv * h->conv_ch);
			F(conv_b, "ssm_conv1d.bias", h->conv_ch);
			F(dt_b, "ssm_dt.bias", h->ssm_heads);
			F(A, "ssm_a", h->ssm_heads);
			F(D, "ssm_d", h->ssm_heads);
			F(ssm_norm, "ssm_norm.weight", h->d_inner);
		}
		F(router, "ffn_gate_inp.weight", h->n_expert * E);
		if (T(gate_e, "ffn_gate_exps.weight") || T(up_e, "ffn_up_exps.weight") ||
		    T(down_e, "ffn_down_exps.weight") || T(gate_s, "ffn_gate_shexp.weight") ||
		    T(up_s, "ffn_up_shexp.weight") || T(down_s, "ffn_down_shexp.weight"))
			return -1;
		L->gate_s.heat = L->up_s.heat = L->down_s.heat = 1;
		L->gate_e.heat = L->up_e.heat = L->down_e.heat =
		    (double)h->n_expert_used / h->n_expert;
#undef T
#undef F
	}
	return 0;
}

void
model_free(model *m)
{
	free(m->L);
	gguf_close(&m->g);
}

/* ---------------------------------------------------------------- */
/* state                                                            */

static float *
fa(size_t n)
{
	return calloc(n ? n : 1, sizeof(float));
}

int
state_init(state *s, const model *m, int ctx)
{
	const hparams *h = &m->hp;
	int i, kvd = h->n_head_kv * h->head_dim;
	size_t big = (size_t)(h->d_in_proj > h->n_vocab ? h->d_in_proj : h->n_vocab);

	memset(s, 0, sizeof(*s));
	s->ctx = ctx;
	s->x = fa((size_t)h->n_embd);
	s->xn = fa((size_t)h->n_embd);
	s->tmp = fa((size_t)h->n_embd);
	s->zxbcdt = fa((size_t)h->d_in_proj);
	s->xbc = fa((size_t)h->conv_ch);
	s->y = fa((size_t)h->d_inner);
	s->q = fa((size_t)h->n_embd);
	s->k = fa((size_t)kvd);
	s->v = fa((size_t)kvd);
	s->att = fa((size_t)h->n_head * (size_t)ctx);
	s->ao = fa((size_t)h->n_embd);
	s->rl = fa((size_t)h->n_expert);
	s->hs_g = fa((size_t)h->n_ff_sh);
	s->hs_u = fa((size_t)h->n_ff_sh);
	s->he_g = fa((size_t)h->n_expert_used * (size_t)h->n_ff);
	s->he_u = fa((size_t)h->n_expert_used * (size_t)h->n_ff);
	s->eo = fa((size_t)h->n_expert_used * (size_t)h->n_embd);
	s->so = fa((size_t)h->n_embd);
	s->logits = fa(big);
	s->conv = calloc((size_t)h->n_layer, sizeof(float *));
	s->ssm = calloc((size_t)h->n_layer, sizeof(float *));
	s->kc = calloc((size_t)h->n_layer, sizeof(float *));
	s->vc = calloc((size_t)h->n_layer, sizeof(float *));
	if (!s->conv || !s->ssm || !s->kc || !s->vc || !s->logits)
		return -1;
	for (i = 0; i < h->n_layer; i++) {
		if (m->L[i].attn) {
			s->kc[i] = fa((size_t)ctx * (size_t)kvd);
			s->vc[i] = fa((size_t)ctx * (size_t)kvd);
			if (!s->kc[i] || !s->vc[i])
				return -1;
		} else {
			s->conv[i] = fa((size_t)(h->d_conv - 1) * (size_t)h->conv_ch);
			s->ssm[i] = fa((size_t)h->d_inner * (size_t)h->d_state);
			if (!s->conv[i] || !s->ssm[i])
				return -1;
		}
	}
	return 0;
}

void
state_reset(state *s, const model *m)
{
	const hparams *h = &m->hp;
	int i;
	s->pos = 0;
	for (i = 0; i < h->n_layer; i++) {
		if (s->conv[i])
			memset(s->conv[i], 0, sizeof(float) * (size_t)(h->d_conv - 1) * (size_t)h->conv_ch);
		if (s->ssm[i])
			memset(s->ssm[i], 0, sizeof(float) * (size_t)h->d_inner * (size_t)h->d_state);
	}
}

void
state_free(state *s, const model *m)
{
	int i;
	for (i = 0; i < m->hp.n_layer; i++) {
		free(s->conv[i]);
		free(s->ssm[i]);
		free(s->kc[i]);
		free(s->vc[i]);
	}
	free(s->conv); free(s->ssm); free(s->kc); free(s->vc);
	free(s->x); free(s->xn); free(s->tmp); free(s->zxbcdt); free(s->xbc);
	free(s->y); free(s->q); free(s->k); free(s->v); free(s->att); free(s->ao);
	free(s->rl); free(s->hs_g); free(s->hs_u); free(s->he_g); free(s->he_u);
	free(s->eo); free(s->so); free(s->logits);
	memset(s, 0, sizeof(*s));
}

/* ---------------------------------------------------------------- */
/* matvec batches                                                   */

#define MV_MAX 32

typedef struct {
	const mv_item *it;
	int n;
	const q8k *xq[MV_MAX];
	size_t *cum;            /* prefix sum of row bytes, in units of rows */
	size_t total;
} mv_job;

static q8k *qbuf;
static size_t qbuf_cap;

static void
mv_worker(void *arg, int ith, int nth)
{
	mv_job *j = arg;
	size_t lo = j->total * (size_t)ith / (size_t)nth;
	size_t hi = j->total * (size_t)(ith + 1) / (size_t)nth;
	int i = 0;

	/* walk the concatenated row space [lo, hi) */
	while (i < j->n && j->cum[i + 1] <= lo)
		i++;
	for (; i < j->n && j->cum[i] < hi; i++) {
		const mv_item *t = &j->it[i];
		const wt *w = t->w;
		size_t r0 = lo > j->cum[i] ? lo - j->cum[i] : 0;
		size_t r1 = (hi < j->cum[i + 1] ? hi : j->cum[i + 1]) - j->cum[i];
		const uint8_t *base = w->data + (size_t)t->e * (size_t)w->N * w->row_bytes;
		size_t r;

		if (g_ref || w->type == GT_F32) {
			for (r = r0; r < r1; r++)
				t->y[r] = dot_row(w->type, base + r * w->row_bytes, t->x, (size_t)w->K);
		} else {
			for (r = r0; r < r1; r++)
				t->y[r] = kq_dot(w->type, base + r * w->row_bytes, j->xq[i], (size_t)w->K);
		}
	}
}

void
mv_run(const mv_item *it, int n)
{
	mv_item cpu[MV_MAX];
	size_t cum[MV_MAX + 1], need = 0, used = 0;
	mv_job job;
	int i, k, nc = 0, ngpu = 0;

	if (vk_ready()) {
		vk_begin();
		for (i = 0; i < n; i++) {
			if (it[i].w->gpu && vk_add(it[i].w, it[i].e, it[i].x, it[i].y) == 0)
				ngpu++;
			else
				cpu[nc++] = it[i];
		}
		if (ngpu)
			vk_submit();
	} else {
		memcpy(cpu, it, sizeof(*it) * (size_t)n);
		nc = n;
	}

	if (nc) {
		memset(&job, 0, sizeof(job));
		job.it = cpu;
		job.n = nc;
		job.cum = cum;
		cum[0] = 0;
		for (i = 0; i < nc; i++) {
			cum[i + 1] = cum[i] + (size_t)cpu[i].w->N;
			need += (size_t)cpu[i].w->K / 256;
		}
		job.total = cum[nc];
		if (!g_ref) {
			if (need > qbuf_cap) {
				free(qbuf);
				qbuf = malloc(need * sizeof(q8k));
				qbuf_cap = qbuf ? need : 0;
				if (!qbuf) {
					fprintf(stderr, "out of memory\n");
					exit(1);
				}
			}
			/* quantize each distinct input once */
			for (i = 0; i < nc; i++) {
				for (k = 0; k < i; k++)
					if (cpu[k].x == cpu[i].x && cpu[k].w->K == cpu[i].w->K)
						break;
				if (k < i) {
					job.xq[i] = job.xq[k];
					continue;
				}
				job.xq[i] = qbuf + used;
				q8k_quantize(cpu[i].x, qbuf + used, (size_t)cpu[i].w->K);
				used += (size_t)cpu[i].w->K / 256;
			}
		}
		pool_run(mv_worker, &job);
	}
	if (ngpu)
		vk_wait();
}

static void
mv1(const wt *w, int e, const float *x, float *y)
{
	mv_item it;
	it.w = w;
	it.e = e;
	it.x = x;
	it.y = y;
	mv_run(&it, 1);
}

/* debug: Q2K_TRACE=1 prints tensor sums named like llama.cpp's graph */
static int trace = -1;

static void
tr(const char *name, int il, const float *v, int n)
{
	double s = 0;
	int i;
	if (trace < 0)
		trace = getenv("Q2K_TRACE") != NULL;
	if (!trace)
		return;
	for (i = 0; i < n; i++)
		s += v[i];
	fprintf(stderr, "%s-%d sum = %f  [%.4f %.4f %.4f]\n", name, il, s, v[0], v[1], v[2]);
}

/* ---------------------------------------------------------------- */
/* elementwise helpers                                              */

static void
rmsnorm(float *o, const float *x, const float *w, int n, float eps)
{
	double ss = 0;
	float sc;
	int i;
	for (i = 0; i < n; i++)
		ss += (double)x[i] * x[i];
	sc = 1.0f / sqrtf((float)(ss / n) + eps);
	for (i = 0; i < n; i++)
		o[i] = x[i] * sc * w[i];
}

static float
silu(float x)
{
	return x / (1.0f + expf(-x));
}

static float
softplus(float x)
{
	return x > 20.0f ? x : log1pf(expf(x));
}

/* ---------------------------------------------------------------- */
/* mamba2 mixer                                                     */

typedef struct {
	const model *m;
	const layer *L;
	state *s;
	const float *dt;        /* softplus'd, per head */
} scan_job;

/* one head: s = s*exp(dt*A) + dt*x*B ; y = s.C + D*x */
static void
scan_worker(void *arg, int ith, int nth)
{
	scan_job *j = arg;
	const hparams *h = &j->m->hp;
	const int P = h->ssm_head_dim, NS = h->d_state;
	const float *x = j->s->xbc, *B = x + h->d_inner, *C = B + NS;
	float *st = j->s->ssm[j->L - j->m->L];
	int hd;

	for (hd = ith; hd < h->ssm_heads; hd += nth) {
		const float dt = j->dt[hd];
		const float dA = expf(dt * j->L->A[hd]);
		const float Dh = j->L->D[hd];
		int p;
		for (p = 0; p < P; p++) {
			const int ii = hd * P + p;
			const float xdt = x[ii] * dt;
			float *sv = st + (size_t)ii * (size_t)NS;
			float acc[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
			float yv;
			int n, l;
			for (n = 0; n < NS; n += 8)
				for (l = 0; l < 8; l++) {
					float v = sv[n + l] * dA + B[n + l] * xdt;
					sv[n + l] = v;
					acc[l] += v * C[n + l];
				}
			yv = ((acc[0] + acc[4]) + (acc[1] + acc[5])) +
			    ((acc[2] + acc[6]) + (acc[3] + acc[7]));
			j->s->y[ii] = yv + Dh * x[ii];
		}
	}
}

static void
mamba2(const model *m, const layer *L, state *s, int il)
{
	const hparams *h = &m->hp;
	const int CH = h->conv_ch, DI = h->d_inner;
	float *z = s->zxbcdt, *xbc_in = s->zxbcdt + DI, *dtr = xbc_in + CH;
	float *cs = s->conv[il];
	float dt[256];
	scan_job sj;
	int c, i;

	mv1(&L->ssm_in, 0, s->xn, s->zxbcdt);

	/* causal depthwise conv over [cs0, cs1, cs2, new], then shift */
	for (c = 0; c < CH; c++) {
		const float *w = L->conv_w + 4 * c;
		float v = cs[c] * w[0] + cs[CH + c] * w[1] + cs[2 * CH + c] * w[2] +
		    xbc_in[c] * w[3] + L->conv_b[c];
		cs[c] = cs[CH + c];
		cs[CH + c] = cs[2 * CH + c];
		cs[2 * CH + c] = xbc_in[c];
		s->xbc[c] = silu(v);
	}
	for (i = 0; i < h->ssm_heads; i++)
		dt[i] = softplus(dtr[i] + L->dt_b[i]);

	sj.m = m;
	sj.L = L;
	sj.s = s;
	sj.dt = dt;
	pool_run(scan_worker, &sj);

	/* gate with silu(z), grouped (n_group = 1) rms norm */
	for (i = 0; i < DI; i++)
		s->y[i] *= silu(z[i]);
	rmsnorm(s->y, s->y, L->ssm_norm, DI, h->eps);

	mv1(&L->ssm_out, 0, s->y, s->tmp);
}

/* ---------------------------------------------------------------- */
/* attention (no positional encoding in this model)                 */

typedef struct {
	const model *m;
	state *s;
	int il, n;
} att_job;

static void
att_worker(void *arg, int ith, int nth)
{
	att_job *j = arg;
	const hparams *h = &j->m->hp;
	const int hd = h->head_dim, kvd = h->n_head_kv * hd;
	const int grp = h->n_head / h->n_head_kv;
	const float *kc = j->s->kc[j->il], *vc = j->s->vc[j->il];
	int hh;

	for (hh = ith; hh < h->n_head; hh += nth) {
		const float *q = j->s->q + hh * hd;
		const int off = (hh / grp) * hd;
		float *a = j->s->att + (size_t)hh * (size_t)j->s->ctx;
		float *o = j->s->ao + hh * hd;
		float mx = -INFINITY, sum = 0;
		int t, d;
		for (t = 0; t < j->n; t++) {
			const float *k = kc + (size_t)t * kvd + off;
			float sc = 0;
			for (d = 0; d < hd; d++)
				sc += q[d] * k[d];
			a[t] = sc * h->attn_scale;
			if (a[t] > mx)
				mx = a[t];
		}
		for (t = 0; t < j->n; t++) {
			a[t] = expf(a[t] - mx);
			sum += a[t];
		}
		for (d = 0; d < hd; d++)
			o[d] = 0;
		for (t = 0; t < j->n; t++) {
			const float *v = vc + (size_t)t * kvd + off;
			const float p = a[t] / sum;
			for (d = 0; d < hd; d++)
				o[d] += p * v[d];
		}
	}
}

static void
attention(const model *m, const layer *L, state *s, int il)
{
	const hparams *h = &m->hp;
	const int kvd = h->n_head_kv * h->head_dim;
	mv_item it[3];
	att_job aj;

	it[0].w = &L->q; it[0].e = 0; it[0].x = s->xn; it[0].y = s->q;
	it[1].w = &L->k; it[1].e = 0; it[1].x = s->xn; it[1].y = s->kc[il] + (size_t)s->pos * kvd;
	it[2].w = &L->v; it[2].e = 0; it[2].x = s->xn; it[2].y = s->vc[il] + (size_t)s->pos * kvd;
	mv_run(it, 3);

	aj.m = m;
	aj.s = s;
	aj.il = il;
	aj.n = s->pos + 1;
	pool_run(att_worker, &aj);

	mv1(&L->o, 0, s->ao, s->tmp);
}

/* ---------------------------------------------------------------- */
/* mixture of experts + shared expert                               */

static void
moe(const model *m, const layer *L, state *s)
{
	const hparams *h = &m->hp;
	const int E = h->n_embd, K = h->n_expert_used, FF = h->n_ff;
	mv_item it[MV_MAX];
	int sel[16], i, k, n;
	float wsum = 0, wk[16], mx;

	/* router: softmax over experts, top-k, renormalise (= softmax over
	 * the chosen logits) */
	for (i = 0; i < h->n_expert; i++) {
		const float *r = L->router + (size_t)i * E;
		float acc = 0;
		int d;
		for (d = 0; d < E; d++)
			acc += r[d] * s->xn[d];
		s->rl[i] = acc;
	}
	for (k = 0; k < K; k++) {
		int best = -1;
		for (i = 0; i < h->n_expert; i++) {
			int used = 0, q;
			for (q = 0; q < k; q++)
				used |= sel[q] == i;
			if (!used && (best < 0 || s->rl[i] > s->rl[best]))
				best = i;
		}
		sel[k] = best;
	}
	mx = s->rl[sel[0]];
	for (k = 0; k < K; k++) {
		wk[k] = expf(s->rl[sel[k]] - mx);
		wsum += wk[k];
	}

	/* gate/up for shared + routed experts in one batch */
	n = 0;
	it[n].w = &L->gate_s; it[n].e = 0; it[n].x = s->xn; it[n].y = s->hs_g; n++;
	it[n].w = &L->up_s; it[n].e = 0; it[n].x = s->xn; it[n].y = s->hs_u; n++;
	for (k = 0; k < K; k++) {
		it[n].w = &L->gate_e; it[n].e = sel[k]; it[n].x = s->xn; it[n].y = s->he_g + k * FF; n++;
		it[n].w = &L->up_e; it[n].e = sel[k]; it[n].x = s->xn; it[n].y = s->he_u + k * FF; n++;
	}
	mv_run(it, n);

	for (i = 0; i < h->n_ff_sh; i++)
		s->hs_g[i] = silu(s->hs_g[i]) * s->hs_u[i];
	for (i = 0; i < K * FF; i++)
		s->he_g[i] = silu(s->he_g[i]) * s->he_u[i];

	n = 0;
	it[n].w = &L->down_s; it[n].e = 0; it[n].x = s->hs_g; it[n].y = s->so; n++;
	for (k = 0; k < K; k++) {
		it[n].w = &L->down_e; it[n].e = sel[k]; it[n].x = s->he_g + k * FF;
		it[n].y = s->eo + k * E; n++;
	}
	mv_run(it, n);

	for (i = 0; i < E; i++) {
		float acc = 0;
		for (k = 0; k < K; k++)
			acc += wk[k] / wsum * s->eo[k * E + i];
		s->tmp[i] = acc;
	}
	tr("ffn_moe_out", (int)(L - m->L), s->tmp, E);
	tr("ffn_shexp", (int)(L - m->L), s->so, E);
	for (i = 0; i < E; i++)
		s->tmp[i] += s->so[i];
}

/* ---------------------------------------------------------------- */

const float *
model_forward(const model *m, state *s, int tok, int want_logits)
{
	const hparams *h = &m->hp;
	const int E = h->n_embd;
	int il, i;

	if (tok < 0 || tok >= h->n_vocab)
		tok = 0;
	dequant_row(m->tok_embd.type, m->tok_embd.data + (size_t)tok * m->tok_embd.row_bytes,
	    s->x, (size_t)E);
	for (i = 0; i < E; i++)
		s->x[i] *= h->emb_scale;
	tr("inp_embd", 0, s->x, E);

	for (il = 0; il < h->n_layer; il++) {
		const layer *L = &m->L[il];

		rmsnorm(s->xn, s->x, L->attn_norm, E, h->eps);
		if (L->attn)
			attention(m, L, s, il);
		else
			mamba2(m, L, s, il);
		tr(L->attn ? "attn_out" : "mamba_out", il, s->tmp, E);
		for (i = 0; i < E; i++)
			s->x[i] += h->res_scale * s->tmp[i];
		tr("ffn_inp", il, s->x, E);

		rmsnorm(s->xn, s->x, L->ffn_norm, E, h->eps);
		moe(m, L, s);
		for (i = 0; i < E; i++)
			s->x[i] += h->res_scale * s->tmp[i];
		tr("l_out", il, s->x, E);
	}
	s->pos++;
	if (!want_logits)
		return NULL;

	rmsnorm(s->xn, s->x, m->out_norm, E, h->eps);
	mv1(&m->tok_embd, 0, s->xn, s->logits);
	for (i = 0; i < h->n_vocab; i++)
		s->logits[i] /= h->logit_scale;
	return s->logits;
}
