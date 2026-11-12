/*
 * Batched prompt processing on the CPU: chunks of up to PB tokens go
 * through each layer together, so every weight row is dotted with all
 * of the chunk's (quantized) tokens while it sits in L1.  Mamba2 conv
 * and scan run per channel / head over the chunk's tokens in order;
 * MoE groups the tokens by expert; attention fills the KV cache and
 * attends causally.  Only the last token's logits are computed.
 *
 * The GPU's share of the recurrent state (split mixers) is fetched
 * before and written back after, so decoding continues seamlessly.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kq.h"
#include "model.h"
#include "pool.h"
#include "quant.h"
#include "vk.h"
#include "vmath.h"

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define PB 128
#define MM_MAX 160

#include <time.h>
static double
now(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}
/* Q2K_PREFILL_PROF: matmul, conv, scan, attention, routing, other */
double pf_t[8];
long pf_rows;

/* ---------------------------------------------------------------- */
/* batched matmul: y[t*ldy + r] = dot(W[r], x_t), rows [r0, r1)      */

typedef struct {
	const wt *w;
	int e;
	const q8k *xq;          /* T vectors of xs blocks each */
	int xs, T;
	float *y;
	int ldy, r0, r1;
} mm_item;

typedef struct {
	const mm_item *it;
	int n;
	size_t cum[MM_MAX + 1];
	long next, chunk;
} mm_job;

static void
mm_worker(void *arg, int ith, int nth)
{
	mm_job *j = arg;
	long c;
	int i = 0;

	(void)ith;
	(void)nth;
	while ((c = pool_take(&j->next, j->chunk, (long)j->cum[j->n])) >= 0) {
		size_t lo = (size_t)c, hi = lo + (size_t)j->chunk;
		if (hi > j->cum[j->n])
			hi = j->cum[j->n];
		while (i > 0 && j->cum[i] > lo)
			i--;
		while (j->cum[i + 1] <= lo)
			i++;
		for (; i < j->n && j->cum[i] < hi; i++) {
			const mm_item *m = &j->it[i];
			const wt *w = m->w;
			size_t r0 = (lo > j->cum[i] ? lo - j->cum[i] : 0) + (size_t)m->r0;
			size_t r1 = (hi < j->cum[i + 1] ? hi : j->cum[i + 1]) - j->cum[i] + (size_t)m->r0;
			const uint8_t *base = w->data + (size_t)m->e * (size_t)w->N * w->row_bytes;
			size_t r;
			int t;

			for (r = r0; r < r1; r++) {
				const uint8_t *row = base + r * w->row_bytes;
				if (m->T > 1) {
					int t0, ok = 1;
					for (t0 = 0; t0 < m->T && ok; t0 += 64)
						ok = kq_dotT(w->type, row, m->xq + (size_t)t0 * (size_t)m->xs,
						    (size_t)m->xs, m->T - t0 > 64 ? 64 : m->T - t0, (size_t)w->K,
						    m->y + (size_t)t0 * (size_t)m->ldy + r, (size_t)m->ldy);
					if (ok)
						continue;
				}
				for (t = 0; t < m->T; t++)
					m->y[(size_t)t * (size_t)m->ldy + r] =
					    kq_dot(w->type, row, m->xq + (size_t)t * (size_t)m->xs, (size_t)w->K);
			}
		}
		if (i > 0)
			i--;
	}
}

static void
mm_run(const mm_item *it, int n)
{
	mm_job j;
	size_t work = 0;
	int i;

	if (n == 0)
		return;
	j.it = it;
	j.n = n;
	j.cum[0] = 0;
	for (i = 0; i < n; i++) {
		j.cum[i + 1] = j.cum[i] + (size_t)(it[i].r1 - it[i].r0);
		work += (size_t)(it[i].r1 - it[i].r0) * it[i].w->row_bytes * (size_t)it[i].T;
	}
	/* chunks of ~256 KiB of row x token work, at least 8 per thread */
	j.chunk = (long)(j.cum[n] * (256u << 10) / (work ? work : 1));
	if (j.chunk > (long)(j.cum[n] / (size_t)(8 * pool_size())))
		j.chunk = (long)(j.cum[n] / (size_t)(8 * pool_size()));
	if (j.chunk < 1)
		j.chunk = 1;
	j.next = 0;
	{
		double t0 = now();
		pool_run(mm_worker, &j);
		pf_t[0] += now() - t0;
		pf_rows += (long)work;
	}
}

static mm_item
mmi(const wt *w, int e, const q8k *xq, int xs, int T, float *y, int ldy)
{
	mm_item m;
	m.w = w;
	m.e = e;
	m.xq = xq;
	m.xs = xs;
	m.T = T;
	m.y = y;
	m.ldy = ldy;
	m.r0 = 0;
	m.r1 = w->N;
	return m;
}

/* ---------------------------------------------------------------- */
/* buffers                                                           */

static struct {
	int ok;
	float *x, *xn, *zx, *xbc, *dt, *y, *tmp, *q, *ao;
	float *hg, *hu, *eo, *sg, *su, *so;
	q8k *xq, *yq, *hq, *sq, *gq;
	int *sel;
	float *wk;
} B;

static int
bufs(const hparams *h)
{
	size_t T = PB, E = (size_t)h->n_embd, K = (size_t)h->n_expert_used;
	if (B.ok)
		return 0;
#define A(p, n) if (!((p) = calloc((n), sizeof(*(p))))) return -1
	A(B.x, T * E); A(B.xn, T * E); A(B.zx, T * (size_t)h->d_in_proj);
	A(B.xbc, T * (size_t)h->conv_ch); A(B.dt, T * (size_t)h->ssm_heads);
	A(B.y, T * (size_t)h->d_inner); A(B.tmp, T * E); A(B.q, T * E); A(B.ao, T * E);
	A(B.hg, T * K * (size_t)h->n_ff); A(B.hu, T * K * (size_t)h->n_ff);
	A(B.eo, T * K * E);
	A(B.sg, T * (size_t)h->n_ff_sh); A(B.su, T * (size_t)h->n_ff_sh); A(B.so, T * E);
	A(B.xq, T * E / 256); A(B.yq, T * (size_t)h->d_inner / 256);
	A(B.hq, T * K * (size_t)h->n_ff / 256); A(B.sq, T * (size_t)h->n_ff_sh / 256);
	A(B.gq, T * K * E / 256);
	A(B.sel, T * K); A(B.wk, T * K);
#undef A
	B.ok = 1;
	return 0;
}

static void
quant_rows(const float *x, q8k *q, int T, int n)
{
	int t;
	for (t = 0; t < T; t++)
		q8k_quantize(x + (size_t)t * (size_t)n, q + (size_t)t * (size_t)(n / 256), (size_t)n);
}

static float
softplus(float x)
{
	return x > 20.0f ? x : log1pf(expf(x));
}

/* ---------------------------------------------------------------- */
/* Mamba2 over T tokens                                              */

typedef struct {
	const model *m;
	const layer *L;
	state *s;
	int il, T;
	long next;
} ssm_job;

/* conv + SiLU per channel over the chunk, in token order */
static void
conv_worker(void *arg, int ith, int nth)
{
	ssm_job *j = arg;
	const hparams *h = &j->m->hp;
	const int CH = h->conv_ch, DI = h->d_inner, DP = h->d_in_proj;
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
				float in = B.zx[(size_t)t * DP + DI + c];
				B.xbc[(size_t)t * CH + c] = s0 * w[0] + s1 * w[1] + s2 * w[2] +
				    in * w[3] + j->L->conv_b[c];
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

/* selective scan per head over the chunk, in token order.  Each
 * channel's state (d_state floats) stays in registers, half a row at a
 * time, across all of the chunk's tokens: per 8 states and token one
 * mul + two FMAs and two loads (B, C), instead of also a load and a
 * store of the state. */
static void
scan_worker(void *arg, int ith, int nth)
{
	ssm_job *j = arg;
	const hparams *h = &j->m->hp;
	const int P = h->ssm_head_dim, NS = h->d_state, CH = h->conv_ch, DI = h->d_inner;
	const int NH = h->ssm_heads, T = j->T;
	float *st = j->s->ssm[j->il];
	float dA[PB], dt[PB];
	long hd;
	int t;

	(void)ith;
	(void)nth;
	while ((hd = pool_take(&j->next, 1, NH)) >= 0) {
		const float A = j->L->A[hd], Dh = j->L->D[hd];
		int p;
		for (t = 0; t < T; t++) {
			dt[t] = B.dt[(size_t)t * NH + hd];
			dA[t] = expf(dt[t] * A);
		}
		for (p = 0; p < P; p++) {
			const int ii = (int)hd * P + p;
			float *sv = st + (size_t)ii * (size_t)NS;
			int half;
			for (t = 0; t < T; t++)
				B.y[(size_t)t * DI + ii] = Dh * B.xbc[(size_t)t * CH + ii];
			for (half = 0; half < NS; half += 64) {
#if defined(__AVX2__) && defined(__FMA__)
				__m256 s0 = _mm256_loadu_ps(sv + half), s1 = _mm256_loadu_ps(sv + half + 8);
				__m256 s2 = _mm256_loadu_ps(sv + half + 16), s3 = _mm256_loadu_ps(sv + half + 24);
				__m256 s4 = _mm256_loadu_ps(sv + half + 32), s5 = _mm256_loadu_ps(sv + half + 40);
				__m256 s6 = _mm256_loadu_ps(sv + half + 48), s7 = _mm256_loadu_ps(sv + half + 56);
				for (t = 0; t < T; t++) {
					const float *x = B.xbc + (size_t)t * CH;
					const float *Bv = x + DI + half, *C = x + DI + NS + half;
					__m256 a = _mm256_set1_ps(dA[t]), xd = _mm256_set1_ps(x[ii] * dt[t]), acc;
					__m128 r;
#define ST(sk, o) sk = _mm256_fmadd_ps(sk, a, _mm256_mul_ps(_mm256_loadu_ps(Bv + o), xd))
					ST(s0, 0); ST(s1, 8); ST(s2, 16); ST(s3, 24);
					ST(s4, 32); ST(s5, 40); ST(s6, 48); ST(s7, 56);
#undef ST
					acc = _mm256_mul_ps(s0, _mm256_loadu_ps(C));
					acc = _mm256_fmadd_ps(s1, _mm256_loadu_ps(C + 8), acc);
					acc = _mm256_fmadd_ps(s2, _mm256_loadu_ps(C + 16), acc);
					acc = _mm256_fmadd_ps(s3, _mm256_loadu_ps(C + 24), acc);
					acc = _mm256_fmadd_ps(s4, _mm256_loadu_ps(C + 32), acc);
					acc = _mm256_fmadd_ps(s5, _mm256_loadu_ps(C + 40), acc);
					acc = _mm256_fmadd_ps(s6, _mm256_loadu_ps(C + 48), acc);
					acc = _mm256_fmadd_ps(s7, _mm256_loadu_ps(C + 56), acc);
					r = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
					r = _mm_add_ps(r, _mm_movehl_ps(r, r));
					r = _mm_add_ss(r, _mm_movehdup_ps(r));
					B.y[(size_t)t * DI + ii] += _mm_cvtss_f32(r);
				}
				_mm256_storeu_ps(sv + half, s0); _mm256_storeu_ps(sv + half + 8, s1);
				_mm256_storeu_ps(sv + half + 16, s2); _mm256_storeu_ps(sv + half + 24, s3);
				_mm256_storeu_ps(sv + half + 32, s4); _mm256_storeu_ps(sv + half + 40, s5);
				_mm256_storeu_ps(sv + half + 48, s6); _mm256_storeu_ps(sv + half + 56, s7);
#else
				int n;
				for (t = 0; t < T; t++) {
					const float *x = B.xbc + (size_t)t * CH;
					const float *Bv = x + DI, *C = x + DI + NS;
					float acc = 0, xd = x[ii] * dt[t];
					for (n = half; n < half + 64 && n < NS; n++) {
						sv[n] = sv[n] * dA[t] + Bv[n] * xd;
						acc += sv[n] * C[n];
					}
					B.y[(size_t)t * DI + ii] += acc;
				}
#endif
			}
		}
	}
}

static void
mamba2_batch(const model *m, const layer *L, state *s, int il, int T)
{
	const hparams *h = &m->hp;
	const int E = h->n_embd, DI = h->d_inner, CH = h->conv_ch, DP = h->d_in_proj;
	const int NH = h->ssm_heads;
	mm_item it[2];
	ssm_job sj;
	int t, i;

	for (t = 0; t < T; t++)
		v_rmsnorm(B.xn + (size_t)t * E, B.x + (size_t)t * E, L->attn_norm, (size_t)E, h->eps);
	quant_rows(B.xn, B.xq, T, E);
	it[0] = mmi(&L->ssm_in, 0, B.xq, E / 256, T, B.zx, DP);
	mm_run(it, 1);

	sj.m = m;
	sj.L = L;
	sj.s = s;
	sj.il = il;
	sj.T = T;
	sj.next = 0;
	{
		double t0 = now();
		pool_run(conv_worker, &sj);
		pf_t[1] += now() - t0;
	}
	for (t = 0; t < T; t++) {
		v_silu(B.xbc + (size_t)t * CH, (size_t)CH);
		for (i = 0; i < NH; i++)
			B.dt[(size_t)t * NH + i] = softplus(B.zx[(size_t)t * DP + DI + CH + i] + L->dt_b[i]);
	}
	sj.next = 0;
	{
		double t0 = now();
		pool_run(scan_worker, &sj);
		pf_t[2] += now() - t0;
	}

	for (t = 0; t < T; t++) {
		float *y = B.y + (size_t)t * DI;
		v_mul_silu(y, B.zx + (size_t)t * DP, (size_t)DI);
		v_rmsnorm(y, y, L->ssm_norm, (size_t)DI, h->eps);
	}
	quant_rows(B.y, B.yq, T, DI);
	it[0] = mmi(&L->ssm_out, 0, B.yq, DI / 256, T, B.tmp, E);
	mm_run(it, 1);
}

/* ---------------------------------------------------------------- */
/* attention over T tokens                                           */

typedef struct {
	const model *m;
	state *s;
	int il, T, pos;
	long next;
} att_job;

static void
att_worker(void *arg, int ith, int nth)
{
	att_job *j = arg;
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
		int t = (int)(w / h->n_head), hh = (int)(w % h->n_head), n = j->pos + t + 1, k;
		const float *q = B.q + (size_t)t * h->n_embd + hh * hd;
		const int off = (hh / grp) * hd;
		float *o = B.ao + (size_t)t * h->n_embd + hh * hd, mx = -INFINITY, sum = 0;
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
	}
	free(a);
}

static void
attention_batch(const model *m, const layer *L, state *s, int il, int T)
{
	const hparams *h = &m->hp;
	const int E = h->n_embd, kvd = h->n_head_kv * h->head_dim;
	mm_item it[3];
	att_job aj;
	int t;

	for (t = 0; t < T; t++)
		v_rmsnorm(B.xn + (size_t)t * E, B.x + (size_t)t * E, L->attn_norm, (size_t)E, h->eps);
	quant_rows(B.xn, B.xq, T, E);
	it[0] = mmi(&L->q, 0, B.xq, E / 256, T, B.q, E);
	it[1] = mmi(&L->k, 0, B.xq, E / 256, T, s->kc[il] + (size_t)s->pos * kvd, kvd);
	it[2] = mmi(&L->v, 0, B.xq, E / 256, T, s->vc[il] + (size_t)s->pos * kvd, kvd);
	mm_run(it, 3);
	aj.m = m;
	aj.s = s;
	aj.il = il;
	aj.T = T;
	aj.pos = s->pos;
	aj.next = 0;
	{
		double t0 = now();
		pool_run(att_worker, &aj);
		pf_t[3] += now() - t0;
	}
	quant_rows(B.ao, B.gq, T, E);
	it[0] = mmi(&L->o, 0, B.gq, E / 256, T, B.tmp, E);
	mm_run(it, 1);
}

/* ---------------------------------------------------------------- */
/* MoE over T tokens: tokens grouped by expert                       */

typedef struct {
	const model *m;
	const layer *L;
	int T;
	long next;
} route_job;

/* router logits, softmax top-k, renormalised weights; per token */
static void
route_worker(void *arg, int ith, int nth)
{
	route_job *j = arg;
	const hparams *h = &j->m->hp;
	const int E = h->n_embd, K = h->n_expert_used, NE = h->n_expert;
	long t;

	(void)ith;
	(void)nth;
	while ((t = pool_take(&j->next, 1, j->T)) >= 0) {
		float rl[256], mx, ws = 0;
		int *sel = B.sel + t * K, e, k, q;
		for (e = 0; e < NE; e++)
			rl[e] = v_dot(j->L->router + (size_t)e * E, B.xn + (size_t)t * E, (size_t)E);
		for (k = 0; k < K; k++) {
			int best = -1;
			for (e = 0; e < NE; e++) {
				int used = 0;
				for (q = 0; q < k; q++)
					used |= sel[q] == e;
				if (!used && (best < 0 || rl[e] > rl[best]))
					best = e;
			}
			sel[k] = best;
		}
		mx = rl[sel[0]];
		for (k = 0; k < K; k++)
			ws += B.wk[t * K + k] = expf(rl[sel[k]] - mx);
		for (k = 0; k < K; k++)
			B.wk[t * K + k] /= ws;
	}
}

static void
moe_batch(const model *m, const layer *L, int T)
{
	const hparams *h = &m->hp;
	const int E = h->n_embd, K = h->n_expert_used, FF = h->n_ff, FS = h->n_ff_sh;
	const int NE = h->n_expert, eb = E / 256, fb = FF / 256;
	int cnt[256], base[257], fill[256], slot_t[PB * 16], slot_k[PB * 16];
	mm_item it[MM_MAX];
	int t, k, e, i, n;

	for (t = 0; t < T; t++)
		v_rmsnorm(B.xn + (size_t)t * E, B.x + (size_t)t * E, L->ffn_norm, (size_t)E, h->eps);
	quant_rows(B.xn, B.xq, T, E);

	/* routing per token: softmax top-k, renormalised */
	double t0 = now();
	{
		route_job rj;
		rj.m = m;
		rj.L = L;
		rj.T = T;
		rj.next = 0;
		pool_run(route_worker, &rj);
	}
	memset(cnt, 0, sizeof(cnt));
	for (i = 0; i < T * K; i++)
		cnt[B.sel[i]]++;
	base[0] = 0;
	for (e = 0; e < NE; e++) {
		base[e + 1] = base[e] + cnt[e];
		fill[e] = 0;
	}
	/* gather each expert's tokens' inputs contiguously */
	for (t = 0; t < T; t++)
		for (k = 0; k < K; k++) {
			int s = base[B.sel[t * K + k]] + fill[B.sel[t * K + k]]++;
			slot_t[s] = t;
			slot_k[s] = k;
			memcpy(B.gq + (size_t)s * eb, B.xq + (size_t)t * eb, sizeof(q8k) * (size_t)eb);
		}

	pf_t[4] += now() - t0;
	/* gate/up: shared expert over all tokens, routed experts over theirs */
	n = 0;
	it[n++] = mmi(&L->gate_s, 0, B.xq, eb, T, B.sg, FS);
	it[n++] = mmi(&L->up_s, 0, B.xq, eb, T, B.su, FS);
	for (e = 0; e < NE; e++)
		if (cnt[e]) {
			it[n++] = mmi(&L->gate_e, e, B.gq + (size_t)base[e] * eb, eb, cnt[e],
			    B.hg + (size_t)base[e] * FF, FF);
			it[n++] = mmi(&L->up_e, e, B.gq + (size_t)base[e] * eb, eb, cnt[e],
			    B.hu + (size_t)base[e] * FF, FF);
		}
	mm_run(it, n);

	v_swiglu(B.sg, B.su, (size_t)T * FS);
	v_swiglu(B.hg, B.hu, (size_t)T * K * FF);
	quant_rows(B.sg, B.sq, T, FS);
	quant_rows(B.hg, B.hq, T * K, FF);

	n = 0;
	it[n++] = mmi(&L->down_s, 0, B.sq, FS / 256, T, B.so, E);
	for (e = 0; e < NE; e++)
		if (cnt[e])
			it[n++] = mmi(&L->down_e, e, B.hq + (size_t)base[e] * fb, fb, cnt[e],
			    B.eo + (size_t)base[e] * E, E);
	mm_run(it, n);

	memcpy(B.tmp, B.so, sizeof(float) * (size_t)T * E);
	for (i = 0; i < T * K; i++)
		v_axpy(B.tmp + (size_t)slot_t[i] * E, B.wk[slot_t[i] * K + slot_k[i]],
		    B.eo + (size_t)i * E, (size_t)E);
}

/* ---------------------------------------------------------------- */

static void
chunk(const model *m, state *s, const int *tok, int T)
{
	const hparams *h = &m->hp;
	const int E = h->n_embd;
	int il, t;

	for (t = 0; t < T; t++) {
		int id = tok[t] < 0 || tok[t] >= h->n_vocab ? 0 : tok[t];
		float *x = B.x + (size_t)t * E;
		int i;
		dequant_row(m->tok_embd.type, m->tok_embd.data + (size_t)id * m->tok_embd.row_bytes,
		    x, (size_t)E);
		for (i = 0; i < E; i++)
			x[i] *= h->emb_scale;
	}
	for (il = 0; il < h->n_layer; il++) {
		const layer *L = &m->L[il];
		if (L->attn)
			attention_batch(m, L, s, il, T);
		else
			mamba2_batch(m, L, s, il, T);
		v_axpy(B.x, h->res_scale, B.tmp, (size_t)T * E);
		moe_batch(m, L, T);
		v_axpy(B.x, h->res_scale, B.tmp, (size_t)T * E);
	}
	s->pos += T;
}

const float *
model_prefill(const model *m, state *s, const int *tok, int n)
{
	const hparams *h = &m->hp;
	int done = 0, il;

	if (n <= 1 || g_ref || bufs(h) != 0) {
		const float *lg = NULL;
		int i;
		for (i = 0; i < n; i++)
			lg = model_forward(m, s, tok[i], i == n - 1);
		return lg;
	}
	/* the GPU's heads' state comes back to the CPU for the batch */
	for (il = 0; il < h->n_layer; il++)
		vk_state_sync(m, s, il, 0, 0, vk_mamba_heads(il));
	/* all but the last token in chunks; the last one through the
	 * regular path, which also produces the logits */
	{
	double t0 = now();
	while (done < n - 1) {
		int T = n - 1 - done > PB ? PB : n - 1 - done;
		chunk(m, s, tok + done, T);
		done += T;
		/* the residual stream of the last token feeds nothing else */
	}
	pf_t[5] += now() - t0;
	if (getenv("Q2K_PREFILL_PROF"))
		fprintf(stderr, "prefill %d tok: total %.3f s: matmul %.3f (%.1f GB/s of row x token work), "
		    "conv %.3f, scan %.3f, attn %.3f, routing %.3f\n", n - 1, pf_t[5], pf_t[0],
		    pf_rows / pf_t[0] / 1e9, pf_t[1], pf_t[2], pf_t[3], pf_t[4]);
	}
	for (il = 0; il < h->n_layer; il++)
		vk_state_sync(m, s, il, 1, 0, vk_mamba_heads(il));
	return model_forward(m, s, tok[n - 1], 1);
}
