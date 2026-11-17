/* q2k: command line front end and server for the GGUF kernel. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "gguf.h"
#include "kq.h"
#include "model.h"
#include "pool.h"
#include "quant.h"
#include "server.h"
#include "tok.h"
#include "vk.h"

#define DEFAULT_MODEL "/tank/models/MiniCPM5-2B-Q8_0.gguf"
#define DEFAULT_SYSTEM "You are a helpful assistant. Please ensure responses are professional, accurate, and safe."

typedef struct {
	const char *model, *prompt, *system, *dump, *transcode, *host, *alias, *api_key;
	int server, port, n_given;
	int n_predict, threads, ctx, gpu_mib, device, chat, raw, info, tokens;
	int bench, check, kbench, top_k, verbose, show_special, logits, noprefill, think, sys_given, nomtp, ndraft;
	float temp, top_p;
	unsigned long seed;
} opts;

static model M;
static tokenizer T;
static state S;
static opts O;

static double
now(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void
usage(void)
{
	fprintf(stderr,
	    "usage: q2k [options] [prompt]\n"
	    "  -m PATH        model (default " DEFAULT_MODEL ", or $Q2K_MODEL)\n"
	    "  -p TEXT        prompt (completion mode)\n"
	    "  -n N           tokens to generate (default 256)\n"
	    "  -i, --chat     interactive chat using the model's template\n"
	    "  -s TEXT        system message for chat\n"
	    "  --think        let thinking models think (chat; qwen35)\n"
	    "  -t N           CPU threads (default: online CPUs - 1)\n"
	    "  -c N           context size (default 4096)\n"
	    "  -g MIB         GPU weight split in MiB (default 512, 0 = CPU only)\n"
	    "  -d N           Vulkan device index\n"
	    "  --split F      GPU share of Mamba2 heads and output rows (0.5)\n"
	    "  --vram MIB     cap on all of our GPU memory (default 1280)\n"
	    "  --stream F     prompts: GPU share of each matrix's rows, streaming\n"
	    "                 what is not resident over PCIe (qwen35; 0.5, 0 = off)\n"
	    "  --stream-mib M VRAM for streamed weights (default 64)\n"
	    "  --temp F       temperature (default 0 = greedy)\n"
	    "  --top-k N      (default 40)  --top-p F (default 0.95)  --seed N\n"
	    "  --ref          reference float kernels on the CPU\n"
	    "  --raw          no chat template for -p (default: raw completion)\n"
	    "  --special      print control tokens\n"
	    "  --info         print model metadata and tensor plan\n"
	    "  --tokens       print the prompt's token ids and exit\n"
	    "  --logits       print top-10 next-token logits after the prompt\n"
	    "  --dump FILE    also write all of them as raw f32\n"
	    "  --bench        time prompt processing and generation\n"
	    "  --check        compare CPU/GPU kernels against the reference\n"
	    "  --kbench       single-thread CPU kernel throughput per type\n"
	    "  --no-batch     feed prompts token by token (no batched prefill)\n"
	    "  --no-mtp       greedy decoding without the MTP draft block (qwen35)\n"
	    "  --draft K      MTP drafts per step, 1..3 (default 3; 0 = --no-mtp)\n"
	    "  --draft-p P    chain another draft while its probability >= P (0.5)\n"
	    "  --transcode F  write the model to F with the codebook i-quants in\n"
	    "                 q2k's faster lossless layouts (X2/X3/X1), then exit\n"
	    "  --server       serve the llama.cpp HTTP API (/completion,\n"
	    "                 /v1/chat/completions, ...); -n, --temp etc. are defaults\n"
	    "  --host H       listen address (default 127.0.0.1)\n"
	    "  --port N       listen port (default 8080)\n"
	    "  --alias NAME   model name reported to clients\n"
	    "  --api-key KEY  require \"Authorization: Bearer KEY\"\n"
	    "  -v             verbose\n");
	exit(2);
}

static const char *
expand_home(const char *p)
{
	static char buf[1024];
	const char *h = getenv("HOME");
	if (p[0] == '~' && p[1] == '/' && h) {
		snprintf(buf, sizeof(buf), "%s%s", h, p + 1);
		return buf;
	}
	return p;
}

static void
parse(int argc, char **argv)
{
	int i;
	long nc = sysconf(_SC_NPROCESSORS_ONLN);

	O.model = getenv("Q2K_MODEL") ? getenv("Q2K_MODEL") : DEFAULT_MODEL;
	O.n_predict = 256;
	O.ndraft = 3;
	/* one core left for the GPU driver, the fence polling and the
	 * rest of the system: measured faster than all of them */
	O.threads = nc > 1 ? (int)nc - 1 : 1;
	O.ctx = 4096;
	O.gpu_mib = 512;
	O.device = -1;
	O.top_k = 40;
	O.top_p = 0.95f;
	O.seed = (unsigned long)time(NULL);
	O.system = DEFAULT_SYSTEM;
	O.host = "127.0.0.1";
	O.port = 8080;
	for (i = 1; i < argc; i++) {
		const char *a = argv[i];
#define ARG() (i + 1 < argc ? argv[++i] : (usage(), ""))
		if (!strcmp(a, "-m")) O.model = ARG();
		else if (!strcmp(a, "-p")) O.prompt = ARG();
		else if (!strcmp(a, "-s")) { O.system = ARG(); O.sys_given = 1; }
		else if (!strcmp(a, "--think")) O.think = 1;
		else if (!strcmp(a, "-n")) { O.n_predict = atoi(ARG()); O.n_given = 1; }
		else if (!strcmp(a, "--server")) O.server = 1;
		else if (!strcmp(a, "--host")) O.host = ARG();
		else if (!strcmp(a, "--port")) O.port = atoi(ARG());
		else if (!strcmp(a, "--alias")) O.alias = ARG();
		else if (!strcmp(a, "--api-key")) O.api_key = ARG();
		else if (!strcmp(a, "-t")) O.threads = atoi(ARG());
		else if (!strcmp(a, "-c")) O.ctx = atoi(ARG());
		else if (!strcmp(a, "-g")) O.gpu_mib = atoi(ARG());
		else if (!strcmp(a, "-d")) O.device = atoi(ARG());
		else if (!strcmp(a, "--split")) g_split = atof(ARG());
		else if (!strcmp(a, "--vram")) vk_vram_limit = (size_t)atoi(ARG()) << 20;
		else if (!strcmp(a, "--stream")) g_stream = atof(ARG());
		else if (!strcmp(a, "--stream-mib")) vk_stream_bytes = (size_t)atoi(ARG()) << 20;
		else if (!strcmp(a, "--temp")) O.temp = (float)atof(ARG());
		else if (!strcmp(a, "--top-k")) O.top_k = atoi(ARG());
		else if (!strcmp(a, "--top-p")) O.top_p = (float)atof(ARG());
		else if (!strcmp(a, "--seed")) O.seed = strtoul(ARG(), NULL, 10);
		else if (!strcmp(a, "-i") || !strcmp(a, "--chat")) O.chat = 1;
		else if (!strcmp(a, "--ref")) g_ref = 1;
		else if (!strcmp(a, "--raw")) O.raw = 1;
		else if (!strcmp(a, "--special")) O.show_special = 1;
		else if (!strcmp(a, "--info")) O.info = 1;
		else if (!strcmp(a, "--tokens")) O.tokens = 1;
		else if (!strcmp(a, "--logits")) O.logits = 1;
		else if (!strcmp(a, "--dump")) { O.dump = ARG(); O.logits = 1; }
		else if (!strcmp(a, "--bench")) O.bench = 1;
		else if (!strcmp(a, "--check")) O.check = 1;
		else if (!strcmp(a, "--kbench")) O.kbench = 1;
		else if (!strcmp(a, "--transcode")) O.transcode = ARG();
		else if (!strcmp(a, "--no-batch")) O.noprefill = 1;
		else if (!strcmp(a, "--no-mtp")) O.nomtp = 1;
		else if (!strcmp(a, "--draft")) O.ndraft = atoi(ARG());
		else if (!strcmp(a, "--draft-p")) g_draft_pmin = (float)atof(ARG());
		else if (!strcmp(a, "-v")) O.verbose = 1;
		else if (a[0] == '-') usage();
		else O.prompt = a;
#undef ARG
	}
	if (O.ctx < 16)
		O.ctx = 16;
}

/* ---------------------------------------------------------------- */
/* sampling                                                         */

static unsigned long long rng;

static double
frand(void)
{
	rng ^= rng << 13;
	rng ^= rng >> 7;
	rng ^= rng << 17;
	return (double)(rng >> 11) / 9007199254740992.0;
}

typedef struct {
	float p;
	int id;
} cand;

static int
cand_desc(const void *a, const void *b)
{
	float x = ((const cand *)a)->p, y = ((const cand *)b)->p;
	return x < y ? 1 : x > y ? -1 : 0;
}

static int
sample(const float *lg, int n)
{
	static cand *c;
	int i, k, best = 0;
	double sum = 0, r, acc = 0;
	float mx;

	for (i = 1; i < n; i++)
		if (lg[i] > lg[best])
			best = i;
	if (O.temp <= 0)
		return best;
	if (!c && !(c = malloc(sizeof(*c) * (size_t)n)))
		return best;
	mx = lg[best];
	for (i = 0; i < n; i++) {
		c[i].id = i;
		c[i].p = expf((lg[i] - mx) / O.temp);
	}
	k = O.top_k > 0 && O.top_k < n ? O.top_k : n;
	qsort(c, (size_t)n, sizeof(*c), cand_desc);
	for (i = 0; i < k; i++)
		sum += c[i].p;
	if (O.top_p > 0 && O.top_p < 1) {
		double cut = 0;
		for (i = 0; i < k; i++) {
			cut += c[i].p;
			if (cut >= O.top_p * sum) {
				k = i + 1;
				sum = cut;
				break;
			}
		}
	}
	r = frand() * sum;
	for (i = 0; i < k; i++) {
		acc += c[i].p;
		if (acc >= r)
			return c[i].id;
	}
	return c[k - 1].id;
}

/* ---------------------------------------------------------------- */

static int *
encode(const char *s, int special, int *n)
{
	int cap = (int)strlen(s) + 16, *ids = malloc(sizeof(int) * (size_t)cap);
	*n = tok_encode(&T, s, strlen(s), special, ids, cap);
	if (*n > cap) {
		cap = *n;
		ids = realloc(ids, sizeof(int) * (size_t)cap);
		*n = tok_encode(&T, s, strlen(s), special, ids, cap);
	}
	return ids;
}

/* feed tokens; returns logits of the last one */
static const float *
feed(const int *ids, int n)
{
	if (n <= 0)
		return NULL;
	if (S.pos + n > S.ctx) {
		fprintf(stderr, "\n[context full]\n");
		return NULL;
	}
	if (O.noprefill) {
		const float *lg = NULL;
		int i;
		for (i = 0; i < n; i++)
			lg = model_forward(&M, &S, ids[i], i == n - 1);
		return lg;
	}
	return model_prefill(&M, &S, ids, n);
}

static int
emit(int id, void *u)
{
	char buf[256];
	int k = tok_piece(&T, id, O.show_special, buf, sizeof(buf));
	(void)u;
	fwrite(buf, 1, (size_t)k, stdout);
	fflush(stdout);
	return 0;
}

static int
is_stop(int id)
{
	return id == T.eos || (T.type[id] == 3 && id != T.bos);
}

static long mtp_steps, mtp_acc, mtp_drafted;
static double mtp_t[4];  /* draft, verify, accept, reject */

#define FED(t) do { if (fed) fed[(*nfed)++] = (t); } while (0)

/* greedy with the MTP block (qwen35): per step one draft and a 2-token
 * verify; the output is the greedy sequence (the verify's row 0 is the
 * model's own prediction, the draft only decides whether row 1 is
 * usable) */
static int
generate_mtp(const float *lg, int n, q2k_emit cb, void *u, int *fed, int *nfed, double *secs)
{
	const int V = M.hp.n_vocab;
	double t0 = now();
	int i = 0, x = sample(lg, V);

	while (i < n) {
		const float *l2;
		int d[MTP_KMAX], a, k, c, j;
		if (is_stop(x))
			break;
		i++;
		if (cb(x, u))
			break;
		if (i >= n || S.pos + MTP_KMAX + 1 >= S.ctx || !qwen35_mtp_ok(&M, &S)) {
			if (S.pos >= S.ctx)
				break;
			lg = model_forward(&M, &S, x, 1);
			FED(x);
			if (i >= n || !lg)
				break;
			x = sample(lg, V);
			continue;
		}
		{
			double ta = now();
			k = qwen35_mtp_draft(&M, &S, x, O.ndraft, d);
			mtp_t[0] += now() - ta;
			ta = now();
			l2 = qwen35_verify(&M, &S, x, d, k);
			mtp_t[1] += now() - ta;
		}
		FED(x);
		/* the longest run of drafts the model agrees with */
		for (c = 0; c < k; c++) {
			a = sample(l2 + (size_t)c * V, V);
			if (a != d[c] || is_stop(d[c]))
				break;
		}
		mtp_steps++;
		mtp_drafted += k;
		mtp_acc += c;
		{
			double ta = now();
			qwen35_settle(&M, &S, d, k, c);
			mtp_t[2] += now() - ta;
		}
		for (j = 0; j < c; j++)
			FED(d[j]);
		for (j = 0; j < c && i < n; j++) {
			i++;
			if (cb(d[j], u))
				goto out;
		}
		x = sample(l2 + (size_t)c * V, V);
	}
out:
	if (secs)
		*secs = now() - t0;
	return i;
}

/* generate up to n tokens after logits lg; returns count */
static int
generate_cb(const float *lg, int n, q2k_emit cb, void *u, int *fed, int *nfed, double *secs)
{
	double t0 = now();
	int i;
	if (nfed)
		*nfed = 0;
	if (lg && M.arch == ARCH_QWEN35 && O.temp <= 0 && qwen35_mtp_ok(&M, &S))
		return generate_mtp(lg, n, cb, u, fed, nfed, secs);
	for (i = 0; i < n && lg; i++) {
		int id = sample(lg, M.hp.n_vocab);
		if (is_stop(id))
			break;
		if (cb(id, u)) {
			i++;
			break;
		}
		if (S.pos >= S.ctx) {
			i++;
			break;
		}
		lg = model_forward(&M, &S, id, 1);
		FED(id);
	}
	if (secs)
		*secs = now() - t0;
	return i;
}

#undef FED

static int
generate(const float *lg, int n, double *secs)
{
	return generate_cb(lg, n, emit, NULL, NULL, NULL, secs);
}

/* ---------------------------------------------------------------- */
/* for the server                                                   */

model *const q2k_m = &M;
tokenizer *const q2k_t = &T;
state *const q2k_s = &S;

void
q2k_set_sampling(float temp, int top_k, float top_p, unsigned long seed)
{
	O.temp = temp;
	O.top_k = top_k;
	O.top_p = top_p;
	rng = seed * 2654435761ull + 1;
}

void
q2k_get_sampling(float *temp, int *top_k, float *top_p)
{
	*temp = O.temp;
	*top_k = O.top_k;
	*top_p = O.top_p;
}

int *
q2k_encode(const char *s, int special, int *n)
{
	return encode(s, special, n);
}

const float *
q2k_feed(const int *ids, int n)
{
	return feed(ids, n);
}

int
q2k_is_stop(int id)
{
	return is_stop(id);
}

int
q2k_generate(const float *lg, int n, q2k_emit cb, void *u, int *fed, int *nfed, double *secs)
{
	return generate_cb(lg, n, cb, u, fed, nfed, secs);
}

static void
print_top(const float *lg)
{
	int i, k, top[10];
	for (k = 0; k < 10; k++) {
		top[k] = -1;
		for (i = 0; i < M.hp.n_vocab; i++) {
			int used = 0, q;
			for (q = 0; q < k; q++)
				used |= top[q] == i;
			if (!used && (top[k] < 0 || lg[i] > lg[top[k]]))
				top[k] = i;
		}
	}
	for (k = 0; k < 10; k++) {
		char buf[128];
		int n = tok_piece(&T, top[k], 1, buf, sizeof(buf) - 1);
		buf[n] = 0;
		printf("%6d %10.4f  '%s'\n", top[k], lg[top[k]], buf);
	}
}

static void
info(void)
{
	const hparams *h = &M.hp;
	size_t tot = 0, dense = 0;
	double hot = 0;
	int i, nattn = 0;

	for (i = 0; i < h->n_layer; i++)
		nattn += M.L[i].attn;
	if (M.arch == ARCH_LLAMA) {
		printf("llama: %d layers, embd %d, vocab %d, ffn %d\n", h->n_layer, h->n_embd,
		    h->n_vocab, h->n_ff);
		printf("  attention: %d heads x %d, %d kv heads, rope %d dims base %g\n",
		    h->n_head, h->head_dim, h->n_head_kv, h->rope_dim, h->rope_base);
	} else if (M.arch == ARCH_QWEN35) {
		printf("qwen35: %d layers (%d gated delta net, %d gated attention), embd %d, vocab %d\n",
		    h->n_layer, h->n_layer - nattn, nattn, h->n_embd, h->n_vocab);
		printf("  attention: %d heads x %d, %d kv heads, rope %d dims base %g\n",
		    h->n_head, h->head_dim, h->n_head_kv, h->rope_dim, h->rope_base);
		printf("  delta net: %d key heads, %d value heads x %d, conv %d; ffn %d\n",
		    h->dn_kh, h->dn_vh, h->dn_dim, h->d_conv, h->n_ff);
	} else {
	printf("granitehybrid: %d layers (%d mamba2, %d attention), embd %d, vocab %d\n",
	    h->n_layer, h->n_layer - nattn, nattn, h->n_embd, h->n_vocab);
	printf("  attention: %d heads x %d, %d kv heads, scale %g, no rope\n",
	    h->n_head, h->head_dim, h->n_head_kv, h->attn_scale);
	printf("  mamba2: d_inner %d, d_state %d, %d heads x %d, conv %d\n",
	    h->d_inner, h->d_state, h->ssm_heads, h->ssm_head_dim, h->d_conv);
	printf("  moe: %d experts (top %d) x ff %d, shared ff %d\n",
	    h->n_expert, h->n_expert_used, h->n_ff, h->n_ff_sh);
	printf("  scales: embedding %g, residual %g, logits 1/%g, eps %g\n",
	    h->emb_scale, h->res_scale, h->logit_scale, h->eps);
	}
	for (i = 0; i < M.n_all; i++) {
		const wt *w = M.all[i];
		tot += w->nbytes;
		if (w->heat >= 1)
			dense += w->nbytes;
		hot += w->heat * (double)w->nbytes;
		if (O.verbose)
			printf("  %-32s %-5s %5d x %6d x %2d %9.2f MiB%s\n", w->name,
			    ggml_type_name(w->type), w->K, w->N, w->E, w->nbytes / 1048576.0,
			    w->gpu_n ? "  [gpu]" : "");
	}
	printf("  quantized weights %.1f MiB (dense %.1f MiB), %.1f MiB read per token\n",
	    tot / 1048576.0, dense / 1048576.0, hot / 1048576.0);
	printf("  cpu kernels: %s, %d threads\n", g_ref ? "reference f32" : kq_isa(), pool_size());
	if (vk_ready())
		printf("  gpu: %s, %.1f MiB resident\n", vk_device_name(), vk_used() / 1048576.0);
}

/* ---------------------------------------------------------------- */
/* kernel check                                                     */

static void
check(void)
{
	uint64_t seen = 0;
	int i, r;
	float *x, *yr, *yk, *yg;

	int kmax = 0, nmax = 0;
	for (i = 0; i < M.n_all; i++) {
		if (M.all[i]->K > kmax)
			kmax = M.all[i]->K;
		if (M.all[i]->N > nmax)
			nmax = M.all[i]->N;
	}
	kmax += 64;
	x = malloc(sizeof(float) * (size_t)kmax);
	yr = malloc(sizeof(float) * (size_t)nmax * 5);
	yk = malloc(sizeof(float) * (size_t)nmax * 5);
	yg = malloc(sizeof(float) * (size_t)nmax * 5);
	srand(1);
	for (i = 0; i < kmax; i++)
		x[i] = (float)((double)rand() / RAND_MAX * 2 - 1);
	printf("%-32s %-5s %10s %10s %10s %9s %9s\n", "tensor", "type",
	    "kq-scalar", "mv_run", "gpu", "cpu ms", "gpu ms");
	for (i = 0; i < M.n_all; i++) {
		wt *w = M.all[i];
		double e_s = 0, e_k = 0, e_g = 0, nr = 0, t0, tc, tg = 0;
		int e = w->E - 1;
		/* an expert that is on the GPU, if any */
		if (w->gpu_n)
			for (e = 0; w->gslot[e] < 0; e++)
				;
		q8k *xq;
		mv_item it;

		/* one tensor per (type, K, gpu) combination */
		uint64_t key = 1ull << (w->type * 2 + (w->gpu_n != 0));
		if ((seen & key) && !O.verbose)
			continue;
		seen |= key;
		xq = malloc(sizeof(q8k) * (size_t)(w->K / 256));
		q8k_quantize(x, xq, (size_t)w->K);
		for (r = 0; r < w->N; r++) {
			const uint8_t *row = w->data + ((size_t)e * w->N + r) * w->row_bytes;
			yr[r] = dot_row(w->type, row, x, (size_t)w->K);
			yk[r] = kq_dot_scalar(w->type, row, xq, (size_t)w->K);
			nr += (double)yr[r] * yr[r];
			e_s += (double)(yk[r] - yr[r]) * (yk[r] - yr[r]);
		}
		it = mv(w, e, x, yk);
		t0 = now();
		mv_run(&it, 1);
		tc = now() - t0;
		for (r = 0; r < w->N; r++)
			e_k += (double)(yk[r] - yr[r]) * (yk[r] - yr[r]);
		if (w->gpu_n) {
			/* the GPU path alone */
			vk_begin();
			vk_add(w, e, x, xq, yg);
			t0 = now();
			vk_submit();
			vk_wait();
			tg = now() - t0;
			for (r = 0; r < w->N; r++)
				e_g += (double)(yg[r] - yr[r]) * (yg[r] - yr[r]);
		}
		{
			/* multi-token kernel vs kq_dot, 5 tokens, first 64 rows */
			q8k *x5 = malloc(sizeof(q8k) * (size_t)(w->K / 256) * 5);
			float o5[5], md = 0, mr = 0;
			int t5, rr;
			for (t5 = 0; t5 < 5; t5++)
				q8k_quantize(x + t5 * 7 % 64, x5 + t5 * (w->K / 256), (size_t)w->K);
			for (rr = 0; rr < 64 && rr < w->N; rr++) {
				const uint8_t *row = w->data + ((size_t)e * w->N + rr) * w->row_bytes;
				if (!kq_dotT(w->type, row, x5, (size_t)(w->K / 256), 5, (size_t)w->K, o5, 1))
					break;
				for (t5 = 0; t5 < 5; t5++) {
					float ref = kq_dot(w->type, row, x5 + t5 * (w->K / 256), (size_t)w->K);
					if (fabsf(o5[t5] - ref) > md)
						md = fabsf(o5[t5] - ref);
					if (fabsf(ref) > mr)
						mr = fabsf(ref);
				}
			}
			if (rr) {
				/* and the xu form (prompts) */
				q8ku *x5u = NULL;
				if (posix_memalign((void **)&x5u, 64, sizeof(q8ku) * (size_t)(w->K / 256) * 5))
					x5u = NULL;
				float mu = 0;
				kq_q8k_xu(x5, (size_t)(w->K / 256), 5, (size_t)(w->K / 256), x5u);
				for (rr = 0; rr < 64 && rr < w->N; rr++) {
					const uint8_t *row = w->data + ((size_t)e * w->N + rr) * w->row_bytes;
					if (!kq_dotTu(w->type, row, x5u, 5, 5, (size_t)w->K, o5, 1))
						break;
					for (t5 = 0; t5 < 5; t5++) {
						float ref = kq_dot(w->type, row, x5 + t5 * (w->K / 256), (size_t)w->K);
						if (fabsf(o5[t5] - ref) > mu)
							mu = fabsf(o5[t5] - ref);
					}
				}
				free(x5u);
				printf("[dotT max|diff| %.2e, xu %.2e of %.2e] ", md, mu, mr);
			}
			if (w->gpu_n) {
				/* GPU tiled GEMM (5 tokens) vs kq_dot */
				float *g5 = malloc(sizeof(float) * 5 * (size_t)w->N);
				float gd = 0, gr = 0;
				vk_matmul_test(w, e, x5, 5, g5);
				for (rr = 0; rr < w->N; rr++)
					for (t5 = 0; t5 < 5; t5++) {
						const uint8_t *row = w->data + ((size_t)e * w->N + rr) * w->row_bytes;
						float ref = kq_dot(w->type, row, x5 + t5 * (w->K / 256), (size_t)w->K);
						if (fabsf(g5[t5 * w->N + rr] - ref) > gd)
							gd = fabsf(g5[t5 * w->N + rr] - ref);
						if (fabsf(ref) > gr)
							gr = fabsf(ref);
					}
				printf("[gpu mm max|diff| %.2e of %.2e] ", gd, gr);
				free(g5);
			}
			free(x5);
		}
		printf("%-32s %-5s %10.2e %10.2e ", w->name, ggml_type_name(w->type),
		    sqrt(e_s / nr), sqrt(e_k / nr));
		if (w->gpu_n)
			printf("%10.2e %9.3f %9.3f\n", sqrt(e_g / nr), tc * 1e3, tg * 1e3);
		else
			printf("%10s %9.3f %9s\n", "-", tc * 1e3, "-");
		free(xq);
	}
	printf("(relative L2 error vs the f32 reference; kq-scalar and the CPU/GPU\n"
	    " kernels use x quantized to 8 bits, so they should agree with each other)\n");
	free(x);
	free(yr);
	free(yk);
	free(yg);
}

/* single thread: cache-hot rows (compute bound) and a full streaming
 * pass (memory bound), in GB/s of weights and ns per 256 weights */
static void
kbench(void)
{
	uint32_t seen = 0;
	float *x, *y;
	volatile float sink = 0;
	int i, r, it, kmax = 0;

	for (i = 0; i < M.n_all; i++)
		if (M.all[i]->K > kmax)
			kmax = M.all[i]->K;
	x = malloc(sizeof(float) * (size_t)kmax);
	y = malloc(sizeof(float) * 200000);
	for (i = 0; i < kmax; i++)
		x[i] = (float)((double)rand() / RAND_MAX * 2 - 1);
	printf("%-5s %6s %10s %10s %12s\n", "type", "K", "hot GB/s", "ns/256", "stream GB/s");
	for (i = 0; i < M.n_all; i++) {
		wt *w = M.all[i];
		q8k *xq = malloc(sizeof(q8k) * (size_t)(w->K / 256));
		int hot_rows, reps;
		double t0, th, ts;
		size_t nbytes;

		if (seen & (1u << w->type))
			continue;
		seen |= 1u << w->type;
		q8k_quantize(x, xq, (size_t)w->K);
		hot_rows = (int)((getenv("KB_HOT") ? atoi(getenv("KB_HOT")) : 256) * 1024 / w->row_bytes);
		if (hot_rows > w->N)
			hot_rows = w->N;
		reps = (int)(400e6 / ((double)hot_rows * w->row_bytes)) + 1;
		for (r = 0; r < hot_rows; r++)   /* warm */
			y[r] = kq_dot(w->type, w->data + r * w->row_bytes, xq, (size_t)w->K);
		t0 = now();
		for (it = 0; it < reps; it++)
			for (r = 0; r < hot_rows; r++)
				y[r] += kq_dot(w->type, w->data + r * w->row_bytes, xq, (size_t)w->K);
		th = now() - t0;
		{
			/* multi-token kernel, T = 64, cache-hot rows */
			static q8ku x64[64 * 80];
			static float o64[64];
			int tt, KT = getenv("KB_T") ? atoi(getenv("KB_T")) : 64;
			double td;
			static q8k x64t[64 * 80];
			for (tt = 0; tt < 64 && w->K <= 80 * 256; tt++)
				memcpy(x64t + tt * (w->K / 256), xq, sizeof(q8k) * (size_t)(w->K / 256));
			if (w->K <= 80 * 256)
				kq_q8k_xu(x64t, (size_t)(w->K / 256), 64, (size_t)(w->K / 256), x64);
			if (w->K <= 80 * 256 &&
			    kq_dotTu(w->type, w->data, x64, 64, 64, (size_t)w->K, o64, 1)) {
				int rr = hot_rows < 64 ? hot_rows : 64;
				t0 = now();
				for (it = 0; it < 20; it++)
					for (r = 0; r < rr; r++)
						kq_dotTu(w->type, w->data + r * w->row_bytes, x64,
						    64, KT, (size_t)w->K, o64, 1);
				td = now() - t0;
				printf("  %s dotT(%d): %.2f ns per 256 weights x token\n", ggml_type_name(w->type),
				    KT, td / (20.0 * rr * KT * (w->K / 256)) * 1e9);
			}
		}
		nbytes = w->row_bytes * (size_t)w->N * (size_t)(w->E > 4 ? 4 : w->E);
		t0 = now();
		for (r = 0; (size_t)r * w->row_bytes < nbytes; r++)
			y[r % 200000] += kq_dot(w->type, w->data + (size_t)r * w->row_bytes, xq, (size_t)w->K);
		ts = now() - t0;
		sink += y[0];
		free(xq);
		printf("%-7s %6d %10.2f %10.2f %12.2f\n", ggml_type_name(w->type), w->K,
		    (double)reps * hot_rows * w->row_bytes / th / 1e9,
		    th / ((double)reps * hot_rows * w->K / 256) * 1e9,
		    nbytes / ts / 1e9);
	}
	(void)sink;
	free(x);
	free(y);
}

/* ---------------------------------------------------------------- */

static void
chat(void)
{
	char line[8192], *msg;
	int n, *ids, first = 1;
	size_t cap = 0;

	printf("q2k chat (empty line or ctrl-d to quit, /reset to clear)\n");
	for (;;) {
		const float *lg;
		double secs;
		int gen, p0;
		printf("\n> ");
		fflush(stdout);
		if (!fgets(line, sizeof(line), stdin))
			break;
		line[strcspn(line, "\n")] = 0;
		if (!line[0])
			break;
		if (!strcmp(line, "/reset")) {
			state_reset(&S, &M);
			first = 1;
			continue;
		}
		cap = strlen(line) + strlen(O.system) + 256;
		msg = malloc(cap);
		if (ARCH_DENSE(M.arch))
			/* ChatML; no default system message (llama/MiniCPM5:
			 * the template starts with the bos token) */
			snprintf(msg, cap, "%.*s%s%s%s<|im_start|>user\n%s<|im_end|>\n"
			    "<|im_start|>assistant\n%s",
			    first && M.arch == ARCH_LLAMA && T.bos >= 0 ? (int)T.text[T.bos].len : 0,
			    T.bos >= 0 ? T.text[T.bos].p : "",
			    first && O.sys_given ? "<|im_start|>system\n" : "",
			    first && O.sys_given ? O.system : "",
			    first && O.sys_given ? "<|im_end|>\n" : "", line,
			    O.think ? "<think>\n" : "<think>\n\n</think>\n\n");
		else
			snprintf(msg, cap, "%s%s%s<|start_of_role|>user<|end_of_role|>%s<|end_of_text|>\n"
			    "<|start_of_role|>assistant<|end_of_role|>",
			    first ? "<|start_of_role|>system<|end_of_role|>" : "",
			    first ? O.system : "", first ? "<|end_of_text|>\n" : "", line);
		ids = encode(msg, 1, &n);
		p0 = S.pos;
		lg = feed(ids, n);
		free(ids);
		free(msg);
		first = 0;
		gen = generate(lg, O.n_predict, &secs);
		/* close the assistant turn in the cache */
		{
			int end = tok_find(&T, ARCH_DENSE(M.arch) ? "<|im_end|>" : "<|end_of_text|>"),
			    nl[4], k;
			k = tok_encode(&T, "\n", 1, 0, nl, 4);
			if (end >= 0)
				feed(&end, 1);
			feed(nl, k);
		}
		if (O.verbose)
			fprintf(stderr, "\n[%d prompt tok, %d gen tok, %.2f tok/s, pos %d]\n",
			    S.pos - p0 - gen, gen, gen / secs, S.pos);
	}
}

int
main(int argc, char **argv)
{
	char err[256];
	double t0;
	int n = 0, *ids = NULL;

	parse(argc, argv);
	rng = O.seed * 2654435761ull + 1;
	if (O.transcode) {
		gguf g;
		extern int gguf_transcode(const gguf *g, const char *out, int verbose);
		if (gguf_open(&g, expand_home(O.model), err, sizeof(err)) != 0) {
			fprintf(stderr, "q2k: %s\n", err);
			return 1;
		}
		return gguf_transcode(&g, O.transcode, O.verbose) ? 1 : 0;
	}
	t0 = now();
	if (model_load(&M, expand_home(O.model), err, sizeof(err)) != 0 ||
	    tok_load(&T, &M.g, err, sizeof(err)) != 0) {
		fprintf(stderr, "q2k: %s\n", err);
		return 1;
	}
	kq_init();
	pool_init(O.threads);
	/* greedy runs keep the MTP block in step from the prompt on */
	if (getenv("Q2K_DRAFT_V"))
		g_draft_vocab = atoi(getenv("Q2K_DRAFT_V"));
	/* (a server decides per request) */
	g_mtp = M.arch == ARCH_QWEN35 && M.mtp && (O.temp <= 0 || O.server) && !O.nomtp &&
	    O.ndraft > 0 &&
	    !O.check && !O.kbench;
	if (g_stream < 0)
		g_stream = ARCH_DENSE(M.arch) && O.gpu_mib > 0 ? 0.5 : 0;
	if (g_stream > 0 && !vk_stream_bytes)
		vk_stream_bytes = (size_t)64 << 20;
	if (O.gpu_mib > 0 && !O.tokens && ARCH_DENSE(M.arch)) {
		if (vk_init((size_t)O.gpu_mib << 20, 0, O.device, O.verbose) == 0)
			vk_plan_rows(&M, O.verbose);
		else
			fprintf(stderr, "q2k: Vulkan unavailable, running on the CPU\n");
	} else if (O.gpu_mib > 0 && !O.tokens) {
		if (vk_init((size_t)O.gpu_mib << 20, model_gpu_state_bytes(&M), O.device,
		    O.verbose) == 0)
		{
			vk_plan_and_upload(&M, O.verbose);
			if (model_prepare_split(&M) != 0) {
				fprintf(stderr, "q2k: out of memory\n");
				return 1;
			}
		}
		else
			fprintf(stderr, "q2k: Vulkan unavailable, running on the CPU\n");
	}
	if (O.verbose)
		fprintf(stderr, "loaded in %.2f s\n", now() - t0);

	if (O.info) {
		info();
		goto out;
	}
	if (O.check) {
		check();
		goto out;
	}
	if (O.kbench && vk_ready()) {
		uint32_t g;
		for (g = 6; g <= 768; g *= 2)
			printf("gpu read %u workgroups: %.1f GB/s (300 MB), %.1f GB/s (3 MB)\n", g,
			    vk_bandwidth(300u << 20, g), vk_bandwidth(3u << 20, g));
	}
	if (O.kbench) {
		kbench();
		goto out;
	}
	if (O.prompt) {
		ids = encode(O.prompt, 1, &n);
		if (O.tokens) {
			int i;
			for (i = 0; i < n; i++) {
				char buf[128];
				int k = tok_piece(&T, ids[i], 1, buf, sizeof(buf) - 1);
				buf[k] = 0;
				printf("%d\t'%s'\n", ids[i], buf);
			}
			goto out;
		}
	}
	if (state_init(&S, &M, O.ctx) != 0) {
		fprintf(stderr, "q2k: out of memory\n");
		return 1;
	}
	if (O.server) {
		server_opts so;
		int r;
		memset(&so, 0, sizeof(so));
		so.host = O.host;
		so.port = O.port;
		so.alias = O.alias;
		so.api_key = O.api_key;
		so.model_path = expand_home(O.model);
		/* like llama-server: until the context is full by default */
		so.n_predict = O.n_given ? O.n_predict : -1;
		so.think = O.think;
		so.verbose = O.verbose;
		r = server_run(&so);
		free(ids);
		vk_shutdown();
		pool_free();
		tok_free(&T);
		model_free(&M);
		return r ? 1 : 0;
	} else if (O.chat) {
		chat();
	} else if (O.prompt && n > 0) {
		const float *lg;
		double tp, tg;
		int gen;

		t0 = now();
		lg = feed(ids, n);
		tp = now() - t0;
		if (O.logits) {
			print_top(lg);
			if (O.dump) {
				FILE *f = fopen(O.dump, "wb");
				if (!f || fwrite(lg, sizeof(float), (size_t)M.hp.n_vocab, f) !=
				    (size_t)M.hp.n_vocab)
					fprintf(stderr, "q2k: cannot write %s\n", O.dump);
				if (f)
					fclose(f);
			}
			goto out;
		}
		if (!O.bench) {
			fwrite(O.prompt, 1, strlen(O.prompt), stdout);
			fflush(stdout);
		}
		if (getenv("Q2K_VBENCH") && lg && qwen35_mtp_ok(&M, &S)) {
			/* verify of T = 1..4 tokens (settled back to nothing accepted) */
			int k, r, nb = atoi(getenv("Q2K_VBENCH")), x0 = sample(lg, M.hp.n_vocab), dd[MTP_KMAX];
			for (k = 0; k < MTP_KMAX; k++)
				dd[k] = x0;
			for (k = 0; k <= MTP_KMAX; k++) {
				double ta = now();
				for (r = 0; r < nb; r++) {
					qwen35_verify(&M, &S, x0, dd, k);
					qwen35_settle(&M, &S, dd, k, 0);
					S.pos--;
					S.mtp_n = S.pos;
				}
				fprintf(stderr, "vbench: verify T=%d %.0f ms\n", k + 1, 1e3 * (now() - ta) / nb);
			}
		}
		gen = generate(lg, O.n_predict, &tg);
		printf("\n");
		if (O.bench) {
			extern double vk_t_wait, vk_t_sub;
			extern long vk_n_sub;
			fprintf(stderr, "time: cpu matvec %.3f s (+%.3f s alongside gpu), "
			    "gpu wait %.3f s, gpu mixer %.3f s, cpu ssm scan %.3f s, attention %.3f s\n",
			    prof_t[PROF_CPU_MV], prof_t[PROF_CPU_MV_OVERLAP],
			    prof_t[PROF_GPU_WAIT], prof_t[PROF_GPU_MIXER], prof_t[PROF_SCAN],
			    prof_t[PROF_ATTN]);
			{
				static const char *nm[] = { "mixer in", "mixer out", "moe gate/up", "moe down" };
				int q;
				fprintf(stderr, "cpu matvec:");
				for (q = 0; q < 4; q++)
					fprintf(stderr, " %s %.3f s %.1f GB/s;", nm[q], prof_t[PROF_MIX_IN + q],
					    prof_b[PROF_MIX_IN + q] / (prof_t[PROF_MIX_IN + q] + 1e-12) / 1e9);
				fprintf(stderr, "\n");
			}
			vk_prof_report();
			if (vk_n_sub)
				fprintf(stderr, "vulkan: %ld batches, %.3f s recording\n",
				    vk_n_sub, vk_t_sub);
		}
		if ((O.bench || O.verbose) && mtp_steps)
			fprintf(stderr, "mtp: %ld passes, %ld of %ld drafts accepted (%.0f%%), %.2f tokens per pass\n",
			    mtp_steps, mtp_acc, mtp_drafted, 100.0 * mtp_acc / (mtp_drafted ? mtp_drafted : 1),
			    (mtp_steps + mtp_acc) / (double)mtp_steps);
		if ((O.bench || O.verbose) && mtp_steps)
			fprintf(stderr, "mtp time: draft %.3f s, verify %.3f s, settle %.3f s"
			    " (%.0f ms per pass)\n", mtp_t[0], mtp_t[1], mtp_t[2],
			    1e3 * (mtp_t[0] + mtp_t[1] + mtp_t[2] + mtp_t[3]) / mtp_steps);
		if (O.bench || O.verbose)
			fprintf(stderr, "prompt: %d tok in %.2f s (%.2f tok/s); "
			    "gen: %d tok in %.2f s (%.2f tok/s)\n",
			    n, tp, n / tp, gen, tg, gen / tg);
	} else
		usage();
out:
	free(ids);
	vk_shutdown();
	pool_free();
	tok_free(&T);
	model_free(&M);
	return 0;
}
