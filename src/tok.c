#include "tok.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* GPT-2 bytes_to_unicode */
static uint32_t b2u[256];
static int u2b[512];

static void
init_bytemap(void)
{
	int b, n = 0;
	for (b = 0; b < 512; b++)
		u2b[b] = -1;
	for (b = 0; b < 256; b++) {
		if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255))
			b2u[b] = (uint32_t)b;
		else
			b2u[b] = 256u + (uint32_t)n++;
		u2b[b2u[b]] = b;
	}
}

static int
ucd_flags(uint32_t cp)
{
	int lo = 0, hi = ucd_nranges - 1;
	if (cp < 0x80) {
		/* fast ASCII path */
		if ((cp | 32) - 'a' < 26)
			return UC_L | UC_DEF;
		if (cp - '0' < 10)
			return UC_N | UC_DEF;
		if (cp == ' ' || (cp >= 9 && cp <= 13))
			return UC_WS | UC_DEF;
		return UC_DEF;
	}
	while (lo <= hi) {
		int mid = (lo + hi) / 2;
		if (cp < ucd_ranges[mid].lo)
			hi = mid - 1;
		else if (cp > ucd_ranges[mid].hi)
			lo = mid + 1;
		else
			return ucd_ranges[mid].f;
	}
	return 0;
}

static uint32_t
fnv(const char *s, size_t n)
{
	uint32_t h = 2166136261u;
	size_t i;
	for (i = 0; i < n; i++)
		h = (h ^ (uint8_t)s[i]) * 16777619u;
	return h;
}

static uint32_t
hpair(int a, int b)
{
	uint32_t h = (uint32_t)a * 0x9e3779b1u ^ ((uint32_t)b + 0x7f4a7c15u) * 0x85ebca77u;
	return h ^ (h >> 15);
}

static int
vfind(const tokenizer *t, const char *s, size_t n)
{
	uint32_t i = fnv(s, n) & t->vmask;
	for (;; i = (i + 1) & t->vmask) {
		int id = t->vh[i];
		if (id < 0)
			return -1;
		if (t->text[id].len == n && memcmp(t->text[id].p, s, n) == 0)
			return id;
	}
}

int
tok_find(const tokenizer *t, const char *text)
{
	return vfind(t, text, strlen(text));
}

static const struct mrg *
mfind(const tokenizer *t, int a, int b)
{
	uint32_t i = hpair(a, b) & t->mmask;
	for (;; i = (i + 1) & t->mmask) {
		const struct mrg *m = &t->mh[i];
		if (m->a < 0)
			return NULL;
		if (m->a == a && m->b == b)
			return m;
	}
}

static int
by_len_desc(const void *x, const void *y, const tokenizer *t)
{
	return (int)t->text[*(const int *)y].len - (int)t->text[*(const int *)x].len;
}

static const tokenizer *sort_t;

static int
cmp_special(const void *x, const void *y)
{
	return by_len_desc(x, y, sort_t);
}

int
tok_load(tokenizer *t, const gguf *g, char *err, size_t errlen)
{
	const gkv *tk = gguf_find(g, "tokenizer.ggml.tokens");
	const gkv *ty = gguf_find(g, "tokenizer.ggml.token_type");
	const gkv *mg = gguf_find(g, "tokenizer.ggml.merges");
	const uint8_t *cur;
	gstr model;
	uint32_t hs, u;
	int i, n;

	memset(t, 0, sizeof(*t));
	init_bytemap();
	if (gguf_get_str(g, "tokenizer.ggml.model", &model) || !gstr_eq(model, "gpt2") ||
	    !tk || tk->type != 9 || tk->atype != GV_STR || !mg || mg->atype != GV_STR) {
		snprintf(err, errlen, "unsupported tokenizer (need gpt2 BPE)");
		return -1;
	}
	n = t->n_vocab = (int)tk->n;
	t->text = calloc((size_t)n, sizeof(gstr));
	t->type = calloc((size_t)n, 1);
	t->special = calloc((size_t)n, sizeof(int));
	for (hs = 1; hs < (uint32_t)n * 2; hs <<= 1)
		;
	t->vmask = hs - 1;
	t->vh = malloc(sizeof(int) * hs);
	if (!t->text || !t->type || !t->special || !t->vh) {
		snprintf(err, errlen, "out of memory");
		return -1;
	}
	for (u = 0; u < hs; u++)
		t->vh[u] = -1;
	cur = tk->val;
	for (i = 0; i < n; i++) {
		uint32_t h;
		t->text[i] = gguf_next_str(&cur);
		t->type[i] = 1;
		if (ty && ty->n == (uint64_t)n && (ty->atype == GV_I32 || ty->atype == GV_U32)) {
			int32_t v;
			memcpy(&v, ty->val + 4 * (size_t)i, 4);
			t->type[i] = (uint8_t)v;
		}
		if (t->type[i] == 3 || t->type[i] == 4)
			t->special[t->n_special++] = i;
		h = fnv(t->text[i].p, t->text[i].len) & t->vmask;
		while (t->vh[h] >= 0)
			h = (h + 1) & t->vmask;
		t->vh[h] = i;
	}
	sort_t = t;
	qsort(t->special, (size_t)t->n_special, sizeof(int), cmp_special);

	for (hs = 1; hs < (uint32_t)mg->n * 2; hs <<= 1)
		;
	t->mmask = hs - 1;
	t->mh = malloc(sizeof(*t->mh) * hs);
	if (!t->mh) {
		snprintf(err, errlen, "out of memory");
		return -1;
	}
	for (u = 0; u < hs; u++)
		t->mh[u].a = -1;
	cur = mg->val;
	for (i = 0; i < (int)mg->n; i++) {
		gstr m = gguf_next_str(&cur);
		char buf[512];
		const char *sp;
		int a, b, r;
		uint32_t h;
		if (m.len >= sizeof(buf))
			continue;
		sp = memchr(m.p + 1, ' ', m.len - 1);
		if (!sp)
			continue;
		a = vfind(t, m.p, (size_t)(sp - m.p));
		b = vfind(t, sp + 1, m.len - (size_t)(sp - m.p) - 1);
		memcpy(buf, m.p, (size_t)(sp - m.p));
		memcpy(buf + (sp - m.p), sp + 1, m.len - (size_t)(sp - m.p) - 1);
		r = vfind(t, buf, m.len - 1);
		if (a < 0 || b < 0 || r < 0)
			continue;
		h = hpair(a, b) & t->mmask;
		while (t->mh[h].a >= 0)
			h = (h + 1) & t->mmask;
		t->mh[h].a = a;
		t->mh[h].b = b;
		t->mh[h].rank = i;
		t->mh[h].id = r;
	}
	{
		gstr pre;
		if (gguf_get_str(g, "tokenizer.ggml.pre", &pre) == 0 && gstr_eq(pre, "qwen35"))
			t->pre = 1;
		else if (gguf_get_str(g, "tokenizer.ggml.pre", &pre) == 0 && gstr_eq(pre, "minicpm5"))
			t->pre = 2;
	}
	t->bos = t->eos = -1;
	if (gguf_get_u32(g, "tokenizer.ggml.bos_token_id", &u) == 0)
		t->bos = (int)u;
	if (gguf_get_u32(g, "tokenizer.ggml.eos_token_id", &u) == 0)
		t->eos = (int)u;
	return 0;
}

void
tok_free(tokenizer *t)
{
	free(t->text);
	free(t->type);
	free(t->special);
	free(t->vh);
	free(t->mh);
	memset(t, 0, sizeof(*t));
}

/* ---------------------------------------------------------------- */

static int
put(int *out, int cap, int n, int id)
{
	if (n < cap)
		out[n] = id;
	return n + 1;
}

static int
utf8_put(char *o, uint32_t c)
{
	if (c < 0x80) {
		o[0] = (char)c;
		return 1;
	}
	if (c < 0x800) {
		o[0] = (char)(0xc0 | c >> 6);
		o[1] = (char)(0x80 | (c & 63));
		return 2;
	}
	o[0] = (char)(0xe0 | c >> 12);
	o[1] = (char)(0x80 | ((c >> 6) & 63));
	o[2] = (char)(0x80 | (c & 63));
	return 3;
}

/* BPE one pre-token (raw bytes) */
static int
bpe_word(const tokenizer *t, const uint8_t *s, size_t len, int *out, int cap, int n)
{
	int stack[256], *sym = stack;
	size_t i;
	int ns = 0;

	if (len > 256 && !(sym = malloc(sizeof(int) * len)))
		return n;
	for (i = 0; i < len; i++) {
		char u[4];
		int k = utf8_put(u, b2u[s[i]]);
		sym[ns++] = vfind(t, u, (size_t)k);
	}
	for (;;) {
		int best = -1, br = 0x7fffffff, bid = -1, k;
		for (k = 0; k + 1 < ns; k++) {
			const struct mrg *m;
			if (sym[k] < 0 || sym[k + 1] < 0)
				continue;
			m = mfind(t, sym[k], sym[k + 1]);
			if (m && m->rank < br) {
				br = m->rank;
				best = k;
				bid = m->id;
			}
		}
		if (best < 0)
			break;
		sym[best] = bid;
		memmove(sym + best + 1, sym + best + 2, sizeof(int) * (size_t)(ns - best - 2));
		ns--;
	}
	for (i = 0; i < (size_t)ns; i++)
		if (sym[i] >= 0)
			n = put(out, cap, n, sym[i]);
	if (sym != stack)
		free(sym);
	return n;
}

static uint32_t
utf8_get(const uint8_t *s, size_t len, size_t *adv)
{
	uint32_t c = s[0];
	int k = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
	int i;
	if (k == 0 || (size_t)k > len) {
		*adv = 1;
		return 0xFFFFFFFEu; /* invalid: its own class (no flags) */
	}
	if (k > 1)
		c &= 0x3fu >> (k - 1);
	for (i = 1; i < k; i++) {
		if ((s[i] & 0xc0) != 0x80) {
			*adv = 1;
			return 0xFFFFFFFEu;
		}
		c = c << 6 | (s[i] & 63);
	}
	*adv = (size_t)k;
	return c;
}

/* llama3 / dbrx pre-tokenizer, ported from llama.cpp's hand-written
 * matcher (pre 0), and the qwen35 variant (pre 1: letter runs include
 * marks \p{M}, digits one at a time, punctuation runs exclude marks)
 * for
 *   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|
 *    ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+               */
static int
pretok(const tokenizer *t, const uint8_t *s, size_t len, int *out, int cap, int n)
{
	const int LM = t->pre == 1 ? (UC_L | UC_M) : UC_L;     /* letter-run class */
	const int ndig = t->pre == 1 ? 1 : 3;                  /* \p{N}{1,ndig} */
	const int PX = t->pre == 1 ? (UC_WS | UC_L | UC_M | UC_N) : (UC_WS | UC_L | UC_N);
	size_t nc = 0, i, *off;
	uint32_t *cp;
	int *fl;
	size_t pos, prev = 0;

	cp = malloc(sizeof(*cp) * (len + 1));
	fl = malloc(sizeof(*fl) * (len + 2));
	off = malloc(sizeof(*off) * (len + 2));
	if (!cp || !fl || !off)
		goto done;
	for (i = 0; i < len;) {
		size_t a;
		cp[nc] = utf8_get(s + i, len - i, &a);
		fl[nc] = cp[nc] == 0xFFFFFFFEu ? 0 : ucd_flags(cp[nc]);
		off[nc++] = i;
		i += a;
	}
	off[nc] = len;
	fl[nc] = 0;

#define CPT(p) ((p) < nc ? cp[p] : 0xFFFFFFFFu)
#define FL(p) ((p) < nc ? fl[p] : 0)
#define EMIT(e) do { size_t e_ = (e); if (e_ > prev) n = bpe_word(t, s + off[prev], \
	off[e_] - off[prev], out, cap, n); prev = e_; } while (0)

	for (pos = 0; pos < nc;) {
		uint32_t c = cp[pos];
		int f = fl[pos], f2;
		size_t nws = 0, last_rn = 0;

		if (c == '\'' && pos + 1 < nc) {
			uint32_t c1 = CPT(pos + 1) | 32;
			if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
				pos += 2;
				EMIT(pos);
				continue;
			}
			if (pos + 2 < nc) {
				uint32_t c2 = CPT(pos + 2) | 32;
				if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') ||
				    (c1 == 'l' && c2 == 'l')) {
					pos += 3;
					EMIT(pos);
					continue;
				}
			}
		}
		if (!(c == '\r' || c == '\n' || (f & UC_N))) {
			if ((f & LM) || (FL(pos + 1) & LM)) {
				pos++;
				while (FL(pos) & LM)
					pos++;
				EMIT(pos);
				continue;
			}
		}
		if (f & UC_N) {
			size_t ini = pos;
			while (FL(pos) & UC_N) {
				if (++pos - ini >= (size_t)ndig) {
					EMIT(pos);
					ini = pos;
				}
			}
			EMIT(pos);
			continue;
		}
		f2 = c == ' ' ? FL(pos + 1) : f;
		if (!(f2 & PX) && f) {
			pos += c == ' ';
			while (!(f2 & PX) && f2)
				f2 = FL(pos + 1), pos++;
			while (CPT(pos) == '\r' || CPT(pos) == '\n')
				pos++;
			EMIT(pos);
			continue;
		}
		while (FL(pos + nws) & UC_WS) {
			uint32_t cc = CPT(pos + nws);
			if (cc == '\r' || cc == '\n')
				last_rn = pos + nws + 1;
			nws++;
		}
		if (last_rn > 0) {
			pos = last_rn;
			EMIT(pos);
			continue;
		}
		if (nws > 1 && CPT(pos + nws) != 0xFFFFFFFFu) {
			pos += nws - 1;
			EMIT(pos);
			continue;
		}
		if (nws > 0) {
			pos += nws;
			EMIT(pos);
			continue;
		}
		/* no match: a single codepoint */
		pos++;
		EMIT(pos);
	}
#undef CPT
#undef FL
#undef EMIT
done:
	free(cp);
	free(fl);
	free(off);
	return n;
}

/* minicpm5 (pre 2): llama.cpp first splits at \p{N}{1,3} and runs the
 * llama3 expression on each piece, so a digit ends the text for the
 * whitespace lookahead (\s+(?!\S)) */
static int
pretok_seg(const tokenizer *t, const uint8_t *s, size_t len, int *out, int cap, int n)
{
	size_t i = 0, start = 0;
	if (t->pre != 2)
		return pretok(t, s, len, out, cap, n);
	while (i < len) {
		size_t a, j, nd = 0;
		uint32_t c = utf8_get(s + i, len - i, &a);
		if (c == 0xFFFFFFFEu || !(ucd_flags(c) & UC_N)) {
			i += a;
			continue;
		}
		n = pretok(t, s + start, i - start, out, cap, n);
		/* up to three digits */
		for (j = i; j < len && nd < 3; nd++) {
			c = utf8_get(s + j, len - j, &a);
			if (c == 0xFFFFFFFEu || !(ucd_flags(c) & UC_N))
				break;
			j += a;
		}
		n = pretok(t, s + i, j - i, out, cap, n);
		i = start = j;
	}
	return pretok(t, s + start, len - start, out, cap, n);
}

int
tok_encode(const tokenizer *t, const char *str, size_t len, int parse_special,
    int *out, int cap)
{
	const uint8_t *s = (const uint8_t *)str;
	size_t i = 0, start = 0;
	int n = 0, k;

	while (i < len) {
		int hit = -1;
		if (parse_special && s[i] == '<') {
			for (k = 0; k < t->n_special; k++) {
				const gstr *g = &t->text[t->special[k]];
				if (g->len && g->len <= len - i && memcmp(s + i, g->p, g->len) == 0) {
					hit = t->special[k];
					break;
				}
			}
		}
		if (hit < 0) {
			i++;
			continue;
		}
		n = pretok_seg(t, s + start, i - start, out, cap, n);
		n = put(out, cap, n, hit);
		i += t->text[hit].len;
		start = i;
	}
	return pretok_seg(t, s + start, len - start, out, cap, n);
}

int
tok_piece(const tokenizer *t, int id, int show_special, char *buf, int cap)
{
	const gstr *g;
	size_t i;
	int n = 0;

	if (id < 0 || id >= t->n_vocab)
		return 0;
	g = &t->text[id];
	if (t->type[id] == 3 || t->type[id] == 4) {
		if (t->type[id] == 3 && !show_special)
			return 0;
		for (i = 0; i < g->len && n < cap; i++)
			buf[n++] = g->p[i];
		return n;
	}
	for (i = 0; i < g->len && n < cap;) {
		size_t a;
		uint32_t c = utf8_get((const uint8_t *)g->p + i, g->len - i, &a);
		i += a;
		if (c < 512 && u2b[c] >= 0)
			buf[n++] = (char)u2b[c];
	}
	return n;
}
