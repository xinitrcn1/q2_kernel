/* HTTP server speaking llama.cpp's llama-server protocol: the native
 * /completion endpoint, the OpenAI-style /v1/completions and
 * /v1/chat/completions (both with SSE streaming), /tokenize,
 * /detokenize, /apply-template, /health, /props, /v1/models, /slots.
 *
 * One model state, so requests run one at a time; idle keep-alive
 * connections are multiplexed with poll() so several clients can hold
 * connections.  A prompt that extends the previous request's tokens
 * (prompt + generated) only feeds the new suffix; anything else resets
 * the recurrent state and feeds the whole prompt. */
#include <errno.h>
#include <math.h>
#include <netdb.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <netinet/in.h>
#include <netinet/tcp.h>

#include "json.h"
#include "server.h"

#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

#define MAX_CONN 64
#define MAX_HEAD (64 << 10)
#define MAX_BODY ((size_t)64 << 20)
#define IDLE_SECS 600

static const server_opts *SO;
static char model_name[256];
static float def_temp, def_top_p;
static int def_top_k;
static unsigned long seed_ctr;

/* tokens in the state (prompt + generated); valid while S.pos == cn */
static int *ctoks, cn, ccap;

typedef struct {
	int fd;
	sbuf in;
	int sent100;
	time_t last;
} conn;

static conn conns[MAX_CONN];
static int nconn;

static double
now(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ---------------------------------------------------------------- */
/* HTTP                                                             */

typedef struct {
	char method[16], path[512];
	const char *body;
	size_t blen;
	int close;              /* Connection: close / HTTP/1.0 */
	char auth[512];
} req;

static int
send_all(int fd, const void *p, size_t n)
{
	const char *c = p;
	while (n) {
		ssize_t k = send(fd, c, n, MSG_NOSIGNAL);
		if (k < 0 && errno == EINTR)
			continue;
		if (k <= 0)
			return -1;
		c += k;
		n -= (size_t)k;
	}
	return 0;
}

static const char *
reason(int code)
{
	switch (code) {
	case 200: return "OK";
	case 204: return "No Content";
	case 400: return "Bad Request";
	case 401: return "Unauthorized";
	case 404: return "Not Found";
	case 405: return "Method Not Allowed";
	case 411: return "Length Required";
	case 413: return "Payload Too Large";
	case 500: return "Internal Server Error";
	case 501: return "Not Implemented";
	default: return "Error";
	}
}

#define CORS "Access-Control-Allow-Origin: *\r\n"

static int
respond(int fd, int code, const char *ctype, const char *body, size_t n, int close)
{
	sbuf h = { 0 };
	int r;
	sb_printf(&h, "HTTP/1.1 %d %s\r\nServer: q2k\r\n" CORS "Content-Type: %s\r\n"
	    "Content-Length: %zu\r\nConnection: %s\r\n\r\n", code, reason(code), ctype, n,
	    close ? "close" : "keep-alive");
	sb_add(&h, body, n);
	r = send_all(fd, h.p, h.n);
	sb_free(&h);
	return r;
}

static int
respond_json(int fd, int code, const sbuf *b, int close)
{
	return respond(fd, code, "application/json; charset=utf-8", b->p ? b->p : "", b->n, close);
}

static int
respond_err(int fd, int code, const char *type, const char *msg, int close)
{
	sbuf b = { 0 };
	int r;
	sb_printf(&b, "{\"error\":{\"code\":%d,\"message\":", code);
	sb_jstr(&b, msg, strlen(msg));
	sb_puts(&b, ",\"type\":");
	sb_jstr(&b, type, strlen(type));
	sb_puts(&b, "}}");
	r = respond_json(fd, code, &b, close);
	sb_free(&b);
	if (SO->verbose)
		fprintf(stderr, "q2k server: %d %s\n", code, msg);
	return r;
}

/* SSE over chunked transfer encoding (the connection stays usable) */
static int
sse_begin(int fd)
{
	static const char h[] = "HTTP/1.1 200 OK\r\nServer: q2k\r\n" CORS
	    "Content-Type: text/event-stream\r\nCache-Control: no-cache\r\n"
	    "Transfer-Encoding: chunked\r\nConnection: keep-alive\r\n\r\n";
	return send_all(fd, h, sizeof(h) - 1);
}

static int
sse_send(int fd, const char *data, size_t n)
{
	sbuf b = { 0 };
	int r;
	sb_printf(&b, "%zx\r\ndata: ", n + 8);
	sb_add(&b, data, n);
	sb_puts(&b, "\n\n\r\n");
	r = send_all(fd, b.p, b.n);
	sb_free(&b);
	return r;
}

static int
sse_end(int fd)
{
	return send_all(fd, "0\r\n\r\n", 5);
}

/* the client went away (only EOF counts: pipelined data does not) */
static int
client_gone(int fd)
{
	struct pollfd p;
	char c;
	p.fd = fd;
	p.events = POLLIN;
	p.revents = 0;
	if (poll(&p, 1, 0) <= 0)
		return 0;
	if (p.revents & (POLLERR | POLLNVAL))
		return 1;
	if (p.revents & (POLLIN | POLLHUP))
		return recv(fd, &c, 1, MSG_PEEK | MSG_DONTWAIT) == 0;
	return 0;
}

/* ---------------------------------------------------------------- */
/* generation                                                       */

enum { K_NATIVE, K_OAI_CMPL, K_OAI_CHAT };
enum { STOP_NONE, STOP_EOS, STOP_LIMIT, STOP_WORD };

typedef struct {
	int kind, stream, fd, dead;
	int reasoning;          /* split <think> text into reasoning_content */
	int think_done, content_started;
	char **stop;
	int nstop;
	sbuf text;              /* all generated bytes (cut at a stop word) */
	size_t done;            /* bytes of text classified so far */
	sbuf rsn, cnt;          /* reasoning and content so far */
	int *toks, ntok, tok_sent;
	int stop_type;
	const char *stop_word;
	int prompt_n, cached_n;
	double t_prompt, t_gen;
	char id[48];
	long created;
	int role_sent;
	const jv *tools;        /* chat with tools: parse <function> calls */
	int in_calls;           /* content from the first "<function" on... */
	sbuf calls;             /* ...is held here, not sent */
	sbuf calls_json;        /* the parsed calls: OpenAI tool_calls items */
	sbuf calls_delta;       /* the same with "index", for the stream */
	int ncalls;
} task;

static void gen_id(char *buf, size_t n, const char *prefix);
static const jv *req_tools(const jv *body);

static size_t
memfind(const char *h, size_t hn, const char *n, size_t nn)
{
	size_t i;
	if (nn == 0 || nn > hn)
		return (size_t)-1;
	for (i = 0; i + nn <= hn; i++)
		if (h[i] == n[0] && !memcmp(h + i, n, nn))
			return i;
	return (size_t)-1;
}

/* how many trailing bytes of s[0..n) could start pat */
static size_t
partial(const char *s, size_t n, const char *pat)
{
	size_t pl = strlen(pat), k;
	for (k = pl - 1 < n ? pl - 1 : n; k > 0; k--)
		if (!memcmp(s + n - k, pat, k))
			return k;
	return 0;
}

/* bytes at the end of s that are an incomplete UTF-8 sequence */
static size_t
utf8_tail(const char *s, size_t n)
{
	size_t i;
	for (i = 1; i <= 3 && i <= n; i++) {
		unsigned char c = (unsigned char)s[n - i];
		if ((c & 0xc0) == 0x80)
			continue;
		if (c >= 0xc0) {
			size_t need = c >= 0xf0 ? 4 : c >= 0xe0 ? 3 : 2;
			return need > i ? i : 0;
		}
		return 0;
	}
	return 0;
}

static void
send_delta(task *t, const char *p, size_t n, int is_rsn)
{
	sbuf b = { 0 };
	if (t->dead || !n)
		return;
	if (t->kind == K_NATIVE) {
		int i;
		sb_puts(&b, "{\"index\":0,\"content\":");
		sb_jstr(&b, p, n);
		sb_puts(&b, ",\"tokens\":[");
		for (i = t->tok_sent; i < t->ntok; i++)
			sb_printf(&b, "%s%d", i > t->tok_sent ? "," : "", t->toks[i]);
		t->tok_sent = t->ntok;
		sb_printf(&b, "],\"stop\":false,\"id_slot\":0,\"tokens_predicted\":%d,"
		    "\"tokens_evaluated\":%d}", t->ntok, t->prompt_n + t->cached_n);
	} else if (t->kind == K_OAI_CMPL) {
		sb_puts(&b, "{\"choices\":[{\"text\":");
		sb_jstr(&b, p, n);
		sb_printf(&b, ",\"index\":0,\"logprobs\":null,\"finish_reason\":null}],"
		    "\"created\":%ld,\"id\":\"%s\",\"model\":", t->created, t->id);
		sb_jstr(&b, model_name, strlen(model_name));
		sb_puts(&b, ",\"system_fingerprint\":\"q2k\",\"object\":\"text_completion\"}");
	} else {
		sb_printf(&b, "{\"choices\":[{\"finish_reason\":null,\"index\":0,\"delta\":{\"%s\":",
		    is_rsn ? "reasoning_content" : "content");
		sb_jstr(&b, p, n);
		sb_printf(&b, "}}],\"created\":%ld,\"id\":\"%s\",\"model\":", t->created, t->id);
		sb_jstr(&b, model_name, strlen(model_name));
		sb_puts(&b, ",\"system_fingerprint\":\"q2k\",\"object\":\"chat.completion.chunk\"}");
	}
	if (sse_send(t->fd, b.p, b.n))
		t->dead = 1;
	sb_free(&b);
}

static void
put(task *t, size_t a, size_t b, int is_rsn)
{
	if (b <= a)
		return;
	sb_add(is_rsn ? &t->rsn : &t->cnt, t->text.p + a, b - a);
	if (t->stream)
		send_delta(t, t->text.p + a, b - a, is_rsn);
}

/* classify text[done..) as reasoning or content, holding back what a
 * stop word, "</think>" or a split UTF-8 character could still change
 * (unless final) */
static void
flush(task *t, int final)
{
	size_t end = t->text.n;
	int i;
	if (!final) {
		size_t hold = 0, k;
		for (i = 0; i < t->nstop; i++)
			if ((k = partial(t->text.p, t->text.n, t->stop[i])) > hold)
				hold = k;
		if (t->reasoning && !t->think_done &&
		    (k = partial(t->text.p, t->text.n, "</think>")) > hold)
			hold = k;
		if (t->tools && !t->in_calls &&
		    (k = partial(t->text.p, t->text.n, "<function")) > hold)
			hold = k;
		end -= hold;
		end -= utf8_tail(t->text.p, end);
	}
	if (end <= t->done)
		return;
	if (t->reasoning && !t->think_done) {
		size_t p = memfind(t->text.p + t->done, t->text.n - t->done, "</think>", 8);
		if (p == (size_t)-1) {
			put(t, t->done, end, 1);
			t->done = end;
			return;
		}
		p += t->done;
		put(t, t->done, p, 1);
		t->done = p + 8;
		t->think_done = 1;
		if (end <= t->done)
			return;
	}
	if (t->reasoning && !t->content_started) {
		while (t->done < end && strchr(" \t\r\n", t->text.p[t->done]))
			t->done++;
		if (t->done < end)
			t->content_started = 1;
	}
	if (t->tools && !t->in_calls) {
		size_t p = memfind(t->text.p + t->done, end - t->done, "<function", 9);
		if (p != (size_t)-1) {
			put(t, t->done, t->done + p, 0);
			t->done += p;
			t->in_calls = 1;
		}
	}
	if (t->in_calls) {
		sb_add(&t->calls, t->text.p + t->done, end - t->done);
		t->done = end;
		return;
	}
	put(t, t->done, end, 0);
	if (end > t->done)
		t->done = end;
}

static int
on_token(int id, void *u)
{
	task *t = u;
	char buf[512];
	size_t old = t->text.n;
	int k = tok_piece(q2k_t, id, q2k_kept(id), buf, sizeof(buf)), i;

	t->toks[t->ntok++] = id;
	sb_add(&t->text, buf, (size_t)k);
	for (i = 0; i < t->nstop; i++) {
		size_t sl = strlen(t->stop[i]), from = old + 1 > sl ? old + 1 - sl : 0, p;
		p = memfind(t->text.p + from, t->text.n - from, t->stop[i], sl);
		if (p != (size_t)-1) {
			t->text.n = from + p;
			t->text.p[t->text.n] = 0;
			t->stop_type = STOP_WORD;
			t->stop_word = t->stop[i];
			return 1;
		}
	}
	flush(t, 0);
	if (t->dead || client_gone(t->fd)) {
		t->dead = 1;
		return 1;
	}
	return 0;
}

/* the JSON schema type of a tool's parameter, or NULL */
static const char *
param_type(const jv *tools, const char *fn, size_t fnl, const char *pn, size_t pnl)
{
	const jv *k, *p;
	for (k = tools->kid; k; k = k->next) {
		const jv *f = json_get(k, "function");
		const char *name = json_str(f ? f : k, "name", "");
		if (strlen(name) != fnl || memcmp(name, fn, fnl))
			continue;
		p = json_get(json_get(f ? f : k, "parameters"), "properties");
		for (p = p && p->t == J_OBJ ? p->kid : NULL; p; p = p->next)
			if (strlen(p->key) == pnl && !memcmp(p->key, pn, pnl))
				return json_str(p, "type", NULL);
	}
	return NULL;
}

/* s starts with lit? then skip it */
static int
eat(const char **s, const char *e, const char *lit)
{
	size_t l = strlen(lit);
	if ((size_t)(e - *s) < l || memcmp(*s, lit, l))
		return 0;
	*s += l;
	return 1;
}

/* parse the held <function name="F"><param name="P">V</param>...
 * </function> calls into OpenAI tool_calls items; a value is a JSON
 * value when the tool's schema gives the parameter a non-string type
 * and it parses, else a string.  Stops at the first malformed call. */
static void
parse_calls(task *t)
{
	const char *s = t->calls.p, *e = s + t->calls.n;
	sbuf args = { 0 }, item = { 0 };
	while (s < e) {
		const char *fn, *fe;
		char cid[40];
		while (s < e && strchr(" \t\r\n", *s))
			s++;
		if (s == e || !eat(&s, e, "<function name=\""))
			break;
		fn = s;
		if (!(fe = memchr(s, '"', (size_t)(e - s))))
			break;
		s = fe + 1;
		if (!eat(&s, e, ">"))
			break;
		args.n = 0;
		sb_puts(&args, "{");
		for (;;) {
			const char *pn, *pe, *v, *ve;
			size_t vl;
			while (s < e && strchr(" \t\r\n", *s))
				s++;
			if (eat(&s, e, "</function>"))
				break;
			if (!eat(&s, e, "<param name=\""))
				goto done;
			pn = s;
			if (!(pe = memchr(s, '"', (size_t)(e - s))))
				goto done;
			s = pe + 1;
			if (!eat(&s, e, ">"))
				goto done;
			if (eat(&s, e, "<![CDATA[")) {
				v = s;
				if ((vl = memfind(s, (size_t)(e - s), "]]>", 3)) == (size_t)-1)
					goto done;
				ve = v + vl;
				s = ve + 3;
				if (!eat(&s, e, "</param>"))
					goto done;
			} else {
				v = s;
				if ((vl = memfind(s, (size_t)(e - s), "</param>", 8)) == (size_t)-1)
					goto done;
				ve = v + vl;
				s = ve + 8;
			}
			if (args.n > 1)
				sb_puts(&args, ",");
			sb_jstr(&args, pn, (size_t)(pe - pn));
			sb_puts(&args, ":");
			{
				const char *ty = param_type(t->tools, fn, (size_t)(fe - fn), pn,
				    (size_t)(pe - pn));
				jv *pj = NULL;
				char err[64];
				if (ty && strcmp(ty, "string") && ve > v)
					pj = json_parse(v, (size_t)(ve - v), err, sizeof(err));
				if (pj) {
					sb_json(&args, pj);
					json_free(pj);
				} else
					sb_jstr(&args, v, (size_t)(ve - v));
			}
		}
		sb_puts(&args, "}");
		gen_id(cid, sizeof(cid), "call_");
		sb_printf(&item, "\"type\":\"function\",\"function\":{\"name\":");
		sb_jstr(&item, fn, (size_t)(fe - fn));
		sb_puts(&item, ",\"arguments\":");
		sb_jstr(&item, args.p, args.n);
		sb_printf(&item, "},\"id\":\"%s\"}", cid);
		sb_puts(&t->calls_json, t->ncalls ? ",{" : "{");
		sb_add(&t->calls_json, item.p, item.n);
		sb_printf(&t->calls_delta, "%s{\"index\":%d,", t->ncalls ? "," : "", t->ncalls);
		sb_add(&t->calls_delta, item.p, item.n);
		item.n = 0;
		t->ncalls++;
	}
done:
	sb_free(&args);
	sb_free(&item);
}

/* after generation: the held calls become tool_calls, or content when
 * none parse */
static void
finish_calls(task *t)
{
	if (!t->in_calls)
		return;
	parse_calls(t);
	if (t->ncalls) {
		/* the whitespace before the first call is not content */
		while (t->cnt.n && strchr(" \t\r\n", t->cnt.p[t->cnt.n - 1]))
			t->cnt.p[--t->cnt.n] = 0;
		if (t->stream && !t->dead) {
			/* one chunk with every call, each with an index */
			sbuf b = { 0 };
			sb_puts(&b, "{\"choices\":[{\"finish_reason\":null,\"index\":0,"
			    "\"delta\":{\"tool_calls\":[");
			sb_add(&b, t->calls_delta.p, t->calls_delta.n);
			sb_printf(&b, "]}}],\"created\":%ld,\"id\":\"%s\",\"model\":", t->created, t->id);
			sb_jstr(&b, model_name, strlen(model_name));
			sb_puts(&b, ",\"system_fingerprint\":\"q2k\",\"object\":\"chat.completion.chunk\"}");
			if (sse_send(t->fd, b.p, b.n))
				t->dead = 1;
			sb_free(&b);
		}
	} else {
		sb_add(&t->cnt, t->calls.p, t->calls.n);
		if (t->stream)
			send_delta(t, t->calls.p, t->calls.n, 0);
	}
}

static void
timings(sbuf *b, const task *t)
{
	sb_printf(b, "\"timings\":{\"cache_n\":%d,\"prompt_n\":%d,\"prompt_ms\":%.3f,"
	    "\"prompt_per_token_ms\":%.3f,\"prompt_per_second\":%.3f,\"predicted_n\":%d,"
	    "\"predicted_ms\":%.3f,\"predicted_per_token_ms\":%.3f,\"predicted_per_second\":%.3f}",
	    t->cached_n, t->prompt_n, t->t_prompt * 1e3,
	    t->prompt_n ? t->t_prompt * 1e3 / t->prompt_n : 0.0,
	    t->t_prompt > 0 ? t->prompt_n / t->t_prompt : 0.0, t->ntok, t->t_gen * 1e3,
	    t->ntok ? t->t_gen * 1e3 / t->ntok : 0.0, t->t_gen > 0 ? t->ntok / t->t_gen : 0.0);
}

static void
usage_obj(sbuf *b, const task *t)
{
	int pt = t->prompt_n + t->cached_n;
	sb_printf(b, "\"usage\":{\"completion_tokens\":%d,\"prompt_tokens\":%d,"
	    "\"total_tokens\":%d,\"prompt_tokens_details\":{\"cached_tokens\":%d}}",
	    t->ntok, pt, pt + t->ntok, t->cached_n);
}

static const char *
finish_reason(const task *t)
{
	return t->ncalls ? "tool_calls" : t->stop_type == STOP_LIMIT ? "length" : "stop";
}

/* the final object (stream: the last chunk) */
static void
final_obj(sbuf *b, const task *t, int return_tokens, float temp, int top_k, float top_p,
    int n_predict)
{
	if (t->kind == K_NATIVE) {
		static const char *st[] = { "none", "eos", "limit", "word" };
		sb_puts(b, "{\"index\":0,\"content\":");
		if (t->stream)
			sb_puts(b, "\"\"");
		else
			sb_jstr(b, t->cnt.p ? t->cnt.p : "", t->cnt.n);
		sb_puts(b, ",\"tokens\":[");
		if (return_tokens || t->stream) {
			int i;
			for (i = t->stream ? t->tok_sent : 0; i < t->ntok; i++)
				sb_printf(b, "%s%d", i > (t->stream ? t->tok_sent : 0) ? "," : "", t->toks[i]);
		}
		sb_puts(b, "],\"id_slot\":0,\"stop\":true,\"model\":");
		sb_jstr(b, model_name, strlen(model_name));
		sb_printf(b, ",\"tokens_predicted\":%d,\"tokens_evaluated\":%d,"
		    "\"generation_settings\":{\"n_predict\":%d,\"temperature\":%g,\"top_k\":%d,"
		    "\"top_p\":%g,\"n_ctx\":%d},\"prompt\":\"\",\"has_new_line\":%s,"
		    "\"truncated\":false,\"stop_type\":\"%s\",\"stopping_word\":",
		    t->ntok, t->prompt_n + t->cached_n, n_predict, temp, top_k, top_p,
		    q2k_s->ctx, t->cnt.p && memchr(t->cnt.p, '\n', t->cnt.n) ? "true" : "false",
		    st[t->stop_type]);
		sb_jstr(b, t->stop_word ? t->stop_word : "", t->stop_word ? strlen(t->stop_word) : 0);
		sb_printf(b, ",\"tokens_cached\":%d,", t->prompt_n + t->cached_n + t->ntok);
		timings(b, t);
		sb_puts(b, "}");
		return;
	}
	sb_puts(b, "{\"choices\":[{\"finish_reason\":");
	sb_printf(b, "\"%s\",\"index\":0,", finish_reason(t));
	if (t->kind == K_OAI_CMPL) {
		sb_puts(b, "\"logprobs\":null,\"text\":");
		if (t->stream)
			sb_puts(b, "\"\"");
		else
			sb_jstr(b, t->cnt.p ? t->cnt.p : "", t->cnt.n);
	} else if (t->stream) {
		sb_puts(b, "\"delta\":{}");
	} else {
		sb_puts(b, "\"message\":{\"role\":\"assistant\",\"content\":");
		sb_jstr(b, t->cnt.p ? t->cnt.p : "", t->cnt.n);
		if (t->reasoning && t->rsn.n) {
			sb_puts(b, ",\"reasoning_content\":");
			sb_jstr(b, t->rsn.p, t->rsn.n);
		}
		if (t->ncalls) {
			sb_puts(b, ",\"tool_calls\":[");
			sb_add(b, t->calls_json.p, t->calls_json.n);
			sb_puts(b, "]");
		}
		sb_puts(b, "}");
	}
	sb_printf(b, "}],\"created\":%ld,\"model\":", t->created);
	sb_jstr(b, model_name, strlen(model_name));
	sb_printf(b, ",\"system_fingerprint\":\"q2k\",\"object\":\"%s\",",
	    t->kind == K_OAI_CMPL ? "text_completion" :
	    t->stream ? "chat.completion.chunk" : "chat.completion");
	usage_obj(b, t);
	sb_printf(b, ",\"id\":\"%s\",", t->id);
	timings(b, t);
	sb_puts(b, "}");
}

static void
gen_id(char *buf, size_t n, const char *prefix)
{
	static const char al[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
	unsigned long long x = (unsigned long long)time(NULL) * 6364136223846793005ull +
	    (++seed_ctr) * 1442695040888963407ull + (unsigned long long)getpid();
	size_t i = strlen(prefix);
	snprintf(buf, n, "%s", prefix);
	for (; i + 1 < n && i < strlen(prefix) + 32; i++) {
		x ^= x >> 33;
		x *= 0xff51afd7ed558ccdull;
		x ^= x >> 29;
		buf[i] = al[x % (sizeof(al) - 1)];
	}
	buf[i] = 0;
}

/* stop strings: a string or an array of strings */
static void
get_stops(const jv *body, task *t)
{
	const jv *s = json_get(body, "stop"), *k;
	t->nstop = 0;
	t->stop = NULL;
	if (!s)
		return;
	if (s->t == J_STR) {
		if (s->len) {
			t->stop = malloc(sizeof(char *));
			t->stop[t->nstop++] = s->s;
		}
		return;
	}
	if (s->t != J_ARR)
		return;
	t->stop = malloc(sizeof(char *) * (s->len + 1));
	for (k = s->kid; k; k = k->next)
		if (k->t == J_STR && k->len)
			t->stop[t->nstop++] = k->s;
}

/* run a request: prompt tokens in, response out over fd */
static int
run(int fd, int kind, const jv *body, int *ptok, int pn, int reasoning, int close)
{
	task t;
	const float *lg;
	float temp = (float)json_num(body, "temperature", def_temp);
	float top_p = (float)json_num(body, "top_p", def_top_p);
	int top_k = (int)json_num(body, "top_k", def_top_k);
	double seed = json_num(body, "seed", -1);
	int n_predict, n, start = 0, *fed, nfed = 0, i, r = 0;
	int cache = json_bool(body, "cache_prompt", 1);
	const jv *so = json_get(body, "stream_options");
	double t0;

	(void)so;
	n_predict = (int)json_num(body, "n_predict", SO->n_predict);
	if (kind != K_NATIVE) {
		n_predict = (int)json_num(body, "max_tokens", n_predict);
		n_predict = (int)json_num(body, "max_completion_tokens", n_predict);
	}
	if (pn <= 0)
		return respond_err(fd, 400, "invalid_request_error", "empty prompt", close);
	if (pn >= q2k_s->ctx)
		return respond_err(fd, 400, "exceed_context_size_error",
		    "the request exceeds the available context size, try increasing it", close);

	memset(&t, 0, sizeof(t));
	t.kind = kind;
	t.fd = fd;
	t.stream = json_bool(body, "stream", 0);
	t.reasoning = reasoning;
	t.created = (long)time(NULL);
	gen_id(t.id, sizeof(t.id), kind == K_OAI_CHAT ? "chatcmpl-" : "cmpl-");
	get_stops(body, &t);
	if (kind == K_OAI_CHAT && strcmp(json_str(body, "tool_choice", "auto"), "none"))
		t.tools = req_tools(body);
	if (t.tools) {
		/* MiniCPM5's call markup is control tokens: output, not stops */
		static const char *const mk[] = { "<function", "</function>", "<param", "</param>" };
		int ids[4], nk = 0, i2;
		for (i2 = 0; i2 < 4; i2++)
			if ((ids[nk] = tok_find(q2k_t, mk[i2])) >= 0)
				nk++;
		q2k_keep_tokens(ids, nk);
	}

	n = q2k_s->ctx - pn;
	if (n_predict >= 0 && n_predict < n)
		n = n_predict;
	t.toks = malloc(sizeof(int) * ((size_t)n + 1));
	fed = malloc(sizeof(int) * ((size_t)n + MTP_KMAX + 2));

	/* the state can only be reused when it holds a prefix of the prompt */
	if (cache && cn > 0 && cn < pn && q2k_s->pos == cn &&
	    !memcmp(ctoks, ptok, sizeof(int) * (size_t)cn))
		start = cn;
	else if (cache && cn > 0 && q2k_s->pos == cn && q2k_m->arch == ARCH_LLAMA) {
		/* pure attention: the KV rows past the common prefix are just
		 * overwritten, so roll back to it (a chat whose re-rendered
		 * history differs from what was generated, an edited earlier
		 * turn) instead of feeding the whole prompt again; at least
		 * one token is fed for the logits */
		int l = 0;
		while (l < cn && l < pn - 1 && ctoks[l] == ptok[l])
			l++;
		q2k_s->pos = l;
		cn = l;
		start = l;
		if (!l)
			state_reset(q2k_s, q2k_m);
	} else {
		state_reset(q2k_s, q2k_m);
		cn = 0;
	}
	t.cached_n = start;
	t.prompt_n = pn - start;
	t0 = now();
	lg = q2k_feed(ptok + start, pn - start);
	t.t_prompt = now() - t0;
	if (!lg) {
		state_reset(q2k_s, q2k_m);
		cn = 0;
		r = respond_err(fd, 500, "server_error", "prompt processing failed", close);
		goto out;
	}
	if (ccap < q2k_s->ctx + MTP_KMAX + 2) {
		ccap = q2k_s->ctx + MTP_KMAX + 2;
		ctoks = realloc(ctoks, sizeof(int) * (size_t)ccap);
	}
	memcpy(ctoks, ptok, sizeof(int) * (size_t)pn);
	cn = pn;

	if (t.stream) {
		if (sse_begin(fd)) {
			r = -1;
			goto out;
		}
		if (kind == K_OAI_CHAT) {
			sbuf b = { 0 };
			sb_printf(&b, "{\"choices\":[{\"finish_reason\":null,\"index\":0,"
			    "\"delta\":{\"role\":\"assistant\",\"content\":null}}],\"created\":%ld,"
			    "\"id\":\"%s\",\"model\":", t.created, t.id);
			sb_jstr(&b, model_name, strlen(model_name));
			sb_puts(&b, ",\"system_fingerprint\":\"q2k\",\"object\":\"chat.completion.chunk\"}");
			if (sse_send(fd, b.p, b.n))
				t.dead = 1;
			sb_free(&b);
		}
	}

	q2k_set_sampling(temp, top_k, top_p, seed >= 0 ? (unsigned long)seed :
	    (unsigned long)time(NULL) ^ (++seed_ctr * 0x9e3779b9ul));
	if (n > 0 && !t.dead)
		q2k_generate(lg, n, on_token, &t, fed, &nfed, &t.t_gen);
	q2k_set_sampling(def_temp, def_top_k, def_top_p, 0);
	for (i = 0; i < nfed; i++)
		ctoks[cn++] = fed[i];
	if (q2k_s->pos != cn)
		cn = 0;         /* lost track: reset next time */
	if (t.stop_type == STOP_NONE)
		t.stop_type = t.ntok >= n ? STOP_LIMIT : STOP_EOS;
	if (t.tools)
		q2k_keep_tokens(NULL, 0);  /* (also at out:, for early exits) */
	flush(&t, 1);
	finish_calls(&t);

	fprintf(stderr, "q2k server: prompt %d tok (%d cached) %.2f s, %.1f tok/s; "
		    "gen %d tok %.2f s, %.2f tok/s%s\n", t.prompt_n, t.cached_n, t.t_prompt,
		    t.t_prompt > 0 ? t.prompt_n / t.t_prompt : 0.0, t.ntok, t.t_gen,
		    t.t_gen > 0 ? t.ntok / t.t_gen : 0.0, t.dead ? " (client gone)" : "");
	if (t.dead) {
		r = -1;
		goto out;
	}
	{
		sbuf b = { 0 };
		final_obj(&b, &t, json_bool(body, "return_tokens", 0), temp, top_k, top_p, n_predict);
		if (t.stream) {
			r = sse_send(fd, b.p, b.n);
			if (!r && kind != K_NATIVE)
				r = sse_send(fd, "[DONE]", 6);
			if (!r)
				r = sse_end(fd);
		} else
			r = respond_json(fd, 200, &b, close);
		sb_free(&b);
	}
out:
	if (t.tools)
		q2k_keep_tokens(NULL, 0);
	free(t.toks);
	free(fed);
	free(t.stop);
	sb_free(&t.text);
	sb_free(&t.rsn);
	sb_free(&t.cnt);
	sb_free(&t.calls);
	sb_free(&t.calls_json);
	sb_free(&t.calls_delta);
	return r;
}

/* ---------------------------------------------------------------- */
/* prompts                                                          */

static void
add_ids(int **ids, int *n, int *cap, const int *src, int k)
{
	if (*n + k > *cap) {
		*cap = (*n + k) * 2 + 16;
		*ids = realloc(*ids, sizeof(int) * (size_t)*cap);
	}
	memcpy(*ids + *n, src, sizeof(int) * (size_t)k);
	*n += k;
}

/* a prompt: a string, token ids, a mix of both, or a list of those
 * (only the first is used); NULL with *err on a bad one */
static int *
prompt_tokens(const jv *p, int *n, const char **err)
{
	int *ids = NULL, cap = 0, i, k;
	const jv *e;
	*n = 0;
	*err = NULL;
	if (!p) {
		*err = "\"prompt\" is required";
		return NULL;
	}
	if (p->t == J_STR) {
		int *x = q2k_encode(p->s, 1, &k);
		add_ids(&ids, n, &cap, x, k);
		free(x);
		return ids;
	}
	if (p->t != J_ARR || !p->kid) {
		*err = "\"prompt\" must be a string or an array";
		return NULL;
	}
	if (p->kid->t == J_ARR || (p->kid->t == J_STR && p->kid->next &&
	    p->kid->next->t == J_STR))
		return prompt_tokens(p->kid, n, err);
	for (e = p->kid; e; e = e->next) {
		if (e->t == J_NUM) {
			i = (int)e->n;
			if (i < 0 || i >= q2k_t->n_vocab) {
				free(ids);
				*err = "token id out of range";
				return NULL;
			}
			add_ids(&ids, n, &cap, &i, 1);
		} else if (e->t == J_STR) {
			int *x = q2k_encode(e->s, 1, &k);
			add_ids(&ids, n, &cap, x, k);
			free(x);
		} else {
			free(ids);
			*err = "bad prompt element";
			return NULL;
		}
	}
	if (!ids)
		ids = malloc(sizeof(int));
	return ids;
}

/* message content: a string or a list of {type: text, text} parts */
static void
content_text(sbuf *b, const jv *c, const char *sep)
{
	const jv *k;
	int first = 1;
	if (!c)
		return;
	if (c->t == J_STR) {
		sb_add(b, c->s, c->len);
		return;
	}
	if (c->t != J_ARR)
		return;
	for (k = c->kid; k; k = k->next) {
		const char *tx = json_str(k, "text", NULL);
		if (tx) {
			if (!first)
				sb_puts(b, sep);
			sb_puts(b, tx);
			first = 0;
		}
	}
}

/* Python's str.strip() on whitespace */
static void
add_trim(sbuf *b, const char *s, size_t n, int left, int right)
{
	while (left && n && strchr(" \t\r\n\f\v", *s)) {
		s++;
		n--;
	}
	while (right && n && strchr(" \t\r\n\f\v", s[n - 1]))
		n--;
	sb_add(b, s, n);
}

static const char *
role_of(const jv *m)
{
	return json_str(m, "role", "");
}

/* a number the way Python prints it: shortest repr that round-trips */
static void
py_num(sbuf *b, double v)
{
	char buf[40];
	int p;
	if (isfinite(v) && v == (double)(long long)v && fabs(v) < 1e15) {
		sb_printf(b, "%lld", (long long)v);
		return;
	}
	for (p = 15; p <= 17; p++) {
		snprintf(buf, sizeof(buf), "%.*g", p, v);
		if (strtod(buf, NULL) == v)
			break;
	}
	sb_puts(b, buf);
}

/* Jinja's tojson (json.dumps): ", " and ": " separators, UTF-8 kept */
static void
py_json(sbuf *b, const jv *v)
{
	const jv *k;
	switch (v->t) {
	case J_NULL: sb_puts(b, "null"); break;
	case J_BOOL: sb_puts(b, v->n ? "true" : "false"); break;
	case J_NUM: py_num(b, v->n); break;
	case J_STR: sb_jstr(b, v->s, v->len); break;
	default:
		sb_add(b, v->t == J_OBJ ? "{" : "[", 1);
		for (k = v->kid; k; k = k->next) {
			if (k != v->kid)
				sb_puts(b, ", ");
			if (v->t == J_OBJ) {
				sb_jstr(b, k->key, strlen(k->key));
				sb_puts(b, ": ");
			}
			py_json(b, k);
		}
		sb_add(b, v->t == J_OBJ ? "}" : "]", 1);
	}
}

/* the non-empty "tools" array of a request, else NULL */
static const jv *
req_tools(const jv *body)
{
	const jv *t = json_get(body, "tools");
	return t && t->t == J_ARR && t->kid ? t : NULL;
}

/* MiniCPM5's tool block (the template's tool_definitions) */
static void
minicpm_tool_defs(sbuf *b, const jv *tools)
{
	const jv *k;
	sb_puts(b, "# Tools\n\nYou are provided with function signatures within <tools></tools> "
	    "XML tags:\n<tools>");
	for (k = tools->kid; k; k = k->next) {
		sb_puts(b, "\n");
		py_json(b, k);
	}
	sb_puts(b, "\n</tools>\n\nTool usage guidelines:\n- You may call zero or more functions. "
	    "If no function calls are needed, just answer normally and do not include any "
	    "<function ... </function>.\n- When calling a function, return an XML object within "
	    "<function ... </function> using:\n<function name=\"function-name\"><param "
	    "name=\"param-name\">param-value</param></function>\n- param-value may be multi-line. "
	    "If it contains <, & or newline characters, wrap it in a CDATA block: <param "
	    "name=\"param-name\"><![CDATA[...multi-line value...]]></param>");
}

/* an assistant message's tool_calls as MiniCPM5 XML.  The GGUF template
 * would emit every call twice (its <tool_sep> pass appends them to the
 * content, then the has_tool_sep pass, with that variable never set,
 * appends them again); we emit them once. */
static void
minicpm_calls(sbuf *out, const jv *calls, int had_content)
{
	const jv *tc;
	int first = 1;
	for (tc = calls && calls->t == J_ARR ? calls->kid : NULL; tc; tc = tc->next) {
		const jv *fn = json_get(tc, "function"), *a, *pv;
		jv *parsed = NULL;
		if (!fn)
			fn = tc;
		if (!first || had_content)
			sb_puts(out, "\n");
		first = 0;
		sb_puts(out, "<function name=\"");
		sb_puts(out, json_str(fn, "name", ""));
		sb_puts(out, "\">");
		/* OpenAI clients send the arguments as a JSON string; the
		 * template iterates them as a mapping */
		a = json_get(fn, "arguments");
		if (a && a->t == J_STR) {
			char err[64];
			a = parsed = json_parse(a->s, a->len, err, sizeof(err));
		}
		for (pv = a && a->t == J_OBJ ? a->kid : NULL; pv; pv = pv->next) {
			sb_printf(out, "<param name=\"%s\">", pv->key);
			if (pv->t == J_STR) {
				int cd = memchr(pv->s, '<', pv->len) || memchr(pv->s, '&', pv->len) ||
				    memchr(pv->s, '\n', pv->len);
				if (cd)
					sb_puts(out, "<![CDATA[");
				sb_add(out, pv->s, pv->len);
				if (cd)
					sb_puts(out, "]]>");
			} else if (pv->t == J_BOOL)
				sb_puts(out, pv->n ? "True" : "False");
			else if (pv->t == J_NULL)
				sb_puts(out, "None");
			else if (pv->t == J_NUM)
				py_num(out, pv->n);
			else
				py_json(out, pv);
			sb_puts(out, "</param>");
		}
		sb_puts(out, "</function>");
		if (parsed)
			json_free(parsed);
	}
}

/* the model's chat template (hand-written per architecture, following
 * the GGUF's Jinja template; tools only for MiniCPM5, no vision); 0 on
 * success */
static int
apply_template(const jv *body, sbuf *out, int *think, const char **err)
{
	const jv *msgs = json_get(body, "messages"), *m, *kw = json_get(body, "chat_template_kwargs");
	const jv *et = json_get(kw, "enable_thinking");
	const char *effort = json_str(kw, "reasoning_effort", json_str(body, "reasoning_effort", NULL));
	sbuf c = { 0 };

	*err = NULL;
	if (!msgs || msgs->t != J_ARR || !msgs->kid) {
		*err = "\"messages\" must be a non-empty array";
		return -1;
	}
	if (!et)
		et = json_get(body, "enable_thinking");
	*think = et && et->t == J_BOOL ? (int)et->n : SO->think;

	if (q2k_m->arch == ARCH_QWEN35) {
		const char *instr = NULL;
		sbuf sys = { 0 };
		int nsys = 0;
		for (m = msgs->kid; m && (!strcmp(role_of(m), "system") ||
		    !strcmp(role_of(m), "developer")); m = m->next, nsys++) {
			c.n = 0;
			content_text(&c, json_get(m, "content"), "");
			if (c.n) {
				size_t before = sys.n;
				if (sys.n)
					sb_puts(&sys, "\n");
				add_trim(&sys, c.p, c.n, 1, 1);
				if (sys.n == before + (before ? 1 : 0))
					sys.n = before;     /* empty after trim */
			}
		}
		if (*think) {
			if (!effort || !strcmp(effort, "xhigh") || !strcmp(effort, "high"))
				instr = "Reasoning effort is set to xhigh. Please think carefully through the "
				    "task, validate key assumptions, consider plausible alternatives, and "
				    "prioritize correctness, consistency, and clarity in the final answer.";
			else if (!strcmp(effort, "low"))
				instr = "Reasoning effort is set to low. Keep your thinking brief and focused, "
				    "moving directly to the conclusion without unnecessary elaboration.";
			else if (strcmp(effort, "medium")) {
				*err = "unexpected reasoning effort (xhigh, medium or low)";
				sb_free(&sys);
				sb_free(&c);
				return -1;
			}
		}
		if (sys.n || instr) {
			sb_puts(out, "<|im_start|>system\n");
			if (instr)
				sb_puts(out, instr);
			if (instr && sys.n)
				sb_puts(out, "\n\n");
			if (sys.n)
				sb_add(out, sys.p, sys.n);
			sb_puts(out, "<|im_end|>\n");
		}
		sb_free(&sys);
		for (; m; m = m->next) {
			const char *r = role_of(m);
			c.n = 0;
			content_text(&c, json_get(m, "content"), "");
			if (!strcmp(r, "user")) {
				sb_puts(out, "<|im_start|>user\n");
				add_trim(out, c.p, c.n, 1, 1);
				sb_puts(out, "<|im_end|>\n");
			} else if (!strcmp(r, "assistant")) {
				const char *rc = json_str(m, "reasoning_content", "");
				sb_puts(out, "<|im_start|>assistant\n<think>\n");
				add_trim(out, rc, strlen(rc), 1, 1);
				sb_puts(out, "\n</think>\n\n");
				add_trim(out, c.p, c.n, 1, 1);
				sb_puts(out, "<|im_end|>\n");
			} else if (!strcmp(r, "tool")) {
				/* opened unless the previous message was a tool's too */
				if (m != msgs->kid) {
					const jv *pv;
					for (pv = msgs->kid; pv->next != m; pv = pv->next)
						;
					if (strcmp(role_of(pv), "tool"))
						sb_puts(out, "<|im_start|>user");
				}
				sb_puts(out, "\n<tool_response>\n");
				add_trim(out, c.p, c.n, 1, 1);
				sb_puts(out, "\n</tool_response>");
				if (!m->next || strcmp(role_of(m->next), "tool"))
					sb_puts(out, "<|im_end|>\n");
			} else if (!strcmp(r, "system") || !strcmp(r, "developer")) {
				*err = "System message must be at the beginning.";
				sb_free(&c);
				return -1;
			} else {
				*err = "Unexpected message role.";
				sb_free(&c);
				return -1;
			}
		}
		sb_puts(out, "<|im_start|>assistant\n");
		sb_puts(out, *think ? "<think>\n" : "<think>\n\n</think>\n\n");
	} else if (q2k_m->arch == ARCH_LLAMA) {
		/* MiniCPM5: ChatML, bos first; with tools the system turn is
		 * messages[0] (if a system message) plus the tool block */
		const jv *pv = NULL, *tools = req_tools(body);
		char bos[64];
		int k = tok_piece(q2k_t, q2k_t->bos, 1, bos, sizeof(bos) - 1);
		sb_add(out, bos, (size_t)k);
		m = msgs->kid;
		if (tools) {
			sbuf defs = { 0 };
			minicpm_tool_defs(&defs, tools);
			sb_puts(out, "<|im_start|>system\n");
			if (!strcmp(role_of(m), "system")) {
				const char *s, *q;
				c.n = 0;
				content_text(&c, json_get(m, "content"), "");
				s = c.p ? c.p : "";
				if (strstr(s, "<tool_def_sep>")) {
					while ((q = strstr(s, "<tool_def_sep>"))) {
						sb_add(out, s, (size_t)(q - s));
						sb_add(out, defs.p, defs.n);
						s = q + 14;
					}
					sb_puts(out, s);
				} else {
					sb_add(out, c.p ? c.p : "", c.n);
					sb_puts(out, "\n\n");
					sb_add(out, defs.p, defs.n);
				}
				pv = m;
				m = m->next;
			} else
				sb_add(out, defs.p, defs.n);
			sb_puts(out, "<|im_end|>\n");
			sb_free(&defs);
		}
		for (; m; pv = m, m = m->next) {
			const char *r = role_of(m);
			const jv *cv = json_get(m, "content");
			c.n = 0;
			if (cv && cv->t == J_STR)
				sb_add(&c, cv->s, cv->len);
			else
				content_text(&c, cv, "");
			if (!c.p)
				sb_add(&c, "", 0);
			if (!strcmp(r, "user") || !strcmp(r, "system")) {
				sb_printf(out, "<|im_start|>%s\n", r);
				sb_add(out, c.p, c.n);
				sb_puts(out, "<|im_end|>\n");
			} else if (!strcmp(r, "assistant")) {
				const jv *rv = json_get(m, "reasoning_content"), *calls;
				sbuf rs = { 0 }, ct = { 0 };
				const char *e = c.n ? strstr(c.p, "</think>") : NULL;
				if (rv && rv->t == J_STR) {
					sb_add(&rs, rv->s, rv->len);
					sb_add(&ct, c.p, c.n);
				} else if (e) {
					/* split '<think>r</think>c' */
					const char *a = c.p, *q;
					for (q = c.p; (q = strstr(q, "<think>")) && q < e; q += 7)
						a = q + 7;
					while (a < e && *a == '\n')
						a++;
					{
						const char *z = e;
						while (z > a && z[-1] == '\n')
							z--;
						sb_add(&rs, a, (size_t)(z - a));
					}
					{
						const char *last = e, *q2 = e;
						while ((q2 = strstr(q2 + 8, "</think>")))
							last = q2;
						last += 8;
						while (*last == '\n')
							last++;
						sb_puts(&ct, last);
					}
				} else
					sb_add(&ct, c.p, c.n);
				if (!ct.p)
					sb_add(&ct, "", 0);
				calls = json_get(m, "tool_calls");
				if (calls && (calls->t != J_ARR || !calls->kid))
					calls = NULL;
				if (rs.n) {
					sb_puts(out, "<|im_start|>assistant\n<think>\n");
					{
						size_t a = 0, z = rs.n;
						while (a < z && rs.p[a] == '\n')
							a++;
						while (z > a && rs.p[z - 1] == '\n')
							z--;
						sb_add(out, rs.p + a, z - a);
					}
					sb_puts(out, "\n</think>\n\n");
					add_trim(out, ct.p, ct.n, 0, 0);
				} else if (!strstr(ct.p, "<think>") && !strstr(ct.p, "</think>")) {
					size_t a = 0;
					sb_puts(out, "<|im_start|>assistant\n<think>\n\n</think>\n\n");
					while (a < ct.n && ct.p[a] == '\n')
						a++;
					sb_add(out, ct.p + a, ct.n - a);
				} else {
					sb_puts(out, "<|im_start|>assistant\n");
					sb_add(out, ct.p, ct.n);
				}
				if (calls)
					minicpm_calls(out, calls, ct.n > 0);
				sb_puts(out, "<|im_end|>\n");
				sb_free(&rs);
				sb_free(&ct);
			} else if (!strcmp(r, "tool")) {
				if (!pv || strcmp(role_of(pv), "tool"))
					sb_puts(out, "<|im_start|>user");
				sb_puts(out, "\n<tool_response>\n");
				if (cv && cv->t != J_STR && cv->t != J_NULL)
					sb_json(out, cv);
				else
					sb_add(out, c.p, c.n);
				sb_puts(out, "\n</tool_response>");
				if (!m->next || strcmp(role_of(m->next), "tool"))
					sb_puts(out, "<|im_end|>\n");
			}
		}
		/* if (pv) the system message (messages[0]) was emitted in order */
		sb_puts(out, "<|im_start|>assistant\n");
		/* the template adds a think block only when enable_thinking is
		 * given; we always make the choice explicit */
		sb_puts(out, *think ? "<think>\n" : "<think>\n\n</think>\n\n");
	} else {
		/* granitehybrid */
		m = msgs->kid;
		sb_puts(out, "<|start_of_role|>system<|end_of_role|>");
		if (!strcmp(role_of(m), "system")) {
			c.n = 0;
			content_text(&c, json_get(m, "content"), "\n");
			sb_add(out, c.p ? c.p : "", c.n);
			m = m->next;
		} else
			sb_puts(out, "You are a helpful assistant. Please ensure responses are "
			    "professional, accurate, and safe.");
		sb_puts(out, "<|end_of_text|>\n");
		{
			const jv *pv = msgs->kid;
			for (; m; m = m->next) {
				const char *r = role_of(m);
				c.n = 0;
				content_text(&c, json_get(m, "content"), "\n");
				if (!strcmp(r, "user") || !strcmp(r, "system") || !strcmp(r, "assistant")) {
					sb_printf(out, "<|start_of_role|>%s<|end_of_role|>", r);
					sb_add(out, c.p ? c.p : "", c.n);
					sb_puts(out, "<|end_of_text|>\n");
				} else if (!strcmp(r, "tool")) {
					if (m == msgs->kid || strcmp(role_of(pv), "tool"))
						sb_puts(out, "<|start_of_role|>user<|end_of_role|>");
					sb_puts(out, "\n<tool_response>\n");
					sb_add(out, c.p ? c.p : "", c.n);
					sb_puts(out, "\n</tool_response>");
					if (!m->next || strcmp(role_of(m->next), "tool"))
						sb_puts(out, "<|end_of_text|>\n");
				}
				pv = m;
			}
		}
		sb_puts(out, "<|start_of_role|>assistant<|end_of_role|>");
		*think = 0;
	}
	sb_free(&c);
	return 0;
}

/* ---------------------------------------------------------------- */
/* endpoints                                                        */

static void
props(sbuf *b)
{
	gstr tpl;
	sb_printf(b, "{\"default_generation_settings\":{\"n_ctx\":%d,\"params\":{\"n_predict\":%d,"
	    "\"temperature\":%g,\"top_k\":%d,\"top_p\":%g}},\"total_slots\":1,\"model_alias\":",
	    q2k_s->ctx, SO->n_predict, def_temp, def_top_k, def_top_p);
	sb_jstr(b, model_name, strlen(model_name));
	sb_puts(b, ",\"model_path\":");
	sb_jstr(b, SO->model_path, strlen(SO->model_path));
	sb_puts(b, ",\"chat_template\":");
	if (gguf_get_str(&q2k_m->g, "tokenizer.chat_template", &tpl) == 0)
		sb_jstr(b, tpl.p, (size_t)tpl.len);
	else
		sb_puts(b, "\"\"");
	sb_puts(b, ",\"modalities\":{\"vision\":false,\"audio\":false},\"build_info\":\"q2k\"}");
}

static void
models(sbuf *b)
{
	long t = (long)time(NULL);
	sb_puts(b, "{\"object\":\"list\",\"data\":[{\"id\":");
	sb_jstr(b, model_name, strlen(model_name));
	sb_printf(b, ",\"object\":\"model\",\"created\":%ld,\"owned_by\":\"q2k\",\"meta\":{"
	    "\"n_vocab\":%d,\"n_ctx\":%d,\"n_embd\":%d}}],\"models\":[{\"name\":", t,
	    q2k_m->hp.n_vocab, q2k_s->ctx, q2k_m->hp.n_embd);
	sb_jstr(b, model_name, strlen(model_name));
	sb_puts(b, ",\"model\":");
	sb_jstr(b, model_name, strlen(model_name));
	sb_puts(b, ",\"type\":\"model\",\"capabilities\":[\"completion\"]}]}");
}

static int
handle(conn *cn_, const req *q)
{
	const int fd = cn_->fd, close = q->close;
	const char *p = q->path;
	int get = !strcmp(q->method, "GET") || !strcmp(q->method, "HEAD");
	int post = !strcmp(q->method, "POST");
	jv *body = NULL;
	sbuf b = { 0 };
	int r = 0;
	char err[128];

	if (!strcmp(q->method, "OPTIONS")) {
		static const char h[] = "HTTP/1.1 204 No Content\r\nServer: q2k\r\n" CORS
		    "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
		    "Access-Control-Allow-Headers: *\r\nAccess-Control-Max-Age: 86400\r\n"
		    "Content-Length: 0\r\n\r\n";
		return send_all(fd, h, sizeof(h) - 1);
	}
	if (!strncmp(p, "/v1/", 4) && strcmp(p, "/v1/health") && strcmp(p, "/v1/models") &&
	    strcmp(p, "/v1/completions") && strcmp(p, "/v1/chat/completions"))
		p += 3;         /* /v1/tokenize etc. */
	if (SO->api_key && strcmp(p, "/health") && strcmp(p, "/v1/health")) {
		const char *a = q->auth;
		if (strncasecmp(a, "Bearer ", 7) || strcmp(a + 7, SO->api_key))
			return respond_err(fd, 401, "authentication_error", "Invalid API Key", close);
	}
	if (get && (!strcmp(p, "/health") || !strcmp(p, "/v1/health"))) {
		sb_puts(&b, "{\"status\":\"ok\"}");
		r = respond_json(fd, 200, &b, close);
		goto out;
	}
	if (get && !strcmp(p, "/")) {
		static const char t[] = "q2k server: POST /completion, /v1/completions, "
		    "/v1/chat/completions, /tokenize, /detokenize, /apply-template; "
		    "GET /health, /props, /v1/models, /slots\n";
		r = respond(fd, 200, "text/plain; charset=utf-8", t, sizeof(t) - 1, close);
		goto out;
	}
	if (get && !strcmp(p, "/props")) {
		props(&b);
		r = respond_json(fd, 200, &b, close);
		goto out;
	}
	if (get && (!strcmp(p, "/models") || !strcmp(p, "/v1/models"))) {
		models(&b);
		r = respond_json(fd, 200, &b, close);
		goto out;
	}
	if (get && !strcmp(p, "/slots")) {
		sb_printf(&b, "[{\"id\":0,\"n_ctx\":%d,\"is_processing\":false,\"n_past\":%d}]",
		    q2k_s->ctx, q2k_s->pos);
		r = respond_json(fd, 200, &b, close);
		goto out;
	}
	if (!post) {
		r = respond_err(fd, get ? 404 : 405, "not_found_error", "File Not Found", close);
		goto out;
	}
	if (!(body = json_parse(q->body, q->blen, err, sizeof(err))) || body->t != J_OBJ) {
		r = respond_err(fd, 400, "invalid_request_error",
		    body ? "request body must be a JSON object" : err, close);
		goto out;
	}
	if (!strcmp(p, "/props")) {
		sb_puts(&b, "{\"success\":true}");
		r = respond_json(fd, 200, &b, close);
	} else if (!strcmp(p, "/completion") || !strcmp(p, "/completions") ||
	    !strcmp(p, "/v1/completions")) {
		const char *e;
		int n, *ids = prompt_tokens(json_get(body, "prompt"), &n, &e);
		if (!ids)
			r = respond_err(fd, 400, "invalid_request_error", e, close);
		else
			r = run(fd, strcmp(p, "/v1/completions") ? K_NATIVE : K_OAI_CMPL, body, ids, n,
			    0, close);
		free(ids);
	} else if (!strcmp(p, "/chat/completions") || !strcmp(p, "/v1/chat/completions") ||
	    !strcmp(p, "/apply-template")) {
		const char *e;
		int think, n, *ids;
		if (apply_template(body, &b, &think, &e)) {
			r = respond_err(fd, 400, "invalid_request_error", e, close);
		} else if (!strcmp(p, "/apply-template")) {
			sbuf o = { 0 };
			sb_puts(&o, "{\"prompt\":");
			sb_jstr(&o, b.p, b.n);
			sb_puts(&o, "}");
			r = respond_json(fd, 200, &o, close);
			sb_free(&o);
		} else {
			const char *rf = json_str(body, "reasoning_format", "auto");
			ids = q2k_encode(b.p, 1, &n);
			r = run(fd, K_OAI_CHAT, body, ids, n, think && strcmp(rf, "none"), close);
			free(ids);
		}
	} else if (!strcmp(p, "/tokenize")) {
		const char *s = json_str(body, "content", "");
		int n, i, *ids = q2k_encode(s, json_bool(body, "parse_special", 1), &n);
		int pieces = json_bool(body, "with_pieces", 0);
		sb_puts(&b, "{\"tokens\":[");
		for (i = 0; i < n; i++) {
			if (i)
				sb_puts(&b, ",");
			if (pieces) {
				char buf[512];
				int k = tok_piece(q2k_t, ids[i], 1, buf, sizeof(buf));
				sb_printf(&b, "{\"id\":%d,\"piece\":", ids[i]);
				sb_jstr(&b, buf, (size_t)k);
				sb_puts(&b, "}");
			} else
				sb_printf(&b, "%d", ids[i]);
		}
		sb_puts(&b, "]}");
		free(ids);
		r = respond_json(fd, 200, &b, close);
	} else if (!strcmp(p, "/detokenize")) {
		const jv *t = json_get(body, "tokens"), *k;
		sbuf s = { 0 };
		for (k = t && t->t == J_ARR ? t->kid : NULL; k; k = k->next)
			if (k->t == J_NUM) {
				char buf[512];
				int n = tok_piece(q2k_t, (int)k->n, 1, buf, sizeof(buf));
				sb_add(&s, buf, (size_t)n);
			}
		sb_puts(&b, "{\"content\":");
		sb_jstr(&b, s.p ? s.p : "", s.n);
		sb_puts(&b, "}");
		sb_free(&s);
		r = respond_json(fd, 200, &b, close);
	} else
		r = respond_err(fd, 404, "not_found_error", "File Not Found", close);
out:
	json_free(body);
	sb_free(&b);
	return r;
}

/* ---------------------------------------------------------------- */
/* connections                                                      */

static void
conn_close(int i)
{
	close(conns[i].fd);
	sb_free(&conns[i].in);
	conns[i] = conns[--nconn];
}

/* parse one request from the buffer: 1 handled (consumed), 0 need more,
 * -1 close the connection */
static int
try_request(conn *c)
{
	char *h = c->in.p, *e, *line, *nl;
	size_t hl, clen = 0, need, he;
	req q;
	int chunked = 0, expect = 0, http10 = 0, r;
	char *body = NULL;
	size_t blen = 0;

	if (!h || (he = memfind(h, c->in.n, "\r\n\r\n", 4)) == (size_t)-1) {
		if (c->in.n > MAX_HEAD) {
			respond_err(c->fd, 413, "invalid_request_error", "header too large", 1);
			return -1;
		}
		return 0;
	}
	e = h + he;
	hl = he + 4;
	memset(&q, 0, sizeof(q));
	/* request line */
	nl = memchr(h, '\n', hl);
	{
		char ver[16] = "";
		char fmt[64];
		snprintf(fmt, sizeof(fmt), "%%15s %%%zus %%15s", sizeof(q.path) - 1);
		if (sscanf(h, fmt, q.method, q.path, ver) < 2) {
			respond_err(c->fd, 400, "invalid_request_error", "bad request line", 1);
			return -1;
		}
		http10 = !strncmp(ver, "HTTP/1.0", 8);
	}
	{
		char *qs = strchr(q.path, '?');
		if (qs)
			*qs = 0;
	}
	q.close = http10;
	for (line = nl + 1; line < e; line = nl + 1) {
		char *colon, *v, *end;
		nl = memchr(line, '\n', (size_t)(e + 2 - line));
		if (!nl)
			break;
		end = nl;
		if (end > line && end[-1] == '\r')
			end--;
		colon = memchr(line, ':', (size_t)(end - line));
		if (!colon)
			continue;
		for (v = colon + 1; v < end && (*v == ' ' || *v == '\t'); v++)
			;
#define HDR(name) ((size_t)(colon - line) == sizeof(name) - 1 && !strncasecmp(line, name, sizeof(name) - 1))
		if (HDR("Content-Length"))
			clen = (size_t)strtoull(v, NULL, 10);
		else if (HDR("Transfer-Encoding"))
			chunked = end - v >= 7 && !strncasecmp(v, "chunked", 7);
		else if (HDR("Expect"))
			expect = end - v >= 12 && !strncasecmp(v, "100-continue", 12);
		else if (HDR("Connection")) {
			if (end - v >= 5 && !strncasecmp(v, "close", 5))
				q.close = 1;
			else if (end - v >= 10 && !strncasecmp(v, "keep-alive", 10))
				q.close = 0;
		} else if (HDR("Authorization")) {
			size_t n = (size_t)(end - v) < sizeof(q.auth) - 1 ? (size_t)(end - v) : sizeof(q.auth) - 1;
			memcpy(q.auth, v, n);
			q.auth[n] = 0;
		}
#undef HDR
	}
	if (clen > MAX_BODY) {
		respond_err(c->fd, 413, "invalid_request_error", "request body too large", 1);
		return -1;
	}
	if (expect && !c->sent100 && c->in.n == hl) {
		send_all(c->fd, "HTTP/1.1 100 Continue\r\n\r\n", 25);
		c->sent100 = 1;
	}
	if (chunked) {
		/* decode in place into a copy once complete */
		size_t at = hl;
		sbuf d = { 0 };
		for (;;) {
			char *le = at < c->in.n ? memchr(h + at, '\n', c->in.n - at) : NULL;
			size_t sz;
			if (!le) {
				sb_free(&d);
				return 0;
			}
			sz = (size_t)strtoull(h + at, NULL, 16);
			at = (size_t)(le - h) + 1;
			if (sz == 0) {
				/* trailers until an empty line */
				for (;;) {
					char *te = at < c->in.n ? memchr(h + at, '\n', c->in.n - at) : NULL;
					if (!te) {
						sb_free(&d);
						return 0;
					}
					if (te == h + at || (te == h + at + 1 && h[at] == '\r')) {
						at = (size_t)(te - h) + 1;
						break;
					}
					at = (size_t)(te - h) + 1;
				}
				break;
			}
			if (d.n + sz > MAX_BODY) {
				sb_free(&d);
				respond_err(c->fd, 413, "invalid_request_error", "request body too large", 1);
				return -1;
			}
			if (c->in.n < at + sz + 2) {
				sb_free(&d);
				return 0;
			}
			sb_add(&d, h + at, sz);
			at += sz + 2;
		}
		need = at;
		body = d.p;
		blen = d.n;
		q.body = body ? body : "";
		q.blen = blen;
	} else {
		need = hl + clen;
		if (c->in.n < need)
			return 0;
		q.body = h + hl;
		q.blen = clen;
	}
	{
		double t0 = now();
		r = handle(c, &q);
		if (SO->verbose)
			fprintf(stderr, "q2k server: %s %s %.3f s\n", q.method, q.path, now() - t0);
	}
	free(body);
	memmove(c->in.p, c->in.p + need, c->in.n - need);
	c->in.n -= need;
	c->sent100 = 0;
	c->last = time(NULL);
	return r || q.close ? -1 : 1;
}

static volatile sig_atomic_t quit;

static void
on_signal(int s)
{
	(void)s;
	quit = 1;
}

int
server_run(const server_opts *o)
{
	struct addrinfo hints, *ai, *a;
	char port[16];
	int ls = -1, one = 1, e;
	struct sigaction sa;

	SO = o;
	q2k_get_sampling(&def_temp, &def_top_k, &def_top_p);
	if (o->alias)
		snprintf(model_name, sizeof(model_name), "%s", o->alias);
	else {
		const char *b = strrchr(o->model_path, '/');
		snprintf(model_name, sizeof(model_name), "%s", b ? b + 1 : o->model_path);
	}
	signal(SIGPIPE, SIG_IGN);
	memset(&sa, 0, sizeof(sa));
	sa.sa_handler = on_signal;
	sigaction(SIGINT, &sa, NULL);
	sigaction(SIGTERM, &sa, NULL);

	memset(&hints, 0, sizeof(hints));
	hints.ai_family = AF_UNSPEC;
	hints.ai_socktype = SOCK_STREAM;
	hints.ai_flags = AI_PASSIVE;
	snprintf(port, sizeof(port), "%d", o->port);
	if ((e = getaddrinfo(o->host, port, &hints, &ai)) != 0) {
		fprintf(stderr, "q2k server: %s: %s\n", o->host, gai_strerror(e));
		return -1;
	}
	for (a = ai; a; a = a->ai_next) {
		ls = socket(a->ai_family, a->ai_socktype, a->ai_protocol);
		if (ls < 0)
			continue;
		setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
		if (bind(ls, a->ai_addr, a->ai_addrlen) == 0 && listen(ls, 64) == 0)
			break;
		close(ls);
		ls = -1;
	}
	freeaddrinfo(ai);
	if (ls < 0) {
		fprintf(stderr, "q2k server: cannot listen on %s:%d: %s\n", o->host, o->port,
		    strerror(errno));
		return -1;
	}
	fprintf(stderr, "q2k server: %s, n_ctx %d, listening on http://%s:%d\n", model_name,
	    q2k_s->ctx, o->host, o->port);

	while (!quit) {
		struct pollfd pf[MAX_CONN + 1];
		int i, n;
		time_t tn = time(NULL);

		pf[0].fd = ls;
		pf[0].events = nconn < MAX_CONN ? POLLIN : 0;
		for (i = 0; i < nconn; i++) {
			pf[i + 1].fd = conns[i].fd;
			pf[i + 1].events = POLLIN;
			pf[i + 1].revents = 0;
		}
		pf[0].revents = 0;
		n = poll(pf, (nfds_t)nconn + 1, 1000);
		if (n < 0) {
			if (errno == EINTR)
				continue;
			perror("poll");
			break;
		}
		/* connections first (their indices shift when one closes) */
		for (i = nconn - 1; i >= 0; i--) {
			conn *c = &conns[i];
			char buf[65536];
			ssize_t k;
			int st;
			if (!(pf[i + 1].revents & (POLLIN | POLLHUP | POLLERR))) {
				if (tn - c->last > IDLE_SECS)
					conn_close(i);
				continue;
			}
			k = recv(c->fd, buf, sizeof(buf), MSG_DONTWAIT);
			if (k < 0 && (errno == EAGAIN || errno == EINTR))
				continue;
			if (k <= 0) {
				conn_close(i);
				continue;
			}
			sb_add(&c->in, buf, (size_t)k);
			c->last = tn;
			while ((st = try_request(c)) == 1)
				;
			if (st < 0)
				conn_close(i);
		}
		if (pf[0].revents & POLLIN) {
			int fd = accept(ls, NULL, NULL);
			if (fd >= 0) {
				struct timeval tv;
				tv.tv_sec = 60;
				tv.tv_usec = 0;
				setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
				setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
				memset(&conns[nconn], 0, sizeof(conn));
				conns[nconn].fd = fd;
				conns[nconn].last = tn;
				nconn++;
			}
		}
	}
	while (nconn)
		conn_close(0);
	close(ls);
	free(ctoks);
	fprintf(stderr, "q2k server: bye\n");
	return 0;
}
