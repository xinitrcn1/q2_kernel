/* Minimal JSON parser and writer (see json.h). */
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "json.h"

#define JSON_DEPTH 64

typedef struct {
	const char *p, *e, *s0;
	char *err;
	size_t errlen;
	int depth;
} jp;

static void
ws(jp *j)
{
	while (j->p < j->e && (*j->p == ' ' || *j->p == '\t' || *j->p == '\n' || *j->p == '\r'))
		j->p++;
}

static jv *
fail(jp *j, const char *msg)
{
	if (j->err && !j->err[0])
		snprintf(j->err, j->errlen, "%s at offset %ld", msg, (long)(j->p - j->s0));
	return NULL;
}

static int
hex4(const char *p, unsigned *out)
{
	unsigned v = 0;
	int i;
	for (i = 0; i < 4; i++) {
		int c = p[i];
		v <<= 4;
		if (c >= '0' && c <= '9') v |= (unsigned)(c - '0');
		else if (c >= 'a' && c <= 'f') v |= (unsigned)(c - 'a' + 10);
		else if (c >= 'A' && c <= 'F') v |= (unsigned)(c - 'A' + 10);
		else return -1;
	}
	*out = v;
	return 0;
}

static size_t
put_utf8(char *o, unsigned c)
{
	if (c < 0x80) { o[0] = (char)c; return 1; }
	if (c < 0x800) { o[0] = (char)(0xc0 | c >> 6); o[1] = (char)(0x80 | (c & 63)); return 2; }
	if (c < 0x10000) {
		o[0] = (char)(0xe0 | c >> 12); o[1] = (char)(0x80 | (c >> 6 & 63));
		o[2] = (char)(0x80 | (c & 63));
		return 3;
	}
	o[0] = (char)(0xf0 | c >> 18); o[1] = (char)(0x80 | (c >> 12 & 63));
	o[2] = (char)(0x80 | (c >> 6 & 63)); o[3] = (char)(0x80 | (c & 63));
	return 4;
}

/* at the opening quote; decoded string into a new buffer */
static char *
pstr(jp *j, size_t *len)
{
	const char *q;
	char *o, *w;
	j->p++;
	for (q = j->p; q < j->e && *q != '"'; q++)
		if (*q == '\\')
			q++;
	if (q >= j->e)
		return (fail(j, "unterminated string"), NULL);
	/* escapes never grow the text (\uXXXX is 6 bytes -> at most 4) */
	o = w = malloc((size_t)(q - j->p) + 1);
	if (!o)
		return NULL;
	while (j->p < q) {
		char c = *j->p++;
		if ((unsigned char)c < 0x20) {
			free(o);
			return (fail(j, "control character in string"), NULL);
		}
		if (c != '\\') {
			*w++ = c;
			continue;
		}
		c = *j->p++;
		switch (c) {
		case '"': case '\\': case '/': *w++ = c; break;
		case 'b': *w++ = '\b'; break;
		case 'f': *w++ = '\f'; break;
		case 'n': *w++ = '\n'; break;
		case 'r': *w++ = '\r'; break;
		case 't': *w++ = '\t'; break;
		case 'u': {
			unsigned u, u2;
			if (q - j->p < 4 || hex4(j->p, &u)) {
				free(o);
				return (fail(j, "bad \\u escape"), NULL);
			}
			j->p += 4;
			if (u >= 0xd800 && u < 0xdc00 && q - j->p >= 6 && j->p[0] == '\\' &&
			    j->p[1] == 'u' && !hex4(j->p + 2, &u2) && u2 >= 0xdc00 && u2 < 0xe000) {
				u = 0x10000 + ((u - 0xd800) << 10) + (u2 - 0xdc00);
				j->p += 6;
			} else if (u >= 0xd800 && u < 0xe000)
				u = 0xfffd;
			w += put_utf8(w, u);
			break;
		}
		default:
			free(o);
			return (fail(j, "bad escape"), NULL);
		}
	}
	j->p = q + 1;
	*w = 0;
	*len = (size_t)(w - o);
	return o;
}

static jv *pval(jp *j);

static jv *
pval(jp *j)
{
	jv *v;
	ws(j);
	if (j->p >= j->e)
		return fail(j, "unexpected end");
	if (!(v = calloc(1, sizeof(*v))))
		return NULL;
	switch (*j->p) {
	case '{': case '[': {
		const int obj = *j->p == '{';
		jv **tail = &v->kid;
		if (++j->depth > JSON_DEPTH) {
			free(v);
			return fail(j, "nesting too deep");
		}
		v->t = obj ? J_OBJ : J_ARR;
		j->p++;
		ws(j);
		if (j->p < j->e && *j->p == (obj ? '}' : ']')) {
			j->p++;
			j->depth--;
			return v;
		}
		for (;;) {
			char *key = NULL;
			jv *k;
			size_t kl;
			ws(j);
			if (obj) {
				if (j->p >= j->e || *j->p != '"' || !(key = pstr(j, &kl))) {
					json_free(v);
					return fail(j, "expected member name");
				}
				ws(j);
				if (j->p >= j->e || *j->p != ':') {
					free(key);
					json_free(v);
					return fail(j, "expected ':'");
				}
				j->p++;
			}
			if (!(k = pval(j))) {
				free(key);
				json_free(v);
				return NULL;
			}
			k->key = key;
			*tail = k;
			tail = &k->next;
			v->len++;
			ws(j);
			if (j->p < j->e && *j->p == ',') {
				j->p++;
				continue;
			}
			if (j->p < j->e && *j->p == (obj ? '}' : ']')) {
				j->p++;
				break;
			}
			json_free(v);
			return fail(j, obj ? "expected ',' or '}'" : "expected ',' or ']'");
		}
		j->depth--;
		return v;
	}
	case '"':
		v->t = J_STR;
		if (!(v->s = pstr(j, &v->len))) {
			free(v);
			return NULL;
		}
		return v;
	case 't': case 'f': case 'n': {
		static const char *w[] = { "true", "false", "null" };
		int i;
		for (i = 0; i < 3; i++) {
			size_t l = strlen(w[i]);
			if ((size_t)(j->e - j->p) >= l && !memcmp(j->p, w[i], l)) {
				j->p += l;
				v->t = i == 2 ? J_NULL : J_BOOL;
				v->n = i == 0;
				return v;
			}
		}
		free(v);
		return fail(j, "bad literal");
	}
	default: {
		char buf[64], *end;
		size_t l = 0;
		while (j->p + l < j->e && l < sizeof(buf) - 1 &&
		    strchr("+-0123456789.eE", j->p[l]))
			l++;
		memcpy(buf, j->p, l);
		buf[l] = 0;
		v->t = J_NUM;
		v->n = strtod(buf, &end);
		if (!l || end != buf + l) {
			free(v);
			return fail(j, "bad value");
		}
		j->p += l;
		return v;
	}
	}
}

jv *
json_parse(const char *s, size_t n, char *err, size_t errlen)
{
	jp j;
	jv *v;
	j.p = j.s0 = s;
	j.e = s + n;
	j.err = err;
	j.errlen = errlen;
	j.depth = 0;
	if (err && errlen)
		err[0] = 0;
	if (!(v = pval(&j)))
		return NULL;
	ws(&j);
	if (j.p != j.e) {
		json_free(v);
		return fail(&j, "trailing data");
	}
	return v;
}

void
json_free(jv *v)
{
	while (v) {
		jv *nx = v->next;
		json_free(v->kid);
		free(v->s);
		free(v->key);
		free(v);
		v = nx;
	}
}

const jv *
json_get(const jv *o, const char *key)
{
	const jv *k;
	if (!o || o->t != J_OBJ)
		return NULL;
	for (k = o->kid; k; k = k->next)
		if (!strcmp(k->key, key))
			return k;
	return NULL;
}

double
json_num(const jv *o, const char *key, double def)
{
	const jv *v = json_get(o, key);
	return v && v->t == J_NUM ? v->n : def;
}

int
json_bool(const jv *o, const char *key, int def)
{
	const jv *v = json_get(o, key);
	return v && v->t == J_BOOL ? (int)v->n : def;
}

const char *
json_str(const jv *o, const char *key, const char *def)
{
	const jv *v = json_get(o, key);
	return v && v->t == J_STR ? v->s : def;
}

/* ---------------------------------------------------------------- */

void
sb_add(sbuf *b, const void *p, size_t n)
{
	if (b->n + n + 1 > b->cap) {
		size_t c = b->cap ? b->cap : 256;
		char *q;
		while (c < b->n + n + 1)
			c *= 2;
		if (!(q = realloc(b->p, c)))
			abort();
		b->p = q;
		b->cap = c;
	}
	memcpy(b->p + b->n, p, n);
	b->n += n;
	b->p[b->n] = 0;
}

void
sb_puts(sbuf *b, const char *s)
{
	sb_add(b, s, strlen(s));
}

void
sb_printf(sbuf *b, const char *fmt, ...)
{
	char buf[512];
	va_list ap;
	int n;
	va_start(ap, fmt);
	n = vsnprintf(buf, sizeof(buf), fmt, ap);
	va_end(ap);
	if (n < 0)
		return;
	if ((size_t)n < sizeof(buf)) {
		sb_add(b, buf, (size_t)n);
		return;
	}
	{
		char *p = malloc((size_t)n + 1);
		if (!p)
			abort();
		va_start(ap, fmt);
		vsnprintf(p, (size_t)n + 1, fmt, ap);
		va_end(ap);
		sb_add(b, p, (size_t)n);
		free(p);
	}
}

/* length of the valid UTF-8 sequence at s (0 if invalid) */
static size_t
u8len(const unsigned char *s, size_t n)
{
	unsigned c = s[0];
	size_t l, i;
	unsigned cp;
	if (c < 0x80) return 1;
	if (c >= 0xc2 && c < 0xe0) { l = 2; cp = c & 31; }
	else if (c >= 0xe0 && c < 0xf0) { l = 3; cp = c & 15; }
	else if (c >= 0xf0 && c < 0xf5) { l = 4; cp = c & 7; }
	else return 0;
	if (n < l)
		return 0;
	for (i = 1; i < l; i++) {
		if ((s[i] & 0xc0) != 0x80)
			return 0;
		cp = cp << 6 | (s[i] & 63);
	}
	if ((l == 3 && (cp < 0x800 || (cp >= 0xd800 && cp < 0xe000))) ||
	    (l == 4 && (cp < 0x10000 || cp > 0x10ffff)))
		return 0;
	return l;
}

void
sb_jstr(sbuf *b, const char *s, size_t n)
{
	const unsigned char *u = (const unsigned char *)s;
	size_t i = 0, run = 0;
	sb_add(b, "\"", 1);
	while (i < n) {
		unsigned c = u[i];
		size_t l;
		const char *esc = NULL;
		char tmp[8];
		if (c >= 0x20 && c < 0x80 && c != '"' && c != '\\') {
			i++;
			continue;
		}
		if (c >= 0x80 && (l = u8len(u + i, n - i)) > 0) {
			i += l;
			continue;
		}
		sb_add(b, s + run, i - run);
		switch (c) {
		case '"': esc = "\\\""; break;
		case '\\': esc = "\\\\"; break;
		case '\n': esc = "\\n"; break;
		case '\r': esc = "\\r"; break;
		case '\t': esc = "\\t"; break;
		default:
			if (c < 0x20)
				snprintf(tmp, sizeof(tmp), "\\u%04x", c);
			else
				strcpy(tmp, "\xef\xbf\xbd");    /* invalid byte */
			esc = tmp;
		}
		sb_puts(b, esc);
		i++;
		run = i;
	}
	sb_add(b, s + run, i - run);
	sb_add(b, "\"", 1);
}

void
sb_json(sbuf *b, const jv *v)
{
	const jv *k;
	switch (v->t) {
	case J_NULL: sb_puts(b, "null"); break;
	case J_BOOL: sb_puts(b, v->n ? "true" : "false"); break;
	case J_NUM:
		if (isfinite(v->n) && v->n == (double)(long long)v->n && fabs(v->n) < 1e15)
			sb_printf(b, "%lld", (long long)v->n);
		else
			sb_printf(b, "%.17g", v->n);
		break;
	case J_STR: sb_jstr(b, v->s, v->len); break;
	default:
		sb_add(b, v->t == J_OBJ ? "{" : "[", 1);
		for (k = v->kid; k; k = k->next) {
			if (k != v->kid)
				sb_add(b, ",", 1);
			if (v->t == J_OBJ) {
				sb_jstr(b, k->key, strlen(k->key));
				sb_add(b, ":", 1);
			}
			sb_json(b, k);
		}
		sb_add(b, v->t == J_OBJ ? "}" : "]", 1);
	}
}

void
sb_free(sbuf *b)
{
	free(b->p);
	b->p = NULL;
	b->n = b->cap = 0;
}
