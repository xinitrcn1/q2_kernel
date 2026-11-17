/* Minimal JSON: a tree parser for request bodies and a string builder
 * for responses. */
#ifndef Q2K_JSON_H
#define Q2K_JSON_H

#include <stddef.h>

enum { J_NULL, J_BOOL, J_NUM, J_STR, J_ARR, J_OBJ };

typedef struct jv {
	int t;
	double n;               /* J_NUM, J_BOOL (0/1) */
	char *s;                /* J_STR: UTF-8, NUL-terminated */
	size_t len;             /* J_STR bytes; J_ARR/J_OBJ children */
	char *key;              /* member name when inside an object */
	struct jv *kid, *next;  /* first child; next sibling */
} jv;

/* NULL on error (message in err) */
jv *json_parse(const char *s, size_t n, char *err, size_t errlen);
void json_free(jv *v);
const jv *json_get(const jv *obj, const char *key);    /* NULL if absent */
/* typed lookups with a default (wrong type counts as absent) */
double json_num(const jv *obj, const char *key, double def);
int json_bool(const jv *obj, const char *key, int def);
const char *json_str(const jv *obj, const char *key, const char *def);

/* growable byte buffer */
typedef struct {
	char *p;
	size_t n, cap;
} sbuf;

void sb_add(sbuf *b, const void *p, size_t n);
void sb_puts(sbuf *b, const char *s);
void sb_printf(sbuf *b, const char *fmt, ...)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 2, 3)))
#endif
    ;
/* a JSON string literal; invalid UTF-8 becomes U+FFFD */
void sb_jstr(sbuf *b, const char *s, size_t n);
/* re-serialize a value (compact) */
void sb_json(sbuf *b, const jv *v);
void sb_free(sbuf *b);

#endif
