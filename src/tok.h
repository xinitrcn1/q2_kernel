/* Byte-level BPE tokenizer (GPT-2 style vocab, llama3/dbrx pre-split). */
#ifndef Q2K_TOK_H
#define Q2K_TOK_H

#include <stddef.h>
#include <stdint.h>

#include "gguf.h"

enum { UC_L = 1, UC_N = 2, UC_WS = 4, UC_DEF = 8, UC_M = 16 };

typedef struct {
	uint32_t lo, hi;
	int f;
} ucd_range;

extern const ucd_range ucd_ranges[];
extern const int ucd_nranges;

typedef struct {
	int n_vocab;
	gstr *text;             /* token text (byte-encoded) */
	uint8_t *type;          /* 1 normal, 3 control, 4 user-defined ... */
	int *special;           /* ids of control/user tokens, longest first */
	int n_special;
	/* hash: token text -> id */
	int *vh;
	uint32_t vmask;
	/* hash: (left,right) -> merge rank and result */
	struct mrg { int a, b, rank, id; } *mh;
	uint32_t mmask;
	int bos, eos;
	int pre;                /* pre-split rules: 0 llama3/dbrx, 1 qwen35 */
} tokenizer;

int tok_load(tokenizer *t, const gguf *g, char *err, size_t errlen);
void tok_free(tokenizer *t);

/* encode UTF-8 text; special tokens are recognised if parse_special.
 * returns count (may exceed cap: call again with a larger buffer) */
int tok_encode(const tokenizer *t, const char *s, size_t len, int parse_special,
    int *out, int cap);

/* decode one token to bytes; returns length (control tokens: their text
 * if show_special, else 0) */
int tok_piece(const tokenizer *t, int id, int show_special, char *buf, int cap);

int tok_find(const tokenizer *t, const char *text);

#endif
