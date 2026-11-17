/* HTTP server (llama.cpp server protocol) and the parts of main.c it
 * drives. */
#ifndef Q2K_SERVER_H
#define Q2K_SERVER_H

#include "model.h"
#include "tok.h"

/* main.c: the loaded model, its tokenizer and the one sequence state */
extern model *const q2k_m;
extern tokenizer *const q2k_t;
extern state *const q2k_s;

/* sampling for the next generate; temp <= 0 is greedy (and lets qwen35
 * speculate with its MTP block) */
void q2k_set_sampling(float temp, int top_k, float top_p, unsigned long seed);
void q2k_get_sampling(float *temp, int *top_k, float *top_p);
int *q2k_encode(const char *s, int special, int *n);
/* consume n tokens; logits of the last (NULL if they do not fit) */
const float *q2k_feed(const int *ids, int n);
int q2k_is_stop(int id);
/* control tokens that generation outputs instead of stopping on (n = 0
 * clears); q2k_kept tells whether id is one */
void q2k_keep_tokens(const int *ids, int n);
int q2k_kept(int id);

/* generate up to n tokens after logits lg.  cb gets every token (not
 * the stop token) and returns nonzero to stop; fed (if not NULL, room
 * for n + MTP_KMAX + 2) records each token that went into the state, in
 * order.  Returns the number of tokens passed to cb. */
typedef int (*q2k_emit)(int id, void *u);
int q2k_generate(const float *lg, int n, q2k_emit cb, void *u, int *fed, int *nfed,
    double *secs);

/* server.c */
typedef struct {
	const char *host, *alias, *api_key, *model_path;
	int port, n_predict, think, verbose;
} server_opts;
int server_run(const server_opts *o);

#endif
