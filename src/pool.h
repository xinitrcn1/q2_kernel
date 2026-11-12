/* Fixed pthread pool: pool_run(fn, arg) calls fn(arg, i, n) on each of
 * the n threads (the caller is thread 0) and returns when all finish. */
#ifndef Q2K_POOL_H
#define Q2K_POOL_H

typedef void (*pool_fn)(void *arg, int ith, int nth);

int pool_init(int nthreads);

/* dynamic scheduling inside a pool_run: claim the next chunk of
 * [0, total) from *next; returns its start, or -1 when done.  Reset
 * *next to 0 before pool_run. */
long pool_take(long *next, long chunk, long total);
void pool_run(pool_fn fn, void *arg);
int pool_size(void);
void pool_free(void);

#endif
