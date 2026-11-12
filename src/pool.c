#include "pool.h"

#include <pthread.h>
#include <stdlib.h>
#include <time.h>

/*
 * Workers spin for a short while before blocking on the condition
 * variable.  A token's forward pass issues a few hundred pool runs a
 * few microseconds apart; blocking between them lets cores fall into
 * deep C-states and clock down, which costs far more than the spin.
 * The spin only reads 'gen'; the job itself is always picked up under
 * the mutex, which orders it.
 */
#define SPIN_NS 2000000L        /* 2 ms */

static struct {
	int n;
	pthread_t *th;
	pthread_mutex_t mu;
	pthread_cond_t go, done;
	volatile unsigned gen;
	volatile int pending;
	volatile int quit;
	pool_fn fn;
	void *arg;
} P;

static long
nsnow(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1000000000L + ts.tv_nsec;
}

/* spin until *v != old or the budget is spent; returns 1 if it changed */
static int
spin_ne(volatile unsigned *v, unsigned old)
{
	long t0 = 0;
	unsigned k;
	for (k = 0;; k++) {
		if (*v != old || P.quit)
			return 1;
		if ((k & 1023) == 1023) {
			long t = nsnow();
			if (t0 == 0)
				t0 = t;
			else if (t - t0 > SPIN_NS)
				return 0;
		}
	}
}

static int
spin_zero(volatile int *v)
{
	long t0 = 0;
	unsigned k;
	for (k = 0;; k++) {
		if (*v == 0)
			return 1;
		if ((k & 1023) == 1023) {
			long t = nsnow();
			if (t0 == 0)
				t0 = t;
			else if (t - t0 > SPIN_NS)
				return 0;
		}
	}
}

static void *
worker(void *p)
{
	int ith = (int)(long)p;
	unsigned seen = 0;

	for (;;) {
		pool_fn fn;
		void *arg;

		spin_ne(&P.gen, seen);
		pthread_mutex_lock(&P.mu);
		while (P.gen == seen && !P.quit)
			pthread_cond_wait(&P.go, &P.mu);
		if (P.quit) {
			pthread_mutex_unlock(&P.mu);
			return NULL;
		}
		seen = P.gen;
		fn = P.fn;
		arg = P.arg;
		pthread_mutex_unlock(&P.mu);

		fn(arg, ith, P.n);

		pthread_mutex_lock(&P.mu);
		if (--P.pending == 0)
			pthread_cond_signal(&P.done);
		pthread_mutex_unlock(&P.mu);
	}
}

int
pool_init(int n)
{
	int i;
	if (n < 1)
		n = 1;
	P.n = n;
	pthread_mutex_init(&P.mu, NULL);
	pthread_cond_init(&P.go, NULL);
	pthread_cond_init(&P.done, NULL);
	P.th = calloc((size_t)n, sizeof(*P.th));
	if (!P.th)
		return -1;
	for (i = 1; i < n; i++)
		if (pthread_create(&P.th[i], NULL, worker, (void *)(long)i) != 0)
			return -1;
	return 0;
}

int
pool_size(void)
{
	return P.n;
}

void
pool_run(pool_fn fn, void *arg)
{
	if (P.n <= 1) {
		fn(arg, 0, 1);
		return;
	}
	pthread_mutex_lock(&P.mu);
	P.fn = fn;
	P.arg = arg;
	P.pending = P.n - 1;
	P.gen++;
	pthread_cond_broadcast(&P.go);
	pthread_mutex_unlock(&P.mu);

	fn(arg, 0, P.n);

	spin_zero(&P.pending);
	pthread_mutex_lock(&P.mu);
	while (P.pending)
		pthread_cond_wait(&P.done, &P.mu);
	pthread_mutex_unlock(&P.mu);
}

void
pool_free(void)
{
	int i;
	if (!P.th)
		return;
	pthread_mutex_lock(&P.mu);
	P.quit = 1;
	pthread_cond_broadcast(&P.go);
	pthread_mutex_unlock(&P.mu);
	for (i = 1; i < P.n; i++)
		pthread_join(P.th[i], NULL);
	free(P.th);
	P.th = NULL;
}
