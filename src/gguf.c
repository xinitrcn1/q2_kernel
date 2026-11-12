#include "gguf.h"

#include <sys/mman.h>
#include <sys/stat.h>

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {
	const uint8_t *p, *end;
	int bad;
} rd;

static int
need(rd *r, uint64_t n)
{
	if (r->bad || (uint64_t)(r->end - r->p) < n) {
		r->bad = 1;
		return 0;
	}
	return 1;
}

static uint32_t
rd_u32(rd *r)
{
	uint32_t v = 0;
	if (need(r, 4)) {
		memcpy(&v, r->p, 4);
		r->p += 4;
	}
	return v;
}

static uint64_t
rd_u64(rd *r)
{
	uint64_t v = 0;
	if (need(r, 8)) {
		memcpy(&v, r->p, 8);
		r->p += 8;
	}
	return v;
}

static gstr
rd_str(rd *r)
{
	gstr s = { 0, "" };
	s.len = rd_u64(r);
	if (need(r, s.len)) {
		s.p = (const char *)r->p;
		r->p += s.len;
	} else
		s.len = 0;
	return s;
}

static size_t
vsize(uint32_t t)
{
	switch (t) {
	case GV_U8: case GV_I8: case GV_BOOL: return 1;
	case GV_U16: case GV_I16: return 2;
	case GV_U32: case GV_I32: case GV_F32: return 4;
	case GV_U64: case GV_I64: case GV_F64: return 8;
	default: return 0;
	}
}

/* skip one value of type t */
static void
skip_val(rd *r, uint32_t t)
{
	if (t == GV_STR) {
		(void)rd_str(r);
		return;
	}
	if (t == GV_ARR) {
		uint32_t at = rd_u32(r);
		uint64_t n = rd_u64(r), i;
		if (at == GV_STR) {
			for (i = 0; i < n && !r->bad; i++)
				(void)rd_str(r);
		} else if (vsize(at) && n <= (uint64_t)(r->end - r->p) / vsize(at)) {
			r->p += n * vsize(at);
		} else
			r->bad = 1;
		return;
	}
	if (vsize(t) && need(r, vsize(t)))
		r->p += vsize(t);
	else
		r->bad = 1;
}

int
gstr_eq(gstr s, const char *c)
{
	size_t n = strlen(c);
	return s.len == n && memcmp(s.p, c, n) == 0;
}

size_t
ggml_type_block(uint32_t t)
{
	switch (t) {
	case GT_F32: case GT_F16: return 1;
	case GT_Q8_0: return 32;
	case GT_Q2_K: case GT_Q3_K: case GT_Q4_K: case GT_Q5_K: case GT_Q6_K:
	case GT_IQ2_XXS: case GT_IQ2_XS: case GT_IQ3_XXS: case GT_IQ1_S:
	case GT_IQ3_S: case GT_IQ2_S: case GT_IQ4_XS:
		return 256;
	default: return 0;
	}
}

size_t
ggml_type_bytes(uint32_t t)
{
	switch (t) {
	case GT_F32: return 4;
	case GT_F16: return 2;
	case GT_Q8_0: return 34;
	case GT_Q2_K: return 84;
	case GT_Q3_K: return 110;
	case GT_Q4_K: return 144;
	case GT_Q5_K: return 176;
	case GT_Q6_K: return 210;
	case GT_IQ2_XXS: return 66;
	case GT_IQ2_XS: return 74;
	case GT_IQ3_XXS: return 98;
	case GT_IQ1_S: return 50;
	case GT_IQ3_S: return 110;
	case GT_IQ2_S: return 82;
	case GT_IQ4_XS: return 136;
	default: return 0;
	}
}

const char *
ggml_type_name(uint32_t t)
{
	switch (t) {
	case GT_F32: return "F32";
	case GT_F16: return "F16";
	case GT_Q8_0: return "Q8_0";
	case GT_Q2_K: return "Q2_K";
	case GT_Q3_K: return "Q3_K";
	case GT_Q4_K: return "Q4_K";
	case GT_Q5_K: return "Q5_K";
	case GT_Q6_K: return "Q6_K";
	case GT_IQ2_XXS: return "IQ2_XXS";
	case GT_IQ2_XS: return "IQ2_XS";
	case GT_IQ3_XXS: return "IQ3_XXS";
	case GT_IQ1_S: return "IQ1_S";
	case GT_IQ3_S: return "IQ3_S";
	case GT_IQ2_S: return "IQ2_S";
	case GT_IQ4_XS: return "IQ4_XS";
	default: return "?";
	}
}

size_t
ggml_row_bytes(uint32_t t, uint64_t ne0)
{
	size_t b = ggml_type_block(t);
	if (b == 0 || ne0 % b)
		return 0;
	return (size_t)(ne0 / b) * ggml_type_bytes(t);
}

int
gguf_open(gguf *g, const char *path, char *err, size_t errlen)
{
	struct stat st;
	rd r;
	uint64_t i;
	const gkv *al;
	size_t off;

	memset(g, 0, sizeof(*g));
	g->fd = open(path, O_RDONLY);
	if (g->fd < 0) {
		snprintf(err, errlen, "%s: cannot open", path);
		return -1;
	}
	if (fstat(g->fd, &st) != 0 || st.st_size < 24) {
		snprintf(err, errlen, "%s: bad file", path);
		goto fail;
	}
	g->size = (size_t)st.st_size;
#ifdef MAP_PREFAULT_READ
	/* map the pages already in memory now, not on first touch (the
	 * first token would otherwise take ~650k soft faults) */
	g->map = mmap(NULL, g->size, PROT_READ, MAP_SHARED | MAP_PREFAULT_READ, g->fd, 0);
#else
	g->map = mmap(NULL, g->size, PROT_READ, MAP_SHARED, g->fd, 0);
#endif
	if (g->map == MAP_FAILED) {
		g->map = NULL;
		snprintf(err, errlen, "%s: mmap failed", path);
		goto fail;
	}
	if (memcmp(g->map, "GGUF", 4) != 0) {
		snprintf(err, errlen, "%s: not GGUF", path);
		goto fail;
	}
	r.p = g->map + 4;
	r.end = g->map + g->size;
	r.bad = 0;
	g->version = rd_u32(&r);
	if (g->version < 2 || g->version > 3) {
		snprintf(err, errlen, "unsupported GGUF version %u", g->version);
		goto fail;
	}
	g->n_tensors = rd_u64(&r);
	g->n_kv = rd_u64(&r);
	if (g->n_kv > 1u << 20 || g->n_tensors > 1u << 20) {
		snprintf(err, errlen, "implausible counts");
		goto fail;
	}
	g->kv = calloc(g->n_kv ? g->n_kv : 1, sizeof(*g->kv));
	g->t = calloc(g->n_tensors ? g->n_tensors : 1, sizeof(*g->t));
	if (!g->kv || !g->t) {
		snprintf(err, errlen, "out of memory");
		goto fail;
	}
	for (i = 0; i < g->n_kv && !r.bad; i++) {
		gkv *kv = &g->kv[i];
		kv->key = rd_str(&r);
		kv->type = rd_u32(&r);
		kv->n = 1;
		if (kv->type == GV_ARR) {
			const uint8_t *save;
			kv->atype = rd_u32(&r);
			kv->n = rd_u64(&r);
			kv->val = r.p;
			/* rewind over the header to reuse skip_val */
			save = r.p;
			r.p = save - 12;
			skip_val(&r, GV_ARR);
		} else {
			kv->val = r.p;
			skip_val(&r, kv->type);
		}
	}
	g->alignment = 32;
	for (i = 0; i < g->n_tensors && !r.bad; i++) {
		gtensor *t = &g->t[i];
		uint32_t d;
		t->name = rd_str(&r);
		t->ndim = rd_u32(&r);
		if (t->ndim > 4) {
			r.bad = 1;
			break;
		}
		for (d = 0; d < 4; d++)
			t->ne[d] = 1;
		for (d = 0; d < t->ndim; d++)
			t->ne[d] = rd_u64(&r);
		t->type = rd_u32(&r);
		t->offset = rd_u64(&r);
	}
	if (r.bad) {
		snprintf(err, errlen, "truncated or malformed header");
		goto fail;
	}
	al = gguf_find(g, "general.alignment");
	if (al && al->type == GV_U32) {
		uint32_t a;
		memcpy(&a, al->val, 4);
		if (a == 0 || (a & (a - 1))) {
			snprintf(err, errlen, "bad alignment");
			goto fail;
		}
		g->alignment = a;
	}
	off = (size_t)(r.p - g->map);
	off = (off + g->alignment - 1) / g->alignment * g->alignment;
	g->data = g->map + off;
	for (i = 0; i < g->n_tensors; i++) {
		gtensor *t = &g->t[i];
		size_t rb = ggml_row_bytes(t->type, t->ne[0]);
		if (rb == 0) {
			snprintf(err, errlen, "tensor %.*s: unsupported type %u",
			    (int)t->name.len, t->name.p, t->type);
			goto fail;
		}
		t->nbytes = rb * t->ne[1] * t->ne[2] * t->ne[3];
		if (t->offset > g->size - off || t->nbytes > g->size - off - t->offset) {
			snprintf(err, errlen, "tensor %.*s out of bounds",
			    (int)t->name.len, t->name.p);
			goto fail;
		}
		t->data = g->data + t->offset;
	}
	return 0;
fail:
	gguf_close(g);
	return -1;
}

void
gguf_close(gguf *g)
{
	if (g->map)
		munmap((void *)g->map, g->size);
	if (g->fd >= 0)
		close(g->fd);
	free(g->kv);
	free(g->t);
	memset(g, 0, sizeof(*g));
	g->fd = -1;
}

const gkv *
gguf_find(const gguf *g, const char *key)
{
	uint64_t i;
	for (i = 0; i < g->n_kv; i++)
		if (gstr_eq(g->kv[i].key, key))
			return &g->kv[i];
	return NULL;
}

const gtensor *
gguf_tensor(const gguf *g, const char *name)
{
	uint64_t i;
	for (i = 0; i < g->n_tensors; i++)
		if (gstr_eq(g->t[i].name, name))
			return &g->t[i];
	return NULL;
}

int
gguf_get_u32(const gguf *g, const char *key, uint32_t *out)
{
	const gkv *kv = gguf_find(g, key);
	if (!kv)
		return -1;
	switch (kv->type) {
	case GV_U32: case GV_I32: memcpy(out, kv->val, 4); return 0;
	case GV_U64: case GV_I64: {
		uint64_t v;
		memcpy(&v, kv->val, 8);
		*out = (uint32_t)v;
		return 0;
	}
	case GV_BOOL: case GV_U8: *out = kv->val[0]; return 0;
	default: return -1;
	}
}

int
gguf_get_f32(const gguf *g, const char *key, float *out)
{
	const gkv *kv = gguf_find(g, key);
	if (!kv)
		return -1;
	if (kv->type == GV_F32) {
		memcpy(out, kv->val, 4);
		return 0;
	}
	if (kv->type == GV_F64) {
		double d;
		memcpy(&d, kv->val, 8);
		*out = (float)d;
		return 0;
	}
	return -1;
}

int
gguf_get_str(const gguf *g, const char *key, gstr *out)
{
	const gkv *kv = gguf_find(g, key);
	const uint8_t *p;
	if (!kv || kv->type != GV_STR)
		return -1;
	p = kv->val;
	*out = gguf_next_str(&p);
	return 0;
}

int
gguf_get_arr_u32(const gguf *g, const char *key, uint64_t i, uint32_t *out)
{
	const gkv *kv = gguf_find(g, key);
	if (!kv)
		return -1;
	if (kv->type != GV_ARR)
		return gguf_get_u32(g, key, out);
	if (i >= kv->n || (kv->atype != GV_U32 && kv->atype != GV_I32))
		return -1;
	memcpy(out, kv->val + 4 * i, 4);
	return 0;
}

gstr
gguf_next_str(const uint8_t **cur)
{
	gstr s;
	memcpy(&s.len, *cur, 8);
	s.p = (const char *)*cur + 8;
	*cur += 8 + s.len;
	return s;
}
