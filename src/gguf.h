/* GGUF v2/v3 reader over a read-only mmap.  Nothing is copied: keys,
 * strings and tensor data all point into the mapping. */
#ifndef Q2K_GGUF_H
#define Q2K_GGUF_H

#include <stddef.h>
#include <stdint.h>

enum gguf_vtype {
	GV_U8 = 0, GV_I8, GV_U16, GV_I16, GV_U32, GV_I32, GV_F32, GV_BOOL,
	GV_STR, GV_ARR, GV_U64, GV_I64, GV_F64
};

/* ggml tensor types used by the kernel */
enum ggml_type {
	GT_F32 = 0, GT_F16 = 1, GT_Q8_0 = 8,
	GT_Q2_K = 10, GT_Q3_K = 11, GT_Q4_K = 12, GT_Q5_K = 13, GT_Q6_K = 14,
	GT_IQ2_XXS = 16, GT_IQ2_XS = 17, GT_IQ3_XXS = 18, GT_IQ1_S = 19,
	GT_IQ3_S = 21, GT_IQ2_S = 22, GT_IQ4_XS = 23
};

typedef struct {
	uint64_t len;
	const char *p; /* not NUL terminated */
} gstr;

typedef struct {
	gstr key;
	uint32_t type;      /* gguf_vtype */
	uint32_t atype;     /* element type when type == GV_ARR */
	uint64_t n;         /* array length (1 for scalars) */
	const uint8_t *val; /* raw value / first array element */
} gkv;

typedef struct {
	gstr name;
	uint32_t ndim;
	uint64_t ne[4];     /* ne[0] is the contiguous dimension */
	uint32_t type;
	uint64_t offset;    /* relative to data section */
	const void *data;
	size_t nbytes;
} gtensor;

typedef struct {
	int fd;
	const uint8_t *map;
	size_t size;
	uint32_t version;
	uint64_t n_kv, n_tensors;
	gkv *kv;
	gtensor *t;
	size_t alignment;
	const uint8_t *data;
} gguf;

int gguf_open(gguf *g, const char *path, char *err, size_t errlen);
void gguf_close(gguf *g);

const gkv *gguf_find(const gguf *g, const char *key);
const gtensor *gguf_tensor(const gguf *g, const char *name);

/* scalar accessors; return 0 on success, -1 if missing or mistyped */
int gguf_get_u32(const gguf *g, const char *key, uint32_t *out);
int gguf_get_f32(const gguf *g, const char *key, float *out);
int gguf_get_str(const gguf *g, const char *key, gstr *out);
/* element i of a u32/i32 array (or the scalar itself, broadcast) */
int gguf_get_arr_u32(const gguf *g, const char *key, uint64_t i, uint32_t *out);

/* string array iteration: *cur starts at kv->val, advances per call */
gstr gguf_next_str(const uint8_t **cur);

size_t ggml_type_block(uint32_t type);  /* elements per block */
size_t ggml_type_bytes(uint32_t type);  /* bytes per block */
const char *ggml_type_name(uint32_t type);
size_t ggml_row_bytes(uint32_t type, uint64_t ne0);

int gstr_eq(gstr s, const char *c);

#endif
