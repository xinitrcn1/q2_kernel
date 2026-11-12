/* Vulkan compute backend: one device-local weight arena (the "split",
 * 512 MiB by default) holding whole tensors, and a host-visible
 * activation arena that matvec shaders read x from and write y to. */
#ifndef Q2K_VK_H
#define Q2K_VK_H

#include <stddef.h>
#include <stdint.h>

#include "kq.h"
#include "model.h"

/* all of our device memory stays within this (1.25 GiB available) */
#define VK_VRAM_LIMIT ((size_t)1280 << 20)

/* state_bytes: device memory for recurrent state (fused Mamba2 path) */
int vk_init(size_t weight_budget, size_t state_bytes, int device_index, int verbose);
int vk_ready(void);
void vk_shutdown(void);
const char *vk_device_name(void);

/* copy whole experts of a tensor (dense: expert 0) into the arena;
 * done once at load, they stay resident.  0 on success, -1 when they
 * do not fit */
int vk_upload(wt *w, const int *experts, int n);
size_t vk_used(void);

/* place whole tensors and whole experts within the budget, upload */
void vk_plan_and_upload(model *m, int verbose);

/* batch recording; vk_submit starts it, vk_wait copies results to y */
void vk_begin(void);
/* x is given both as the f32 key (for dedupe) and its q8k blocks */
int vk_add(const wt *w, int e, const float *xkey, const q8k *xq, float *y);
void vk_submit(void);
void vk_wait(void);

/* split Mamba2 mixer: the GPU's heads (0 if none); vk_mamba_submit
 * starts the GPU half, vk_wait delivers its partial ssm_out and its
 * sum of squares */
int vk_mamba_ok(int il);
int vk_mamba_heads(int il);
void vk_mamba_submit(const model *m, int il, const float *x, float *out, float *ss);
void vk_prof_collect(void);
void vk_prof_report(void);
void vk_state_reset(void);
/* GPU rows [0, N) of the output projection, or NULL */
const wt *vk_lm(void);

/* MoE experts resident on the GPU (gate/up -> SwiGLU -> down each),
 * run asynchronously while the CPU does the rest; vk_wait collects. */
void vk_moe_begin(void);
int vk_moe_add(const wt *gate, const wt *up, const wt *down, int e, float *out);
void vk_moe_submit(const q8k *xq, int K);

#endif
