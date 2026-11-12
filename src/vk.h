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

int vk_init(size_t weight_budget, int device_index, int verbose);
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

#endif
