/* i-quant codebooks (generated into iqtab.c by tools/gen_iqtab.py) */
#ifndef Q2K_IQTAB_H
#define Q2K_IQTAB_H

#include <stdint.h>

#define IQ1S_DELTA 0.125f

extern const uint8_t kmask_iq2xs[8];
extern const uint8_t ksigns_iq2xs[128];
extern const uint64_t iq2xxs_grid[256];
extern const uint64_t iq2xs_grid[512];
extern const uint64_t iq2s_grid[1024];
extern const uint32_t iq3xxs_grid[256];
extern const uint32_t iq3s_grid[512];
extern const int8_t kvalues_iq4nl[16];
extern const uint64_t iq1s_grid[2048];

#endif
