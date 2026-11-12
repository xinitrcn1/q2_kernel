/* Build requirements, included first in every file (Makefile: -include):
 * LLVM/clang 23 or newer, and an x86-64 with AVX2, FMA and F16C. */
#ifndef Q2K_REQ_H
#define Q2K_REQ_H

#if !defined(__clang__) || __clang_major__ < 23
#error "q2k needs clang 23 or newer (CC=clang23)"
#endif

#if !defined(__AVX2__) || !defined(__FMA__) || !defined(__F16C__)
#error "q2k needs AVX2, FMA and F16C (Haswell / Excavator or newer): build with -march=native or -march=haswell"
#endif

#endif
