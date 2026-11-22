#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0 — (c) 2026 Maher El Hmissi ; the embedded SFPU model is a port of tt-isa-documentation (c) Tenstorrent AI ULC, Apache-2.0
"""
erf_tt.py — Test décisif « prime Tenstorrent » : erf float32 sur le SFPU (Blackhole / Wormhole).

Ce que fait le script :
  1. Compile un modèle BIT-EXACT de l'arithmétique SFPU (SFPMAD + SFPARECIP, d'après tt-isa-documentation),
     sur GPU (nvcc) ou à défaut sur CPU (gcc + OpenMP).
  2. Mesure l'erf actuelle de tt-metal (rationnelle n16/d16 sur [-10, 10]) : contrôle positif.
  3. Cherche de nouvelles formules (rationnelles plus petites sur [0, 3.92] + étape de correction),
     ajuste puis POLIT les coefficients bit à bit avec le modèle exact, en recherchant les pires entrées.
  4. Vérifie de façon EXHAUSTIVE toutes les entrées float32 positives (2,1 milliards) — erf est impaire.
  5. Compare au critère fixé d'avance et écrit les coefficients prêts pour le noyau C++.

Usage : python erf_tt.py [--minutes 40] [--cpu] [--rapide]
"""
import os, sys, time, json, math, ctypes, shutil, subprocess, argparse, tempfile
import numpy as np

# ----------------------------------------------------------------------------------------------
# CRITÈRE FIXÉ D'AVANCE (ne pas modifier après avoir vu les résultats)
#   GROS LOT  : erreur max <= 1.0 ULP (exhaustif, Blackhole bit-exact) ET coût <= coût actuel
#   PETIT LOT : (max <= 2.0 ULP ET coût <= coût actuel) OU (max <= 1.0 ULP ET coût <= 1.2 x coût actuel)
#   Contrôle positif obligatoire : l'erf actuelle doit mesurer >= 3 ULP (signalement #58282 : 6,3 sur ttsim)
# ----------------------------------------------------------------------------------------------
SEUIL_GROS, SEUIL_PETIT, MARGE_COUT = 1.0, 2.0, 1.2

SRC_H = r'''// Modèle bit-exact de l'arithmétique SFPU Tenstorrent (portage de tt-isa-documentation/Miscellaneous/FMA/fma.c
// et de SFPARECIP.md), utilisable en C (gcc) et en CUDA (nvcc).
#pragma once
#include <stdint.h>
#include <math.h>

#ifdef __CUDACC__
#define HD __host__ __device__ __forceinline__
#else
#define HD static inline
#endif

#ifdef __CUDA_ARCH__
#define CLZ32(v) __clz((int)(v))
#else
#define CLZ32(v) __builtin_clz(v)
#endif

HD float as_f(uint32_t u) { union { uint32_t u; float f; } c; c.u = u; return c.f; }
HD uint32_t as_u(float f) { union { uint32_t u; float f; } c; c.f = f; return c.u; }

#define SEMI_STICKY_SHIFT(var, amount) \
  do { int32_t s_ = (amount); \
       if (s_ >= 64) { var = 0; } \
       else { uint64_t orig_ = var; var >>= s_; if (var) var |= ((var << s_) != orig_); } } while (0)

// ---- Blackhole SFPMAD / fma.s : x*y+z ----
HD uint32_t fma_bh(uint32_t x, uint32_t y, uint32_t z) {
  int32_t x_e = (x >> 23) & 255; uint32_t x_m = (x & 0x7fffff) ^ 0x800000; if (x_e == 0) x_m = 0;
  int32_t y_e = (y >> 23) & 255; uint32_t y_m = (y & 0x7fffff) ^ 0x800000; if (y_e == 0) y_m = 0;
  int32_t z_e = (z >> 23) & 255; uint32_t z_m = (z & 0x7fffff) ^ 0x800000; if (z_e == 0) z_m = 0;
  uint32_t z_sign = z & 0x80000000u;
  uint32_t p_sign = (x ^ y) & 0x80000000u;
  uint64_t p_m = (uint64_t)x_m * (uint64_t)y_m;
  int32_t p_e = x_e + y_e - 23 - 127;
  p_m <<= 3; z_m <<= 3;
  p_m = (p_m >> 23) | ((p_m & 0x7fffff) != 0);
  p_e += 23;
  if (x_e == 255 || y_e == 255 || p_e >= 255 || z_e == 255) {
    if ((x_e == 255 && (x_m != 0x800000 || y_m == 0)) || (y_e == 255 && (y_m != 0x800000 || x_m == 0)) ||
        (z_e == 255 && z_m != 0x4000000) || (z_e == 255 && (x_e == 255 || y_e == 255) && (z_sign != p_sign)))
      return 0x7fc00000u;
    else if (z_e == 255) return z;
    else return p_sign | 0x7f800000u;
  }
  if (p_m == 0 || p_e < 0) return z_m ? z : (z_sign & p_sign);
  uint64_t zz = z_m;
  int32_t r_e = p_e > z_e ? p_e : z_e;
  if (p_e < r_e) SEMI_STICKY_SHIFT(p_m, r_e - p_e);
  if (z_e < r_e) SEMI_STICKY_SHIFT(zz, r_e - z_e);
  uint32_t pm = (uint32_t)p_m, zm = (uint32_t)zz;
  uint32_t r_sign = pm >= zm ? p_sign : z_sign;
  if (z_sign != r_sign) zm = ~zm;
  if (p_sign != r_sign) pm = ~pm;
  uint32_t r_m = zm + pm + (p_sign != z_sign);
  if (r_m == 0) return z_sign & p_sign;
  int32_t n = 5 - CLZ32(r_m);
  r_e += n;
  if (r_e >= 255) return r_sign | 0x7f800000u;
  if (r_e <= 0) { n += 1 - r_e; r_e = 0; }
  if (n <= 0) r_m <<= -n; else r_m = (r_m >> n) | ((r_m & (n | 1)) != 0);
  uint32_t r = ((uint32_t)r_e << 23) + ((r_m >> 3) & 0x7fffff);
  r += (((r_m & 7) + (r & 1)) > 4);
  if (!(r >> 23)) r = 0;
  return r_sign | r;
}

// ---- Wormhole SFPMAD : x*y+z ----
HD uint32_t fma_wh(uint32_t x, uint32_t y, uint32_t z) {
  int32_t x_e = (x >> 23) & 255; uint32_t x_m = (x & 0x7fffff) ^ 0x800000; if (x_e == 0) x_m = 0;
  int32_t y_e = (y >> 23) & 255; uint32_t y_m = (y & 0x7fffff) ^ 0x800000; if (y_e == 0) y_m = 0;
  int32_t z_e = (z >> 23) & 255; uint32_t z_m = (z & 0x7fffff) ^ 0x800000; if (z_e == 0) z_m = 0;
  uint32_t z_sign = z & 0x80000000u;
  uint32_t p_sign = (x ^ y) & 0x80000000u;
  uint64_t p_m = (uint64_t)x_m * (uint64_t)y_m;
  int32_t p_e = x_e + y_e - 23 - 127;
  p_m <<= 3; z_m <<= 3;
  p_m = (p_m >> 23) | ((p_m & 0x7fffff) != 0);
  p_e += 23;
  uint32_t nan_result = 0;
  if (x_e == 255 || y_e == 255 || p_e >= 255 || z_e == 255) {
    if ((x_e == 255 && (x_m != 0x800000 || y_m == 0)) || (y_e == 255 && (y_m != 0x800000 || x_m == 0)) ||
        (z_e == 255 && z_m == 0x4000000 && (x_e == 255 || y_e == 255 || p_e >= 255) && (z_sign != p_sign)))
      nan_result = p_sign | 0x7f800001u;
    else if (z_e == 255 && z_m != 0x4000000) nan_result = z_sign | 0x7f800001u;
    else if (z_e == 255) return z;
    else return p_sign | 0x7f800000u;
    if (p_e > 255) p_e = 255;
  }
  if (p_m == 0 || p_e < 0) {
    if (nan_result) { p_m = 0; p_e = 0; }
    else return z_m ? z : 0;
  }
  uint64_t zz = z_m;
  int32_t r_e = p_e > z_e ? p_e : z_e;
  if (p_e < r_e) SEMI_STICKY_SHIFT(p_m, r_e - p_e);
  if (z_e < r_e) SEMI_STICKY_SHIFT(zz, r_e - z_e);
  uint32_t pm = (uint32_t)p_m, zm = (uint32_t)zz;
  uint32_t r_sign = pm >= zm ? p_sign : z_sign;
  if (z_sign != r_sign) zm = ~zm;
  if (p_sign != r_sign) pm = ~pm;
  uint32_t r_m = zm + pm + (p_sign != z_sign);
  if (r_m == 0) return nan_result;
  int32_t n = 5 - CLZ32(r_m);
  r_e += n;
  if (r_e >= 255) return nan_result ? nan_result : (r_sign | 0x7f800000u);
  if (r_e < 0) return nan_result;
  if (n <= 0) r_m <<= -n; else r_m = (r_m >> n) | (r_m & 1);
  uint32_t r = ((uint32_t)r_e << 23) + ((r_m >> 3) & 0x7fffff);
  r += (((r_m & 7) + (r & 1)) > 4);
  if (!(r >> 23)) return nan_result;
  return (nan_result ? nan_result : r_sign) | r;
}

HD uint32_t fma_m(int model, uint32_t a, uint32_t b, uint32_t c) {
  return model ? fma_wh(a, b, c) : fma_bh(a, b, c);
}

// ---- SFPARECIP (mode RECIP), Blackhole ----
#define RECIP_LUT_VALUES { \
  127, 125, 123, 121, 119, 117, 116, 114, 112, 110, 109, 107, 105, 104, 102, 100, 99, \
  97, 96, 94, 93, 91, 90, 88, 87, 85, 84, 83, 81, 80, 79, 77, 76, 75, 74, 72, 71, 70, \
  69, 68, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, \
  47, 46, 45, 44, 43, 42, 41, 40, 40, 39, 38, 37, 36, 35, 35, 34, 33, 32, 31, 31, 30, \
  29, 28, 28, 27, 26, 25, 25, 24, 23, 23, 22, 21, 21, 20, 19, 19, 18, 17, 17, 16, 15, \
  15, 14, 14, 13, 12, 12, 11, 11, 10, 9, 9, 8, 8, 7, 7, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, \
  1, 0}
static const uint8_t RECIP_LUT_H[128] = RECIP_LUT_VALUES;
#ifdef __CUDACC__
__constant__ uint8_t RECIP_LUT_D[128] = RECIP_LUT_VALUES;
#endif
#ifdef __CUDA_ARCH__
#define RECIP_LUT RECIP_LUT_D
#else
#define RECIP_LUT RECIP_LUT_H
#endif

HD uint32_t approx_recip(uint32_t x) {
  uint32_t sign = x & 0x80000000u; uint32_t a = x - sign, r;
  if (a < 0x00800000u) r = 0x7f800000u;
  else if (a < 0x7e800000u) r = ((253u - (a >> 23)) << 23) | ((uint32_t)RECIP_LUT[(a >> 16) & 0x7f] << 16);
  else r = 0;
  return sign + r;
}

// sfpu_reciprocal_iter<2> (Blackhole), tel que dans ckernel_sfpu_recip.h
#define F_M2   0xC0000000u   /* -2.0 */
#define F_NZ   0x80000000u   /* -0.0 */
#define F_PZ   0x00000000u   /* +0.0 */
#define F_ONE  0x3f800000u
HD uint32_t recip_seq(int model, uint32_t x) {
  uint32_t y = approx_recip(x);
  uint32_t t = fma_m(model, x, y, F_M2);                // t = x*y - 2
  uint32_t y1 = fma_m(model, y, t ^ 0x80000000u, F_NZ);  // y1 = y * -t - 0
  if (t & 0x80000000u) {                                 // v_if (t < 0)
    t = fma_m(model, x, y1, F_M2);
    y = fma_m(model, y1, t ^ 0x80000000u, F_NZ);
  }
  return y;
}

#define MAXC 18
typedef struct {
  int n, d;            // degrés de N(u) et D(u) en u = x^2 (D(0)=b[0])
  float a[MAXC];       // numérateur : y = x * sum a_k u^k
  float b[MAXC];       // dénominateur : sum b_k u^k
  float xclamp;        // |x| est borné à xclamp (clamp symétrique, exact)
  int corr;            // 1 = une étape de correction résiduelle après la division
  int clampout;        // 1 = sortie bornée à [-1, 1]
  int model;           // 0 = Blackhole, 1 = Wormhole (SFPMAD), réciproque = séquence BH
  int kind;            // 0 = rationnelle x*N(x²)/D(x²) ; 1 = deux morceaux (x*P(x²) si |x|<split, sinon Q(|x|-center))
  float split, center;
} Design;

HD uint32_t eval_erf2(const Design* D, uint32_t xb) {
  int m = D->model;
  uint32_t ax = xb & 0x7fffffffu, s = xb & 0x80000000u, cl = as_u(D->xclamp);
  if (ax > 0x7f800000u) return 0x7fc00000u;
  if (ax > cl) ax = cl;                                        // min(|x|, X) : exact
  uint32_t u = fma_m(m, ax, ax, F_PZ);                         // u = x²
  uint32_t P = as_u(D->a[D->n]);
  for (int k = D->n - 1; k >= 0; k--) P = fma_m(m, P, u, as_u(D->a[k]));
  // x + x*P(x²) : une seule erreur d'arrondi. Mise à l'échelle exacte par 2^24 (SFPDIVP2) pour que le produit
  // x*P ne soit jamais « sous-normal » (le SFPU le remplacerait par zéro pour les x minuscules).
  uint32_t xs = fma_m(m, ax, 0x4b800000u, F_PZ);
  uint32_t ys = fma_m(m, xs, P, xs);
  uint32_t yA = fma_m(m, ys, 0x33800000u, F_PZ);
  uint32_t t = fma_m(m, ax, F_ONE, as_u(D->center) ^ 0x80000000u);  // t = |x| - center
  uint32_t yB = as_u(D->b[D->d]);
  for (int k = D->d - 1; k >= 0; k--) yB = fma_m(m, yB, t, as_u(D->b[k]));
  uint32_t y = (as_f(ax) < D->split) ? yA : yB;                // v_if (|x| < split)
  if (D->clampout && (y & 0x7fffffffu) > F_ONE && (y & 0x7fffffffu) <= 0x7f800000u) y = F_ONE;
  return (y & 0x7fffffffu) | s;                                // setsgn
}

HD uint32_t eval_erf(const Design* D, uint32_t xb) {
  if (D->kind == 1) return eval_erf2(D, xb);
  int m = D->model;
  // symmetric_clamp(x, xclamp) : exact
  uint32_t ax = xb & 0x7fffffffu, s = xb & 0x80000000u, cl = as_u(D->xclamp);
  if (ax > 0x7f800000u) return 0x7fc00000u;   // NaN : hors sujet ici
  if (ax > cl) xb = s | cl;
  uint32_t u = fma_m(m, xb, xb, F_PZ);                       // u = x*x
  uint32_t N = as_u(D->a[D->n]);
  for (int k = D->n - 1; k >= 0; k--) N = fma_m(m, N, u, as_u(D->a[k]));
  uint32_t Q = as_u(D->b[D->d]);
  for (int k = D->d - 1; k >= 0; k--) Q = fma_m(m, Q, u, as_u(D->b[k]));
  uint32_t xN = fma_m(m, N, xb, F_PZ);                        // numer * x
  uint32_t r = recip_seq(m, Q);
  uint32_t y = fma_m(m, xN, r, F_PZ);                         // numer * recip(denom)
  if (D->corr) {
    uint32_t e = fma_m(m, y ^ 0x80000000u, Q, xN);            // e = xN - y*Q
    y = fma_m(m, r, e, y);                                    // y = y + r*e
  }
  if (D->clampout) {
    uint32_t ay = y & 0x7fffffffu;
    if (ay > F_ONE && ay <= 0x7f800000u) y = (y & 0x80000000u) | F_ONE;
  }
  return y;
}

// Erreur en ULP float32 (réelle) par rapport à erf exacte (double) de l'entrée float32 exacte.
HD double ulp_err(uint32_t xb, uint32_t yb) {
  double x = (double)as_f(xb);
  if (((xb >> 23) & 255) == 0) x = 0.0 * x;                   // entrée dénormale : traitée comme 0 par le SFPU
  double ref = erf(x);
  double y = (double)as_f(yb);
  if (y != y) return 1e30;
  double aref = fabs(ref);
  if (aref < 1.1754943508222875e-38) return (y == 0.0 || fabs(y - ref) < 1.1754943508222875e-38) ? 0.0 : 1e30;
  int e; frexp(aref, &e);
  double ulp = ldexp(1.0, e - 24);
  return fabs(y - ref) / ulp;
}
'''
SRC_CU = r'''// Bibliothèque d'évaluation exhaustive : compilée par nvcc (GPU) ou gcc -x c (CPU, OpenMP).
#include "sfpu_model.h"
#include <stdlib.h>
#include <string.h>

typedef struct {
  double maxerr; uint32_t argmax; uint32_t pad;
  unsigned long long n_gt1, n_gt05, count, n_absgt1;
  double sumerr;
} Stats;

#ifdef __CUDACC__
#define EXPORT extern "C"
#else
#define EXPORT
#endif

#ifdef __CUDACC__
#include <cuda_runtime.h>
static float __uint_as_float_host(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

__device__ __forceinline__ void acc_point(const Design* D, uint32_t x, double* lmax, uint32_t* larg,
                                          unsigned long long* g1, unsigned long long* g05,
                                          unsigned long long* ag1, double* lsum) {
  uint32_t y = eval_erf(D, x);
  double e = ulp_err(x, y);
  if (e > *lmax || (e == *lmax && x < *larg)) { *lmax = e; *larg = x; }
  *g1 += (e > 1.0); *g05 += (e > 0.5); *lsum += e;
  *ag1 += ((y & 0x7fffffffu) > F_ONE && (y & 0x7fffffffu) <= 0x7f800000u);
}

__device__ void flush(double lmax, uint32_t larg, unsigned long long g1, unsigned long long g05,
                      unsigned long long ag1, double lsum, unsigned long long cnt,
                      unsigned long long* key, unsigned long long* cg1, unsigned long long* cg05,
                      unsigned long long* cag1, double* csum, unsigned long long* ccnt) {
  float ef = (float)lmax; if ((double)ef < lmax) ef = nextafterf(ef, 3.0e38f);
  unsigned long long k = ((unsigned long long)__float_as_uint(ef) << 32) | (unsigned long long)(0xffffffffu - larg);
  atomicMax(key, k);
  atomicAdd(cg1, g1); atomicAdd(cg05, g05); atomicAdd(cag1, ag1); atomicAdd(csum, lsum); atomicAdd(ccnt, cnt);
}

__global__ void k_sweep(Design D, uint32_t lo, unsigned long long cnt, uint32_t stride, unsigned long long* key,
                        unsigned long long* cg1, unsigned long long* cg05, unsigned long long* cag1, double* csum,
                        unsigned long long* ccnt) {
  double lmax = -1; uint32_t larg = 0; unsigned long long g1 = 0, g05 = 0, ag1 = 0, c = 0; double lsum = 0;
  for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < cnt;
       i += (unsigned long long)gridDim.x * blockDim.x) {
    uint32_t x = (uint32_t)(lo + i * (unsigned long long)stride);
    acc_point(&D, x, &lmax, &larg, &g1, &g05, &ag1, &lsum); c++;
  }
  if (c) flush(lmax, larg, g1, g05, ag1, lsum, c, key, cg1, cg05, cag1, csum, ccnt);
}

__global__ void k_list(Design D, const uint32_t* xs, int m, double* errs, unsigned long long* key,
                       unsigned long long* cg1, unsigned long long* cg05, unsigned long long* cag1, double* csum,
                       unsigned long long* ccnt) {
  double lmax = -1; uint32_t larg = 0; unsigned long long g1 = 0, g05 = 0, ag1 = 0, c = 0; double lsum = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < m; i += gridDim.x * blockDim.x) {
    uint32_t x = xs[i];
    uint32_t y = eval_erf(&D, x);
    double e = ulp_err(x, y);
    if (errs) errs[i] = e;
    if (e > lmax || (e == lmax && x < larg)) { lmax = e; larg = x; }
    g1 += (e > 1.0); g05 += (e > 0.5); lsum += e; c++;
    ag1 += ((y & 0x7fffffffu) > F_ONE && (y & 0x7fffffffu) <= 0x7f800000u);
  }
  if (c) flush(lmax, larg, g1, g05, ag1, lsum, c, key, cg1, cg05, cag1, csum, ccnt);
}

static unsigned long long* d_acc = 0; static double* d_sum = 0;
static uint32_t* d_xs = 0; static double* d_errs = 0; static int d_cap = 0;

static int prep() {
  if (!d_acc) { if (cudaMalloc(&d_acc, 5 * sizeof(unsigned long long)) != cudaSuccess) return -1;
                if (cudaMalloc(&d_sum, sizeof(double)) != cudaSuccess) return -1; }
  cudaMemset(d_acc, 0, 5 * sizeof(unsigned long long)); cudaMemset(d_sum, 0, sizeof(double));
  return 0;
}
static int finish(Stats* out) {
  if (cudaDeviceSynchronize() != cudaSuccess) return -2;
  unsigned long long h[5]; double s;
  cudaMemcpy(h, d_acc, sizeof h, cudaMemcpyDeviceToHost); cudaMemcpy(&s, d_sum, sizeof s, cudaMemcpyDeviceToHost);
  uint32_t ef = (uint32_t)(h[0] >> 32);
  out->maxerr = (double)__uint_as_float_host(ef);
  out->argmax = 0xffffffffu - (uint32_t)(h[0] & 0xffffffffu);
  out->n_gt1 = h[1]; out->n_gt05 = h[2]; out->n_absgt1 = h[3]; out->count = h[4]; out->sumerr = s;
  return 0;
}
#endif


EXPORT int has_gpu() {
#ifdef __CUDACC__
  int n = 0; if (cudaGetDeviceCount(&n) != cudaSuccess) return 0; return n > 0;
#else
  return 0;
#endif
}

EXPORT int sweep(const Design* D, uint32_t lo, uint32_t hi, uint32_t stride, Stats* out) {
  unsigned long long cnt = ((unsigned long long)hi - lo) / stride + 1;
#ifdef __CUDACC__
  if (prep()) return -1;
  k_sweep<<<1024, 256>>>(*D, lo, cnt, stride, d_acc, d_acc + 1, d_acc + 2, d_acc + 3, d_sum, d_acc + 4);
  return finish(out);
#else
  double gmax = -1; uint32_t garg = 0; unsigned long long g1 = 0, g05 = 0, ag1 = 0; double gs = 0;
  #pragma omp parallel
  {
    double lmax = -1; uint32_t larg = 0;
    #pragma omp for reduction(+:g1,g05,ag1,gs) schedule(static)
    for (long long i = 0; i < (long long)cnt; i++) {
      uint32_t x = (uint32_t)(lo + (unsigned long long)i * stride);
      uint32_t y = eval_erf(D, x);
      double e = ulp_err(x, y);
      if (e > lmax || (e == lmax && x < larg)) { lmax = e; larg = x; }
      g1 += (e > 1.0); g05 += (e > 0.5); gs += e;
      ag1 += ((y & 0x7fffffffu) > F_ONE && (y & 0x7fffffffu) <= 0x7f800000u);
    }
    #pragma omp critical
    { if (lmax > gmax || (lmax == gmax && larg < garg)) { gmax = lmax; garg = larg; } }
  }
  out->maxerr = gmax; out->argmax = garg; out->n_gt1 = g1; out->n_gt05 = g05; out->n_absgt1 = ag1;
  out->count = cnt; out->sumerr = gs;
  return 0;
#endif
}

EXPORT int eval_list(const Design* D, const uint32_t* xs, int m, Stats* out, double* errs) {
#ifdef __CUDACC__
  if (prep()) return -1;
  if (m > d_cap) {
    if (d_xs) { cudaFree(d_xs); cudaFree(d_errs); }
    if (cudaMalloc(&d_xs, (size_t)m * 4) != cudaSuccess) return -1;
    if (cudaMalloc(&d_errs, (size_t)m * 8) != cudaSuccess) return -1;
    d_cap = m;
  }
  cudaMemcpy(d_xs, xs, (size_t)m * 4, cudaMemcpyHostToDevice);
  k_list<<<512, 256>>>(*D, d_xs, m, errs ? d_errs : 0, d_acc, d_acc + 1, d_acc + 2, d_acc + 3, d_sum, d_acc + 4);
  int rc = finish(out);
  if (!rc && errs) cudaMemcpy(errs, d_errs, (size_t)m * 8, cudaMemcpyDeviceToHost);
  return rc;
#else
  double gmax = -1; uint32_t garg = 0; unsigned long long g1 = 0, g05 = 0, ag1 = 0; double gs = 0;
  #pragma omp parallel
  {
    double lmax = -1; uint32_t larg = 0;
    #pragma omp for reduction(+:g1,g05,ag1,gs) schedule(static)
    for (int i = 0; i < m; i++) {
      uint32_t x = xs[i];
      uint32_t y = eval_erf(D, x);
      double e = ulp_err(x, y);
      if (errs) errs[i] = e;
      if (e > lmax || (e == lmax && x < larg)) { lmax = e; larg = x; }
      g1 += (e > 1.0); g05 += (e > 0.5); gs += e;
      ag1 += ((y & 0x7fffffffu) > F_ONE && (y & 0x7fffffffu) <= 0x7f800000u);
    }
    #pragma omp critical
    { if (lmax > gmax || (lmax == gmax && larg < garg)) { gmax = lmax; garg = larg; } }
  }
  out->maxerr = gmax; out->argmax = garg; out->n_gt1 = g1; out->n_gt05 = g05; out->n_absgt1 = ag1;
  out->count = m; out->sumerr = gs;
  return 0;
#endif
}

// Contrôle positif : notre portage des FMA contre le fichier de référence (compilé à part).
EXPORT uint32_t fma_port(int model, uint32_t a, uint32_t b, uint32_t c) { return fma_m(model, a, b, c); }
EXPORT uint32_t eval_one(const Design* D, uint32_t x) { return eval_erf(D, x); }

// Collecte des entrées dont l'erreur dépasse un seuil (pour alimenter le polissage).
#ifdef __CUDACC__
__global__ void k_collect(Design D, uint32_t lo, unsigned long long cnt, uint32_t stride, double thr,
                          uint32_t* out, int cap, int* nout) {
  for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < cnt;
       i += (unsigned long long)gridDim.x * blockDim.x) {
    uint32_t x = (uint32_t)(lo + i * (unsigned long long)stride);
    double e = ulp_err(x, eval_erf(&D, x));
    if (e > thr) { int k = atomicAdd(nout, 1); if (k < cap) out[k] = x; }
  }
}
#endif
EXPORT int collect(const Design* D, uint32_t lo, uint32_t hi, uint32_t stride, double thr, uint32_t* out, int cap,
                   int* nout) {
  unsigned long long cnt = ((unsigned long long)hi - lo) / stride + 1;
#ifdef __CUDACC__
  uint32_t* d_out; int* d_n;
  if (cudaMalloc(&d_out, (size_t)cap * 4) != cudaSuccess) return -1;
  if (cudaMalloc(&d_n, sizeof(int)) != cudaSuccess) return -1;
  cudaMemset(d_n, 0, sizeof(int));
  k_collect<<<1024, 256>>>(*D, lo, cnt, stride, thr, d_out, cap, d_n);
  if (cudaDeviceSynchronize() != cudaSuccess) return -2;
  int n; cudaMemcpy(&n, d_n, sizeof(int), cudaMemcpyDeviceToHost);
  int m = n < cap ? n : cap;
  cudaMemcpy(out, d_out, (size_t)m * 4, cudaMemcpyDeviceToHost);
  cudaFree(d_out); cudaFree(d_n);
  *nout = n; return 0;
#else
  int n = 0;
  #pragma omp parallel for schedule(static)
  for (long long i = 0; i < (long long)cnt; i++) {
    uint32_t x = (uint32_t)(lo + (unsigned long long)i * stride);
    double e = ulp_err(x, eval_erf(D, x));
    if (e > thr) { int k;
      #pragma omp atomic capture
      k = n++;
      if (k < cap) out[k] = x; }
  }
  *nout = n; return 0;
#endif
}
'''

MAXC = 18
class Design(ctypes.Structure):
    _fields_ = [('n', ctypes.c_int), ('d', ctypes.c_int), ('a', ctypes.c_float * MAXC), ('b', ctypes.c_float * MAXC),
                ('xclamp', ctypes.c_float), ('corr', ctypes.c_int), ('clampout', ctypes.c_int), ('model', ctypes.c_int),
                ('kind', ctypes.c_int), ('split', ctypes.c_float), ('center', ctypes.c_float)]

class Stats(ctypes.Structure):
    _fields_ = [('maxerr', ctypes.c_double), ('argmax', ctypes.c_uint32), ('pad', ctypes.c_uint32),
                ('n_gt1', ctypes.c_ulonglong), ('n_gt05', ctypes.c_ulonglong), ('count', ctypes.c_ulonglong),
                ('n_absgt1', ctypes.c_ulonglong), ('sumerr', ctypes.c_double)]

def f2b(v): return int(np.array(v, dtype=np.float32).view(np.uint32))
def b2f(b): return float(np.array(b & 0xffffffff, dtype=np.uint32).view(np.float32))
def log(*a):
    print(*a, flush=True)

# ----------------------------------------------------------------------------------------------
def construire(force_cpu):
    wd = os.path.join(tempfile.gettempdir(), 'erf_tt_build')
    os.makedirs(wd, exist_ok=True)
    open(os.path.join(wd, 'sfpu_model.h'), 'w').write(SRC_H)
    open(os.path.join(wd, 'erf_lib.cu'), 'w').write(SRC_CU)
    if not force_cpu and shutil.which('nvcc'):
        so = os.path.join(wd, 'liberf_gpu.so')
        archs = ['-arch=native']
        try:
            cc = subprocess.run(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
                                capture_output=True, text=True).stdout.split()[0].replace('.', '')
            archs.append(f'-arch=sm_{cc}')
        except Exception:
            pass
        for arch in archs:
            r = subprocess.run(['nvcc', '-O3', arch, '-Xcompiler', '-fPIC', '-shared', '-o', so, 'erf_lib.cu'],
                               cwd=wd, capture_output=True, text=True)
            if r.returncode == 0:
                lib = ctypes.CDLL(so)
                if lib.has_gpu():
                    return declarer(lib), 'GPU'
            else:
                log(f"  (nvcc {arch} a échoué : {r.stderr.strip()[:300]})")
        log("  GPU indisponible : repli sur le CPU.")
    so = os.path.join(wd, 'liberf_cpu.so')
    r = subprocess.run(['gcc', '-O3', '-march=native', '-fopenmp', '-shared', '-fPIC', '-x', 'c', 'erf_lib.cu',
                        '-o', so, '-lm'], cwd=wd, capture_output=True, text=True)
    if r.returncode:
        sys.exit("Compilation impossible :\n" + r.stderr)
    return declarer(ctypes.CDLL(so)), 'CPU'

def declarer(lib):
    P = ctypes.POINTER
    lib.sweep.argtypes = [P(Design), ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, P(Stats)]
    lib.eval_list.argtypes = [P(Design), P(ctypes.c_uint32), ctypes.c_int, P(Stats), P(ctypes.c_double)]
    lib.collect.argtypes = [P(Design), ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_double,
                            P(ctypes.c_uint32), ctypes.c_int, P(ctypes.c_int)]
    lib.eval_one.argtypes = [P(Design), ctypes.c_uint32]
    lib.eval_one.restype = ctypes.c_uint32
    return lib

# ----------------------------------------------------------------------------------------------
class Banc:
    def __init__(self, lib, mode, rapide):
        self.lib, self.mode, self.rapide = lib, mode, rapide
    def sweep(self, D, lo, hi, stride=1):
        st = Stats()
        if self.lib.sweep(ctypes.byref(D), lo, hi, stride, ctypes.byref(st)):
            sys.exit("Erreur GPU pendant le balayage")
        return st
    def evalue(self, D, xs, avec_err=False):
        st = Stats()
        errs = np.empty(len(xs), dtype=np.float64) if avec_err else None
        self.lib.eval_list(ctypes.byref(D), xs.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)), len(xs),
                           ctypes.byref(st), errs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) if avec_err else None)
        return st, errs
    def collecte(self, D, lo, hi, stride, seuil, cap=200000):
        out = np.empty(cap, dtype=np.uint32); n = ctypes.c_int(0)
        self.lib.collect(ctypes.byref(D), lo, hi, stride, seuil, out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                         cap, ctypes.byref(n))
        return out[:min(n.value, cap)].copy(), n.value
    # pas de balayage : exhaustif sur GPU, échantillonné sur CPU
    def pas_complet(self, hi):
        if self.mode == 'GPU':
            return 1
        return max(1, hi // (400_000 if self.rapide else 4_000_000))

# ----------------------------------------------------------------------------------------------
def coeffs_actuels():
    A = [1.1283791065e+00, 2.1477432549e-01, 6.2133435160e-02, 5.6230435148e-03, 6.1307044234e-04,
         1.7678321456e-05, 2.7384647439e-08, -2.8632063387e-10]
    B = [1.0, 5.2367275953e-01, 1.2961706519e-01, 1.9642570987e-02, 1.9545555115e-03, 1.3179056987e-04,
         1.3156344494e-06, -3.5153888689e-09, -6.7350725691e-12]
    return A, B

def faire_design(a, b, xclamp, corr, clampout, model=0):
    D = Design(); D.n = len(a) - 1; D.d = len(b) - 1
    for i, v in enumerate(a): D.a[i] = float(np.float32(v))
    for i, v in enumerate(b): D.b[i] = float(np.float32(v))
    D.xclamp = float(np.float32(xclamp)); D.corr = corr; D.clampout = clampout; D.model = model
    return D

def copie(D, **kw):
    E = Design(); ctypes.memmove(ctypes.byref(E), ctypes.byref(D), ctypes.sizeof(Design))
    for k, v in kw.items(): setattr(E, k, v)
    return E

def charge(v):
    """Instructions SFPLOADI pour charger une constante : 0 (registre constant), 1 (exacte en bf16), 2 sinon."""
    v = float(np.float32(v))
    if v in (0.0, 1.0, -1.0): return 0
    return 1 if (f2b(v) & 0xffff) == 0 else 2

def cout(D):
    a = [D.a[i] for i in range(D.n + 1)]; b = [D.b[i] for i in range(D.d + 1)]
    if D.kind == 1:
        # load, store, abs, min(|x|,X), x², x*P, t=|x|-m, sélection (cmp+mov+push/pop), setsgn
        fixe = 1 + 1 + 1 + 2 + 1 + 1 + 1 + 4 + 1 + 2   # + 2 mises à l'échelle exactes (SFPDIVP2)
        return (fixe + charge(D.xclamp) + charge(D.split) + charge(D.center) + D.n + D.d + 2 * D.clampout
                + sum(charge(v) for v in a) + sum(charge(v) for v in b))
    fixe = 1 + 1 + 2 + 1 + 1 + 7 + 1   # load, store, clamp symétrique, x², numer·x, réciproque, produit
    return (fixe + charge(D.xclamp) + D.n + D.d + 2 * D.corr + 2 * D.clampout
            + sum(charge(v) for v in a) + sum(charge(v) for v in b))

# ----------------------------------------------------------------------------------------------
def seuil_saturation():
    """Plus petit float32 x tel que erf(x) s'arrondit à 1.0f, puis le plus petit bf16 >= x."""
    lo, hi = f2b(3.0), f2b(4.5)
    while lo < hi:
        m = (lo + hi) // 2
        if math.erf(b2f(m)) >= 1.0 - 2.0 ** -25: hi = m
        else: lo = m + 1
    x1 = b2f(lo)
    xc = lo if (lo & 0xffff) == 0 else ((lo | 0xffff) + 1)
    return x1, b2f(xc)

def ajuste(n, d, X, iters=60):
    """Rationnelle minimax en ULP float32 : erf(x) ~ x*N(x²)/D(x²), D(0)=1, sur [0, X] (Sanathanan-Koerner + Lawson)."""
    xs = np.unique(np.concatenate([np.geomspace(1e-6, 0.5, 3000), np.linspace(0.3, X, 9000),
                                   X * (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, 4000)))]))
    xs = xs[xs > 0]
    f = np.array([math.erf(v) for v in xs])
    wu = 2.0 ** (np.floor(np.log2(np.abs(f))) - 23)
    s = X * X; us = xs * xs / s
    Den = np.ones_like(xs); w = np.ones_like(xs) / len(xs); best = (1e99, None)
    Pn = np.stack([us ** k for k in range(n + 1)], 1); Pd = np.stack([us ** k for k in range(1, d + 1)], 1)
    for _ in range(iters):
        A = np.concatenate([xs[:, None] * Pn, -f[:, None] * Pd], 1)
        W = np.sqrt(w) / (wu * Den)
        sol, *_ = np.linalg.lstsq(A * W[:, None], f * W, rcond=None)
        N = Pn @ sol[:n + 1]; Dd = 1 + Pd @ sol[n + 1:]
        if np.any(Dd <= 0):
            Den = np.abs(Dd) + 1e-9; continue
        e = (xs * N / Dd - f) / wu; m = np.max(np.abs(e))
        if m < best[0]: best = (m, sol.copy())
        Den = Dd; w = w * (np.abs(e) + 1e-300); w /= w.sum()
    m, sol = best
    if sol is None:
        raise RuntimeError("ajustement impossible (pôle dans l'intervalle)")
    a = [sol[k] / s ** k for k in range(n + 1)]
    b = [1.0] + [sol[n + k] / s ** k for k in range(1, d + 1)]
    return a, b, m

def lawson(B, f, unite, iters=80):
    w = np.ones(len(f)) / len(f); best = (1e99, None)
    for _ in range(iters):
        W = np.sqrt(w) / unite
        cf, *_ = np.linalg.lstsq(B * W[:, None], f * W, rcond=None)
        e = (B @ cf - f) / unite; m = np.max(np.abs(e))
        if m < best[0]: best = (m, cf.copy())
        w = w * (np.abs(e) + 1e-300); w /= w.sum()
    return best

def bf16_proche(v):
    b = f2b(v); return b2f((b + 0x8000) & 0xffff0000)

def ajuste2(c, dA, dB, X):
    """Deux morceaux : erf(x) = x + x*P(x²) sur [0,c] (erreur relative), erf(x) = Q(x-m) sur [c,X] (erreur absolue en ULP)."""
    xa = np.unique(np.concatenate([np.geomspace(1e-6, c, 3000), c * (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, 3000)))]))
    xa = xa[xa > 0]
    fa = np.array([math.erf(v) for v in xa]) / xa - 1.0      # erf(x) = x + x*P(x²)
    ua = 2.0 ** (np.floor(np.log2((fa + 1.0) * xa)) - 23) / xa
    mA, cA = lawson(np.stack([(xa * xa) ** k for k in range(dA + 1)], 1), fa, ua)
    m = bf16_proche(0.5 * (c + X))
    xb = np.unique(np.concatenate([np.linspace(c, X, 6000), c + (X - c) * (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, 3000)))]))
    fb = np.array([math.erf(v) for v in xb])
    ub = 2.0 ** (np.floor(np.log2(fb)) - 23)
    mB, cB = lawson(np.stack([(xb - m) ** k for k in range(dB + 1)], 1), fb, ub)
    return list(cA), list(cB), m, max(mA, mB)

# ----------------------------------------------------------------------------------------------
def echantillon_initial(X, rapide, rng):
    pts = []
    for e in range(-126, 2):                       # chaque binade, des plus petites à [2, 4)
        k = 48 if e < -20 else (256 if rapide else 1024)
        man = rng.integers(0, 1 << 23, k, dtype=np.uint64)
        pts.append(((np.uint64(e + 127) << np.uint64(23)) | man).astype(np.uint32))
    dense = rng.uniform(0.25, X, 30000 if rapide else 200000).astype(np.float32).view(np.uint32)
    pts.append(dense)
    pts.append(np.arange(f2b(X) - 4096, f2b(X) + 1, dtype=np.uint32))
    return np.unique(np.concatenate(pts)).astype(np.uint32)

class Polisseur:
    def __init__(self, banc, D, S):
        self.banc, self.D, self.S = banc, D, S
        self.b0 = 0 if D.kind == 1 else 1     # rationnelle : b0 = 1 figé
        self.params = [f2b(D.a[i]) for i in range(D.n + 1)] + [f2b(D.b[i]) for i in range(self.b0, D.d + 1)]
        self.gel = set()        # indices figés (contrainte bf16)
        self.nevals = 0
    def applique(self, p):
        E = copie(self.D)
        for i in range(self.D.n + 1): E.a[i] = b2f(p[i])
        for j in range(self.b0, self.D.d + 1): E.b[j] = b2f(p[self.D.n + 1 + j - self.b0])
        return E
    def score(self, p):
        st, _ = self.banc.evalue(self.applique(p), self.S); self.nevals += 1
        return (round(st.maxerr, 7), st.sumerr / st.count)
    def polit(self, budget_s, rng):
        t0 = time.time(); best = self.score(self.params)
        pas = [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 4096, 16384]
        amelioration = True
        while amelioration and time.time() - t0 < budget_s * 0.7:
            amelioration = False
            for i in range(len(self.params)):
                if i in self.gel: continue
                for st in pas:
                    ok = False
                    for sg in (1, -1):
                        c = list(self.params); c[i] = (c[i] + sg * st) & 0xffffffff
                        sc = self.score(c)
                        if sc < best: self.params, best, ok, amelioration = c, sc, True, True; break
                    if ok: break
                if time.time() - t0 > budget_s * 0.7: break
        libres = [i for i in range(len(self.params)) if i not in self.gel]
        while time.time() - t0 < budget_s and libres:      # stratégie d'évolution (petites mutations)
            c = list(self.params)
            for i in rng.choice(libres, size=min(len(libres), int(rng.integers(1, 4))), replace=False):
                c[i] = (c[i] + int(rng.choice([-1, 1])) * int(rng.integers(1, 9)) * int(rng.choice([1, 1, 4, 32]))) & 0xffffffff
            sc = self.score(c)
            if sc < best: self.params, best = c, sc
        return best

def chasse(banc, D, X):
    """Vérification complète sur [0, X] et collecte des pires points."""
    hi = f2b(X); stp = banc.pas_complet(hi)
    st = banc.sweep(D, 0, hi, stp)
    pts, ntot = banc.collecte(D, 0, hi, stp, max(0.6, 0.8 * st.maxerr))
    return st, pts, ntot

def optimise(banc, nom, plan, X, budget_s, rng, rapide):
    t0 = time.time()
    if plan[0] == 'R':
        _, n, d, corr = plan
        a, b, approx = ajuste(n, d, X)
        D = faire_design(a, b, X, corr, 0)
    else:
        _, c, n, d = plan
        a, b, m, approx = ajuste2(c, n, d, X)
        D = faire_design(a, b, X, 0, 0); D.kind = 1; D.split = c; D.center = m
    S = echantillon_initial(X, rapide, rng)
    st0, _ = banc.evalue(D, S)
    log(f"  [{nom}] ajustement : erreur d'approximation {approx:.3f} ULP ; après arrondi float32 : {st0.maxerr:.3f} ULP")
    pol = Polisseur(banc, D, S)
    tours = 0; best_full = None
    while time.time() - t0 < budget_s and tours < 12:
        tours += 1
        reste = budget_s - (time.time() - t0)
        sc = pol.polit(min(reste * 0.45, 25 if rapide else 90), rng)
        D = pol.applique(pol.params)
        st, pts, ntot = chasse(banc, D, X)
        best_full = st
        log(f"  [{nom}] tour {tours} : échantillon {sc[0]:.3f} ULP | vérif. [0,{X}] {st.maxerr:.3f} ULP "
            f"(pire x={b2f(st.argmax):.9g}), {ntot} points chauds, {len(pol.S)} pts, {pol.nevals} évals")
        if len(pts):
            pick = pts if len(pts) <= 40000 else rng.choice(pts, 40000, replace=False)
            voisins = (np.int64(st.argmax) + np.arange(-64, 65)).astype(np.uint32)
            pol.S = np.unique(np.concatenate([pol.S, pick.astype(np.uint32), voisins]))
        if st.maxerr <= 0.5:
            break
    # tentative de réduction du coût : arrondir en bf16 les coefficients de plus haut degré
    cible = SEUIL_GROS if best_full.maxerr <= SEUIL_GROS else (SEUIL_PETIT if best_full.maxerr <= SEUIL_PETIT else None)
    if cible is not None and time.time() - t0 < budget_s * 1.3:
        nb = len(pol.params)
        if D.kind == 1:   # trois plus hauts degrés de chaque polynôme
            ordre = [D.n, nb - 1, D.n - 1, nb - 2, D.n - 2, nb - 3, nb - 4, D.n - 3, nb - 5]
        else:
            ordre = [D.n, nb - 1]
        for idx in ordre:
            sauve = list(pol.params)
            pol.params[idx] = (pol.params[idx] + 0x8000) & 0xffff0000   # arrondi au bf16 le plus proche
            pol.gel.add(idx)
            pol.polit(10 if rapide else 45, rng)
            E = pol.applique(pol.params)
            st, _, _ = chasse(banc, E, X)
            if st.maxerr <= cible:
                D, best_full = E, st
                log(f"  [{nom}] coefficient {idx} en bf16 : OK ({st.maxerr:.3f} ULP), coût {cout(D)}")
            else:
                pol.params = sauve; pol.gel.discard(idx)
                log(f"  [{nom}] coefficient {idx} en bf16 : refusé ({st.maxerr:.3f} ULP)")
    return pol.applique(pol.params) if best_full is None else D

def verifie_final(banc, D, nom):
    """Exhaustif sur toutes les entrées positives (0 .. +inf) pour Blackhole et Wormhole."""
    res = {}
    hi = 0x7f800000; stp = banc.pas_complet(hi)
    for model, arch in ((0, 'BH'), (1, 'WH')):
        E = copie(D, model=model)
        t = time.time(); st = banc.sweep(E, 0, hi, stp)
        if st.n_absgt1 and not E.clampout:
            E = copie(E, clampout=1); st = banc.sweep(E, 0, hi, stp)
        res[arch] = dict(max=st.maxerr, pire=b2f(st.argmax), moy=st.sumerr / st.count,
                         pct_gt1=100.0 * st.n_gt1 / st.count, n=st.count, clampout=E.clampout, s=time.time() - t)
    return res

def hexf(v):
    h = float.hex(float(np.float32(v)))
    m, e = h.split('p'); m = m.rstrip('0')
    return (m + '0' if m.endswith('.') else m) + 'p' + e + 'f'

# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--minutes', type=float, default=40)
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--rapide', action='store_true', help='petit test (quelques minutes)')
    args = ap.parse_args()
    rng = np.random.default_rng(2026)
    T0 = time.time()

    log("=" * 96)
    log("TEST DÉCISIF — erf float32 sur SFPU Tenstorrent (modèle bit-exact SFPMAD + SFPARECIP)")
    log(f"Critère fixé d'avance : GROS LOT = max <= {SEUIL_GROS} ULP et coût <= actuel ; PETIT LOT = max <= "
        f"{SEUIL_PETIT} ULP et coût <= actuel, ou max <= {SEUIL_GROS} ULP et coût <= {MARGE_COUT} x actuel")
    lib, mode = construire(args.cpu)
    banc = Banc(lib, mode, args.rapide)
    log(f"Moteur : {mode}" + ("" if mode == 'GPU' else "  (vérifications ÉCHANTILLONNÉES, pas exhaustives)"))

    # contrôle de cohérence du modèle : une entrée connue
    X1, Xc = seuil_saturation()
    log(f"Saturation : erf(x) s'arrondit à 1.0f dès x = {X1:.9g} ; borne utilisée (exacte en bf16) : {Xc}")

    # 1) CONTRÔLE POSITIF : l'erf actuelle de tt-metal
    A, B = coeffs_actuels()
    base = faire_design(A, B, 10.0, 0, 1)
    log("\n[1] Contrôle positif : erf actuelle de tt-metal (n16/d16 sur [-10, 10], clamp de sortie)")
    for lo, hi in ((0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 4.0)):
        st = banc.sweep(base, f2b(lo), f2b(hi), max(1, (f2b(hi) - f2b(lo)) // 2_000_000))
        log(f"    [{lo}, {hi}] max {st.maxerr:6.3f} ULP  moyenne {st.sumerr / st.count:.3f}  "
            f"> 1 ULP : {100 * st.n_gt1 / st.count:5.1f} %")
    log("    (signalement #58282, mesuré sur ttsim : moyennes 0.647 / 0.943 / 0.973 / 1.073 / 1.122 / 1.127, "
        "parts 22.5 / 40.8 / 41.2 / 45.8 / 49.0 / 46.8 %)")
    ref = verifie_final(banc, base, 'actuelle')
    cout_base = cout(base)
    log(f"    Exhaustif : Blackhole max {ref['BH']['max']:.3f} ULP ({ref['BH']['pct_gt1']:.2f} % > 1 ULP), "
        f"Wormhole max {ref['WH']['max']:.3f} ULP | coût estimé {cout_base} instructions")
    controle_ok = ref['BH']['max'] >= 3.0

    # 2) RECHERCHE
    if args.rapide:
        plans = [('2M c=1.5 A8/B12', ('K', 1.5, 8, 12))]
    else:
        plans = [('2M c=1.5 A8/B12', ('K', 1.5, 8, 12)), ('2M c=1.25 A7/B13', ('K', 1.25, 7, 13)),
                 ('2M c=1.75 A8/B11', ('K', 1.75, 8, 11)), ('2M c=1.5 A7/B12', ('K', 1.5, 7, 12)),
                 ('2M c=1.0 A6/B14', ('K', 1.0, 6, 14)), ('2M c=1.5 A7/B10', ('K', 1.5, 7, 10)),
                 ('R5/6+corr', ('R', 5, 6, 1))]
    budget_total = args.minutes * 60 - (time.time() - T0) - 60
    budget = max(30.0, budget_total * 0.85 / len(plans))
    log(f"\n[2] Recherche : {len(plans)} familles de formules, ~{budget / 60:.1f} min chacune")
    resultats = []
    for nom, plan in plans:
        if plan[0] == 'R':
            log(f"\n  --- {nom} : x·N(x²)/D(x²), N degré {plan[1]}, D degré {plan[2]}, correction={'oui' if plan[3] else 'non'} ---")
        else:
            log(f"\n  --- {nom} : x + x·P(x²) si |x|<{plan[1]}, sinon Q(|x|-m) ; degrés {plan[2]} et {plan[3]} ---")
        try:
            D = optimise(banc, nom, plan, Xc, budget, rng, args.rapide)
        except RuntimeError as ex:
            log(f"  [{nom}] abandonné : {ex}"); continue
        v = verifie_final(banc, D, nom)
        D = copie(D, clampout=v['BH']['clampout'])
        c = cout(D)
        n, d = D.n, D.d; corr = D.corr
        log(f"  [{nom}] FINAL : Blackhole max {v['BH']['max']:.3f} ULP ({v['BH']['pct_gt1']:.3f} % > 1 ULP, moy "
            f"{v['BH']['moy']:.3f}) | Wormhole max {v['WH']['max']:.3f} ULP | coût {c} vs {cout_base} actuel")
        resultats.append(dict(nom=nom, kind=D.kind, n=n, d=d, corr=corr, cout=c, BH=v['BH'], WH=v['WH'],
                              a=[D.a[i] for i in range(D.n + 1)], b=[D.b[i] for i in range(D.d + 1)],
                              xclamp=D.xclamp, clampout=D.clampout, split=D.split, center=D.center))

    # 3) CONTRÔLE NÉGATIF : on abîme la meilleure formule, l'erreur doit exploser
    meilleur = min(resultats, key=lambda r: (r['BH']['max'] > SEUIL_GROS, r['cout'], r['BH']['max']))
    Dm = faire_design(meilleur['a'], meilleur['b'], meilleur['xclamp'], meilleur['corr'], meilleur['clampout'])
    Dm.kind = meilleur['kind']; Dm.split = meilleur['split']; Dm.center = meilleur['center']
    Dn = copie(Dm); Dn.a[0] = b2f(f2b(Dm.a[0]) + 256)
    stn = banc.sweep(Dn, f2b(0.5), f2b(4.0), max(1, (f2b(4.0) - f2b(0.5)) // 2_000_000))
    log(f"\n[3] Contrôle négatif : {meilleur['nom']} avec a0 décalé de 256 ULP -> max {stn.maxerr:.2f} ULP "
        f"(doit être nettement pire)")

    # 4) VERDICT
    log("\n" + "=" * 96)
    log(f"{'formule':<14}{'coût':>6}{'BH max':>10}{'BH >1ULP':>11}{'BH moy':>9}{'WH max':>10}")
    log(f"{'ACTUELLE':<14}{cout_base:>6}{ref['BH']['max']:>10.3f}{ref['BH']['pct_gt1']:>10.2f}%{ref['BH']['moy']:>9.3f}"
        f"{ref['WH']['max']:>10.3f}")
    for r in sorted(resultats, key=lambda r: r['cout']):
        log(f"{r['nom']:<14}{r['cout']:>6}{r['BH']['max']:>10.3f}{r['BH']['pct_gt1']:>10.3f}%{r['BH']['moy']:>9.3f}"
            f"{r['WH']['max']:>10.3f}")
    gros = [r for r in resultats if r['BH']['max'] <= SEUIL_GROS and r['cout'] <= cout_base]
    petit = [r for r in resultats if (r['BH']['max'] <= SEUIL_PETIT and r['cout'] <= cout_base) or
             (r['BH']['max'] <= SEUIL_GROS and r['cout'] <= MARGE_COUT * cout_base)]
    exhaustif = (mode == 'GPU')
    if not controle_ok:
        verdict = "CONTRÔLE POSITIF ÉCHOUÉ : le banc ne reproduit pas l'erreur publiée, ne rien conclure."
    elif gros:
        g = min(gros, key=lambda r: (r['cout'], r['BH']['max']))
        verdict = (f"GROS LOT ATTEINT : {g['nom']} — {g['BH']['max']:.3f} ULP max (au lieu de {ref['BH']['max']:.2f}), "
                   f"{g['cout']} instructions au lieu de {cout_base} ({100 * (1 - g['cout'] / cout_base):.0f} % de moins)")
    elif petit:
        g = min(petit, key=lambda r: (r['BH']['max'], r['cout']))
        verdict = f"PETIT LOT ATTEINT : {g['nom']} — {g['BH']['max']:.3f} ULP max, {g['cout']} instructions (actuel {cout_base})"
    else:
        verdict = "NON ATTEINT : aucune formule ne bat l'actuelle selon le critère."
    if not exhaustif:
        verdict += "  [CPU : vérification échantillonnée seulement — à confirmer sur GPU]"
    log("VERDICT : " + verdict)
    log("=" * 96)

    # 5) SORTIES
    out = dict(date=time.strftime('%Y-%m-%d %H:%M'), moteur=mode, exhaustif=exhaustif, critere=dict(
        gros=SEUIL_GROS, petit=SEUIL_PETIT, marge_cout=MARGE_COUT), saturation=X1, borne=Xc,
        actuelle=dict(cout=cout_base, **{k: ref[k] for k in ref}), formules=resultats, verdict=verdict,
        controle_negatif=stn.maxerr)
    json.dump(out, open('erf_tt_resultats.json', 'w'), indent=1, default=float)
    with open('erf_tt_noyau.txt', 'w') as fo:
        for r in sorted(resultats, key=lambda r: (r['BH']['max'] > SEUIL_GROS, r['cout'])):
            fo.write(f"// {r['nom']} : max {r['BH']['max']:.3f} ULP (Blackhole, exhaustif={exhaustif}), "
                     f"{r['BH']['pct_gt1']:.3f} % > 1 ULP, coût estimé {r['cout']} (actuel {cout_base})\n")
            if r['kind'] == 1:
                fo.write(f"//   ax = min(|x|, {r['xclamp']}) ; si ax < {r['split']} : y = 2^-24*(xs + xs*P(ax²)), xs = 2^24*ax ; sinon y = Q(ax - {r['center']}) ;"
                         + (" borné à 1 ;" if r['clampout'] else "") + " y = copysign(y, x)\n")
                fo.write("//   P (degré 0 -> n, en x²) : " + ", ".join(hexf(v) for v in r['a']) + "\n")
                fo.write("//   Q (degré 0 -> d, en ax-m) : " + ", ".join(hexf(v) for v in r['b']) + "\n\n")
            else:
                fo.write(f"//   x borné à ±{r['xclamp']} ; y = x*N(x²) * recip(D(x²))"
                         + (" ; e = x*N - y*D ; y += recip*e" if r['corr'] else "")
                         + (" ; sortie bornée à [-1,1]" if r['clampout'] else "") + "\n")
                fo.write("//   N (degré 0 -> n) : " + ", ".join(hexf(v) for v in r['a']) + "\n")
                fo.write("//   D (degré 0 -> d) : " + ", ".join(hexf(v) for v in r['b']) + "\n\n")
    log("Fichiers écrits : erf_tt_resultats.json, erf_tt_noyau.txt")
    log(f"Durée totale : {(time.time() - T0) / 60:.1f} min")

if __name__ == '__main__':
    main()
