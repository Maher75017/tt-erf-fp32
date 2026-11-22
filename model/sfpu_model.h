// SPDX-License-Identifier: Apache-2.0
// Port of tt-isa-documentation/Miscellaneous/FMA/fma.c (SFPMAD, Blackhole and Wormhole) and of the SFPARECIP
// description in tt-isa-documentation, (c) Tenstorrent AI ULC, Apache-2.0.
// Modifications (C/CUDA port, erf evaluation, ULP metric): (c) 2026 Maher El Hmissi, Apache-2.0.
// Modèle bit-exact de l'arithmétique SFPU Tenstorrent (portage de tt-isa-documentation/Miscellaneous/FMA/fma.c
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
