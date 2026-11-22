// Évalue erf (rationnelle) avec la VRAIE réciproque Wormhole (graine quadratique + 2 Newton + échelle),
// telle que dans tt_metal/hw/ckernels/wormhole_b0/.../ckernel_sfpu_recip.h, et la réciproque Blackhole
// (SFPARECIP + Newton) pour comparaison.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "sfpu_model.h"
#include "designs.h"
static uint32_t k0, k1, k2;
static inline uint32_t recip_wh(uint32_t in) {
  uint32_t negx = 0xBF800000u | (in & 0x007FFFFFu);          // copyman(-1.0f, in)
  uint32_t y = fma_wh(k0, negx, k1);                          // y = K1 + K0*negx
  uint32_t sb = ~in;                                           // scale_bits = ~in
  y = fma_wh(y, negx, k2);                                     // y = K2 + y*negx
  uint32_t scale = sb & 0xFF800000u;                           // setman(scale_bits, 0)
  uint32_t t = fma_wh(negx, y, F_ONE);                         // t = 1 + negx*y
  scale = fma_wh(scale, 0x3F000000u, F_PZ);                    // scale *= 0.5
  y = fma_wh(y, t, y);                                         // y = y + y*t
  t = fma_wh(negx, y, F_ONE);                                  // 2e itération
  y = fma_wh(y, t, y);
  y = fma_wh(y, scale, F_PZ);                                  // y = y * scale
  return (y & 0x7FFFFFFFu) | (in & 0x80000000u);               // copysgn(y, in)
}
static inline uint32_t eval_rat(int wh, int whrecip, const float* A, int n, const float* B, int d,
                                float xclamp, int corr, uint32_t xb) {
  uint32_t ax = xb & 0x7fffffffu, s = xb & 0x80000000u, cl = as_u(xclamp);
  if (ax > cl) xb = s | cl;
  uint32_t u = fma_m(wh, xb, xb, F_PZ);
  uint32_t N = as_u(A[n]); for (int k = n - 1; k >= 0; k--) N = fma_m(wh, N, u, as_u(A[k]));
  uint32_t Q = as_u(B[d]); for (int k = d - 1; k >= 0; k--) Q = fma_m(wh, Q, u, as_u(B[k]));
  uint32_t xN = fma_m(wh, N, xb, F_PZ);
  uint32_t r = whrecip ? recip_wh(Q) : recip_seq(wh, Q);
  uint32_t y = fma_m(wh, xN, r, F_PZ);
  if (corr) { uint32_t e = fma_m(wh, y ^ 0x80000000u, Q, xN); y = fma_m(wh, r, e, y); }
  uint32_t ay = y & 0x7fffffffu; if (ay > F_ONE && ay <= 0x7f800000u) y = (y & 0x80000000u) | F_ONE;
  return y;
}
int main(int argc, char** argv) {
  float f0 = 0.3232325017452239990234375f, f1 = 1.4545459747314453125f, f2 = 2.121212482452392578125f;
  k0 = as_u(f0); k1 = as_u(f1); k2 = as_u(f2);
  long step = argc > 1 ? atol(argv[1]) : 1;   // 1 = exhaustif
  struct { const char* nom; int wh, whr; const float* A; int n; const float* B; int d; float xc; int corr; } C[] = {
    {"actuel  BH (SFPARECIP)", 0, 0, CUR_A, 7, CUR_B, 8, 10.0f, 0},
    {"propose BH (SFPARECIP)", 0, 0, PRO_A, 5, PRO_B, 6, 3.921875f, 1},
    {"actuel  WH (ancien modele : recip BH)", 1, 0, CUR_A, 7, CUR_B, 8, 10.0f, 0},
    {"propose WH (ancien modele : recip BH)", 1, 0, PRO_A, 5, PRO_B, 6, 3.921875f, 1},
    {"actuel  WH (vraie recip WH)", 1, 1, CUR_A, 7, CUR_B, 8, 10.0f, 0},
    {"propose WH (vraie recip WH)", 1, 1, PRO_A, 5, PRO_B, 6, 3.921875f, 1},
  };
  int nc = sizeof(C) / sizeof(C[0]);
  for (int c = 0; c < nc; c++) {
    double mx = 0, sum = 0; long gt1 = 0, cnt = 0; uint32_t worst = 0;
    #pragma omp parallel
    {
      double lmx = 0, lsum = 0; long lgt = 0, lcnt = 0; uint32_t lw = 0;
      #pragma omp for schedule(static)
      for (long i = 0; i <= 0x7f800000L; i += step) {
        uint32_t xb = (uint32_t)i;
        uint32_t y = eval_rat(C[c].wh, C[c].whr, C[c].A, C[c].n, C[c].B, C[c].d, C[c].xc, C[c].corr, xb);
        double e = ulp_err(xb, y);
        if (e > lmx) { lmx = e; lw = xb; }
        lsum += e; lgt += e > 1.0; lcnt++;
      }
      #pragma omp critical
      { if (lmx > mx) { mx = lmx; worst = lw; } sum += lsum; gt1 += lgt; cnt += lcnt; }
    }
    printf("%-40s max %7.3f ULP (x=%.9g) | > 1 ULP %6.3f %% | moyenne %.3f | %ld entrées\n",
           C[c].nom, mx, as_f(worst), 100.0 * gt1 / cnt, sum / cnt, cnt);
    fflush(stdout);
  }
}
