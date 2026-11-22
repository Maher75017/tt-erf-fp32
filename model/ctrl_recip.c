// Contrôle : la réciproque Wormhole modélisée doit être proche de 1/x (quelques ULP au plus).
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "sfpu_model.h"
static uint32_t k0, k1, k2;
static inline uint32_t recip_wh(uint32_t in) {
  uint32_t negx = 0xBF800000u | (in & 0x007FFFFFu);
  uint32_t y = fma_wh(k0, negx, k1);
  uint32_t sb = ~in;
  y = fma_wh(y, negx, k2);
  uint32_t scale = sb & 0xFF800000u;
  uint32_t t = fma_wh(negx, y, F_ONE);
  scale = fma_wh(scale, 0x3F000000u, F_PZ);
  y = fma_wh(y, t, y);
  t = fma_wh(negx, y, F_ONE);
  y = fma_wh(y, t, y);
  y = fma_wh(y, scale, F_PZ);
  return (y & 0x7FFFFFFFu) | (in & 0x80000000u);
}
int main(void) {
  float f0 = 0.3232325017452239990234375f, f1 = 1.4545459747314453125f, f2 = 2.121212482452392578125f;
  k0 = as_u(f0); k1 = as_u(f1); k2 = as_u(f2);
  double mw = 0, mb = 0; uint32_t ww = 0;
  for (uint32_t xb = 0x3F800000u; xb < 0x4C000000u; xb += 7) {   // x de 1 à 2^25
    double ex = 1.0 / (double)as_f(xb);
    float fe = (float)ex; double u = nextafterf(fe, INFINITY) - (double)fe;
    double ew = fabs((double)as_f(recip_wh(xb)) - ex) / u;
    double eb = fabs((double)as_f(recip_seq(0, xb)) - ex) / u;
    if (ew > mw) { mw = ew; ww = xb; } if (eb > mb) mb = eb;
  }
  printf("réciproque WH modélisée : max %.3f ULP (x=%.9g) | réciproque BH : max %.3f ULP\n", mw, as_f(ww), mb);
  // contrôle négatif : K1 décalé d'un bit de poids fort -> doit être nettement pire
  k1 ^= 0x00400000u; double mn = 0;
  for (uint32_t xb = 0x3F800000u; xb < 0x4C000000u; xb += 7) {
    double ex = 1.0 / (double)as_f(xb); float fe = (float)ex; double u = nextafterf(fe, INFINITY) - (double)fe;
    double e = fabs((double)as_f(recip_wh(xb)) - ex) / u; if (e > mn) mn = e;
  }
  printf("contrôle négatif (K1 faussé) : max %.1f ULP\n", mn);
}
