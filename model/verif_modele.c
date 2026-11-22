// SPDX-License-Identifier: Apache-2.0
// (c) 2026 Maher El Hmissi
// Vérifie que le modèle SFPMAD (sfpu_model.h) reproduit bit à bit le modèle de référence de Tenstorrent
// (fma_ref.c, tiré de tt-isa-documentation/Miscellaneous/FMA/fma.c, Apache-2.0), Blackhole et Wormhole.
// Opérandes : bits tirés au hasard (xorshift), avec une part d'exposants proches pour exercer les annulations.
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "sfpu_model.h"
uint32_t fma_model_bh(uint32_t, uint32_t, uint32_t);
uint32_t fma_model_wh(uint32_t, uint32_t, uint32_t);
static uint64_t s = 0x9E3779B97F4A7C15ull;
static uint32_t r32(void) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (uint32_t)(s >> 16); }
int main(int argc, char** argv) {
  long n = argc > 1 ? atol(argv[1]) : 20000000L, ebh = 0, ewh = 0;
  for (long i = 0; i < n; i++) {
    uint32_t x = r32(), y = r32(), z = r32();
    if (i & 1) {  // une moitié : exposants rapprochés (annulations, arrondis délicats)
      uint32_t e = 100 + (r32() % 56);
      x = (x & 0x807fffffu) | (e << 23); y = (y & 0x807fffffu) | ((254 - e + (r32() % 5) - 2 + 0) << 23 & 0x7f800000u);
      z = (z & 0x807fffffu) | (((r32() % 9) + 123) << 23);
    }
    if (fma_bh(x, y, z) != fma_model_bh(x, y, z)) ebh++;
    if (fma_wh(x, y, z) != fma_model_wh(x, y, z)) ewh++;
  }
  printf("triplets testés : %ld | écarts Blackhole : %ld | écarts Wormhole : %ld\n", n, ebh, ewh);
  return (ebh || ewh) ? 1 : 0;
}
