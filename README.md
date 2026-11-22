# tt-erf-fp32

FP32 `erf` for the Tenstorrent SFPU (Blackhole and Wormhole): fewer instructions and a lower worst-case error than the current tt-metal kernel, checked on **every** positive FP32 input with a bit-exact model of the hardware arithmetic.

Context: tt-metal issue [#58282](https://github.com/tenstorrent/tt-metal/issues/58282) (`ttnn.erf` FP32 accuracy), and [#61160](https://github.com/tenstorrent/tt-metal/issues/61160) (a possible PRECISE tier).
Author: Maher El Hmissi ([Maher75017](https://github.com/Maher75017)), November 2026. Français : [README.fr.md](README.fr.md).

## Results at a glance

Exhaustive evaluation of all 2,139,095,041 positive FP32 inputs (0 to +inf; `erf` is odd), on bit-exact models of both chips.

| Kernel | SFPU instructions per element (estimate) | Max error, Blackhole | Max error, Wormhole | Inputs above 1 ULP (BH / WH) | Mean error |
| --- | --- | --- | --- | --- | --- |
| Current tt-metal (rational n16/d16 on [-10, 10], output clamp) | 64 | 8.000 ULP | 7.000 ULP | 8.97 % / 9.05 % | 0.341 ULP |
| **Proposed default** (`R5/6+corr`: x·N(x²)/D(x²) + one correction step) | **54** | **6.011 ULP** | **6.011 ULP** | 8.75 % / 8.75 % | 0.335 ULP |
| Precise variant (`2M c=1.0 A6/B14`: two polynomials, split at \|x\| = 1) | 80 | **1.000 ULP** | **1.000 ULP** | **0 % / 0 %** | 0.374 ULP |

Other variants (2.2 ULP at 73 instructions, 1.6 ULP at 77, …) are listed in `results/erf_tt_noyau.txt`, with every coefficient in hexadecimal.

## Correction (22 November 2026)

The first version of this page said the Wormhole figures were identical to the Blackhole ones. That was wrong for kernels that divide. The 19 November run modelled Wormhole's SFPMAD, but used the Blackhole reciprocal sequence on both chips; tt-metal's Wormhole `sfpu_reciprocal` is a different algorithm (quadratic seed, two Newton steps, power-of-two scaling). With it modelled step by step (`model/wh_recip.c`, results in `results/wormhole_vraie_reciproque.txt`):

- the current kernel on Wormhole is **7.000 ULP** max with 9.05 % of inputs above 1 ULP (not 8.000 ULP and 8.97 %);
- the proposed kernel is unchanged on Wormhole (6.011 ULP, 8.75 %): its correction step absorbs the difference between the two reciprocals;
- the precise variant does not divide, so its figures were already right for both chips.

A sample of 214,489 inputs puts the current kernel's worst case at 4.5 ULP; the exhaustive run finds 8.0 ULP on Blackhole (at x = 5.8636, where `erf` rounds to 1.0f). Sampling does not find the worst cases.

## Caveats, stated upfront

- **Instruction counts are estimates** (SFPU instructions per element, including constant loads), not measured cycles. Cycles per tile will be measured on ttsim for the tt-metal pull request.
- The gain of the proposed default is mostly in the worst case; its main benefit is about **15 % fewer instructions**.
- The precise variant costs about 25 % more instructions. Given the performance concern in #58282, it is not proposed as the default.
- My own success criterion, fixed before the search (in `erf/erf_tt.py`), was stricter: at most 1 ULP at no extra cost, or at most 2 ULP at no extra cost, or at most 1 ULP for at most 1.2 × the cost. **It was not met.** The results above are what the search found, reported as they are.

## How it was verified

1. **Bit-exact model of the arithmetic.** `model/sfpu_model.h` ports Tenstorrent's reference model of SFPMAD (Blackhole and Wormhole, `model/fma_ref.c` from tt-isa-documentation, unchanged) and the SFPARECIP description. `model/verif_modele.c` compares the port with the reference on 20 million random operand triples, half of them with close exponents to exercise cancellations: **0 mismatches** on Blackhole and on Wormhole. Control: comparing the Wormhole port against the Blackhole reference gives 118,031 mismatches, so the test does detect differences.
2. **Wormhole reciprocal.** `model/wh_recip.c` follows tt-metal's Wormhole `sfpu_reciprocal` instruction by instruction (`ckernel_sfpu_recip.h`, commit a695405). Control (`model/ctrl_recip.c`): on every 7th input from 1 to 2^25, the modelled reciprocal is within 0.892 ULP of 1/x; with one constant corrupted it is off by about 10^8 ULP.
3. **Metric.** Error in ULP against `erf` computed in float64 on the exact input, with the ULP size taken in the binade of the exact value. This is slightly stricter than the metric used in #58282.
4. **Positive control.** The model of the current kernel reproduces the ttsim figures published in #58282, band by band:

| \|x\| band | Mean error, this model | Mean error, #58282 (ttsim) | Above 1 ULP, this model | Above 1 ULP, #58282 |
| --- | --- | --- | --- | --- |
| [0.5, 1.0] | 0.653 | 0.647 | 22.2 % | 22.5 % |
| [1.0, 1.5] | 0.946 | 0.943 | 40.7 % | 40.8 % |
| [1.5, 2.0] | 0.972 | 0.973 | 41.6 % | 41.2 % |
| [2.0, 2.5] | 1.065 | 1.073 | 45.7 % | 45.8 % |
| [2.5, 3.0] | 1.135 | 1.122 | 48.5 % | 49.0 % |
| [3.0, 4.0] | 1.146 | 1.127 | 47.7 % | 46.8 % |

5. **Negative control.** Shifting one coefficient of the precise variant by 256 ULP raises its maximum error to 64.9 ULP.
6. **Saturation.** `erf(x)` rounds to 1.0f from x = 3.9192059; the inputs are clamped at 3.921875, exactly representable in bf16.

## Proof of date

The SHA-256 of `results/erf_tt_noyau.txt` is
`7f33cf4e3fcfecf4593419aa21735197aa21e6b0edf4b0c9a7ed905188fbac01`.
The same fingerprint was sent to Tenstorrent on 19 November 2026, with the proposal for #58282. The file is published unchanged.

## Reproduce

```sh
# 1. Model check (any machine with gcc), about one second
cd model && gcc -O2 -o verif_modele verif_modele.c fma_ref.c -lm && ./verif_modele 20000000

# 2. Full pipeline: positive control, coefficient fitting and bit-exact polishing,
#    exhaustive check of 2.1 billion inputs on both models, negative control.
#    Python 3 with numpy; an NVIDIA GPU with CUDA is strongly recommended
#    (47 minutes on an RTX 5090). --cpu uses gcc + OpenMP instead (much slower);
#    --rapide runs a short test.
cd ../erf && python3 erf_tt.py --minutes 40

# 3. Wormhole with its true reciprocal (gcc + OpenMP): control in a few seconds,
#    exhaustive run of the 6 cases in about 25 minutes on 2 cores. ./wh_recip 97 samples every 97th input.
cd ../model && gcc -O2 -o ctrl_recip ctrl_recip.c -lm && ./ctrl_recip
gcc -O3 -march=native -fopenmp -o wh_recip wh_recip.c -lm && ./wh_recip 1
```

The log of the reference run is `results/erf_tt.log`; all figures and coefficients are in `results/erf_tt_resultats.json`. The script's messages are in French.

## Next step

A pull request against tt-metal, referencing #58282: the kernel change for Blackhole and Wormhole, tests, the accuracy-budget entries, and cycles per tile measured on ttsim rather than estimated. Note: tt-llk's Float32 sweeps currently compile the bf16 path of `erf` ([#56492](https://github.com/tenstorrent/tt-metal/issues/56492)), so the pull request's measurements will define `INP_FLOAT32`.

## Layout

```
model/fma_ref.c        Tenstorrent's reference SFPMAD models (tt-isa-documentation, Apache-2.0, unchanged)
model/sfpu_model.h     bit-exact C/CUDA port: SFPMAD (Blackhole, Wormhole), SFPARECIP, erf evaluation, ULP metric
model/verif_modele.c   port vs reference on 20 million random triples
model/wh_recip.c       exhaustive check with the true Wormhole reciprocal (designs.h: coefficients)
model/ctrl_recip.c     control of the modelled Wormhole reciprocal against 1/x
erf/erf_tt.py          full pipeline (control, search and polishing, exhaustive check)
results/               reference run: log, JSON results, coefficient file; wormhole_vraie_reciproque.txt
SHA256SUMS             fingerprints of every file
```

See also [exact-exp32](https://github.com/Maher75017/exact-exp32): correctly rounded exponentials checked on all 2^32 inputs, with the same method.

## License

Apache License 2.0 (see `LICENSE`), like tt-metal and tt-isa-documentation. `model/fma_ref.c` is © Tenstorrent AI ULC and is included unchanged; `model/sfpu_model.h` is a port of it and keeps the attribution.
