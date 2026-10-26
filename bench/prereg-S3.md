# Preregistration S3: vattn_fwd speed at the real Wan shape (frozen before any real-shape timing)

Shape: L=32,760, H=40, D=128, B=1, f16, non-causal. 2.198e13 FLOP per call.
Reference arm: torch SDPA (AOTriton `attn_fwd`), called as ComfyUI calls it (`tools/sdpa-bench.py`).
Known from the 2026-10-06 step profile: AOTriton ~0.885 s per call (35.4 s / 40), ~25 TFLOPS.
Roof: 125.4 TFLOPS measured WMMA f16 (mojo-baro `wmma_peak`).

## Predictions

| arm | predicted | falsifier |
|---|---|---|
| AOTriton on the dump (`sdpa-bench.py`) | 0.80-0.95 s/call (matches the profile) | outside -> the bench does not reproduce ComfyUI's call; fix the bench before any comparison |
| S2 kernel as committed (091f8b4: BM 64, BN 32, single LDS buffer, 208 VGPR) | 15-35 TFLOPS (0.63-1.47 s/call) | > 35 -> the knob sweep is optional; < 15 -> look at LDS bank conflicts and the V^T scatter store first |
| best swept config (BM {64,128} x BN {32,64} x LDS double buffer) | >= 45 TFLOPS | < 45 after the sweep -> stop sweeping, read isa-loops of the inner loop |

## Adoption rule (S3 -> S4)

Ship a config only if, in ONE gpu-wait job with gates first (fail-closed):
1. parity (`bench/parity.sh`, synthetic cases + the real dump) PASS, real dump max abs err <= 1.5x SDPA's own max err vs the f32 reference;
2. device time <= 1/1.8 of AOTriton's (>= 1.8x), 5 repeats alternating arm order, spread <= 10 % per arm;
3. ISA receipt: 0 spills.
The sweep picks the candidate; only the confirmation table decides.
