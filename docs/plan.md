# Plan: Mojo flash-attention kernel for Wan 2.2 i2v on gfx1100

## Context

Profile of one Wan 2.2 i2v denoise step (832x480, 81 frames, 2026-10-06, Brain note
`gpu-media/ComfyUI/notes/2026-10-06-wan-i2v-kernel-memory-investigation.md`):

| kernel | s/step | share | rate |
|---|---|---|---|
| AOTriton `attn_fwd` (torch SDPA flash) | 35.4 | 74.9 % | ~25 TFLOPS, ~20 % of the 125.4 TFLOPS measured WMMA roof |
| GEMM (hipBLASLt Cijk) | 11.1 | 23.6 % | ~83 TFLOPS, ~67 % |
| everything else | 0.8 | 1.6 % | |

Attention is self-attention, non-causal, no mask, B=1 (cfg 1), H=40, D=128, L=32,760 (1560 tokens x 21
latent frames). 2.2e13 FLOP per call, 40 calls per step, 4 steps per clip.
Target: attention at >= 60 % of roof, so 35.4 s drops to <= 12 s per step, and the clip goes from ~240 s
to ~145 s at full quality. Outcome = faster Wan clips for every video pipeline (doar.bio template,
fly-connectome story videos), at unchanged quality.

## What exists (reuse, do not rewrite)

- `~/Projects/mojo/mojo-baro/kernels/attn.mojo` `amar_attn_prefill_wmma` (~L510-660): Mojo WMMA prefill
  flash attention. S^T = K.Q^T with 2 accumulator chains, lane-owns-query softmax (xor-16 shuffle),
  P^T rebuilt in registers (no LDS round trip), V^T in LDS, f16 in / f32 acc. Causal, HD=256,
  paged KV, 16-key tiles, 28 spills. This is the template.
- `kernels/matmul_wmma_pipe.mojo`: LDS double buffer, 2-deep register prefetch, XOR swizzle, 16-B
  vector loads, `rocdl.flat_work_group_size` metadata for the 256-VGPR cap. This is the K/V tile loader.
- `docs/BASELINE.md` L307-330: gfx1100 WMMA fragment lane layout (use `mma()` directly; `TensorCore.load_a` fails on RDNA).
- `kernels/wmma_peak.mojo` + `bench/wmma-peak.sh`: roof 125.4 TFLOPS.
- Tools: `tools/isa-receipt.py` (VGPR/LDS/spills), `~/iTools/bin/isa-loops`, `~/iTools/bin/rocprof-kernels`,
  `bench/build-variant.sh` (comptime knob sweeps), `bench/PROTOCOL-RULES.md` (prereg, arm read-back).
- ComfyUI hook: `ModelPatcher.set_model_optimized_attention` / `transformer_options["optimized_attention_override"]`
  (`comfy/model_patcher.py:688`, `comfy/ldm/modules/attention.py:171-204` `wrap_attn`); precedent
  `comfy_extras/nodes_sparse_attention.py:195-225`. Wan call site `comfy/ldm/wan/model.py:84-90`,
  q/k/v arrive flat `[B, L, H*D]`, contiguous, rope applied.
- Kernel-loading precedent: `comfy_kitchen` `_C.abi3.so` via DLPack capsule + int stream
  (`.venv/.../comfy_kitchen/backends/hip/__init__.py:107-134, 237-291`).
- Telemetry + profiler already landed: `comfy-run --telemetry` (iTools 072c063),
  `ComfyUI/custom_nodes/step_profiler.py` (`COMFY_PROFILE_STEP`).

Skills that govern the work: `inference-kernels`, `kernel-parity`, `kernel-arm-round`,
`persistent-kernel-gfx11`, `optimize-for-rocm`, `llm-benchmark-method`, `memory-optimization`.

## Design

### Kernel `vattn_fwd` (Mojo 1.0, new repo `~/Projects/mojo/mojo-vattn`, `-I` mojo-baro `kernels/`)

```
grid  = (ceil(L / BM), H)     head-major order: all q-blocks of one head resident together,
                              so that head's K+V (16.8 MB) stays in the 96 MB Infinity Cache
block = BM/16 waves (wave32); each wave owns 16 queries
per wave regs: Q frags 16x128 f16 (pre-scaled by log2e/sqrt(128)) | O acc 16x128 f32 | S 16xBN f32
loop over key tiles of BN:
   LDS K tile [BN,128] + V^T tile [128,BN], double-buffered, XOR-swizzled, 16-B loads (from matmul_wmma_pipe)
   S^T = K.Q^T (WMMA 16x16x16, 8 k-steps over D)
   online softmax in f32 with exp2: m_new, alpha = exp2(m_old - m_new), l = l*alpha + rowsum(P)
   P -> f16 in registers (PS scale knob, default 1), O^T = V^T.P^T + O^T*alpha
tail tile: keys >= L masked to -inf (32,760 = 64*511 + 56)
epilogue: O / l, write f16 to [B, L, H, D] (same flat layout Wan consumes; no transposes)
```

Memory-optimization notes (name the hot path, measure bytes): no transpose copies on input or output
(torch path makes `[B,H,L,D]` strided views; measure whether SDPA copies them); K/V re-read per q-block
is served from Infinity Cache only with head-major grid order (receipt: rocprof FETCH_SIZE or
effective GB/s vs roof); BM=128 halves K/V traffic vs BM=64 at a VGPR cost.

Sweep knobs (comptime, `build-variant.sh`): BM {64,128}, BN {32,64}, waves {4,8}, LDS double buffer on/off,
PS {1, 2^15}, flat_work_group_size. LDS budget <= 32 KB/block for 2 blocks per WGP (64 KB limit).
Every instantiation gets an `isa-receipt` (VGPR, scratch, LDS) BEFORE timing; spills = reject.

### Python bridge (decided by stage S0)

Route A: `max.experimental.torch.CustomOpLibrary` (needs `max` wheel for the ComfyUI venv's Python 3.13).
Route B: `mojo build --emit shared-lib` + `PythonModuleBuilder` (pattern: `~/iTools/llm/gguf-sweep/`),
taking `data_ptr()` ints + torch's current HIP stream; if Mojo cannot adopt torch's stream,
`torch.cuda.current_stream().synchronize()` around the call (80 calls per step: overhead budget < 1 ms each).
Custom node `ComfyUI/custom_nodes/mojo_vattn/`: installs the override on the Wan model only; gate
`q.shape[1] == k.shape[1] >= 8192 and mask is None and dim_head == 128 and dtype fp16/bf16`,
else delegate to the original function (cross-attention stays on SDPA). Env `VATTN=0` disables.

## Stages (each: preregistered prediction + falsifier in `bench/prereg-<stage>.md`, committed before running)

```
S-1 cheap alternative ─┐
S0 bridge spike ───────┼─► S2 correct kernel ─► S3 tuned kernel ─► S4 ComfyUI node ─► S5 clip gate
S1 real-data harness ──┘
```

| stage | what | gate | size |
|---|---|---|---|
| S-1 | One clip with `--use-ck-attention` (comfy_kitchen int8 attention, already on gfx1100) | if >= 1.8x and frames pass the S5 quality gate, report and ask whether to stop here | XS, 1 GPU job |
| S0 | Mojo vector-add GPU kernel called from the ComfyUI venv on torch tensors, routes A then B | output exact, runs on torch's stream or a measured sync cost < 1 ms | S |
| S1 | Dump one real self-attention call (layer 20, step 3) q,k,v,o as f16 `.npy` via the custom-node hook; fp32 reference O via torch math SDPA; AOTriton error vs fp32 recorded as the band | files + band numbers in `bench/` | S |
| S2 | Port `amar_attn_prefill_wmma` to HD=128, non-causal, contiguous [L,H,D], tail mask, BM=64 BN=32 | max abs/rel error vs fp32 reference <= 1.5x AOTriton's own error, no NaN/inf, on the real dump + 3 random shapes incl. L not a multiple of BM | M |
| S3 | Knob sweep (table above), head-major grid, then double buffer + swizzle | prereg: >= 1.8x AOTriton device time at the real shape (one process, both arms, rocprof-kernels, warmup excluded, 20 timed calls); parity from S2 still green | L |
| S4 | Custom node + bridge, step profile rerun | `attn_fwd` gone from profile, `vattn_fwd` time matches S3 within 10 % | S |
| S5 | Same-seed clip c1: AOTriton vs Mojo, plus a control arm torch math SDPA vs AOTriton | Mojo-vs-AOTriton frame PSNR/SSIM within the math-vs-AOTriton band; per-step wall from `--telemetry`; frame strip looked at; user judges the clip | S |

Execution substrate: herdr lanes per CLAUDE.md section 4a. S0 + S1 in parallel (sonnet, effort high), S2-S3 one opus
lane (effort high, judgment-heavy kernel work), S4-S5 by me. Every GPU job through `gpu-wait`, `--timeout` each,
S3 sweeps on subsets first, `gpu-wait stats --days 1` in lane reports.

## Files

- New: `~/Projects/mojo/mojo-vattn/` (`vattn.mojo`, `test_vattn.mojo`, `bench/`, `tools/vattn-ref.py`, `docs/`, README)
- New: `~/Projects/imports/ComfyUI/custom_nodes/mojo_vattn/` (override node + bridge loader)
- Reused read-only: mojo-baro `kernels/attn.mojo`, `kernels/matmul_wmma_pipe.mojo`, `docs/BASELINE.md`, `tools/isa-receipt.py`
- Brain: `~/Brain/mojo-vattn/` (whiteboard, prereg results), ledger entries for misses

## Verification (end to end)

1. S2 parity test `mojo run test_vattn.mojo` under gpu-wait: error table vs fp32 reference and AOTriton band.
2. S3 bench: rocprof-kernels device time per call, both arms in one process, VGPR/spill receipts.
3. S4: `COMFY_PROFILE_STEP=3` profile shows `vattn_fwd` replacing `attn_fwd`.
4. S5: `comfy-run --telemetry` same-seed clip, per-step wall before/after, PSNR/SSIM vs control band,
   frame strip opened, clip shown to the user. Done only when the user has seen the clip.
