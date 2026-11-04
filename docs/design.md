# Design

`vattn_fwd` is a fixed-shape kernel: every dimension Wan 2.2 self-attention needs is a compile-time
constant, and everything else is out of scope. That trade buys fully unrolled inner loops and no
shape branches. It has not bought speed: tuned variants tied or lost to PyTorch SDPA on the target
shape (README, Performance). The full plan, including unshipped stages, is in [plan.md](plan.md).

## The shape is fixed at compile time

Non-causal, no mask, batch 1, head dimension 128 (`comptime D = 128`), f16 in and out, f32
accumulate, layout `[L, H, D]`. Only `L` and `H` are runtime values. Loops over D and the tiles are
`comptime for`.

## Head-major grid keeps K and V in Infinity Cache

- Grid `(ceil(L / BM), H)`: all query blocks of one head run together, so that head's K and V are
  re-read from the 96 MB Infinity Cache, not from VRAM.
- Block: `BM / 16` waves of 32 lanes; each wave owns 16 query rows. `BM` (default 64) and `BN`
  (default 32) are comptime knobs: `-D VATTN_BM=128 -D VATTN_BN=64`.
- `flat_work_group_size = THREADS` via `@__llvm_metadata`, so register allocation sees the real
  block size.

## One key tile per step, softmax online in f32

1. K tile `[BN, 128]` and V^T tile `[128, BN]` go to LDS with 16-byte loads, then `barrier()`.
   Rows are padded by 8 elements (`KS = D + 8`, `VS = BN + 8`) to spread them across LDS banks:
   18,944 bytes per block at the defaults.
2. `S = Q K^T` with WMMA 16x16x16 (`layout.tensor_core.mma`), 8 steps over D.
3. Q carries the scale `log2(e) / sqrt(128)`, so the softmax uses `exp2`:
   `m_new = max(m, rowmax S)`, `alpha = exp2(m - m_new)`, `l = l * alpha + rowsum P`.
4. `O = O * alpha + P V` with WMMA; P becomes f16 in registers, no LDS round trip.

Keys at or past `L` are set to `-3e38` before the softmax and rows past `L` are not written, so
any `L` works. The epilogue divides by `l` and writes f16 straight to `[L, H, D]`: no transposes.

## Not shipped

- LDS double buffering and XOR swizzle from `plan.md`: the release uses one buffer and padding.
- A Python or ComfyUI bridge: no payoff while SDPA is as fast or faster.
- Causal masks, other head sizes, batch > 1.

## How to prove it wrong

`scripts/gpu-gates.sh` compares against an fp64 reference (`tools/vattn-ref.py`) at L = 129, 1000
and 4097 and fails above a max abs error of 2e-3. It needs gfx1100; GitHub CI has no GPU and only
builds (`scripts/ci.sh`). Other GPUs are untested.
