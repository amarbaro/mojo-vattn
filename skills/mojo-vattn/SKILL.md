---
name: mojo-vattn
description: "Flash-attention forward kernel in Mojo for gfx1100 (RX 7900 XTX): non-causal, f16, head_dim 128, [L, H, D] layout. Use when you need a reference WMMA attention kernel or parity harness for RDNA3 in Mojo, not as a production attention backend."
---

# mojo-vattn

One Mojo WMMA flash-attention forward kernel for gfx1100 with an fp64 parity harness.

## Install

`uv sync`, then `mojo build -I <repo>/src app.mojo`, or copy `src/vattn.mojo`.

## Use

```mojo
from vattn import vattn_fwd, BM, THREADS

ctx.enqueue_function[vattn_fwd](
    q.unsafe_ptr(), k.unsafe_ptr(), v.unsafe_ptr(), o.unsafe_ptr(),
    Int32(L), Int32(H), scale_log2,
    grid_dim=(ceildiv(L, BM), H), block_dim=THREADS,
)
```

## Rules

- f16 `[L, H, 128]`, contiguous, non-causal, no mask. Other shapes or dtypes are unsupported.
- `scale_log2 = log2(e) / sqrt(128)`.
- gfx1100 only. Run `scripts/gpu-gates.sh` after any kernel change; `scripts/ci.sh` only builds.
- Do not claim a speedup: the kernel does not beat PyTorch SDPA.
- Tile sizes are comptime: `-D VATTN_BM=`, `-D VATTN_BN=`. Check ISA for VGPR and scratch spills
  before timing a new pair.
