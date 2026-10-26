# mojo-vattn

A flash-attention forward kernel in Mojo for the self-attention of video diffusion transformers
(Wan 2.2), written for the AMD RX 7900 XTX (gfx1100, RDNA3 WMMA). Non-causal, no mask, f16 in and
out, f32 accumulate, head dimension 128, layout `[L, H, D]`. It is a kernel and a parity harness,
not a drop-in attention backend: it is not wired into any framework, and it is not faster than
PyTorch's SDPA.

```mojo
from vattn import vattn_fwd, BM, THREADS

# q, k, v, o: device buffers of f16, L * H * D elements each
ctx.enqueue_function[vattn_fwd](
    q.unsafe_ptr(), k.unsafe_ptr(), v.unsafe_ptr(), o.unsafe_ptr(),
    Int32(L), Int32(H), scale_log2,          # scale_log2 = log2(e) / sqrt(128)
    grid_dim=(ceildiv(L, BM), H), block_dim=THREADS,
)
```

## Install

Requires Mojo through the `max` 26.6.0 wheel (`uv sync`). Building needs no GPU; running does.

| Way | Command |
|---|---|
| Source path | `mojo build -I path/to/mojo-vattn/src app.mojo` |
| Vendor | copy `src/vattn.mojo` into your project |

`scripts/ci.sh` builds the harness (default tiles and a BM 128 / BN 64 variant) and the example.
It is the CPU check and runs on GitHub: without a GPU it builds everything and does not run the example.

## What you get

- `vattn_fwd`: one kernel, 16 queries per wave, WMMA 16x16x16, online softmax in f32 with
  `exp2`, K tile and V^T tile staged in LDS. Tail rows and keys past `L` are masked, so any `L`
  works. Output goes straight to `[L, H, D]` with no transposes.
- Comptime tile knobs `VATTN_BM` (default 64) and `VATTN_BN` (default 32), passed as
  `mojo build -D VATTN_BM=128`.
- `tests/run_vattn.mojo`: harness that loads `q.bin k.bin v.bin`, runs the kernel, writes
  `o_mojo.bin`, and times `ITERS` launches.
- `tools/vattn-ref.py`: fp64 reference over the same f16 bytes (`gen`, `check`).

The API is in [docs/api.md](docs/api.md). The design and the stage plan are in
[docs/plan.md](docs/plan.md).

## Performance

There is no speed claim in this release. The repo holds the preregistered protocol
(`bench/prereg-S3.md`, `bench/arms.sh`) for timing against PyTorch SDPA, but no timing receipt.
Context, from `docs/plan.md`: in one Wan 2.2 denoise step, AOTriton attention took 35.4 s of
47.3 s (75 %), at about 20 % of the measured 125.4 TFLOPS WMMA roof. Tuned variants of this
kernel tied or lost to PyTorch SDPA on that shape; those runs are not part of this tree.

Correctness: `scripts/gpu-gates.sh` compares the kernel with the fp64 reference on three
generated cases (L = 129, 1000, 4097) and fails above a max abs error of 2e-3. Run it locally on
gfx1100; GitHub runners have no GPU.

## Limits

- Verified on one GPU only: gfx1100 (RX 7900 XTX). Other GPUs are untested.
- head_dim 128, f16, non-causal, no mask, batch 1. Anything else is not supported.
- Not faster than PyTorch SDPA; no framework integration (no ComfyUI node, no bridge).
- `scripts/ci.sh` checks that everything builds, not that it computes the right answer; the
  parity gate is local.
- `bench/arms.sh` and `tools/sdpa-*.py` need a PyTorch ROCm environment, set by `$HOME/...` path
  in `bench/arms.sh`; they are the author's local harness.
