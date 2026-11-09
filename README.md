# mojo-vattn

<!-- badges -->
[![ci](https://github.com/amarbaro/mojo-vattn/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/amarbaro/mojo-vattn/actions/workflows/ci.yml) [![latest release](https://img.shields.io/github/v/release/amarbaro/mojo-vattn)](https://github.com/amarbaro/mojo-vattn/releases/latest) [![license](https://img.shields.io/github/license/amarbaro/mojo-vattn)](https://github.com/amarbaro/mojo-vattn/blob/main/LICENSE)
<!-- /badges -->

A flash-attention forward kernel in Mojo for the self-attention of video diffusion transformers
(Wan 2.2), written for the AMD RX 7900 XTX (gfx1100, RDNA3 WMMA). Non-causal, no mask, f16 in and
out, f32 accumulate, head dimension 128, layout `[L, H, D]`. It is a kernel and a parity harness,
not a drop-in attention backend: it is not wired into any framework, and it is not faster than
PyTorch's SDPA.

Install: `pixi workspace channel add https://conda.modular.com/max https://prefix.dev/amarbaro/amarbaro`, then `pixi add mojo-vattn` (import `vattn`). From source: `git clone https://github.com/amarbaro/mojo-vattn`, then `mojo run -I mojo-vattn/src app.mojo`.
Toolchains: Mojo 1.2 stable, and the nightly 1.3.0.dev2026110805 in the weekly [mojo-libs](https://github.com/amarbaro/mojo-libs) matrix (36 libraries on stable, 35 on nightly).

```mojo
from max.gpu.host import DeviceContext
from std.math import ceildiv, sqrt
from std.sys import has_accelerator
from vattn import vattn_fwd, f16, BM, THREADS, D


def main() raises:
    comptime L = 64
    comptime H = 1
    comptime N = L * H * D
    comptime if not has_accelerator():
        print("SKIP quickstart: no GPU, kernel compiled only")
        return
    var ctx = DeviceContext()
    var q = ctx.enqueue_create_buffer[f16](N)
    var k = ctx.enqueue_create_buffer[f16](N)
    var v = ctx.enqueue_create_buffer[f16](N)
    var o = ctx.enqueue_create_buffer[f16](N)
    q.enqueue_fill(0)
    k.enqueue_fill(0)
    v.enqueue_fill(1)
    var scale_log2 = Float32(1.4426950408889634) / sqrt(Float32(D))
    ctx.enqueue_function[vattn_fwd](
        q.unsafe_ptr(), k.unsafe_ptr(), v.unsafe_ptr(), o.unsafe_ptr(), Int32(L), Int32(H), scale_log2,
        grid_dim=(ceildiv(L, BM), H), block_dim=THREADS,
    )
    with o.map_to_host() as h:
        print("o[0] =", h[0], "(uniform attention over V = 1, expect 1)")
```

## Install

Requires Mojo through the `max` 26.7.0 wheel (`uv sync`). Building needs no GPU; running does.

| Way | Command |
|---|---|
| Source path | `mojo build -I path/to/mojo-vattn/src app.mojo` |
| Vendor | copy `src/vattn/` into your project |

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

The API is in [docs/api.md](docs/api.md). The design is in [docs/design.md](docs/design.md); the stage plan is in
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
