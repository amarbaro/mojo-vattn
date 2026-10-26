# API

`from vattn import vattn_fwd, f16, BM, THREADS, D`

## vattn_fwd

```
vattn_fwd(Q, K, V, O: MutPointer[Scalar[f16]], seq: Int32, heads: Int32, scale_log2: Float32)
```

GPU kernel, launched with `ctx.enqueue_function[vattn_fwd]`.

| Argument | Meaning |
|---|---|
| `Q`, `K`, `V` | f16, `[seq, heads, 128]`, contiguous |
| `O` | f16 output, same layout |
| `seq` | sequence length L, any positive value |
| `heads` | H |
| `scale_log2` | `log2(e) / sqrt(128)`, folded into Q |

Launch: `grid_dim=(ceildiv(L, BM), H)`, `block_dim=THREADS`. Softmax is non-causal, no mask,
f32 accumulate; output is `softmax(Q K^T / sqrt(128)) V`.

## Constants

| Name | Value |
|---|---|
| `D` | 128 |
| `BM` | queries per block, `-D VATTN_BM=`, default 64 |
| `BN` | keys per tile, `-D VATTN_BN=`, default 32 |
| `THREADS` | `BM / 16 * 32` |

## Checked by

`scripts/gpu-gates.sh` (fp64 reference, max abs error under 2e-3, no NaN or inf) and
`examples/quickstart.mojo` (uniform attention over V = 1 returns 1).
