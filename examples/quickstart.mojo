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
