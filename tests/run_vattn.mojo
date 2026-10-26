from max.gpu.host import DeviceContext, DeviceBuffer
from std.math import ceildiv, sqrt
from std.memory import unsafe_memcpy
from std.sys import argv
from std.time import perf_counter_ns
from vattn import vattn_fwd, f16, BM, THREADS, D


def load(ctx: DeviceContext, path: String, n: Int) raises -> DeviceBuffer[f16]:
    var b: List[UInt8]
    with open(path, "r") as f:
        b = f.read_bytes()
    if len(b) != n * 2:
        raise Error(String("FAIL load: ") + path + " has " + String(len(b)) + " bytes, want " + String(n * 2))
    var h = ctx.enqueue_create_host_buffer[f16](n)
    ctx.synchronize()
    unsafe_memcpy(dest=h.unsafe_ptr().unsafe_bitcast[UInt8](), src=b.unsafe_ptr(), count=n * 2)
    var d = ctx.enqueue_create_buffer[f16](n)
    ctx.enqueue_copy(dst_buf=d, src_buf=h)
    return d


def launch(
    ctx: DeviceContext, mut q: DeviceBuffer[f16], mut k: DeviceBuffer[f16], mut v: DeviceBuffer[f16], mut o: DeviceBuffer[f16],
    L: Int, H: Int, scale_log2: Float32,
) raises:
    ctx.enqueue_function[vattn_fwd](
        q.unsafe_ptr(), k.unsafe_ptr(), v.unsafe_ptr(), o.unsafe_ptr(), Int32(L), Int32(H), scale_log2,
        grid_dim=(ceildiv(L, BM), H), block_dim=THREADS,
    )


def main() raises:
    var a = argv()
    if len(a) != 5:
        raise Error("usage: run_vattn DIR L H ITERS")
    var dir = String(a[1])
    var L = Int(a[2])
    var H = Int(a[3])
    var iters = Int(a[4])
    var n = L * H * D
    var ctx = DeviceContext()
    var q = load(ctx, dir + "/q.bin", n)
    var k = load(ctx, dir + "/k.bin", n)
    var v = load(ctx, dir + "/v.bin", n)
    var o = ctx.enqueue_create_buffer[f16](n)
    ctx.enqueue_memset(o, 0)
    var scale_log2 = Float32(1.4426950408889634) / sqrt(Float32(D))
    launch(ctx, q, k, v, o, L, H, scale_log2)
    ctx.synchronize()
    var oh = ctx.enqueue_create_host_buffer[f16](n)
    ctx.enqueue_copy(dst_buf=oh, src_buf=o)
    ctx.synchronize()
    var out = List[UInt8](length=n * 2, fill=0)
    unsafe_memcpy(dest=out.unsafe_ptr(), src=oh.unsafe_ptr().unsafe_bitcast[UInt8](), count=n * 2)
    with open(dir + "/o_mojo.bin", "w") as f:
        f.write_bytes(Span(out))
    if iters > 0:
        for _ in range(2):
            launch(ctx, q, k, v, o, L, H, scale_log2)
        ctx.synchronize()
        var t = perf_counter_ns()
        for _ in range(iters):
            launch(ctx, q, k, v, o, L, H, scale_log2)
        ctx.synchronize()
        var ms = Float64(perf_counter_ns() - t) / 1e6 / Float64(iters)
        var tflops = 4.0 * Float64(L) * Float64(L) * Float64(D) * Float64(H) / (ms * 1e-3) / 1e12
        print("vattn_fwd L=", L, " H=", H, " ms/call=", ms, " TFLOPS=", tflops)
