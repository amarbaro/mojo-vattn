from max.gpu import block_idx, thread_idx, WARP_SIZE
from max.gpu.primitives import warp
from std.math import exp2
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from layout import TileTensor, row_major, stack_allocation
from layout.tensor_core import mma
from std.utils import StaticTuple
from std.sys import get_defined_int

comptime f16 = DType.float16
comptime f32 = DType.float32
comptime D = 128
comptime BM = get_defined_int["VATTN_BM", 64]()
comptime BN = get_defined_int["VATTN_BN", 32]()
comptime WAVES = BM // 16
comptime THREADS = WAVES * WARP_SIZE
comptime KS = D + 8
comptime VS = BN + 8
comptime NEG = Float32(-3.0e38)


@__llvm_metadata(`rocdl.flat_work_group_size`=StaticTuple[Int32, 1](Int32(THREADS)))
def vattn_fwd(
    Q: MutPointer[Scalar[f16], MutAnyOrigin],
    K: MutPointer[Scalar[f16], MutAnyOrigin],
    V: MutPointer[Scalar[f16], MutAnyOrigin],
    O: MutPointer[Scalar[f16], MutAnyOrigin],
    seq: Int32,
    heads: Int32,
    scale_log2: Float32,
):
    var L = Int(seq)
    var H = Int(heads)
    var head = Int(block_idx.y)
    var tid = Int(thread_idx.x)
    var wave = tid // WARP_SIZE
    var lane = tid % WARP_SIZE
    var h = lane % 16
    var half = lane // 16
    var row = Int(block_idx.x) * BM + wave * 16 + h
    var valid = row < L

    var ks = stack_allocation[f16, address_space = AddressSpace.SHARED](row_major[BN, KS]())
    var vt = stack_allocation[f16, address_space = AddressSpace.SHARED](row_major[D, VS]())
    var ksv = ks.vectorize[1, 8]()
    var vtv = vt.vectorize[1, 8]()

    var qf = Array[SIMD[f16, 16], D // 16](fill=SIMD[f16, 16](0))
    if valid:
        var Qr = TileTensor(Q.unsafe_offset((row * H + head) * D), row_major[D]()).vectorize[8]()
        comptime for c in range(D // 16):
            qf[c] = rebind[SIMD[f16, 8]](Qr[2 * c]).join(rebind[SIMD[f16, 8]](Qr[2 * c + 1]))

    var m_run = NEG
    var l_run = Float32(0)
    var acc = Array[SIMD[f32, 8], D // 16](fill=SIMD[f32, 8](0))
    var t0 = 0
    while t0 < L:
        barrier()
        comptime for s in range(BN * D // 8 // THREADS):
            var v = tid + s * THREADS
            var key = v // (D // 8)
            var c = v % (D // 8)
            var p = t0 + key
            var x = SIMD[f16, 8](0)
            if p < L:
                var Kr = TileTensor(K.unsafe_offset((p * H + head) * D), row_major[D]()).vectorize[8]()
                x = rebind[SIMD[f16, 8]](Kr[c])
            ksv[key, c] = rebind[ksv.ElementType](x)
        comptime for s in range(BN * D // 8 // THREADS):
            var v = tid + s * THREADS
            var key = v % BN
            var c = v // BN
            var p = t0 + key
            var x = SIMD[f16, 8](0)
            if p < L:
                var Vr = TileTensor(V.unsafe_offset((p * H + head) * D), row_major[D]()).vectorize[8]()
                x = rebind[SIMD[f16, 8]](Vr[c])
            comptime for i in range(8):
                vt[c * 8 + i, key] = rebind[vt.ElementType](x[i])
        barrier()

        var sc = Array[SIMD[f32, 8], BN // 16](fill=SIMD[f32, 8](0))
        comptime for u in range(BN // 16):
            var s0 = SIMD[f32, 8](0)
            var s1 = SIMD[f32, 8](0)
            comptime for c in range(D // 16):
                var a = rebind[SIMD[f16, 8]](ksv[u * 16 + h, 2 * c]).join(rebind[SIMD[f16, 8]](ksv[u * 16 + h, 2 * c + 1]))
                var t = SIMD[f32, 8](0)
                comptime if c % 2 == 0:
                    mma(t, a, qf[c], s0)
                    s0 = t
                else:
                    mma(t, a, qf[c], s1)
                    s1 = t
            sc[u] = (s0 + s1) * scale_log2

        var mx = NEG
        comptime for u in range(BN // 16):
            comptime for i in range(8):
                if t0 + u * 16 + 2 * i + half < L:
                    mx = max(mx, sc[u][i])
        mx = max(mx, warp.shuffle_xor(mx, UInt32(16)))
        var m_new = max(m_run, mx)
        var alpha = exp2(m_run - m_new)
        var ps = Float32(0)
        var bf = Array[SIMD[f16, 16], BN // 16](fill=SIMD[f16, 16](0))
        comptime for u in range(BN // 16):
            var pv = SIMD[f32, 8](0)
            comptime for i in range(8):
                if t0 + u * 16 + 2 * i + half < L:
                    pv[i] = exp2(sc[u][i] - m_new)
            ps += pv.reduce_add()
            var b = SIMD[f16, 16](0)
            comptime for i in range(8):
                var own = pv[i].cast[f16]()
                var oth = warp.shuffle_xor(pv[i], UInt32(16)).cast[f16]()
                b[2 * i] = own if half == 0 else oth
                b[2 * i + 1] = oth if half == 0 else own
            bf[u] = b
        ps += warp.shuffle_xor(ps, UInt32(16))
        l_run = l_run * alpha + ps
        m_run = m_new

        comptime for dt in range(D // 16):
            var o = acc[dt] * alpha
            comptime for u in range(BN // 16):
                var a = rebind[SIMD[f16, 8]](vtv[dt * 16 + h, 2 * u]).join(rebind[SIMD[f16, 8]](vtv[dt * 16 + h, 2 * u + 1]))
                var t = SIMD[f32, 8](0)
                mma(t, a, bf[u], o)
                o = t
            acc[dt] = o
        t0 += BN

    if valid:
        var inv = 1 / l_run
        var base = (row * H + head) * D
        comptime for dt in range(D // 16):
            comptime for i in range(8):
                O[unsafe_offset=base + dt * 16 + 2 * i + half] = (acc[dt][i] * inv).cast[f16]()
