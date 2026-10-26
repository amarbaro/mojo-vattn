"""Oracle for vattn_fwd. Inputs are raw f16 [L,H,D] files; the reference is fp64 softmax(QK^T/sqrt(D))V
over the SAME f16 bytes the kernel reads.

  vattn-ref.py gen DIR L H [SEED]     write q.bin k.bin v.bin + o_ref64.bin (f64) for a synthetic case
  vattn-ref.py check DIR [BAND]       compare DIR/o_mojo.bin (and o_sdpa.bin if present) against the
                                      reference (o_ref64.bin, else o_ref.bin f32); exit 1 on NaN/inf or
                                      when max abs err > BAND x (SDPA err if present, else 2e-3)
"""
import json, sys
import numpy as np

D = 128


def ref(q, k, v):
    L, H, _ = q.shape
    out = np.empty((L, H, D))
    for h in range(H):
        qh, kh, vh = (x[:, h].astype(np.float64) for x in (q, k, v))
        for s in range(0, L, 2048):
            z = qh[s:s + 2048] @ kh.T / np.sqrt(D)
            z = np.exp(z - z.max(axis=1, keepdims=True))
            out[s:s + 2048, h] = (z / z.sum(axis=1, keepdims=True)) @ vh
    return out


def gen(d, L, H, seed=0):
    rng = np.random.default_rng(seed)
    q, k, v = (rng.standard_normal((L, H, D)).astype(np.float16) for _ in range(3))
    q *= np.float16(2.0)  # sharper softmax than unit normal, closer to real attention
    for n, x in (("q", q), ("k", k), ("v", v)):
        x.tofile(f"{d}/{n}.bin")
    ref(q, k, v).tofile(f"{d}/o_ref64.bin")
    json.dump({"L": L, "H": H, "D": D, "seed": seed}, open(f"{d}/meta.json", "w"))


def err(a, b):
    e = np.abs(a.astype(np.float64) - b)
    return float(e.max()), float(e.mean())


def check(d, band=1.5):
    m = json.load(open(f"{d}/meta.json"))
    L, H = m["L"], m["H"]
    try:
        r = np.fromfile(f"{d}/o_ref64.bin", np.float64)
    except FileNotFoundError:
        r = np.fromfile(f"{d}/o_ref.bin", np.float32).astype(np.float64)
    o = np.fromfile(f"{d}/o_mojo.bin", np.float16)
    if o.size != L * H * D or r.size != o.size:
        print(f"FAIL {d}: size o={o.size} ref={r.size} want {L * H * D}")
        return False
    if not np.isfinite(o).all():
        print(f"FAIL {d}: {int((~np.isfinite(o)).sum())} non-finite outputs")
        return False
    mx, mean = err(o, r)
    try:
        smx, smean = err(np.fromfile(f"{d}/o_sdpa.bin", np.float16), r)
        lim, what = band * smx, f"sdpa max {smx:.3e} mean {smean:.3e}, gate {band}x"
    except FileNotFoundError:
        lim, what = 2e-3, "no sdpa arm, gate 2e-3 abs"
    ok = mx <= lim
    print(f"{'PASS' if ok else 'FAIL'} {d}: L={L} H={H} mojo max {mx:.3e} mean {mean:.3e} | {what}")
    return ok


if __name__ == "__main__":
    if sys.argv[1] == "gen":
        gen(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]) if len(sys.argv) > 5 else 0)
    else:
        sys.exit(0 if check(sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 1.5) else 1)
