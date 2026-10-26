"""Reference arm: time torch SDPA (AOTriton on this box) on a dumped case exactly the way ComfyUI calls it
([B,L,H*D] -> [B,H,L,D] strided views -> scaled_dot_product_attention -> back). Run with the ComfyUI venv.
Usage: sdpa-bench.py DIR ITERS"""
import json, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

# same priority list ComfyUI installs (comfy/ops.py:68-73)
PRIO = [SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]

d, iters = sys.argv[1], int(sys.argv[2])
m = json.load(open(f"{d}/meta.json"))
L, H, D = m["L"], m["H"], m.get("D", 128)
q, k, v = (torch.from_numpy(np.fromfile(f"{d}/{n}.bin", np.float16)).cuda().view(1, L, H, D) for n in "qkv")


def call():
    with sdpa_kernel(PRIO, set_priority=True):
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    return o.transpose(1, 2).reshape(1, L, H * D)


for _ in range(2):
    call()
torch.cuda.synchronize()
t = time.perf_counter()
for _ in range(iters):
    call()
torch.cuda.synchronize()
ms = (time.perf_counter() - t) / iters * 1e3
print(f"sdpa L={L} H={H} ms/call={ms:.2f} TFLOPS={4 * L * L * D * H / (ms * 1e-3) / 1e12:.1f}")
