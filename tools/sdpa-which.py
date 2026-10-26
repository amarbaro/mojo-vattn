import json, sys, numpy as np, torch, torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
d=".work/real"; m=json.load(open(f"{d}/meta.json")); L,H,D=m["L"],m["H"],128
q,k,v=(torch.from_numpy(np.fromfile(f"{d}/{n}.bin",np.float16)).cuda().view(1,L,H,D) for n in "qkv")
P=[SDPBackend.FLASH_ATTENTION,SDPBackend.CUDNN_ATTENTION,SDPBackend.EFFICIENT_ATTENTION,SDPBackend.MATH]
def call():
    with sdpa_kernel(P,set_priority=True):
        return F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2))
call(); torch.cuda.synchronize()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA,torch.profiler.ProfilerActivity.CPU]) as p:
    call(); torch.cuda.synchronize()
print(p.key_averages().table(sort_by="self_device_time_total",row_limit=8,max_name_column_width=70))
print("strides", q.transpose(1,2).stride(), "contig", q.transpose(1,2).is_contiguous())
