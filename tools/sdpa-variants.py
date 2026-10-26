import json, time, numpy as np, torch, torch.nn.functional as F
d=".work/real"; L,H,D=32760,40,128
ld=lambda n: torch.from_numpy(np.fromfile(f"{d}/{n}.bin",np.float16)).cuda().view(1,L,H,D)
q,k,v=ld("q"),ld("k"),ld("v")
def t(name,Q,K,V,n=3):
    F.scaled_dot_product_attention(Q,K,V); torch.cuda.synchronize(); s=time.perf_counter()
    for _ in range(n): F.scaled_dot_product_attention(Q,K,V)
    torch.cuda.synchronize(); print(f"{name:34s} {(time.perf_counter()-s)/n*1e3:8.1f} ms")
tr=lambda x: x.transpose(1,2)
t("dump, strided [B,H,L,D] views",tr(q),tr(k),tr(v))
t("dump, contiguous [B,H,L,D]",tr(q).contiguous(),tr(k).contiguous(),tr(v).contiguous())
r=lambda: torch.randn(1,L,H,D,dtype=torch.float16,device="cuda")
t("randn, strided",tr(r()),tr(r()),tr(r()))
torch.backends.cudnn.enabled=False
t("dump, strided, cudnn off",tr(q),tr(k),tr(v))
t("dump, bf16 strided",tr(q.bfloat16()),tr(k.bfloat16()),tr(v.bfloat16()))
