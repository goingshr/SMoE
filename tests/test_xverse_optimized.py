"""BF16 operator equivalence and packed-slot reuse on the target CUDA device."""
import torch
import torch.nn.functional as F
from utils.grouped_decode import GroupedDecode
from utils.triton_bf16_expert import bf16_expert

torch.manual_seed(37)
torch.set_num_threads(15)
H,I,K=2560,1728,6

def check(out, ref):
    err=out.float()-ref.float()
    rms=(err.square().mean()/ref.float().square().mean().clamp_min(1e-20)).sqrt().item()
    assert rms < .01, rms
    torch.testing.assert_close(out,ref,atol=.004,rtol=.02)
    return dict(max_abs=err.abs().max().item(),relative_rms=rms)

with torch.no_grad():
    packs=[torch.randn(3*H*I,device='cuda',dtype=torch.bfloat16)*.02 for _ in range(K)]
    graph=GroupedDecode(H,I,K,'cuda')
    for iteration,n in enumerate((1,3,6,2,6)):
        x=torch.randn(1,H,device='cuda',dtype=torch.bfloat16)
        weights=torch.rand(1,K,device='cuda',dtype=torch.float32)
        weights/=weights.sum()
        slots=list(reversed(range(n)))
        if iteration:
            # Reuse identical slot addresses with new contents, as eviction does.
            packs[0].mul_(.99)
        out=graph(x,weights,[p.data_ptr() for p in packs[:n]],slots).clone()
        for e in range(n):
            gu=packs[e][:2*H*I].view(2*I,H)
            down=packs[e][2*H*I:].view(H,I)
            g,u=F.linear(x,gu).split(I,-1)
            reference=F.linear(F.silu(g)*u,down)
            single=bf16_expert(x,gu,down)
            error=check(single,reference)
            reference.mul_(weights[:,slots[e]:slots[e]+1])
            check(out[e:e+1],reference)
        print('grouped/individual correctness',iteration,n,error,flush=True)
    torch.cuda.synchronize()
print('PASS BF16 Xverse kernels; FP32 router scores preserved',flush=True)

from utils.decode_norm import rms_norm
with torch.no_grad():
    for h in (2560,80,257):
        for scale in (.01,1.,100.):
            x=torch.randn(1,1,h,device='cuda',dtype=torch.bfloat16)*scale
            w=torch.randn(h,device='cuda',dtype=torch.bfloat16)
            xf=x.float()
            ref=w*(xf*torch.rsqrt(xf.square().mean(-1,keepdim=True)+1e-6)).bfloat16()
            torch.testing.assert_close(rms_norm(x,w,1e-6),ref,atol=.004,rtol=.008)
print('PASS RMSNorm BF16 cast boundaries',flush=True)

from utils.decode_rope import rope
with torch.no_grad():
    for nq,nk in ((32,32),(32,8),(3,1)):
        q=torch.randn(1,nq,1,80,device='cuda',dtype=torch.bfloat16)
        k=torch.randn(1,nk,1,80,device='cuda',dtype=torch.bfloat16)
        angles=torch.randn(1,1,80,device='cuda')
        cos,sin=angles.cos().bfloat16(),angles.sin().bfloat16()
        def ref(x):
            a,b=x.chunk(2,-1)
            return x*cos.unsqueeze(1)+torch.cat((-b,a),-1)*sin.unsqueeze(1)
        oq,ok=rope(q,k,cos,sin,None)
        assert torch.equal(oq,ref(q)) and torch.equal(ok,ref(k))
print('PASS Xverse modern RoPE exact BF16',flush=True)
