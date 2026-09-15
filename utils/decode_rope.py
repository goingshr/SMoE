"""Single-token RoPE with separate BF16 products before BF16 addition."""
import logging
import sys
import torch
import triton
import triton.language as tl

enabled = False


@triton.jit
def _rope(Q,K,Cos,Sin,Pos,OQ,OK,D:tl.constexpr,NQ:tl.constexpr,NK:tl.constexpr,
          BLOCK:tl.constexpr):
    r=tl.arange(0,BLOCK)
    col=r%D
    partner=r-col+(col+D//2)%D
    sign=tl.where(col<D//2,-1.,1.)
    position=tl.load(Pos)
    c=tl.load(Cos+position*D+col).to(tl.float32)
    s=tl.load(Sin+position*D+col).to(tl.float32)
    q=tl.load(Q+r,r<NQ,other=0).to(tl.float32)
    qr=tl.load(Q+partner,r<NQ,other=0).to(tl.float32)*sign
    k=tl.load(K+r,r<NK,other=0).to(tl.float32)
    kr=tl.load(K+partner,r<NK,other=0).to(tl.float32)*sign
    qo=(q*c).to(tl.bfloat16).to(tl.float32)+(qr*s).to(tl.bfloat16).to(tl.float32)
    ko=(k*c).to(tl.bfloat16).to(tl.float32)+(kr*s).to(tl.bfloat16).to(tl.float32)
    tl.store(OQ+r,qo.to(tl.bfloat16),r<NQ)
    tl.store(OK+r,ko.to(tl.bfloat16),r<NK)


def rope(q,k,cos,sin,position):
    oq,ok=torch.empty_like(q),torch.empty_like(k)
    _rope[(1,)](q,k,cos,sin,position,oq,ok,q.shape[-1],q.numel(),k.numel(),
        triton.next_power_of_2(max(q.numel(),k.numel())),num_warps=4,enable_fp_fusion=False)
    return oq,ok


def install(model):
    runtime=sys.modules[type(model).__module__]
    if hasattr(runtime,'_smoe_rope_reference'): return
    reference=runtime.apply_rotary_pos_emb
    runtime._smoe_rope_reference=reference
    def forward(q,k,cos,sin,position_ids,unsqueeze_dim=1):
        tensors=(q,k,cos,sin)
        if (enabled and not torch.is_grad_enabled() and unsqueeze_dim==1
                and q.ndim==k.ndim==4 and q.shape[0]==k.shape[0]==1
                and q.shape[2]==k.shape[2]==1 and q.shape[-1]==k.shape[-1]
                and q.shape[-1]%2==0 and cos.ndim==sin.ndim==2
                and cos.shape==sin.shape and cos.shape[-1]==q.shape[-1]
                and position_ids.numel()==1 and position_ids.device==q.device
                and all(t.is_cuda and t.dtype==torch.bfloat16 and t.device==q.device
                        and t.is_contiguous() for t in tensors)):
            return rope(q,k,cos,sin,position_ids)
        return reference(q,k,cos,sin,position_ids,unsqueeze_dim)
    runtime.apply_rotary_pos_emb=forward
    logging.getLogger(__name__).info('[decode RoPE] BF16 product casts retained')
