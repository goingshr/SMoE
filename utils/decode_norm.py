"""BF16 one-token RMSNorm with the checkpoint's intermediate cast intact."""
import logging
import torch
import triton
import triton.language as tl

enabled = False


@triton.jit
def _rms(X, W, Out, H: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.arange(0, BLOCK)
    x = tl.load(X+r, r < H, other=0).to(tl.float32)
    variance = tl.sum(x*x,0) / H
    norm = (x*tl.rsqrt(variance+EPS)).to(tl.bfloat16).to(tl.float32)
    weight = tl.load(W+r,r < H,other=0).to(tl.float32)
    tl.store(Out+r,(norm*weight).to(tl.bfloat16),r < H)


def rms_norm(x, weight, eps):
    h = weight.numel()
    out = torch.empty_like(x)
    _rms[(1,)](x,weight,out,h,eps,triton.next_power_of_2(h),num_warps=4,enable_fp_fusion=False)
    return out


def install(model):
    count = 0
    for module in model.modules():
        if type(module).__name__ not in {'DeepseekRMSNorm', 'XverseRMSNorm'} or hasattr(module,'_smoe_norm_reference'):
            continue
        module._smoe_norm_reference = module.forward
        def forward(x, _module=module):
            w = _module.weight
            if (enabled and not torch.is_grad_enabled() and x.is_cuda
                    and x.dtype == w.dtype == torch.bfloat16 and x.device == w.device
                    and x.numel() == w.numel() and x.is_contiguous() and w.is_contiguous()):
                return rms_norm(x,w,_module.variance_epsilon)
            return _module._smoe_norm_reference(x)
        module.forward = forward
        count += 1
    logging.getLogger(__name__).info('[decode RMSNorm] patched=%d BF16 casts retained',count)
