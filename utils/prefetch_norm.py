"""Fuse the existing DeepSeek look-ahead normalization, retaining BF16 casts."""
import torch
import triton
import triton.language as tl


@triton.jit
def _norms(Raw, Shared, Residual, W1, W2, Out, H: tl.constexpr,
           EPS1: tl.constexpr, EPS2: tl.constexpr, HAS_RESIDUAL: tl.constexpr,
           BLOCK: tl.constexpr):
    index = tl.arange(0, BLOCK)
    mask = index < H
    raw = tl.load(Raw + index, mask, other=0).to(tl.float32)
    shared = tl.load(Shared + index, mask, other=0).to(tl.float32)
    h = (raw + shared).to(tl.bfloat16).to(tl.float32)
    if HAS_RESIDUAL:
        residual = tl.load(Residual + index, mask, other=0).to(tl.float32)
        h = (h + residual).to(tl.bfloat16).to(tl.float32)
    variance = tl.sum(h * h, 0) / H
    norm = (h * tl.rsqrt(variance + EPS1)).to(tl.bfloat16).to(tl.float32)
    weight1 = tl.load(W1 + index, mask, other=0).to(tl.float32)
    norm = (norm * weight1).to(tl.bfloat16).to(tl.float32)
    h = (norm + h).to(tl.bfloat16).to(tl.float32)
    variance = tl.sum(h * h, 0) / H
    norm = (h * tl.rsqrt(variance + EPS2)).to(tl.bfloat16).to(tl.float32)
    weight2 = tl.load(W2 + index, mask, other=0).to(tl.float32)
    tl.store(Out + index, (norm * weight2).to(tl.bfloat16), mask)


def fused_prefetch_norms(raw, shared, residual, norm1, norm2):
    h = raw.shape[-1]
    tensors = [raw, shared, norm1.weight, norm2.weight]
    if residual is not None:
        tensors.append(residual)
    if (raw.numel() != h or any(t.numel() != h or not t.is_contiguous()
            or t.device != raw.device or t.dtype != torch.bfloat16 for t in tensors)
            or raw.device.type != 'cuda'):
        return None
    output = torch.empty_like(raw)
    _norms[(1,)](raw, shared, residual if residual is not None else raw,
        norm1.weight, norm2.weight, output, h, norm1.variance_epsilon,
        norm2.variance_epsilon, residual is not None, triton.next_power_of_2(h),
        num_warps=4, enable_fp_fusion=False)
    return output
