"""One-token BF16 expert kernels; retain projection/activation rounding points."""
import torch
import triton
import triton.language as tl


@triton.jit
def _gate_up(X, W, Mid, H: tl.constexpr, I: tl.constexpr,
             ROWS: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    c = tl.arange(0, BLOCK)
    x = tl.load(X + c, c < H, other=0).to(tl.float32)
    g = tl.load(W + r[:, None]*H + c[None, :],
                (r[:, None] < I) & (c[None, :] < H), other=0).to(tl.float32)
    u = tl.load(W + (r[:, None]+I)*H + c[None, :],
                (r[:, None] < I) & (c[None, :] < H), other=0).to(tl.float32)
    g = tl.sum(g*x[None, :], 1).to(tl.bfloat16).to(tl.float32)
    u = tl.sum(u*x[None, :], 1).to(tl.bfloat16).to(tl.float32)
    act = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
    tl.store(Mid + r, (act*u).to(tl.bfloat16), r < I)


@triton.jit
def _down(Mid, W, Y, H: tl.constexpr, I: tl.constexpr,
          ROWS: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    c = tl.arange(0, BLOCK)
    x = tl.load(Mid + c, c < I, other=0).to(tl.float32)
    w = tl.load(W + r[:, None]*I + c[None, :],
                (r[:, None] < H) & (c[None, :] < I), other=0).to(tl.float32)
    y = tl.sum(w*x[None, :], 1).to(tl.bfloat16)
    tl.store(Y + r, y, r < H)


def bf16_expert(x, gate_up, down, rows=4):
    h, i = down.shape
    if (not x.is_cuda or x.dtype != torch.bfloat16 or x.numel() != h
            or tuple(gate_up.shape) != (2*i, h)
            or any(t.dtype != x.dtype or t.device != x.device or not t.is_contiguous()
                   for t in (x, gate_up, down))):
        raise ValueError('Triton expert requires contiguous one-token BF16 CUDA tensors')
    mid = torch.empty((i,), dtype=x.dtype, device=x.device)
    out = torch.empty_like(x)
    _gate_up[(triton.cdiv(i, rows),)](x, gate_up, mid, h, i, rows,
            triton.next_power_of_2(h), num_warps=4, enable_fp_fusion=False)
    _down[(triton.cdiv(h, rows),)](mid, down, out, h, i, rows,
            triton.next_power_of_2(i), num_warps=4, enable_fp_fusion=False)
    return out
