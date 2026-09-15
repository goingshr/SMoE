"""Graph replay of all cached experts for one BF16 DeepSeek decode layer.

Weight addresses refer to pinned-in-cache slots; the caller establishes each
slot's load dependency and retains its normal ready_compute/end_compute lease.
Every layer owns its input/output buffers. All replay/copies use the caller's
default stream, and the host metadata cannot be reused before copy completion.
"""
import torch
import triton
import triton.language as tl
from utils.triton_bf16_expert import _gate_up


@triton.jit
def _group_gate(X, Ptrs, Mid, H: tl.constexpr, I: tl.constexpr, ROWS: tl.constexpr,
                BLOCK: tl.constexpr):
    e = tl.program_id(1)
    addr = tl.load(Ptrs + e)
    if addr != 0:
        w = addr.to(tl.pointer_type(tl.bfloat16))
        _gate_up(X, w, Mid + e*I, H, I, ROWS, BLOCK)


@triton.jit
def _group_down(Mid, Ptrs, Slots, Weights, Out, H: tl.constexpr, I: tl.constexpr,
                ROWS: tl.constexpr, BLOCK: tl.constexpr):
    e = tl.program_id(1)
    addr = tl.load(Ptrs + e)
    if addr != 0:
        w = addr.to(tl.pointer_type(tl.bfloat16)) + 2*I*H
        r = tl.program_id(0)*ROWS + tl.arange(0, ROWS)
        c = tl.arange(0, BLOCK)
        x = tl.load(Mid + e*I + c, c < I, other=0).to(tl.float32)
        v = tl.load(w+r[:, None]*I+c[None, :],
                    (r[:, None] < H) & (c[None, :] < I), other=0).to(tl.float32)
        out = tl.sum(v*x[None, :], 1).to(tl.bfloat16).to(tl.float32)
        slot = tl.load(Slots + e)
        score = tl.load(Weights + slot).to(tl.float32)
        tl.store(Out + e*H + r, (out*score).to(tl.bfloat16), r < H)


class GroupedDecode:
    def __init__(self, hidden, intermediate, top_k, device):
        self.h, self.i, self.k = hidden, intermediate, top_k
        self.x = torch.zeros((1,hidden), dtype=torch.bfloat16, device=device)
        self.weights = torch.zeros((top_k,), dtype=torch.bfloat16, device=device)
        self.mid = torch.empty((top_k,intermediate), dtype=torch.bfloat16, device=device)
        self.out = torch.empty((top_k,hidden), dtype=torch.bfloat16, device=device)
        self.host_meta = torch.zeros((2,top_k), dtype=torch.int64, pin_memory=True)
        self.meta = torch.zeros_like(self.host_meta, device=device)
        self._host_array = self.host_meta.numpy()
        self.copy_done = torch.cuda.Event()
        self.engaged = False
        # Compile before capture; addresses are read at replay, never constants.
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            self._launch()
        stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self._launch()
        stream.synchronize()

    def _launch(self):
        _group_gate[(triton.cdiv(self.i,4),self.k)](self.x,self.meta,self.mid,
            self.h,self.i,4,triton.next_power_of_2(self.h),num_warps=4,enable_fp_fusion=False)
        _group_down[(triton.cdiv(self.h,4),self.k)](self.mid,self.meta,self.meta[1],self.weights,
            self.out,self.h,self.i,4,triton.next_power_of_2(self.i),num_warps=4,enable_fp_fusion=False)

    def __call__(self, x, weights, addresses, slots):
        if len(addresses) > self.k or len(slots) != len(addresses):
            raise ValueError('invalid grouped expert metadata')
        self.copy_done.synchronize()
        self._host_array.fill(0)
        self._host_array[0,:len(addresses)] = addresses
        self._host_array[1,:len(slots)] = slots
        self.meta.copy_(self.host_meta, non_blocking=True)
        self.copy_done.record(torch.cuda.current_stream(x.device))
        self.x.copy_(x)
        self.weights.copy_(weights.reshape(-1))
        self.graph.replay()
        if not self.engaged:
            print('[Grouped Triton] BF16 cached experts graph replay engaged', flush=True)
            self.engaged = True
        return self.out
