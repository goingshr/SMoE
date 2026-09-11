"""Graph replay must observe changed activations and cache-slot weights."""
from types import SimpleNamespace
import torch
from MoEModule.qwen_moe import Qwen2MoeMLP
from MoEModule.decode_graph import DecodeExpertGraph


def main():
    torch.manual_seed(2026)
    torch.set_num_threads(2)
    cfg = SimpleNamespace(hidden_size=3584, moe_intermediate_size=2560,
                          hidden_act='silu', device='cuda:0')
    modules = [Qwen2MoeMLP(cfg) for _ in range(2)]
    for m in modules:
        backing = torch.cat([m.gate_proj.weight, m.up_proj.weight]).detach()
        m.gate_proj.weight.data = backing[:2560]
        m.up_proj.weight.data = backing[2560:]
        m._gpu_fuse_gate_up = True
        m.configure_gpu_bf16_gate_up(backing)
    before = torch.cuda.memory_allocated()
    graphs = [DecodeExpertGraph(m, 3584, 'cuda:0') for m in modules]
    overhead = torch.cuda.memory_allocated() - before
    count = 0
    with torch.no_grad():
        for step in range(8):
            slot = (step // 2) % 2
            m = modules[slot]
            x = torch.randn(1, 3584, dtype=torch.bfloat16, device='cuda')
            if step % 2:
                storage = torch.empty(1, 7168, dtype=x.dtype, device=x.device)
                storage[:, ::2].copy_(x)
                x = storage[:, ::2]
            if step % 3 == 0:
                # Equivalent to a load-stream DMA replacement followed by B8.
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    for param in m.parameters():
                        param.normal_(std=0.02)
                stream.synchronize()
            fused_weight = m._gpu_gate_up_weight
            m._gpu_gate_up_weight = None
            expected = m(x)
            m._gpu_gate_up_weight = fused_weight
            actual = graphs[slot](x)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            actual.mul_(0.5)  # Production weights outputs in place.
            count += 1
        for g, m in zip(graphs, modules):
            x = torch.randn(7, 3584, dtype=torch.bfloat16, device='cuda')
            torch.testing.assert_close(g(x), m(x), rtol=0, atol=0)
            assert g.engaged
    print(f'PASS {count} replay/slot-replacement cases and prefill fallback; '
          f'exact BF16; graph_allocated_overhead={overhead}')


if __name__ == '__main__':
    main()
