"""Check fused BF16 projections, including in-place cache-slot replacement."""
from types import SimpleNamespace
import torch
from MoEModule.qwen_moe import Qwen2MoeMLP


def main():
    torch.manual_seed(2026)
    torch.set_num_threads(2)
    cfg = SimpleNamespace(hidden_size=3584, moe_intermediate_size=2560,
                          hidden_act='silu', device='cuda:0')
    expert = Qwen2MoeMLP(cfg)
    expert._gpu_fuse_gate_up = True
    backing = torch.cat([expert.gate_proj.weight, expert.up_proj.weight]).detach()
    expert.gate_proj.weight.data = backing[:2560]
    expert.up_proj.weight.data = backing[2560:]
    worst = 0.0
    count = 0
    with torch.no_grad():
        for generation in range(2):
            if generation:
                # Slot ownership changes by copying into the same storage.
                backing.normal_(std=0.02)
            for rows in [1, 2, 7, 128]:
                x = torch.randn(rows, 3584, device='cuda', dtype=torch.bfloat16)
                for noncontiguous in [False, True]:
                    if noncontiguous:
                        storage = torch.empty((rows, 7168), device='cuda', dtype=x.dtype)
                        storage[:, ::2].copy_(x)
                        x = storage[:, ::2]
                        assert not x.is_contiguous()
                    expert._gpu_gate_up_weight = None
                    expected = expert(x)
                    assert expert.configure_gpu_bf16_gate_up(backing)
                    actual = expert(x)
                    # Single-token fusion; multi-token inputs use the reference.
                    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.002)
                    worst = max(worst, (actual.float()-expected.float()).abs().max().item())
                    count += 1
    print(f'PASS {count} GPU gate/up cases; max_abs={worst}; rtol=.02 atol=.002')


if __name__ == '__main__':
    main()
