"""Compare the production dispatch/combine path with its eager reference."""
import logging
from types import SimpleNamespace
from unittest.mock import patch

import torch

from MoEModule.SMoE_base import AbstractMoELayer
import utils.expertcache as ec


class Cache:
    cache_window = None
    LoadTimeOneExpert = [0.002]

    def __init__(self, cpu_experts):
        self.cpu_experts = cpu_experts
        self.load_stream = torch.cuda.Stream()
        self.pins = []

    def ready_compute(self, uid):
        self.pins.append(uid)

    def end_compute(self, uid):
        self.pins.remove(uid)

    def query_expert(self, uid):
        return uid[1] not in self.cpu_experts

    def get_compute_expert(self, uid, offload=False):
        return lambda x: x * (1 + uid[1] / 16)

    def clear_queue(self):
        pass

    def wait_until_queue_empty(self):
        pass


class Layer(AbstractMoELayer):
    def __init__(self, hidden, cpu_experts):
        super().__init__(SimpleNamespace(device='cuda:0', if_usecpu=True, num_hidden_layers=28),
                         Cache(cpu_experts), 0)
        self.gate = torch.nn.Linear(hidden, 8, bias=False,
                                    device='cuda', dtype=torch.bfloat16)

    def get_gate(self):
        return self.gate

    def get_num_experts(self):
        return 8

    def get_top_k(self):
        return 8

    def get_norm_topk_prob(self):
        return self.normalize

    def compute_shared_expert(self, x):
        return x * 0.25


def main():
    logging.disable(logging.INFO)
    torch.manual_seed(2026)
    torch.set_num_threads(2)
    ec.tokens = 1
    count = 0
    # Same production pipeline with all GPU, mixed CPU/GPU, and all CPU
    # assignments. M>1 exercises the original ragged/prefill fallback.
    with torch.no_grad(), patch('MoEModule.SMoE_base.CPU_load_management',
                               side_effect=lambda u, *args: ([], list(u))):
        for h in [17, 3584]:
            for cpu_experts in [set(), {1, 3, 6}, set(range(8))]:
                layer = Layer(h, cpu_experts)
                for norm in [False, True]:
                    layer.normalize = norm
                    for tokens in [0, 1, 2, 7]:
                        x = torch.randn(1, tokens, h, device='cuda', dtype=torch.bfloat16)
                        layer._decode_direct = False
                        layer._decode_stream_chain = False
                        layer._pinned_decode = False
                        ref, ref_logits = layer.run_with_cache(x)
                        layer._decode_direct = True
                        layer._decode_stream_chain = True
                        layer._pinned_decode = True
                        actual, logits = layer.run_with_cache(x)
                        torch.cuda.synchronize()
                        torch.testing.assert_close(actual, ref, rtol=0, atol=0)
                        torch.testing.assert_close(logits, ref_logits, rtol=0, atol=0)
                        assert not layer.ExpertCache.pins
                        count += 1
    print(f'PASS {count} production dispatch/combine cases; exact BF16 outputs')


if __name__ == '__main__':
    main()
