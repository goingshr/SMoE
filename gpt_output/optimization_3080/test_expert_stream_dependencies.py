"""GPU consumer and slot reuse must wait for their exact dependency events."""
from collections import deque
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from utils.expertcache import ExpertCache


def check_dependencies():
    class Expert(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.ones((16, 16), device='cuda', dtype=torch.bfloat16)
            self.storage = self.weight.untyped_storage()

        def forward(self, x):
            return torch.nn.functional.linear(x, self.weight)

    cache = ExpertCache.__new__(ExpertCache)
    cache.device = torch.device('cuda:0')
    cache.mtx = threading.Lock()
    cache.load_stream = torch.cuda.Stream()
    cache.main_modules = [Expert()]
    source = torch.full((16, 16), 2.0, dtype=torch.bfloat16)
    cache.offloaded_storages = [SimpleNamespace(storage=source.untyped_storage())]
    cache.registered_experts = {(0, 0): SimpleNamespace(index=0, offload_index=0)}
    cache._slot_load_done = [None]
    cache._slot_compute_done = [None]
    cache.measure_dma = False
    cache.pinned_staging = False
    cache._dma_timings = deque()
    x = torch.ones((1, 16), device='cuda', dtype=torch.bfloat16)
    torch.cuda.synchronize()
    with torch.cuda.stream(cache.load_stream):
        torch.cuda._sleep(20000000)
    cache._swap(0, 0)
    out = cache.get_compute_expert((0, 0))(x)
    assert (out.cpu() == 32).all().item(), 'read before H2D completion'

    torch.cuda._sleep(20000000)
    out = cache.main_modules[0](x)
    cache.record_expert_use([(0, 0)], torch.cuda.current_stream())
    source.fill_(3)
    cache._swap(0, 0)
    assert (out.cpu() == 32).all().item(), 'weight overwritten before consumer finished'
    cache.load_stream.synchronize()
    assert (cache.main_modules[0].weight.cpu() == 3).all().item()
    print('[stream test] H2D-to-compute and compute-to-reuse ordering passed', flush=True)


if __name__ == '__main__':
    check_dependencies()
