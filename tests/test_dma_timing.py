"""Exercise the production raw-storage DMA and its completed-event estimator."""
from collections import deque
from types import SimpleNamespace
import torch
from utils.expertcache import ExpertCache


torch.set_num_threads(2)
source = torch.empty(3584 * 2560 * 3, dtype=torch.bfloat16, pin_memory=True).fill_(0.5)
destination = torch.empty_like(source, device='cuda')
cache = SimpleNamespace(
    main_modules=[SimpleNamespace(storage=destination.untyped_storage())],
    offloaded_storages=[SimpleNamespace(storage=source.untyped_storage())],
    load_stream=torch.cuda.Stream(), measure_dma=True,
    _dma_timings=deque(), LoadTimeOneExpert=[0.002],
    measured_dma_copies=0,
)
for value in [0.5, -0.25, 1.0]:
    source.fill_(value)
    event = ExpertCache._swap(cache, 0, 0)
    cache.load_stream.synchronize()
    assert event.query()
    ExpertCache.consume_dma_timings(cache)
    assert not cache._dma_timings
    assert cache.LoadTimeOneExpert[-1] > 0
    assert torch.equal(destination.cpu(), source)
print(f'PASS full BF16 storage DMA, completed events, estimator drain; '
      f'durations_s={cache.LoadTimeOneExpert[1:]}')
assert cache.measured_dma_copies == 3
