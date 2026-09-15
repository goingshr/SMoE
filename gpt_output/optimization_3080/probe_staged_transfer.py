"""Three-expert transfer batches; call only between generation measurements."""
import ctypes
import json
import statistics
import time

import torch
from utils import expertcache as ec
from utils.pinned_staging import PinnedStagingRing


def probe():
    if not ec._CUDART_AVAILABLE:
        print('[staging probe] skipped: libcudart unavailable', flush=True)
        return
    nbytes = 3 * 2048 * 1408 * 2
    sources = [torch.full((nbytes,), i + 11, dtype=torch.uint8) for i in range(3)]
    targets = [torch.empty(nbytes, device='cuda', dtype=torch.uint8) for _ in sources]
    ring = PinnedStagingRing(nbytes)
    stream = torch.cuda.Stream()
    for mode in ('pageable', 'memmove_ring'):
        walls = []
        for iteration in range(15):
            stream.synchronize()
            begin = time.perf_counter()
            for source, target in zip(sources, targets):
                if mode == 'memmove_ring':
                    index, staged = ring.stage(source.untyped_storage())
                    ptr = staged.data_ptr()
                else:
                    ptr = source.data_ptr()
                error = ec._cudaMemcpyAsync(ctypes.c_void_p(target.data_ptr()),
                    ctypes.c_void_p(ptr), ctypes.c_size_t(nbytes), ctypes.c_int(1),
                    ctypes.c_void_p(stream.cuda_stream))
                if error:
                    raise RuntimeError(f'cudaMemcpyAsync failed: {error}')
                done = torch.cuda.Event()
                done.record(stream)
                if mode == 'memmove_ring':
                    ring.release_after(index, done)
            stream.synchronize()
            if iteration >= 5:
                walls.append(time.perf_counter() - begin)
        for source, target in zip(sources, targets):
            assert torch.equal(source, target.cpu())
        print('[staging probe]', json.dumps(dict(mode=mode, bytes_per_expert=nbytes,
            experts_per_batch=3, iterations=10,
            median_batch_wall_ms=1000 * statistics.median(walls),
            median_per_expert_wall_ms=1000 * statistics.median(walls) / 3)), flush=True)


if __name__ == '__main__':
    probe()
