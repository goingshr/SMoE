"""Check ring wraparound while asynchronous DMA still owns host buffers."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from utils.pinned_staging import PinnedStagingRing


def check_ring():
    nbytes = 65536
    ring = PinnedStagingRing(nbytes)
    source = torch.empty(nbytes, dtype=torch.uint8)
    stream = torch.cuda.Stream()
    destinations = []
    for i in range(9):
        source.fill_(17 + i)
        index, buffer = ring.stage(source.untyped_storage())
        with torch.cuda.stream(stream):
            # An explicit delay makes premature host-buffer reuse observable.
            torch.cuda._sleep(200000)
            destination = torch.empty(nbytes, dtype=torch.uint8, device='cuda')
            destination.copy_(buffer, non_blocking=True)
            done = torch.cuda.Event()
            done.record()
        ring.release_after(index, done)
        destinations.append(destination)
    ring.synchronize()
    for i, destination in enumerate(destinations):
        assert (destination.cpu() == 17 + i).all().item(), i
    print('[staging test] byte-exact after 9 submissions through 2 buffers', flush=True)
    # This diagnostic includes host staging and stream completion in wall time.
    # It runs before the following policy's warmups, never during measurement.
    from gpt_output.optimization_3080.probe_staged_transfer import probe
    probe()


if __name__ == '__main__':
    check_ring()
