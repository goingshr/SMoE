"""Fixed-shape expert graphs over stable cache-slot weight addresses.

Capture happens during model construction, before inference workers submit
CUDA work. Each slot owns its graph buffers; no shared graph memory pool.
"""
import logging
import torch

logger = logging.getLogger(__name__)


class DecodeExpertGraph:
    _capture_streams = {}

    def __init__(self, forward, hidden_size, device):
        self.forward = forward
        self.input = torch.zeros((1, hidden_size), dtype=torch.bfloat16, device=device)
        # Reuse the capture stream's cuBLAS handle/workspace across slots.
        stream = self._capture_streams.get(self.input.device)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            self._capture_streams[self.input.device] = stream
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.no_grad(), torch.cuda.stream(stream):
            for _ in range(3):
                forward(self.input)
        stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(self.graph, stream=stream):
            self.output = forward(self.input)
        stream.synchronize()
        self.engaged = False

    def __call__(self, x):
        if (torch.is_grad_enabled() or x.device != self.input.device
                or x.dtype != self.input.dtype or x.shape != self.input.shape):
            return self.forward(x)
        self.input.copy_(x)
        self.graph.replay()
        if not self.engaged:
            logger.debug("[expert graph] replay engaged on %s", x.device)
            self.engaged = True
        # The cache pins this slot until all expert outputs have been consumed.
        # Caller must not retain output across a subsequent call to this slot.
        return self.output
