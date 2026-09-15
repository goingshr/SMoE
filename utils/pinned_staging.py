"""Bounded host staging for immutable expert weights, owned by one loader."""
import ctypes

import torch


class PinnedStagingRing:
    def __init__(self, nbytes, slots=2):
        if slots < 1 or nbytes < 1:
            raise ValueError('positive buffer size and slot count required')
        self.buffers = [torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
                        for _ in range(slots)]
        self.events = [None] * slots
        self.index = 0

    def stage(self, storage):
        index = self.index
        buffer = self.buffers[index]
        if storage.device.type != 'cpu' or storage.nbytes() != buffer.numel():
            raise ValueError('staging requires a matching CPU expert storage')
        if self.events[index] is not None:
            self.events[index].synchronize()
        # libc memcpy/memmove avoids a Torch parallel-copy launch on the
        # loader's single core. CDLL releases the GIL during this byte copy.
        ctypes.memmove(buffer.data_ptr(), storage.data_ptr(), buffer.numel())
        return index, buffer

    def release_after(self, index, event):
        # Reuse is permitted only after the stream has consumed every byte.
        self.events[index] = event
        self.index = (index + 1) % len(self.buffers)

    def synchronize(self):
        for event in self.events:
            if event is not None:
                event.synchronize()
