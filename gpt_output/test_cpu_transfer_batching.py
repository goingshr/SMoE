#!/usr/bin/env python3
"""CPU-only semantic test for repeated activation reuse and output batching."""

from types import SimpleNamespace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

import MoEModule.SMoE_base as smoe_base


class RecordingExpert:
    def __init__(self, scale):
        self.scale = scale
        self.input_ptrs = []

    def __call__(self, value):
        self.input_ptrs.append(value.data_ptr())
        return value * self.scale


class FakeCache:
    def __init__(self, experts):
        self.experts = experts

    def get_compute_expert(self, uid, offload=False):
        assert offload
        return self.experts[uid]


class CpuComputeHarness:
    _cpu_compute = smoe_base.AbstractMoELayer._cpu_compute
    _cpu_compute_reference = smoe_base.AbstractMoELayer._cpu_compute_reference
    _record_cpu_compute = smoe_base.AbstractMoELayer._record_cpu_compute
    _record_cpu_transfer = staticmethod(
        smoe_base.AbstractMoELayer._record_cpu_transfer)

    def __init__(self, experts, batched):
        self.ExpertCache = FakeCache(experts)
        self.config = SimpleNamespace(device="cpu")
        self._batch_cpu_transfers = batched
        self._staged_decode_input = None
        self._decode_minmax = False
        self.CPUComputeTimeOneExpertOneBatch = [0.05]


def make_entries():
    token0_a = torch.tensor(
        [[1.0, -2.0, 3.0, -4.0]], dtype=torch.bfloat16)
    token0_b = token0_a.clone()
    token12 = torch.tensor([[2.0, 1.0, -1.0, 0.5],
                            [-3.0, 4.0, 2.0, -0.5]], dtype=torch.bfloat16)
    return {
        (7, 1): [token0_a, torch.tensor([[0.25]], dtype=torch.bfloat16),
                 torch.tensor([0]), [0]],
        (7, 2): [token0_b, torch.tensor([[0.75]], dtype=torch.bfloat16),
                 torch.tensor([0]), [0]],
        (7, 3): [token12, torch.tensor([[0.5], [0.125]], dtype=torch.bfloat16),
                 torch.tensor([1, 2]), [1, 2]],
    }


def reset_counters():
    smoe_base._cpu_ms_cur_token_samples.clear()
    smoe_base.cpu_activation_d2h_copies = 0
    smoe_base.cpu_activation_d2h_bytes = 0
    smoe_base.cpu_output_h2d_copies = 0
    smoe_base.cpu_output_h2d_bytes = 0


def run(batched):
    experts = {
        (7, 1): RecordingExpert(2.0),
        (7, 2): RecordingExpert(-1.0),
        (7, 3): RecordingExpert(0.5),
    }
    harness = CpuComputeHarness(experts, batched)
    outputs = {}
    reset_counters()
    harness._cpu_compute(list(experts), make_entries(), outputs)
    counters = (
        smoe_base.cpu_activation_d2h_copies,
        smoe_base.cpu_activation_d2h_bytes,
        smoe_base.cpu_output_h2d_copies,
        smoe_base.cpu_output_h2d_bytes,
    )
    return experts, outputs, counters


def main():
    reference_experts, reference, reference_counts = run(False)
    batched_experts, batched, batched_counts = run(True)

    assert reference.keys() == batched.keys()
    for uid in reference:
        assert torch.equal(reference[uid], batched[uid]), uid

    # The first two experts consume the same logical token slice.  The batched
    # path must reuse one CPU tensor even though their source tensors differ.
    assert (batched_experts[(7, 1)].input_ptrs[0]
            == batched_experts[(7, 2)].input_ptrs[0])
    assert (reference_experts[(7, 1)].input_ptrs[0]
            != reference_experts[(7, 2)].input_ptrs[0])

    assert reference_counts[0] == 3
    assert batched_counts[0] == 2
    assert reference_counts[2] == 3
    assert batched_counts[2] == 1
    assert batched_counts[1] < reference_counts[1]
    assert batched_counts[3] == reference_counts[3]

    reset_counters()
    empty = CpuComputeHarness({}, True)
    empty_outputs = {}
    empty._cpu_compute([], {}, empty_outputs)
    assert not empty_outputs
    assert smoe_base.cpu_activation_d2h_copies == 0
    assert smoe_base.cpu_output_h2d_copies == 0

    single_uid = (7, 1)
    single_experts = {single_uid: RecordingExpert(2.0)}
    single = CpuComputeHarness(single_experts, True)
    single_outputs = {}
    single._cpu_compute(
        [single_uid], {single_uid: make_entries()[single_uid]}, single_outputs)
    assert smoe_base.cpu_activation_d2h_copies == 1
    assert smoe_base.cpu_output_h2d_copies == 1
    print("PASS: CPU transfer batching preserves outputs and reduces copy calls")


if __name__ == "__main__":
    main()
