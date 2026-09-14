"""Check zero-copy BF16 decode projection for DeepSeek and Xverse experts."""

from pathlib import Path
from types import SimpleNamespace
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from MoEModule.deepseek_moe import DeepseekMLP
from MoEModule.xverse_moe import XverseMLP
from utils.model_loader import ExpertWrapper


def check(cls, name, device):
    config = SimpleNamespace(
        hidden_size=128, moe_intermediate_size=64, intermediate_size=64,
        hidden_act="silu", device=device, pretraining_tp=1,
    )
    torch.manual_seed(20260914)
    expert = cls(config).eval()
    wrapper = ExpertWrapper(expert, name, torch.device(device), tocpu=device == "cpu")
    gate, up = expert.gate_proj.weight, expert.up_proj.weight
    fused = (expert._cpu_gate_up_weight if device == "cpu"
             else expert._gpu_gate_up_weight)
    assert fused is not None
    assert gate.data_ptr() + gate.nbytes == up.data_ptr()
    assert fused.data_ptr() == gate.data_ptr()
    assert fused.nbytes == gate.nbytes + up.nbytes
    assert (wrapper.cpu_bf16_gate_up_fused if device == "cpu"
            else wrapper.gpu_bf16_gate_up_fused)

    for tokens in (1, 2, 17):
        x = torch.randn((tokens, 128), device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            if device == "cpu":
                expert._cpu_gate_up_weight = None
            else:
                expert._gpu_gate_up_weight = None
            reference = expert(x)
            if device == "cpu":
                expert._cpu_gate_up_weight = fused
            else:
                expert._gpu_gate_up_weight = fused
            actual = expert(x)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    config.pretraining_tp = 2
    with torch.no_grad():
        expert(torch.randn((1, 1, 128), device=device, dtype=torch.bfloat16))


if __name__ == "__main__":
    os.environ["SMOE_EXPERT_GRAPH"] = "0"
    for cls, name in ((DeepseekMLP, "deepseekmoe"), (XverseMLP, "xversemoe")):
        check(cls, name, "cpu")
        if torch.cuda.is_available():
            check(cls, name, "cuda:0")
    print("PASS: DeepSeek and Xverse BF16 fusion")
