"""Zero-copy BF16 gate/up projection for cache-backed MoE experts."""

import os

import torch
import torch.nn.functional as F


class FusedGateUpMixin:
    def _init_fused_gate_up(self):
        self._cpu_gate_up_weight = None
        self._gpu_gate_up_weight = None
        self._cpu_fuse_gate_up = os.environ.get(
            "SMOE_CPU_BF16_FUSED_GATE_UP", "1"
        ).strip().lower() not in {"0", "false", "off", "no"}
        self._gpu_fuse_gate_up = os.environ.get(
            "SMOE_GPU_BF16_FUSED_GATE_UP", "1"
        ).strip().lower() not in {"0", "false", "off", "no"}

    def configure_cpu_bf16_gate_up(self, weight: torch.Tensor) -> bool:
        return self._configure_fused_gate_up(weight, "cpu", self._cpu_fuse_gate_up)

    def configure_gpu_bf16_gate_up(self, weight: torch.Tensor) -> bool:
        return self._configure_fused_gate_up(weight, "cuda", self._gpu_fuse_gate_up)

    def _configure_fused_gate_up(self, weight, device_type, enabled):
        if not enabled:
            return False
        expected = (2 * self.intermediate_size, self.hidden_size)
        if (weight.device.type != device_type or weight.dtype != torch.bfloat16
                or tuple(weight.shape) != expected or not weight.is_contiguous()):
            raise ValueError(
                f"{device_type} gate/up fusion requires contiguous BF16 "
                f"weight with shape {expected}"
            )
        if device_type == "cpu":
            self._cpu_gate_up_weight = weight
        else:
            self._gpu_gate_up_weight = weight
        return True

    def _forward_fused_gate_up(self, x):
        # The decode path has one token. Keep the existing projections for
        # prefill, training, and tensor-parallel weights.
        if (self.config.pretraining_tp != 1 or torch.is_grad_enabled()
                or x.numel() != self.hidden_size):
            return None
        weight = (self._gpu_gate_up_weight if x.device.type == "cuda"
                  else self._cpu_gate_up_weight if x.device.type == "cpu"
                  else None)
        if weight is None:
            return None
        gate, up = F.linear(x, weight).split(self.intermediate_size, dim=-1)
        return self.down_proj(self.act_fn(gate) * up)
