"""Zero-copy BF16 gate/up projection for cache-backed MoE experts."""

import os

import torch
import torch.nn.functional as F

CPU_AVX2_GEMV = os.environ.get("SMOE_CPU_AVX2_GEMV", "0") == "1"
GPU_TRITON_EXPERT = os.environ.get("SMOE_GPU_TRITON_EXPERT", "0") == "1"
_avx2_gemv = None


class FusedGateUpMixin:
    def _init_fused_gate_up(self):
        global _avx2_gemv
        if CPU_AVX2_GEMV and _avx2_gemv is None:
            from utils.cpu_bf16_kernel import load_kernel
            _avx2_gemv = load_kernel()
            print("[CPU AVX2 GEMV] loaded: BF16 storage, FP32 accumulation", flush=True)
        self._cpu_gate_up_weight = None
        self._gpu_gate_up_weight = None
        self._cpu_fuse_gate_up = os.environ.get(
            "SMOE_CPU_BF16_FUSED_GATE_UP", "1"
        ).strip().lower() not in {"0", "false", "off", "no"}
        self._gpu_fuse_gate_up = os.environ.get(
            "SMOE_GPU_BF16_FUSED_GATE_UP", "1"
        ).strip().lower() not in {"0", "false", "off", "no"}
        # AVX2 hosts can prefer GEMV for one-token BF16 experts. Keep an
        # explicit gate: CPUs with native BF16/AMX need a separate benchmark.
        self._cpu_bf16_mv = os.environ.get("SMOE_CPU_BF16_MV", "0") == "1"

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
        if (x.device.type == "cpu" and CPU_AVX2_GEMV
                and x.dtype == torch.bfloat16 and x.is_contiguous()
                and self.down_proj.weight.is_contiguous()):
            gate, up = _avx2_gemv(weight, x.reshape(-1)).split(self.intermediate_size)
            output = _avx2_gemv(self.down_proj.weight, self.act_fn(gate) * up)
            return output.reshape(x.shape)
        if (x.device.type == "cuda" and GPU_TRITON_EXPERT
                and self.config.hidden_act == "silu" and x.is_contiguous()
                and x.dtype == torch.bfloat16):
            from utils.triton_bf16_expert import bf16_expert
            return bf16_expert(x, weight, self.down_proj.weight)
        if x.device.type == "cpu" and self._cpu_bf16_mv:
            gate, up = torch.mv(weight, x.reshape(-1)).split(self.intermediate_size)
            output = torch.mv(self.down_proj.weight, self.act_fn(gate) * up)
            return output.reshape(x.shape)
        gate, up = F.linear(x, weight).split(self.intermediate_size, dim=-1)
        return self.down_proj(self.act_fn(gate) * up)
