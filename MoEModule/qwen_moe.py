"""
qwen_moe.py — SMoE expert module for Qwen2 MoE.

Provides:
  - Qwen2MoeMLP                        plain MLP expert
  - Qwen2MoeSparseMoeBlockwithCache    cache-aware MoE block (inherits AbstractMoELayer)
"""

import logging
import os

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from typing import List, Optional, Tuple

from utils.expertcache import (
    ExpertCache,
    replaceset_between_tokens,
)
from MoEModule.SMoE_base import AbstractMoELayer

logger = logging.getLogger(__name__)

ExpertUID = Tuple[int, int]


class Qwen2MoeMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size       = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size,
                                   bias=False, dtype=torch.bfloat16, device=config.device)
        self.up_proj   = nn.Linear(self.hidden_size, self.intermediate_size,
                                   bias=False, dtype=torch.bfloat16, device=config.device)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size,
                                   bias=False, dtype=torch.bfloat16, device=config.device)
        self.act_fn = ACT2FN[config.hidden_act]
        self._cpu_gate_up_weight = None
        self._gpu_gate_up_weight = None
        self._gpu_fuse_gate_up = os.environ.get(
            "SMOE_GPU_BF16_FUSED_GATE_UP", "1"
        ).strip().lower() not in {"0", "false", "off", "no"}
        self._cpu_fuse_gate_up = os.environ.get(
            "SMOE_CPU_BF16_FUSED_GATE_UP", "1"
        ).strip().lower() not in {"0", "false", "off", "no"}

    def configure_cpu_bf16_gate_up(self, weight: torch.Tensor) -> bool:
        """Attach a zero-copy BF16 gate/up view over the expert backing store."""
        if not self._cpu_fuse_gate_up:
            return False
        expected_shape = (2 * self.intermediate_size, self.hidden_size)
        if (weight.device.type != "cpu"
                or weight.dtype != torch.bfloat16
                or tuple(weight.shape) != expected_shape
                or not weight.is_contiguous()):
            raise ValueError(
                "CPU gate/up fusion requires contiguous BF16 weight with "
                f"shape={expected_shape}, got device={weight.device}, "
                f"dtype={weight.dtype}, shape={tuple(weight.shape)}"
            )
        self._cpu_gate_up_weight = weight
        return True

    def configure_gpu_bf16_gate_up(self, weight: torch.Tensor) -> bool:
        """Use the existing adjacent gate/up storage without duplicating weights."""
        if not self._gpu_fuse_gate_up:
            return False
        if (weight.device.type != "cuda" or weight.dtype != torch.bfloat16
                or tuple(weight.shape) != (2 * self.intermediate_size, self.hidden_size)
                or not weight.is_contiguous()):
            raise ValueError("GPU gate/up fusion requires contiguous CUDA BF16 weights")
        self._gpu_gate_up_weight = weight
        return True

    def forward(self, x):
        if (x.device.type == "cuda" and self._gpu_gate_up_weight is not None
                and x.numel() == self.hidden_size and not torch.is_grad_enabled()):
            gate_up = F.linear(x, self._gpu_gate_up_weight)
            gate, up = gate_up.split(self.intermediate_size, dim=-1)
            return self.down_proj(self.act_fn(gate) * up)
        if x.device.type == "cpu" and self._cpu_gate_up_weight is not None:
            gate_up = F.linear(x, self._cpu_gate_up_weight)
            gate, up = gate_up.split(self.intermediate_size, dim=-1)
            return self.down_proj(self.act_fn(gate) * up)
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen2MoeSparseMoeBlockwithCache(AbstractMoELayer):
    """
    Qwen2 MoE block with expert cache.

    Qwen2-specific attributes vs AbstractMoELayer:
      - gate: nn.Linear
      - shared_expert: Qwen2MoeMLP + shared_expert_gate (gated output)
      - norm_topk_prob: from config
      - prefetch: lightweight predict (no attention, layernorm-only approx)
    """

    def __init__(self, config, expertcache_: ExpertCache, layerid,
                 gate: nn.Linear, shared_experts: Qwen2MoeMLP,
                 shared_expert_gate: nn.Linear,
                 next_attention, next_gate_weight,
                 next_input_layernorm, next_post_attention_layernorm):
        super().__init__(config, expertcache_, layerid)

        self.num_experts    = config.num_experts
        self.top_k          = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        self.gate               = gate
        self.shared_expert      = shared_experts
        self.shared_expert_gate = shared_expert_gate
        self._shared_graph = None
        if os.environ.get("SMOE_EXPERT_GRAPH", "1") == "1":
            from MoEModule.decode_graph import DecodeExpertGraph
            self._shared_graph = DecodeExpertGraph(
                self._compute_shared_eager, config.hidden_size, config.device)

        # Next-layer modules for prefetch prediction
        self.next_attention                = next_attention
        self.next_gate_weight              = next_gate_weight
        self.next_input_layernorm          = next_input_layernorm
        self.next_post_attention_layernorm = next_post_attention_layernorm

    # ------------------------------------------------------------------
    # AbstractMoELayer interface
    # ------------------------------------------------------------------

    def get_gate(self) -> nn.Module:
        return self.gate

    def get_num_experts(self) -> int:
        return self.num_experts

    def get_top_k(self) -> int:
        return self.top_k

    def get_norm_topk_prob(self) -> bool:
        return self.norm_topk_prob

    def compute_shared_expert(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._shared_graph is not None:
            return self._shared_graph(hidden_states)
        return self._compute_shared_eager(hidden_states)

    def _compute_shared_eager(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shared_out = self.shared_expert(hidden_states)
        return F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_out

    def predict_next_layer_experts(self, residual_cur, identity, bsh,
                                   shared_expert_output) -> Optional[List[int]]:
        if self.next_gate_weight is None:
            return None
        return self._predict_next_layer_experts(
            residual_cur, identity, bsh, shared_expert_output)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, hidden_states, residual=None, attn_weights=None,
                present_key_value=None, attention_mask=None, position_ids=None,
                output_attentions=False, cache_position=None,
                position_embeddings=None):

        final_hidden_states, router_logits = self.run_with_cache(
            hidden_states,
            residual=residual,
            attn_weights=attn_weights,
            present_key_value=present_key_value,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        return final_hidden_states, router_logits

    # ------------------------------------------------------------------
    # Predict next-layer top experts (Qwen2: lightweight layernorm-only approx)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _predict_next_layer_experts(self, residual_cur, identity, bsh,
                                    shared_expert_output) -> List[int]:
        """
        Approximate next-layer router scores using current residual stream.
        Skips attention (too expensive) — uses layernorm + gate only.
        Returns list of predicted top expert IDs for layer+1.
        """
        h = identity + shared_expert_output   # MoE output (approx)
        h = h.reshape(*bsh)
        if residual_cur is not None:
            h = residual_cur + h              # residual add
        next_residual = h
        h = self.next_input_layernorm(h)
        h = h + next_residual                 # skip attention
        h = self.next_post_attention_layernorm(h)

        logits = self.next_gate_weight(h.view(-1, bsh[2]))
        scores = F.softmax(logits, dim=1, dtype=torch.float)

        top_experts, _ = replaceset_between_tokens(
            scores.tolist(), self.replaceScoreRatio, self.top_k)
        return top_experts
