"""
xverse_moe.py — SMoE expert module implementation for Xverse MoE.

Provides:
  - XverseMLP              plain MLP expert
  - XverseMoEMLPwithCache  cache-aware MoE block (inherits AbstractMoELayer)

All common inference logic (B0–B14, persistent bg-thread, CPU compute) is
implemented in SMoE_base.AbstractMoELayer. This file only contains the
Xverse-specific parts:
  - Gate: self.router (nn.Linear)
  - Shared experts: self.shared_experts (optional)
  - Two calling conventions: patcher mode (SMoECache) and direct mode
  - Prefetch predict: full attention run (similar to DeepSeek)
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
from MoEModule.fused_gate_up import FusedGateUpMixin

logger = logging.getLogger(__name__)

ExpertUID = Tuple[int, int]


class XverseMLP(FusedGateUpMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size       = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False, dtype=torch.bfloat16, device=config.device)
        self.up_proj   = nn.Linear(self.hidden_size, self.intermediate_size, bias=False, dtype=torch.bfloat16, device=config.device)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False, dtype=torch.bfloat16, device=config.device)
        self.act_fn = ACT2FN[config.hidden_act]
        self._init_fused_gate_up()

    def forward(self, x):
        fused = self._forward_fused_gate_up(x)
        if fused is not None:
            return fused
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices   = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)
            gate_proj = torch.cat([F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)
            up_proj   = torch.cat([F.linear(x, up_proj_slices[i])   for i in range(self.config.pretraining_tp)], dim=-1)
            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = sum(F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp))
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class XverseMoEMLPwithCache(AbstractMoELayer):
    """
    Xverse MoE block with expert cache.

    Xverse-specific attributes vs AbstractMoELayer:
      - router: nn.Linear (gate)
      - shared_experts: optional shared MLP
      - routing weights normalized manually (norm_topk_prob=True equivalent)
      - prefetch uses layernorm-only approximation (same as Qwen, no attention run)
    """

    def __init__(self, config, expertcache_: ExpertCache, layerid, gate: nn.Linear,
                 shared_experts, next_attention, next_gate_weight,
                 next_input_layernorm, next_post_attention_layernorm):
        super().__init__(config, expertcache_, layerid)

        self.num_experts        = config.num_experts
        self.top_k              = config.moe_top_k
        self.num_shared_experts = config.num_shared_experts

        self.router         = gate
        self.shared_experts = shared_experts
        self._shared_graph = None
        self._validate_shared = os.environ.get('SMOE_VALIDATE_GROUPED', '0') == '1'
        if shared_experts is not None and os.environ.get('SMOE_XVERSE_SHARED_GRAPH', '0') == '1':
            from MoEModule.decode_graph import DecodeExpertGraph
            self._shared_graph = DecodeExpertGraph(shared_experts, config.hidden_size, config.device)

        # Next-layer modules for prefetch prediction
        self.next_attention                = next_attention
        self.next_gate_weight              = next_gate_weight
        self.next_input_layernorm          = next_input_layernorm
        self.next_post_attention_layernorm = next_post_attention_layernorm

    # ------------------------------------------------------------------
    # AbstractMoELayer interface
    # ------------------------------------------------------------------

    def get_gate(self) -> nn.Module:
        return self.router

    def get_num_experts(self) -> int:
        return self.num_experts

    def get_top_k(self) -> int:
        return self.top_k

    def get_norm_topk_prob(self) -> bool:
        # Xverse normalizes routing weights manually (same as DeepSeek)
        return True

    def compute_shared_expert(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states is [T, H] (already flattened in run_with_cache).
        if self.num_shared_experts is not None and self.shared_experts is not None:
            T, H = hidden_states.shape
            if self._shared_graph is not None and T == 1:
                result = self._shared_graph(hidden_states).view(T, H)
                if self._validate_shared:
                    reference = self.shared_experts(hidden_states.view(1, T, H)).view(T, H)
                    torch.testing.assert_close(result, reference, rtol=0, atol=0)
                    logger.info('[Shared graph validation] layer=%d exact', self.layerid)
                    self._validate_shared = False
                return result
            return self.shared_experts(hidden_states.view(1, T, H)).view(T, H)
        return torch.zeros_like(hidden_states)

    def predict_next_layer_experts(self, residual_cur, identity, bsh,
                                   shared_expert_output) -> Optional[List[int]]:
        if self.next_gate_weight is None:
            return None
        return self.get_next_top_expert(
            residual_cur,
            hidden_states=shared_expert_output,
            raw_hidden=identity,
            bsh=bsh)

    # ------------------------------------------------------------------
    # forward — supports two calling conventions
    # ------------------------------------------------------------------

    def forward(self, hidden_states, residual_or_cache=None,
                attn_weights=None, present_key_value=None,
                attention_mask=None, position_ids=None,
                output_attentions=False, cache_position=None):
        """
        Two calling conventions:
          1. Patcher mode: (hidden_states, cache=SMoECache) — attn context from cache
          2. Direct mode:  (hidden_states, residual, attn_weights, present_key_value, ...)
        """
        from utils.cache import SMoECache
        if isinstance(residual_or_cache, SMoECache):
            cache                 = residual_or_cache
            ctx                   = cache.get_attn_context(self.layerid)
            residual_cur          = ctx.get('residual')
            attn_weights_cur      = ctx.get('attn_weights')
            present_key_value_cur = cache
            attention_mask_       = ctx.get('attention_mask')
            position_ids_         = ctx.get('position_ids')
            output_attentions_    = ctx.get('output_attentions', False)
            cache_position_       = ctx.get('cache_position')
        else:
            residual_cur          = residual_or_cache
            attn_weights_cur      = attn_weights
            present_key_value_cur = present_key_value
            attention_mask_       = attention_mask
            position_ids_         = position_ids
            output_attentions_    = output_attentions
            cache_position_       = cache_position

        final_hidden_states, router_logits = self.run_with_cache(
            hidden_states,
            residual=residual_cur,
            attn_weights=attn_weights_cur,
            present_key_value=present_key_value_cur,
            attention_mask=attention_mask_,
            position_ids=position_ids_,
            output_attentions=output_attentions_,
            cache_position=cache_position_,
        )
        return final_hidden_states, router_logits

    # ------------------------------------------------------------------
    # Predict next-layer top experts (layernorm-only approximation, no attention)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_next_top_expert(self, residual_cur, hidden_states, raw_hidden, bsh):
        h = raw_hidden + hidden_states   # approx MoE output
        h = h.reshape(*bsh)
        if residual_cur is not None:
            h = residual_cur + h         # residual add

        next_residual = h
        h = self.next_input_layernorm(h)
        h = h + next_residual            # skip attention
        h = self.next_post_attention_layernorm(h)

        batch_size, sequence_length, hidden_dim = bsh
        logits = self.next_gate_weight(h.view(-1, hidden_dim))
        scores = logits.softmax(dim=-1)
        top_experts, _ = replaceset_between_tokens(
            scores.tolist(), self.replaceScoreRatio, self.top_k)
        return top_experts
