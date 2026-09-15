"""
deepseek_moe.py — SMoE expert module implementation for DeepSeek MoE.

Provides:
  - DeepseekMLP              plain MLP expert
  - DeepseekMoEwithCache     cache-aware MoE block (inherits AbstractMoELayer)
"""

import time
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
import utils.expertcache as expertcache
from MoEModule.SMoE_base import AbstractMoELayer
from MoEModule.fused_gate_up import FusedGateUpMixin

logger = logging.getLogger(__name__)

ExpertUID = Tuple[int, int]


class DeepseekMLP(FusedGateUpMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size       = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
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


class DeepseekMoEwithCache(AbstractMoELayer):
    """
    DeepSeek MoE block with expert cache.

    DeepSeek-specific attributes vs AbstractMoELayer:
      - router: nn.Parameter (raw weight matrix, not nn.Linear)
      - shared_experts: optional shared MLP (n_shared_experts)
      - routing weights are manually normalized (norm_topk_prob=True equivalent)
      - prefetch uses layernorm-only approximation (same as Qwen, no attention run)
    """

    def __init__(self, config, expertcache_: ExpertCache, layerid, gate: nn.Parameter,
                 shared_experts, next_attention, next_gate_weight,
                 next_input_layernorm, next_post_attention_layernorm):
        super().__init__(config, expertcache_, layerid)

        self.num_experts        = config.n_routed_experts
        self.top_k              = config.num_experts_per_tok
        self.num_shared_experts = config.n_shared_experts

        self.router         = gate           # nn.Parameter (raw weight)
        self.shared_experts = shared_experts
        self._prefetch_score_order = os.environ.get("SMOE_PREFETCH_SCORE_ORDER", "0") == "1"
        self._prefetch_fused_norms = os.environ.get("SMOE_PREFETCH_FUSED_NORMS", "0") == "1"
        self._prefetch_fused_logged = False

        # Next-layer modules for prefetch prediction
        self.next_attention                = next_attention
        self.next_gate_weight              = next_gate_weight
        self.next_input_layernorm          = next_input_layernorm
        self.next_post_attention_layernorm = next_post_attention_layernorm

    # ------------------------------------------------------------------
    # AbstractMoELayer interface
    # ------------------------------------------------------------------

    def get_gate(self) -> nn.Module:
        # DeepSeek stores gate as a raw Parameter; wrap into a callable
        class _GateLinear:
            def __init__(self, weight):
                self.weight = weight
            def __call__(self, x):
                return F.linear(x, self.weight, None)
        return _GateLinear(self.router)

    def get_num_experts(self) -> int:
        return self.num_experts

    def get_top_k(self) -> int:
        return self.top_k

    def get_norm_topk_prob(self) -> bool:
        # Match MoEGate.forward: some DeepSeek checkpoints deliberately leave
        # the selected softmax mass below one (the supplied config uses False).
        return self.config.norm_topk_prob

    def get_router_softmax_dtype(self):
        # The checkpoint's MoEGate.forward keeps the logits dtype.
        return None

    def compute_shared_expert(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states is [T, H] (already flattened in run_with_cache).
        # DeepSeek shared_experts expects [B, S, H]; we treat T as B=1, S=T.
        if self.num_shared_experts is not None and self.shared_experts is not None:
            T, H = hidden_states.shape
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
    # forward
    # ------------------------------------------------------------------

    def forward(self, hidden_states, cache=None):
        ctx = cache.get_attn_context(self.layerid) if cache is not None else {}
        residual_cur          = ctx.get('residual')
        attn_weights_cur      = ctx.get('attn_weights')
        present_key_value_cur = cache
        attention_mask        = ctx.get('attention_mask')
        position_ids          = ctx.get('position_ids')
        output_attentions     = ctx.get('output_attentions', False)

        final_hidden_states, router_logits = self.run_with_cache(
            hidden_states,
            residual=residual_cur,
            attn_weights=attn_weights_cur,
            present_key_value=present_key_value_cur,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
        )
        return final_hidden_states, router_logits

    # ------------------------------------------------------------------
    # Predict next-layer top experts (layernorm-only approximation, no attention)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_next_top_expert(self, residual_cur, hidden_states, raw_hidden, bsh):
        h = None
        if self._prefetch_fused_norms and bsh[0] * bsh[1] == 1:
            from utils.prefetch_norm import fused_prefetch_norms
            h = fused_prefetch_norms(raw_hidden, hidden_states, residual_cur,
                                    self.next_input_layernorm, self.next_post_attention_layernorm)
            if h is not None and not self._prefetch_fused_logged:
                logger.info("[prefetch fused norms] layer=%d engaged", self.layerid)
                self._prefetch_fused_logged = True
        if h is None:
            h = raw_hidden + hidden_states   # approx MoE output
            h = h.reshape(*bsh)
            if residual_cur is not None:
                h = residual_cur + h         # residual add

            next_residual = h
            h = self.next_input_layernorm(h)
            h = h + next_residual            # skip attention
            h = self.next_post_attention_layernorm(h)

        batch_size, sequence_length, hidden_dim = bsh
        logits = F.linear(h.view(-1, hidden_dim), self.next_gate_weight, None)
        scores = logits.softmax(dim=-1)
        scores_list = scores.tolist()
        top_experts, _ = replaceset_between_tokens(
            scores_list, self.replaceScoreRatio, self.top_k)
        if self._prefetch_score_order and batch_size * sequence_length == 1:
            top_experts.sort(key=lambda eid: scores_list[0][eid], reverse=True)
        return top_experts
