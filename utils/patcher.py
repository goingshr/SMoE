"""
SMoE patcher — monkey-patches loaded causal-LM instances (DeepseekForCausalLM,
Qwen2MoeForCausalLM, XverseForCausalLM) so that SMoE expert caching statistics
work without any modification to modeling_*.py files.

Usage (inside utils.build_model after MoE layers are replaced):
    from utils.patcher import patch_model_forward
    patch_model_forward(model, model_type)   # model_type: "deepseekmoe" | "qwenmoe" | "xversemoe"
"""

import time
import functools
import warnings
import logging
import torch
from transformers.cache_utils import Cache, DynamicCache

from utils.cache import SMoECache
import utils.expertcache as expertcachewithscorefullcpu

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compatibility shim: transformers>=4.38 renamed get_max_length→get_seq_length
# Some model files (e.g. DeepSeek's modeling_deepseek.py) still call the old
# name.  Patch it onto DynamicCache and SMoECache so we don't need to touch
# any model file.
# ---------------------------------------------------------------------------
if not hasattr(DynamicCache, "get_max_length"):
    DynamicCache.get_max_length = DynamicCache.get_seq_length


def patch_deepseek_model(model):
    """
    Patch a DeepseekForCausalLM instance in-place.
    Must be called after DeepseekMoEwithCache layers have been installed by build_model().
    """
    from MoEModule import DeepseekMoEwithCache

    for layer_idx, layer in enumerate(model.model.layers):
        _patch_sdpa_attention(layer.self_attn)
        _patch_decoder_layer(layer, layer_idx, DeepseekMoEwithCache)

    _patch_inner_model_forward(model.model, use_smoe_cache=True)


def patch_model_forward(model, model_type: str):
    """
    Universal entry point: patch the inner model's forward for any supported model_type.
    Injects prefill/decode timing and per-token expert hit-rate statistics —
    works for deepseekmoe, qwenmoe, and xversemoe.

    All three models use SMoECache so attention patcher can store KV context.
    For deepseekmoe, also patches attention and decoder layers.
    """
    if model_type == "deepseekmoe":
        patch_deepseek_model(model)
    elif model_type in ("qwenmoe", "xversemoe"):
        _patch_inner_model_forward(model.model, use_smoe_cache=True)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ---------------------------------------------------------------------------
# Attention patch
# ---------------------------------------------------------------------------

def _patch_sdpa_attention(attn_module):
    """
    Wrap attention forward to accept an if_update kwarg.
    When if_update=False, sets cache._readonly=True so that SMoECache.update()
    returns existing KV without writing — allowing prefetch attention runs.
    """
    original_forward = attn_module.forward

    @functools.wraps(original_forward)
    def patched_forward(hidden_states, attention_mask=None, position_ids=None,
                        past_key_value=None, output_attentions=False, use_cache=False,
                        if_update=True, **kwargs):
        readonly_set = (not if_update) and isinstance(past_key_value, SMoECache)
        if readonly_set:
            past_key_value._readonly = True
        try:
            return original_forward(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )
        finally:
            if readonly_set:
                past_key_value._readonly = False

    attn_module.forward = patched_forward


# ---------------------------------------------------------------------------
# Decoder layer patch
# ---------------------------------------------------------------------------

def _patch_decoder_layer(layer, layer_idx, DeepseekMoEwithCache):
    """
    Replace each decoder layer's forward with a version that:
    - Stores attention context in SMoECache (residual, attn_weights, mask, etc.)
    - Calls self.mlp(hidden_states, cache) for MoE layers so they can do prefetching.
    layer_idx is passed explicitly because DeepseekDecoderLayer has no layer_idx attribute.
    """
    original_forward = layer.forward

    @functools.wraps(original_forward)
    def patched_forward(hidden_states, attention_mask=None, position_ids=None,
                        past_key_value=None, output_attentions=False, use_cache=False,
                        **kwargs):
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. "
                "Please make sure use `attention_mask` instead."
            )

        s1 = time.time()
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = layer.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            if_update=True,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        e1 = time.time()
        logger.debug("timeofattention %s %s", layer_idx, e1 - s1)

        if isinstance(layer.mlp, DeepseekMoEwithCache):
            if isinstance(past_key_value, SMoECache):
                past_key_value.set_attn_context(
                    layer_idx,
                    residual=residual,
                    attn_weights=self_attn_weights,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    output_attentions=output_attentions,
                )
            hidden_states, _ = layer.mlp(hidden_states, past_key_value)
        else:
            hidden_states = layer.mlp(hidden_states)

        s2 = time.time()
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        e2 = time.time()
        logger.debug("timeexceptffn %s %s", layer_idx, e2 - s2)
        return outputs

    layer.forward = patched_forward


# ---------------------------------------------------------------------------
# Inner model (DeepseekModel) forward patch
# ---------------------------------------------------------------------------

def _patch_inner_model_forward(inner_model, use_smoe_cache: bool = False):
    """
    Wrap the inner model's forward to inject prefill/decode timing and
    per-token expert GPU hit-rate statistics (logged at INFO level).

    use_smoe_cache=True  → also wraps past_key_values in SMoECache (deepseek only).
    use_smoe_cache=False → statistics only; no KV-cache replacement (qwen/xverse).
    """
    original_forward = inner_model.forward

    @functools.wraps(original_forward)
    def patched_forward(input_ids=None, attention_mask=None, position_ids=None,
                        past_key_values=None, inputs_embeds=None, use_cache=None,
                        output_attentions=None, output_hidden_states=None,
                        return_dict=None, **kwargs):
        s = time.time()

        # SMoECache wrap only for deepseek (attention patcher requires it)
        if use_smoe_cache and use_cache and not isinstance(past_key_values, SMoECache):
            past_key_values = SMoECache.from_legacy_cache(past_key_values)

        result = original_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        e = time.time()
        ec = expertcachewithscorefullcpu
        if ec.tokens == 0:
            # ── Prefill complete (first forward pass) ────────────────────
            prefill_elapsed = e - s
            ec.prefill_time = prefill_elapsed
            ec.tokens      += 1
            logger.info("[SMoE] prefill_time=%.4f s", prefill_elapsed)
            # MoE layers update the same counters during prefill.  They must
            # not leak into token=1 or prompt-level decode hit-rate metrics.
            ec.cache_hits_per_token = 0
            ec.cache_total_per_token = 0
        else:
            # ── Decode token ─────────────────────────────────────────────
            token_elapsed  = e - s
            ec.decode_time += token_elapsed
            ec.tokens      += 1
            decode_idx      = ec.tokens - 1          # 1-based token index
            avg_decode      = ec.decode_time / decode_idx

            hits     = ec.cache_hits_per_token
            total    = ec.cache_total_per_token
            hit_rate = hits / total if total > 0 else float('nan')
            ec.cache_hits_prompt += hits
            ec.cache_total_prompt += total

            logger.info(
                "[SMoE] token=%d  decode=%.4f s  avg_decode=%.4f s  "
                "gpu_hit_rate=%.3f (%d/%d)",
                decode_idx, token_elapsed, avg_decode,
                hit_rate, hits, total,
            )

            # Reset per-token accumulators for the next token
            ec.cache_hits_per_token  = 0
            ec.cache_total_per_token = 0

        return result

    inner_model.forward = patched_forward
