"""Avoid rebuilding an 8192-square mask for short dynamic-cache sequences.

Consumers index absolute cache positions. Keep rows 0..key_length-1 and the
same additive mask values, padding rule and SDPA fully-masked-row handling.
Static caches and unusual position layouts retain the checkpoint reference.
"""
import logging
import functools
import torch
from transformers.cache_utils import StaticCache


def compact_mask(binary, attention_mask, input_tensor, sdpa):
    length=attention_mask.shape[-1]
    dtype=input_tensor.dtype
    minimum=torch.finfo(dtype).min
    causal=binary[:length,:length][None,None].expand(input_tensor.shape[0],1,length,length).to(dtype)
    causal=causal * minimum
    padding=causal.eq(0.) * attention_mask[:,None,None,:].eq(0.)
    causal=causal.masked_fill(padding,minimum)
    if sdpa and torch.any(attention_mask != 1):
        causal=causal.mul(~torch.all(causal == minimum,dim=-1,keepdim=True)).to(dtype)
    return causal


def install(model):
    inner=model.model
    reference=inner._update_causal_mask
    inner._smoe_mask_reference=reference
    inner._smoe_compact_mask_allowed=False
    inner._smoe_compact_mask_engaged=False
    def before(module,args,kwargs):
        attention=kwargs.get('attention_mask')
        cache=kwargs.get('past_key_values')
        position=kwargs.get('cache_position')
        allowed=(not torch.is_grad_enabled() and attention is not None and attention.ndim==2
                 and module.config._attn_implementation in ('sdpa','eager')
                 and attention.shape[-1] <= module.causal_mask.shape[-1]
                 and not isinstance(cache,StaticCache)
                 and not any(isinstance(getattr(l.self_attn,'past_key_value',None),StaticCache)
                             for l in module.layers))
        if allowed and position is not None:
            # Preserve fallback for callers with absolute positions beyond the
            # provided mask, rather than assuming all calls came from generate.
            allowed=bool(torch.all((position>=0)&(position<attention.shape[-1])).item())
        module._smoe_compact_mask_allowed=allowed
    original_forward = inner.forward
    @functools.wraps(original_forward)
    def forward(*args, **kwargs):
        before(inner, args, kwargs)
        return original_forward(*args, **kwargs)
    # Install before the model timing wrapper, so the eligibility checks are
    # included in measured decode latency as well as the mask construction.
    inner.forward = forward
    def update(attention,input_tensor):
        if not inner._smoe_compact_mask_allowed:
            return reference(attention,input_tensor)
        if not inner._smoe_compact_mask_engaged:
            logging.getLogger(__name__).info('[compact causal mask] actual sequence extent engaged')
            inner._smoe_compact_mask_engaged=True
        return compact_mask(inner.causal_mask,attention,input_tensor,
                            inner.config._attn_implementation=='sdpa')
    inner._update_causal_mask=update
