from MoEModule import (
    DeepseekMLP, DeepseekMoEwithCache,
    Qwen2MoeMLP, Qwen2MoeSparseMoeBlockwithCache,
    XverseMLP, XverseMoEMLPwithCache,
)
from typing import Tuple
from utils.expertcache import ExpertCache
import os
import json
from dataclasses import dataclass
import torch
from configs.configuration_deepseek import DeepseekConfig
from configs.qwen2_config import Qwen2MoeConfig
from configs.configuration_xverse import XverseConfig
from safetensors.torch import load_file

from torch import nn
from tqdm.auto import trange
from contextlib import contextmanager
# Qwen2MoE is in the official transformers library — import directly.
from models.modeling_qwen import Qwen2MoeForCausalLM
# DeepSeek / Xverse are not in official transformers — loaded lazily from model_path.
# The actual classes are resolved inside build_model() after model_path is known.
import psutil
import gc,sys
import copy
import numpy as np
import logging

logger = logging.getLogger(__name__)
current_supportmoe = ["deepseekmoe","qwenmoe","xversemoe"]
ExpertUID = Tuple[int,int]

@contextmanager
def with_default_dtype(dtype):
    _dtype_original = torch.get_default_dtype()

    try:
        torch.set_default_dtype(dtype)
        yield
    finally:
        torch.set_default_dtype(_dtype_original)

def nested_flatten(t):
    if isinstance(t, (list, tuple)):
        for x in t:
            yield from nested_flatten(x)
    elif isinstance(t, dict):
        for k, v in sorted(t.items()):
            yield from nested_flatten(v)
    else:
        yield t

def nested_pack(flat, structure):
    """
    Restore nested structure from flattened state
    :param flat: result of nested_flatten
    :param structure: used as example when recovering structure
    :returns: nested structure like :structure: filled with elements of :flat:
    """
    return _nested_pack(iter(flat), structure)

def _nested_pack(flat_iter, structure):
    if is_namedtuple(structure):
        return type(structure)(*[_nested_pack(flat_iter, x) for x in structure])
    elif isinstance(structure, (list, tuple)):
        return type(structure)(_nested_pack(flat_iter, x) for x in structure)
    elif isinstance(structure, dict):
        return {k: _nested_pack(flat_iter, v) for k, v in sorted(structure.items())}
    else:
        return next(flat_iter)

def is_namedtuple(x):
    """Checks if x is a namedtuple instance. Taken from https://stackoverflow.com/a/2166841 ."""
    t = type(x)
    b = t.__bases__
    if len(b) != 1 or b[0] != tuple:
        return False
    f = getattr(t, "_fields", None)
    if not isinstance(f, tuple):
        return False
    return all(type(n) == str for n in f)


@dataclass(frozen=True)
class OffloadConfig:
    main_size: int
    offload_size: int

def make_empty_expert(
    model_config,
    model_type: str
):
    # logger.debug(model_type)
    if model_type == "deepseekmoe":
        return DeepseekMLP(model_config)
    elif model_type == "qwenmoe":
        # logger.debug(111111111111)
        return Qwen2MoeMLP(model_config)
    elif model_type == "xversemoe":
        return XverseMLP(model_config)
    else:
        raise ValueError("This model is not supported")


def _read_weight_map(states_dir: str) -> dict:
    # Try safetensors index first, then PyTorch bin index
    st_index = os.path.join(states_dir, "model.safetensors.index.json")
    bin_index = os.path.join(states_dir, "pytorch_model.bin.index.json")
    if os.path.exists(st_index):
        with open(st_index) as f:
            return json.load(f)["weight_map"]
    elif os.path.exists(bin_index):
        with open(bin_index) as f:
            return json.load(f)["weight_map"]
    else:
        raise FileNotFoundError(
            f"No weight index found in {states_dir}. "
            "Expected model.safetensors.index.json or pytorch_model.bin.index.json"
        )


def _load_shard(full_path: str, device: str = "cpu") -> dict:
    """Load a weight shard — supports both .safetensors and .bin formats."""
    if full_path.endswith(".safetensors"):
        return load_file(full_path, device=device)
    else:
        import torch as _torch
        return _torch.load(full_path, map_location=device, weights_only=True)


def _is_original_hf_format(weight_map: dict) -> bool:
    """True if this is an original HF sharded checkpoint rather than pre-split by div_tensors.py."""
    for key, fpath in weight_map.items():
        if "mlp.experts" in key:
            # div_tensors.py names files "expert_<layer>_<idx>.safetensors"
            return not fpath.startswith("expert_")
    return False  # no expert keys found


def _load_non_expert_params(states_dir: str, weight_map: dict) -> dict:
    """
    Collect all non-expert parameters from an original HF sharded checkpoint.
    Loads each shard at most once, then frees it.
    Works for all three model types (deepseekmoe / qwenmoe / xversemoe) because
    all three store routed experts under 'mlp.experts.*'.
    """
    shard_to_keys: dict[str, list] = {}
    for key, fpath in weight_map.items():
        if "mlp.experts" not in key:
            shard_to_keys.setdefault(fpath, []).append(key)

    combined = {}
    for fpath, keys in shard_to_keys.items():
        shard = _load_shard(os.path.join(states_dir, fpath), device="cpu")
        for key in keys:
            combined[key] = shard[key]
        del shard
    return combined


def _extract_expert_dict(shard: dict, layer_idx: int, expert_idx: int) -> dict:
    """Strip the full-path prefix from one expert's weights inside an original HF shard."""
    prefix = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}."
    return {k[len(prefix):]: v for k, v in shard.items() if k.startswith(prefix)}


def make_and_load_expert_wrapper(
    config,
    states_dir: str,
    expert_uid: tuple[int, int],
    model_type,
    device: torch.device,
    _weight_map: dict = None,
    _shard_cache: dict = None,
):
    layer_idx, expert_idx = expert_uid

    if _weight_map is None:
        _weight_map = _read_weight_map(states_dir)

    prefix = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}"

    # Collect all distinct shards containing any weight for this expert
    expert_key_to_shard = {k: v for k, v in _weight_map.items() if k.startswith(prefix + ".")}
    shard_files = set(expert_key_to_shard.values())

    # Primary shard (the one we'd like to keep in the cache)
    primary_shard_fpath = _weight_map[f"{prefix}.up_proj.weight"]

    state_dict = {}
    for shard_fpath in shard_files:
        full_path = os.path.join(states_dir, shard_fpath)
        if _shard_cache is not None and shard_fpath == primary_shard_fpath:
            # Keep the primary shard in cache (evict previous if different)
            if shard_fpath not in _shard_cache:
                _shard_cache.clear()
                _shard_cache[shard_fpath] = _load_shard(full_path, device="cpu")
            shard = _shard_cache[shard_fpath]
        else:
            # Secondary shard (rare cross-shard case): load on demand, don't cache
            shard = _load_shard(full_path, device="cpu")

        # Detect original HF format: full-path keys like "model.layers.X..."
        # Can't use next(iter()) safely since lm_head.weight may come first
        is_full_path = any(k.startswith("model.layers") for k in shard)
        if is_full_path:
            state_dict.update(_extract_expert_dict(shard, layer_idx, expert_idx))
        else:
            state_dict.update(shard)

    if model_type == "deepseekmoe" and os.environ.get("SMOE_HOST_EXPERT_INIT", "1") == "1":
        # These experts are immediately packed into CPU backing storage.
        # Construct on meta and attach checkpoint tensors to avoid random
        # initialization and a full CPU->GPU->CPU weight round trip per expert.
        host_config = copy.copy(config)
        host_config.device = "meta"
        expert = make_empty_expert(host_config, model_type)
        expert.load_state_dict(state_dict, strict=True, assign=True)
        expert.config = config
    else:
        expert = make_empty_expert(config, model_type)
        expert.load_state_dict(state_dict, strict=True)
    return ExpertWrapper(expert, model_type, device, tocpu=True)


def load_00_expert_state_dict(states_dir: str, model_type: str, device: torch.device):
    # deepseekmoe MoE starts at layer 1; qwenmoe and xversemoe start at layer 0
    first_layer = 1 if model_type == "deepseekmoe" else 0
    if model_type not in ("deepseekmoe", "qwenmoe", "xversemoe"):
        raise ValueError("This model is not supported")

    weight_map = _read_weight_map(states_dir)
    lookup_key = f"model.layers.{first_layer}.mlp.experts.0.gate_proj.weight"
    shard = _load_shard(os.path.join(states_dir, weight_map[lookup_key]), device=str(device))

    if _is_original_hf_format(weight_map):
        # Original HF format: strip prefix
        return _extract_expert_dict(shard, first_layer, 0)
    return shard


class ExpertWrapper(nn.Module):
    def __init__(
            self,
            expert_module: DeepseekMLP,
            model_type:str,
            device: torch.device,
            tocpu:bool
    ):
        super().__init__()
        self.model_type = model_type
        self.replace_layer_storage = None
        if model_type == "deepseekmoe":
            self.replace_layer_storage = self.replace_layer_storage_deepseekmoe
        if model_type == "qwenmoe":
        # qwenmoe expert structure is the same as deepseekmoe.
            self.replace_layer_storage = self.replace_layer_storage_deepseekmoe
        if model_type == "xversemoe":
        # xversemoe expert structure is the same as deepseekmoe.
            self.replace_layer_storage = self.replace_layer_storage_deepseekmoe
        if self.replace_layer_storage == None:
            raise ValueError("This model is not supported")
        expert_module, self.storage = self.replace_layer_storage(expert_module, device,tocpu)
        self.cpu_bf16_gate_up_fused = (
            getattr(expert_module, "_cpu_gate_up_weight", None) is not None
        )
        self.gpu_bf16_gate_up_fused = (
            getattr(expert_module, "_gpu_gate_up_weight", None) is not None
        )
        self.expert_module = lambda *args, **kwargs: expert_module(*args, **kwargs)
        self.decode_graph = None
        if (not tocpu and model_type == "qwenmoe"
                and os.environ.get("SMOE_EXPERT_GRAPH", "1") == "1"):
            from MoEModule.decode_graph import DecodeExpertGraph
            self.decode_graph = DecodeExpertGraph(
                self.expert_module, expert_module.hidden_size, device)
        self._register_state_dict_hook(self._add_storage_to_state_dict_hook)
        self._register_load_state_dict_pre_hook(self._load_storage_from_state_dict_hook)

    @staticmethod
    def _add_storage_to_state_dict_hook(self, state_dict, prefix, local_metadata):
        state_dict[prefix + 'storage'] = torch.as_tensor(self.storage, dtype=torch.uint8)
        return state_dict

    def _load_storage_from_state_dict_hook(self, state_dict, prefix, local_metadata, strict, missing_keys,
                                           unexpected_keys, error_msgs):
        self.storage.copy_(state_dict[prefix + 'storage'].storage().untyped())
        del state_dict[prefix + 'storage']

    def forward(self, *args, **kwargs):
        if self.decode_graph is not None and len(args) == 1 and not kwargs:
            return self.decode_graph(args[0])
        return self.expert_module(*args, **kwargs)

    def replace_layer_storage_deepseekmoe(self,
            layer: DeepseekMLP,
            device: torch.device,
            tocpu:bool
    ):
        # logger.debug("------")
        # logger.debug(f"Current CPU memory usage: {psutil.Process().memory_info().rss / (1024 ** 2):.2f} MB")
        if self.model_type in {"qwenmoe", "deepseekmoe", "xversemoe"}:
            # Adjacent gate/up weights expose one zero-copy BF16 matrix to the
            # fused CPU projection. GPU slots use the same raw storage layout.
            states = [
                ("gate_proj", getattr(layer, "gate_proj").weight.data),
                ("up_proj", getattr(layer, "up_proj").weight.data),
                ("down_proj", getattr(layer, "down_proj").weight.data),
            ]
        else:
            states = [
                ("gate_proj", getattr(layer, "gate_proj").weight.data),
                ("down_proj", getattr(layer, "down_proj").weight.data),
                ("up_proj", getattr(layer, "up_proj").weight.data),
            ]

        storage_size = 0
        offsets = [0]

        for k,x in states:
            if not isinstance(x, torch.Tensor):
                continue
            storage_size += x.nbytes
            offsets.append(storage_size)
        if tocpu:
            # Every GPU slot has CPU backing for later eviction. DeepSeek's
            # full expert set needs pageable memory to use host RAM + swap;
            # keep pinned transfers for the smaller Qwen/Xverse workloads.
            cpu_tensor = torch.empty(
                storage_size, dtype=torch.uint8, device="cpu",
                pin_memory=(self.model_type != "deepseekmoe"
                            and torch.cuda.is_available()),
            )
            storage = cpu_tensor.untyped_storage()
        else:
            storage = torch.UntypedStorage(storage_size, device=device)
        # logger.debug(f"Current CPU memory usage1: {psutil.Process().memory_info().rss / (1024 ** 2):.2f} MB")
        i = 0
        # logger.debug("------")
        newtensors = dict()
        for k,x in states:
            if not isinstance(x, torch.Tensor):
                continue
            # logger.debug(k)
            start = offsets[i]
            end = offsets[i + 1]
            if tocpu:
                a_view = torch.as_tensor(storage[start:end], dtype=x.dtype, device="cpu").view(x.shape)
            else:
                a_view = torch.as_tensor(storage[start:end], dtype=x.dtype, device=device).view(x.shape)
            a_view[...] = x
            assert a_view.data_ptr() == storage.data_ptr() + start
            i += 1
            newtensors[k]=a_view

        for k, newtensor in newtensors.items():
            patched = getattr(layer, k)
            patched.weight.data = newtensor
            assert patched.weight.data.data_ptr() == newtensor.data_ptr()

        configure_gate_up = getattr(layer, "configure_cpu_bf16_gate_up" if tocpu
                                    else "configure_gpu_bf16_gate_up", None)
        if configure_gate_up is not None:
            gate = newtensors["gate_proj"]
            up = newtensors["up_proj"]
            assert gate.data_ptr() + gate.nbytes == up.data_ptr()
            gate_up_storage = storage[offsets[0]:offsets[2]]
            gate_up = torch.as_tensor(
                gate_up_storage, dtype=gate.dtype, device=gate.device
            ).view(gate.size(0) + up.size(0), gate.size(1))
            configure_gate_up(gate_up)
        return layer, storage
def _make_module_cuda(model_path,model_type,device,state_dict_00):
    config = None
    if model_type == "deepseekmoe":
        config = DeepseekConfig.from_pretrained(model_path,torch_dtype=torch.bfloat16,device_map=device)
    if model_type == "qwenmoe":
        config = Qwen2MoeConfig.from_pretrained(model_path,torch_dtype=torch.bfloat16,device_map=device)
    if model_type == "xversemoe":
        config = XverseConfig.from_pretrained(model_path,torch_dtype=torch.bfloat16,device_map=device)
    with torch.no_grad():
        expert = make_empty_expert(config, model_type)
        expert.load_state_dict(state_dict_00)
    expertwrapper = ExpertWrapper(expert, model_type,device=device,tocpu=False)
    return expertwrapper
def _make_module_cpu(model_path,model_type,device,state_dict_00):
    config = None
    if model_type == "deepseekmoe":
        config = DeepseekConfig.from_pretrained(model_path,torch_dtype=torch.bfloat16,device_map="cpu")
    if model_type == "qwenmoe":
        config = Qwen2MoeConfig.from_pretrained(model_path,torch_dtype=torch.bfloat16,device_map="cpu")
    if model_type == "xversemoe":
        config = XverseConfig.from_pretrained(model_path,torch_dtype=torch.bfloat16,device_map="cpu")
    expert = make_empty_expert(config, model_type)
    expert.load_state_dict(state_dict_00)
    expertwrapper = ExpertWrapper(expert, model_type,device=device,tocpu=True)
    return expertwrapper


def _compute_expert_size_bytes(model_config, model_type: str) -> int:
    """Return the number of GPU bytes occupied by one expert's weight matrices (bfloat16).

    Each expert has 3 weight matrices (gate_proj / up_proj / down_proj),
    each of shape (intermediate_size, hidden_size) in bfloat16 = 2 bytes.
    """
    hidden = model_config.hidden_size
    if model_type in ("deepseekmoe", "qwenmoe"):
        inter = model_config.moe_intermediate_size
    elif model_type == "xversemoe":
        inter = model_config.intermediate_size
    else:
        raise ValueError(f"_compute_expert_size_bytes: unknown model_type={model_type}")
    return 3 * hidden * inter * 2


def build_model(
    model_path: str,
    model_type: str,
    device: torch.device,
    main_size: int = None,
    config_path: str = None,
    gpu_mem_gb: float = None,
    # predictor1: Predictor,
    # predictor2: Predictor
):
    # config_path overrides where config.json is loaded from (weights still from model_path)
    cfg_path = config_path if config_path else model_path
    state_dict_00 = load_00_expert_state_dict(model_path,model_type, device)
    if model_type not in current_supportmoe:
        raise ValueError("This model is not supported")
    logger.info("GPU memory allocation for common params begins.")
    logger.info(f"Loading config from: {cfg_path}")
    logger.info(f"Loading weights from: {model_path}")
    if model_type == "deepseekmoe":
        # DeepSeek is not in the official transformers library.
        # Load the model class from the model directory (trust_remote_code) or
        # from the lazy proxy that resolves it from model_path.
        import models.modeling_deepseek as _ds_mod
        _ds_mod.set_model_path(model_path)
        DeepseekForCausalLM = _ds_mod.DeepseekForCausalLM
        with device, with_default_dtype(torch.bfloat16):
            oldconfig = DeepseekConfig.from_pretrained(
                    cfg_path,
                    n_routed_experts=0,
                    torch_dtype=torch.bfloat16,
                    device_map=device,
                    trust_remote_code=True
                )
            model = DeepseekForCausalLM(oldconfig)
        model_config = DeepseekConfig.from_pretrained(cfg_path,trust_remote_code=True)
    elif model_type == "qwenmoe":
        with device, with_default_dtype(torch.bfloat16):
            # num_experts=1 (not 0) so that transformers creates Qwen2MoeSparseMoeBlock
            # (which has shared_expert / shared_expert_gate / gate attributes).
            # With num_experts=0 transformers falls back to a plain dense MLP,
            # which is missing those attributes that model_loader needs to splice
            # in the Qwen2MoeSparseMoeBlockwithCache replacements.
            oldconfig = Qwen2MoeConfig.from_pretrained(
                    cfg_path,
                    num_experts=1,
                    torch_dtype=torch.bfloat16,
                    device_map=device,
                    trust_remote_code=True
                )
            model = Qwen2MoeForCausalLM(oldconfig)
        model_config = Qwen2MoeConfig.from_pretrained(cfg_path,trust_remote_code=True)
    elif model_type == "xversemoe":
        # Xverse is not in the official transformers library.
        import models.modeling_xverse as _xv_mod
        _xv_mod.set_model_path(model_path)
        XverseForCausalLM = _xv_mod.XverseForCausalLM
        with device, with_default_dtype(torch.bfloat16):
            oldconfig = XverseConfig.from_pretrained(
                    cfg_path,
                    num_experts=0,
                    torch_dtype=torch.bfloat16,
                    device_map=device,
                    trust_remote_code=True
                )
            model = XverseForCausalLM(oldconfig)
        model_config = XverseConfig.from_pretrained(cfg_path,trust_remote_code=True)
    print("Init GPU memory cost", torch.cuda.memory_summary())
    print("GPU memory allocation for common params ends.")
    torch.cuda.empty_cache()
    
    offload_size_ = None
    if model_type == "deepseekmoe":
        offload_size_ = (model_config.num_hidden_layers-1)*model_config.n_routed_experts
    if model_type == "qwenmoe":
        offload_size_ = model_config.num_hidden_layers*model_config.num_experts
    if model_type == "xversemoe":
        offload_size_ = model_config.num_hidden_layers*model_config.num_experts
    if offload_size_ == None:
        raise ValueError("This model is not supported")
    
    # ── Auto-compute main_size if not explicitly provided ─────────────────────
    # Measured after the model skeleton has been allocated on GPU but before
    # ExpertCache is created.  At this point the skeleton already holds all
    # non-expert parameters (attention, embedding, layer-norms), so
    # torch.cuda.memory_allocated() gives a good estimate of that overhead.
    if main_size is None:
        if gpu_mem_gb is None:
            raise ValueError(
                "build_model: either main_size or gpu_mem_gb must be provided."
            )
        expert_bytes = _compute_expert_size_bytes(model_config, model_type)
        total_budget = int(gpu_mem_gb * (1024 ** 3))
        if torch.cuda.is_available() and str(device) != 'cpu':
            skeleton_used = torch.cuda.memory_allocated(device)
        else:
            skeleton_used = 0
        available = max(0, total_budget - skeleton_used)
        main_size = max(1, int(available / expert_bytes))
        logger.info(
            "[AUTO CACHE] gpu_mem=%.1f GB  skeleton_used=%.2f GB  "
            "expert_size=%.2f MB  -> cache_size=%d",
            gpu_mem_gb,
            skeleton_used / (1024 ** 3),
            expert_bytes / (1024 ** 2),
            main_size,
        )

    # wrap make_module_* so they load config from cfg_path, weights from model_path
    def _make_module_cuda_cfg(mp, mt, dev, sd):
        return _make_module_cuda(cfg_path, mt, dev, sd)
    def _make_module_cpu_cfg(mp, mt, dev, sd):
        return _make_module_cpu(cfg_path, mt, dev, sd)

    if model_type == "deepseekmoe":
        for layer_idx in range(model_config.first_k_dense_replace, model_config.num_hidden_layers):
            model.model.layers[layer_idx].mlp.gate.weight = nn.Parameter(torch.empty((model_config.n_routed_experts, model_config.hidden_size),device = model_config.device,dtype=torch.bfloat16))
    if model_type == "qwenmoe":
        for layer_idx in range(0, model_config.num_hidden_layers):
            model.model.layers[layer_idx].mlp.gate = nn.Linear(model_config.hidden_size, model_config.num_experts, bias=False,dtype=torch.bfloat16,device=model_config.device)
    if model_type == "xversemoe":
        for layer_idx in range(0, model_config.num_hidden_layers):
            model.model.layers[layer_idx].mlp.router = nn.Linear(model_config.hidden_size, model_config.num_experts, bias=False,dtype=torch.float,device=model_config.device)
    weight_map = _read_weight_map(model_path)
    original_hf = _is_original_hf_format(weight_map)

    logger.info("loading common params...")
    # xversemoe bin checkpoints contain rotary_emb.inv_freq buffers not in the architecture.
    # qwenmoe: zero-expert model has dense MLP keys (gate_proj/up_proj/down_proj) but the
    # checkpoint stores shared_expert.* / shared_expert_gate.* / gate.* — mismatched by design
    # because these layers are replaced with Qwen2MoeSparseMoeBlockwithCache right after.
    # Use strict=False for both qwenmoe and xversemoe.
    _strict = model_type == "deepseekmoe"
    if original_hf:
        # Original HF shards mix expert and non-expert params; collect only non-expert keys.
        common_state_dict = _load_non_expert_params(model_path, weight_map)
        model.load_state_dict(common_state_dict, strict=_strict)
        del common_state_dict
    else:
        # Pre-split format: common_params.safetensors contains exactly the non-expert params.
        trunk_state_path = os.path.join(model_path, weight_map["model.embed_tokens.weight"])
        model.load_state_dict(_load_shard(trunk_state_path), strict=_strict)
    device = next(model.parameters()).device
    logger.info(f"Model is on device: {device}")
    logger.debug("Common params have loaded.")
    # Load transient checkpoint shards before allocating the complete host
    # expert backing. On 32 GiB hosts the opposite order causes swap storms.
    expert_cache = ExpertCache(
        model_config,
        make_module_cuda=_make_module_cuda_cfg,
        make_module_cpu=_make_module_cpu_cfg,
        main_size=main_size,
        offload_size=offload_size_,
        window_size=model_config.window_size,
        state_dict_00=state_dict_00,
        model_type=model_type,
        model_path=model_path,
    )
    _shard_cache: dict = {} if original_hf else None
    # Replace each layer with the cache-enabled implementation
    if model_type == "deepseekmoe":
        init_size = 0
        for layer_idx in trange(1,model_config.num_hidden_layers, desc="Loading experts"):
            curr_layer = model.model.layers[layer_idx]
            if layer_idx <27:
                next_layer = model.model.layers[layer_idx+1]
                next_attention = next_layer.self_attn
                next_gate_weight = next_layer.mlp.gate.weight
                next_input_layernorm = next_layer.input_layernorm
                next_post_attention_layernorm = next_layer.post_attention_layernorm
                curr_layer.mlp = DeepseekMoEwithCache(
                    model_config,
                    expert_cache,
                    layer_idx,
                    curr_layer.mlp.gate.weight,
                    curr_layer.mlp.shared_experts,
                    next_attention,
                    next_gate_weight,
                    next_input_layernorm,
                    next_post_attention_layernorm
                )
            else:
                curr_layer.mlp = DeepseekMoEwithCache(
                    model_config,
                    expert_cache,
                    layer_idx,
                    curr_layer.mlp.gate.weight,
                    curr_layer.mlp.shared_experts,
                    None,None,None,None
                )
            
            for expert_idx in range(model_config.n_routed_experts):
                # logger.debug("---------")
                # logger.debug(f"Current CPU memory usage: {psutil.Process().memory_info().rss / (1024 ** 2):.2f} MB")
                do_offload = init_size >= main_size

                expert_wrapper = make_and_load_expert_wrapper(
                    config=model_config,
                    states_dir=model_path,
                    expert_uid=(layer_idx, expert_idx),
                    model_type=model_type,
                    device=device,
                    _weight_map=weight_map,
                    _shard_cache=_shard_cache,
                )
                # logger.debug(f"Current CPU memory usage: {psutil.Process().memory_info().rss / (1024 ** 2):.2f} MB")
                expert_cache.add_expert(
                    uid=(layer_idx, expert_idx),
                    module=expert_wrapper,
                    offload=do_offload,
                )
                del expert_wrapper
                init_size+=1
            gc.collect()
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
    if model_type == "qwenmoe":
        init_size=0
        for layer_idx in trange(0,model_config.num_hidden_layers, desc="Loading experts"):
            curr_layer = model.model.layers[layer_idx]
            if layer_idx <27:
                next_layer = model.model.layers[layer_idx+1]
                next_attention = next_layer.self_attn
                next_gate_weight = next_layer.mlp.gate
                next_input_layernorm = next_layer.input_layernorm
                next_post_attention_layernorm = next_layer.post_attention_layernorm

                curr_layer.mlp = Qwen2MoeSparseMoeBlockwithCache(
                    model_config,
                    expert_cache,
                    layer_idx,
                    curr_layer.mlp.gate,
                    curr_layer.mlp.shared_expert,
                    curr_layer.mlp.shared_expert_gate,
                    next_attention,
                    next_gate_weight,
                    next_input_layernorm,
                    next_post_attention_layernorm
                )
            else:
                curr_layer.mlp = Qwen2MoeSparseMoeBlockwithCache(
                    model_config,
                    expert_cache,
                    layer_idx,
                    curr_layer.mlp.gate,
                    curr_layer.mlp.shared_expert,
                    curr_layer.mlp.shared_expert_gate,
                    None,None,None,None
                )
            for expert_idx in range(model_config.num_experts):
                # logger.debug("---------")
                # logger.debug(f"Current CPU memory usage: {psutil.Process().memory_info().rss / (1024 ** 2):.2f} MB")
                do_offload = init_size >= main_size

                expert_wrapper = make_and_load_expert_wrapper(
                    config=model_config,
                    states_dir=model_path,
                    expert_uid=(layer_idx, expert_idx),
                    model_type=model_type,
                    device=device,
                    _weight_map=weight_map,
                    _shard_cache=_shard_cache,
                )
                # logger.debug(f"Current CPU memory usage: {psutil.Process().memory_info().rss / (1024 ** 2):.2f} MB")
                expert_cache.add_expert(
                    uid=(layer_idx, expert_idx),
                    module=expert_wrapper,
                    offload=do_offload,
                )
                del expert_wrapper
                init_size+=1
            gc.collect()
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()

    if model_type == "xversemoe":
        init_size=0
        for layer_idx in trange(0,model_config.num_hidden_layers, desc="Loading experts"):
            curr_layer = model.model.layers[layer_idx]
            if layer_idx <27:
                next_layer = model.model.layers[layer_idx+1]
                next_attention = next_layer.self_attn
                next_gate_weight = next_layer.mlp.router
                next_input_layernorm = next_layer.input_layernorm
                next_post_attention_layernorm = next_layer.post_attention_layernorm

                curr_layer.mlp = XverseMoEMLPwithCache(
                    model_config,
                    expert_cache,
                    layer_idx,
                    curr_layer.mlp.router,
                    curr_layer.mlp.shared_experts,
                    next_attention,
                    next_gate_weight,
                    next_input_layernorm,
                    next_post_attention_layernorm
                )
            else:
                curr_layer.mlp = XverseMoEMLPwithCache(
                    model_config,
                    expert_cache,
                    layer_idx,
                    curr_layer.mlp.router,
                    curr_layer.mlp.shared_experts,
                    None,None,None,None
                )
            for expert_idx in range(model_config.num_experts):
                # logger.debug("---------")
                # logger.debug(f"Current CPU memory usage: {psutil.Process().memory_info().rss / (1024 ** 2):.2f} MB")
                do_offload = init_size >= main_size

                expert_wrapper = make_and_load_expert_wrapper(
                    config=model_config,
                    states_dir=model_path,
                    expert_uid=(layer_idx, expert_idx),
                    model_type=model_type,
                    device=device,
                    _weight_map=weight_map,
                    _shard_cache=_shard_cache,
                )
                # logger.debug(f"Current CPU memory usage: {psutil.Process().memory_info().rss / (1024 ** 2):.2f} MB")
                expert_cache.add_expert(
                    uid=(layer_idx, expert_idx),
                    module=expert_wrapper,
                    offload=do_offload,
                )
                del expert_wrapper
                init_size+=1
            gc.collect()
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
    logger.info(f"Current CPU memory usage: {psutil.Process().memory_info().rss / (1024 ** 2):.2f} MB")
    allocated_memory = torch.cuda.memory_allocated(0)  # check GPU device 0
    logger.info(f"Allocated GPU memory on device 0: {allocated_memory / (1024 ** 2):.2f} MB")

    if model_type == "deepseekmoe":
        from utils.patcher import patch_deepseek_model
        patch_deepseek_model(model)
        logger.info("SMoE patches applied to DeepseekForCausalLM.")

    # Inject prefill/decode timing + per-token expert hit-rate stats for all models.
    # For deepseekmoe this is already done inside patch_deepseek_model (via
    # _patch_inner_model_forward); for qwenmoe and xversemoe we apply it here.
    if model_type in ("qwenmoe", "xversemoe"):
        from utils.patcher import patch_model_forward
        patch_model_forward(model, model_type)
        logger.info("SMoE inner-model forward patched for %s.", model_type)

    if model_type == "deepseekmoe" and os.environ.get("SMOE_TRIM_HOST_HEAP", "1") == "1":
        # Checkpoint views and temporary expert packs are dead after loading.
        # Release glibc's unused arenas before memory-constrained decoding;
        # this never discards live tensor storage or changes model arithmetic.
        if _shard_cache is not None:
            _shard_cache.clear()
        gc.collect()
        import ctypes
        libc = ctypes.CDLL(None)
        trim = getattr(libc, "malloc_trim", None)
        if trim is not None:
            trim.argtypes = [ctypes.c_size_t]
            trim.restype = ctypes.c_int
            before = psutil.Process().memory_info().rss
            released = trim(0)
            after = psutil.Process().memory_info().rss
            logger.info("[host heap] malloc_trim=%d rss_before=%d rss_after=%d",
                        released, before, after)

    if model_type == "deepseekmoe":
        logger.info("[CPU BF16 ready] experts=%d fused_gate_up=%d",
                    len(expert_cache.offloaded_storages),
                    sum(getattr(m, "cpu_bf16_gate_up_fused", False)
                        for m in expert_cache.offloaded_storages))
    return model
