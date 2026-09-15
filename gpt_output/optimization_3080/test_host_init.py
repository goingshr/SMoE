"""Check that direct checkpoint attachment preserves packed expert weights."""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from safetensors.torch import save_file
from utils.model_loader import make_and_load_expert_wrapper
from utils.expertcache import ExpertCache, EvictionInfo

torch.set_num_threads(1)
torch.manual_seed(41)
cfg = SimpleNamespace(device='cpu', hidden_size=128, moe_intermediate_size=64,
                      pretraining_tp=1, hidden_act='silu')
weights = {f'model.layers.1.mlp.experts.0.{name}.weight': torch.randn(
    shape, dtype=torch.bfloat16) for name, shape in (
        ('gate_proj', (64, 128)), ('up_proj', (64, 128)),
        ('down_proj', (128, 64)))}
with tempfile.TemporaryDirectory() as d:
    save_file(weights, str(Path(d) / 'model.safetensors'))
    (Path(d) / 'model.safetensors.index.json').write_text(json.dumps({
        'weight_map': {k: 'model.safetensors' for k in weights}}))
    os.environ['SMOE_HOST_EXPERT_INIT'] = '0'
    reference = make_and_load_expert_wrapper(cfg, d, (1, 0), 'deepseekmoe', 'cpu')
    os.environ['SMOE_HOST_EXPERT_INIT'] = '1'
    actual = make_and_load_expert_wrapper(cfg, d, (1, 0), 'deepseekmoe', 'cpu')
    assert bytes(reference.storage) == bytes(actual.storage)
    cache = ExpertCache.__new__(ExpertCache)
    cache.module_type = type(actual)
    cache.module_size = len(actual.storage)
    cache.registered_experts = {}
    cache.offloaded_storages = [None, None]
    cache.offloaded_infos = [0, 0]
    cache.cache_window = None
    cache.cache_infos = EvictionInfo()
    cache._make_host_backing = lambda: make_and_load_expert_wrapper(
        cfg, d, (1, 0), 'deepseekmoe', 'cpu')
    cache.add_expert((1, 0), actual, offload=True)
    assert cache.offloaded_storages[0] is actual
    assert cache.registered_experts[(1, 0)].offload_index == 0
    cache.add_expert_storage((1, 1), reference.storage, offload=True)
    assert bytes(cache.offloaded_storages[1].storage) == bytes(reference.storage)
    for n in (1, 7):
        x = torch.randn(n, 128, dtype=torch.bfloat16)
        with torch.no_grad():
            torch.testing.assert_close(actual(x), reference(x), rtol=0, atol=0)
    os.environ['SMOE_CPU_BF16_MV'] = '1'
    mv = make_and_load_expert_wrapper(cfg, d, (1, 0), 'deepseekmoe', 'cpu')
    for n in (1, 7):
        x = torch.randn(n, 128, dtype=torch.bfloat16)
        with torch.no_grad():
            torch.testing.assert_close(mv(x), reference(x), rtol=0.016, atol=0.016)
print('PASS: host initialization preserves packed BF16 bytes and forwards; MV and prefill fallback checked')
