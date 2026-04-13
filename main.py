import numpy
import sys
import torch
from transformers import AutoTokenizer, TextStreamer
from utils.model_loader import build_model
import time
import psutil
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--model_name",    type=str, default='qwenmoe')
parser.add_argument("--model_path",    type=str, default='')
parser.add_argument("--config_path",   type=str, default='')
parser.add_argument("--input_num",     type=int, default=20)
parser.add_argument("--dataset_path",  type=str, default='wic')
parser.add_argument("--batch_size",    type=int, default=1)
parser.add_argument("--debug",         type=bool, default=False)
parser.add_argument("--output_len",    type=int, default=100)
parser.add_argument("--GPU_mem",       type=int, default=24)
parser.add_argument("--cpu_cores",     type=int, default=16)

args = parser.parse_args()

import os as _os

def _pick_idle_cores(n):
    """Sample 0.2s, return the n CPU core IDs with the lowest utilization."""
    per_core = psutil.cpu_percent(percpu=True, interval=0.2)
    ranked = sorted(range(len(per_core)), key=lambda i: per_core[i])
    return ranked[:n]

_cores         = _pick_idle_cores(args.cpu_cores)
_compute_cores = _cores[:-1]   # n-1 cores: CPU matmul
_shared_core   = _cores[-1]    # 1 core: loading + bg_worker

try:
    _os.sched_setaffinity(0, set(_compute_cores))  # main process uses compute_cores only
except Exception:
    pass

# Write expertcache module variables (read by loading / bg_worker on startup)
import utils.expertcache as _ecpre
_ecpre._shared_core   = _shared_core
_ecpre._compute_cores = _compute_cores

print(f"[AFFINITY] n={args.cpu_cores}  compute={_compute_cores}  shared={_shared_core}")

torch.set_num_threads(len(_compute_cores))  # intra-op = n-1
torch.set_num_interop_threads(1)            # interop fixed at 1, isolated from intra-op


class StopWatch(TextStreamer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_prefilling    = None
        self.prefilling_time     = None
        self.start_decoding      = None
        self.decoding_time       = None
        self.decoding_iterations = 0

    def put(self, value):
        if self.start_prefilling is None:
            self.start_prefilling = time.time()
            return
        elif self.prefilling_time is None:
            self.prefilling_time = time.time() - self.start_prefilling
            self.start_decoding  = time.time()
        self.decoding_iterations += 1

        if self.decoding_iterations % 10 == 0:
            current_time = time.time()
            logger.info("Prefilling time: %.4f s", self.prefilling_time)
            logger.info("Decoding time per iteration: %.4f s",
                        (current_time - self.start_decoding) / self.decoding_iterations)

        return super().put(value)

    def end(self):
        if self.decoding_time is None and self.start_decoding is not None:
            self.decoding_time = time.time() - self.start_decoding
            current_time = time.time()
            logger.info("Prefilling time: %.4f s", self.prefilling_time)
            logger.info("Decoding time per iteration: %.4f s",
                        (current_time - self.start_decoding) / self.decoding_iterations)
        return super().end()


# ── Model selection ──────────────────────────────────────────────────────────

from download import ensure_model

if args.model_name == 'deepseekmoe':
    model_name = ensure_model('deepseekmoe', args.model_path)
    model_type = "deepseekmoe"
    from models import modeling_deepseek
    import MoEModule.deepseek_moe as deepseek_moe
elif args.model_name == 'xversemoe':
    model_name = ensure_model('xversemoe', args.model_path)
    model_type = "xversemoe"
    from models import modeling_xverse
    import MoEModule.xverse_moe as xverse_moe
elif args.model_name == 'qwenmoe':
    model_name = ensure_model('qwenmoe', args.model_path)
    model_type = "qwenmoe"
    from models import modeling_qwen
    import MoEModule.qwen_moe as qwen_moe
else:
    assert False, f'invalid model: {args.model_name}'

if args.debug:
    import debugpy
    try:
        debugpy.listen(("localhost", 9501))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
    except Exception:
        pass

# ── Tokenizer ────────────────────────────────────────────────────────────────

# ── Tokenizer ────────────────────────────────────────────────────────────────

if args.model_name == 'xversemoe':
    # xversemoe tokenizer.json uses old 'add_prefix_space' field incompatible
    # with newer tokenizers library; patch it on the fly into a temp directory.
    import json as _jt, os as _ost, shutil as _shut, tempfile as _tmpt
    _tok_src = model_name
    _tok_dst = '/tmp/xverse_tokenizer_fixed'
    _ost.makedirs(_tok_dst, exist_ok=True)
    for _fn in ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']:
        _fp = _ost.path.join(_tok_src, _fn)
        if _ost.path.exists(_fp):
            _shut.copy(_fp, _tok_dst)
    def _fix_ms(obj):
        if isinstance(obj, dict):
            if obj.get('type') == 'Metaspace' and 'add_prefix_space' in obj:
                obj['prepend_scheme'] = 'never' if not obj.pop('add_prefix_space') else 'first'
            for v in obj.values(): _fix_ms(v)
        elif isinstance(obj, list):
            for item in obj: _fix_ms(item)
    with open(_ost.path.join(_tok_dst, 'tokenizer.json')) as _f:
        _td = _jt.load(_f)
    _fix_ms(_td)
    with open(_ost.path.join(_tok_dst, 'tokenizer.json'), 'w') as _f:
        _jt.dump(_td, _f, ensure_ascii=False)
    tokenizer = AutoTokenizer.from_pretrained(_tok_dst)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

# ── Dataset loading ───────────────────────────────────────────────────────────

from utils.load_dataset import load_all

dataset_path = args.dataset_path
all_inputs   = load_all(dataset_path, args.batch_size, args.input_num)

# ── Model initialization ─────────────────────────────────────────────────────

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# cache_size = 300 for all models (as specified)
cache_size = 470 if args.model_name == 'qwenmoe' else 300

_cfg_path = args.config_path if args.config_path else None
model = build_model(
    model_path=model_name,
    model_type=model_type,
    device=device,
    main_size=cache_size,
    config_path=_cfg_path,
)
model = model.to(device)

# ── Print config summary to log ──────────────────────────────────────────────
import json as _json
_config_file = _cfg_path if _cfg_path else model_name + "/config.json"
try:
    with open(_config_file) as _cf:
        _cfg_dict = _json.load(_cf)
    _smoe_keys = ['replaceScoreRatio', 'window_size', 'if_prefetch',
                  'if_usecpu', 'if_replace']
    _smoe_cfg = {k: _cfg_dict[k] for k in _smoe_keys if k in _cfg_dict}
    print(f"[CONFIG] model_name={args.model_name}  model_path={model_name}")
    print(f"[CONFIG] config_path={_config_file}")
    print(f"[CONFIG] cache_size={cache_size}  output_len={args.output_len}  input_num={args.input_num}")
    print(f"[CONFIG] SMoE fields: {_smoe_cfg}")
except Exception as _e:
    print(f"[CONFIG] Failed to read config: {_e}")

logger.info(f"Current CPU memory usage: {psutil.Process().memory_info().rss / (1024 ** 2):.2f} MB")

output_len = args.output_len

# ── Inference loop ───────────────────────────────────────────────────────────

import utils.expertcache as expertcache
import MoEModule.SMoE_base as _smoe_base

for i, _ in enumerate(all_inputs):
    # Reset per-prompt statistics (patcher reads these each token)
    expertcache.tokens       = 0
    expertcache.decode_time  = 0.0
    expertcache.prefill_time = 0.0
    expertcache.cache_hits_per_token  = 0
    expertcache.cache_total_per_token = 0
    expertcache.prefetch_loaded_by_layer = {}
    expertcache.prefetch_start_time      = {}
    _smoe_base.cpu_compute_ms_per_token.clear()
    _smoe_base._cpu_ms_cur_token_samples.clear()
    _smoe_base._cpu_ms_cur_token_idx = -1
    texts  = all_inputs[i]
    print('=' * 20, flush=True)
    print(f"input_id: {i}")
    print(f'text: {texts}', flush=True)

    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items() if k != "token_type_ids"}

    with torch.no_grad():
        start   = time.time()
        outputs = model.generate(**inputs, max_new_tokens=output_len)
        end     = time.time()

    # Patcher already logged per-token prefill/decode/hit-rate via logger.info.
    # Flush the last token's CPU compute_ms (no next-token boundary to trigger it).
    if _smoe_base._cpu_ms_cur_token_samples:
        _smoe_base.cpu_compute_ms_per_token.append(
            sum(_smoe_base._cpu_ms_cur_token_samples) /
            len(_smoe_base._cpu_ms_cur_token_samples))
        _smoe_base._cpu_ms_cur_token_samples = []

    print(f"[CPU compute_ms per token] prompt={i}: {_smoe_base.cpu_compute_ms_per_token}")

    # Print prompt-level totals here.
    decode_tokens   = expertcache.tokens - 1   # subtract 1 for prefill token
    avg_decode_time = (expertcache.decode_time / decode_tokens
                       if decode_tokens > 0 else float('nan'))
    logger.info("[SMoE] prompt=%d  prefill=%.4f s  avg_decode=%.4f s  "
                "total=%.4f s  decode_tokens=%d",
                i, expertcache.prefill_time, avg_decode_time,
                end - start, decode_tokens)

    results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    logger.warning("results: %s", results)
    print('=' * 20, flush=True)
