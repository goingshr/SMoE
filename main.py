import numpy
import sys
import torch
from transformers import AutoTokenizer, TextStreamer
from utils.model_loader import build_model
import time
import psutil
import argparse
import logging
from utils.cpu_affinity import select_cpu_placement

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--model_name",    type=str, default='qwenmoe')
parser.add_argument("--model_path",    type=str, default='')
parser.add_argument("--config_path",   type=str, default='')
parser.add_argument("--input_num",     type=int, default=20)
parser.add_argument("--input_len",     type=int, default=None)
parser.add_argument("--warmup_num",    type=int, default=0)
parser.add_argument("--dataset_path",  type=str, default='wic')
parser.add_argument("--batch_size",    type=int, default=1)
parser.add_argument("--debug",         type=bool, default=False)
parser.add_argument("--output_len",    type=int, default=100)
parser.add_argument("--GPU_mem",       type=float, default=10)
parser.add_argument("--cpu_cores",     type=int, default=16)
parser.add_argument("--seed",          type=int, default=37)

args = parser.parse_args()
torch.manual_seed(args.seed)
numpy.random.seed(args.seed)

if args.input_num < 1:
    parser.error("--input_num must be at least 1")
if args.input_len is not None and args.input_len < 1:
    parser.error("--input_len must be at least 1")
if args.warmup_num < 0:
    parser.error("--warmup_num cannot be negative")

import os as _os

print(f"[PYTHON] gil_enabled={getattr(sys, '_is_gil_enabled', lambda: True)()} "
      f"switch_interval={sys.getswitchinterval()}")

_placement     = select_cpu_placement(args.cpu_cores)
_compute_cores = list(_placement.compute_cores)  # n-1 physical cores: CPU matmul
_shared_core   = _placement.shared_core          # 1 core: loading + bg_worker

try:
    _os.sched_setaffinity(0, set(_compute_cores))  # main process uses compute_cores only
except Exception:
    pass

# Write expertcache module variables (read by loading / bg_worker on startup)
import utils.expertcache as _ecpre
_ecpre._shared_core   = _shared_core
_ecpre._compute_cores = _compute_cores

print(
    f"[AFFINITY] n={args.cpu_cores}  compute={_compute_cores} "
    f"compute_packages={list(_placement.compute_packages)}  "
    f"shared={_shared_core} shared_package={_placement.shared_package}"
)

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
all_inputs = load_all(
    dataset_path, args.batch_size, args.input_num + args.warmup_num)

# ── Model initialization ─────────────────────────────────────────────────────

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_cfg_path = args.config_path if args.config_path else None
model = build_model(
    model_path=model_name,
    model_type=model_type,
    device=device,
    gpu_mem_gb=args.GPU_mem,
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
    print(f"[CONFIG] cache_size=auto(gpu_mem={args.GPU_mem}GB)  "
          f"input_len={args.input_len}  output_len={args.output_len}  "
          f"input_num={args.input_num}  warmup_num={args.warmup_num}")
    print(f"[CONFIG] SMoE fields: {_smoe_cfg}")
except Exception as _e:
    print(f"[CONFIG] Failed to read config: {_e}")

logger.info(f"Current CPU memory usage: {psutil.Process().memory_info().rss / (1024 ** 2):.2f} MB")

output_len = args.output_len

# ── Inference loop ───────────────────────────────────────────────────────────

import utils.expertcache as expertcache
import MoEModule.SMoE_base as _smoe_base
from utils import decode_metrics

_measured_records = []
for i, _ in enumerate(all_inputs):
    _phase = "warmup" if i < args.warmup_num else "measure"
    decode_metrics.reset()
    _sample_id = i if _phase == "warmup" else i - args.warmup_num
    # Sampling is reproducible independently of model construction and warmup.
    torch.manual_seed(args.seed + i)
    # Reset per-prompt statistics (patcher reads these each token)
    expertcache.tokens       = 0
    expertcache.decode_time  = 0.0
    expertcache.prefill_time = 0.0
    expertcache.cache_hits_per_token  = 0
    expertcache.cache_total_per_token = 0
    expertcache.cache_hits_prompt  = 0
    expertcache.cache_total_prompt = 0
    expertcache.prefetch_loaded_by_layer = {}
    expertcache.prefetch_start_time      = {}
    _smoe_base.cpu_compute_ms_per_token.clear()
    _smoe_base.cpu_compute_token_indices.clear()
    _smoe_base._cpu_ms_cur_token_samples.clear()
    _smoe_base._cpu_ms_cur_token_idx = -1
    _smoe_base.cpu_activation_d2h_copies = 0
    _smoe_base.cpu_activation_d2h_bytes = 0
    _smoe_base.cpu_output_h2d_copies = 0
    _smoe_base.cpu_output_h2d_bytes = 0
    texts  = all_inputs[i]
    print('=' * 20, flush=True)
    print(f"[RUN] phase={_phase} sample_id={_sample_id} dataset_index={i}", flush=True)
    print(f"input_id: {i}")
    print(f'text: {texts}', flush=True)

    _tokenizer_kwargs = {
        "return_tensors": "pt",
        "padding": True,
        "truncation": args.input_len is not None,
    }
    if args.input_len is not None:
        _tokenizer_kwargs["max_length"] = args.input_len
    inputs = tokenizer(texts, **_tokenizer_kwargs)
    _actual_input_len = int(inputs["input_ids"].shape[-1])
    logger.info(
        "[INPUT] phase=%s sample_id=%d input_tokens=%d requested_max=%s",
        _phase, _sample_id, _actual_input_len, args.input_len,
    )
    inputs = {k: v.to(device) for k, v in inputs.items() if k != "token_type_ids"}

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
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
        _smoe_base.cpu_compute_token_indices.append(
            _smoe_base._cpu_ms_cur_token_idx)
        _smoe_base._cpu_ms_cur_token_samples = []

    _cpu_decode_ms = [
        value for token_idx, value in zip(
            _smoe_base.cpu_compute_token_indices,
            _smoe_base.cpu_compute_ms_per_token,
        )
        if token_idx > 0
    ]
    if _cpu_decode_ms:
        logger.info(
            "[CPU expert] prompt=%d decode_forward_mean=%.3f ms "
            "median=%.3f ms p95=%.3f ms sampled_tokens=%d",
            i,
            float(numpy.mean(_cpu_decode_ms)),
            float(numpy.median(_cpu_decode_ms)),
            float(numpy.percentile(_cpu_decode_ms, 95)),
            len(_cpu_decode_ms),
        )

    # Print prompt-level totals here.
    decode_tokens   = expertcache.tokens - 1   # subtract 1 for prefill token
    avg_decode_time = (expertcache.decode_time / decode_tokens
                       if decode_tokens > 0 else float('nan'))
    logger.info("[SMoE] prompt=%d  prefill=%.4f s  avg_decode=%.6f s  "
                "total=%.4f s  decode_tokens=%d",
                i, expertcache.prefill_time, avg_decode_time,
                end - start, decode_tokens)
    _gpu_hit_rate = (expertcache.cache_hits_prompt / expertcache.cache_total_prompt
                     if expertcache.cache_total_prompt > 0 else float('nan'))
    logger.info(
        "[GPU cache] prompt=%d decode_hit_rate=%.6f hits=%d total=%d",
        i, _gpu_hit_rate, expertcache.cache_hits_prompt,
        expertcache.cache_total_prompt,
    )
    if decode_metrics.enabled:
        # Drain possible final-layer prefetch before resolving event samples.
        _cache_metrics = model.model.layers[0].mlp.ExpertCache
        _cache_metrics.wait_until_queue_empty()
        _cache_metrics.load_stream.synchronize()
        _cache_metrics.consume_dma_timings()
        logger.info('[Decode metrics] prompt=%d %s', i,
                    _json.dumps(decode_metrics.summary(decode_tokens), sort_keys=True))
    logger.info(
        "[CPU transfer] prompt=%d activation_d2h=%d/%dB output_h2d=%d/%dB",
        i,
        _smoe_base.cpu_activation_d2h_copies,
        _smoe_base.cpu_activation_d2h_bytes,
        _smoe_base.cpu_output_h2d_copies,
        _smoe_base.cpu_output_h2d_bytes,
    )

    results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    if model_type in ("qwenmoe", "xversemoe"):
        _cache = model.model.layers[0].mlp.ExpertCache
        _replayed_slots = sum(
            m.decode_graph is not None and m.decode_graph.engaged
            for m in _cache.main_modules)
        _replayed_shared = sum(
            getattr(layer.mlp, "_shared_graph", None) is not None
            and layer.mlp._shared_graph.engaged
            for layer in model.model.layers)
        logger.info(
            "[GPU decode] prompt=%d cache_slots=%d replayed_slots=%d shared_layers=%d "
            "peak_allocated=%d peak_reserved=%d",
            i, len(_cache.main_modules), _replayed_slots, _replayed_shared,
            torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved())
        if _cache.measure_dma:
            logger.info("[DMA timing] completed_copies=%d estimate_ms=%.4f",
                        _cache.measured_dma_copies,
                        1000 * sum(_cache.LoadTimeOneExpert) / len(_cache.LoadTimeOneExpert))
    if _phase == "measure":
        _measured_records.append({
            "decode": avg_decode_time,
            "cpu_forward": float(numpy.mean(_cpu_decode_ms)) if _cpu_decode_ms else float('nan'),
            "gpu_hit_rate": _gpu_hit_rate,
            "prefill": expertcache.prefill_time,
            "total": end - start,
            "decode_tokens": decode_tokens,
            "input_tokens": _actual_input_len,
        })
    logger.warning("results: %s", results)
    print('=' * 20, flush=True)

if _measured_records:
    def _mean(key):
        return float(numpy.nanmean([record[key] for record in _measured_records]))

    logger.warning(
        "[SUMMARY] measured_prompts=%d mean_decode=%.6f s/token "
        "mean_cpu_expert_forward=%.3f ms mean_gpu_hit_rate=%.6f "
        "mean_prefill=%.4f s mean_total=%.4f s",
        len(_measured_records), _mean("decode"), _mean("cpu_forward"),
        _mean("gpu_hit_rate"), _mean("prefill"), _mean("total"),
    )
