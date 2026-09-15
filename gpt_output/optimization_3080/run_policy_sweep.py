"""Reuse one loaded model for controlled scheduling/prefetch diagnostics.

All variants run the same WiC prompts and max_new_tokens=100, with separately
logged warmups. A final acceptance run still uses 10 measured prompts per core.
"""
import argparse
import json
import os
from pathlib import Path
import runpy
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
p = argparse.ArgumentParser()
p.add_argument('--outdir', type=Path, required=True)
p.add_argument('--cores', type=int, default=3)
p.add_argument('--input-num', type=int, default=3)
p.add_argument('--warmup-num', type=int, default=2)
p.add_argument('--score-window', type=int, default=16)
p.add_argument('--layer-cache-floor', type=int, default=12)
p.add_argument('--policies', nargs='+', choices=['legacy', 'highscore', 'cpu', 'minmax', 'prefetch', 'staged',
                                               'score', 'score_highscore', 'score_staged', 'score_cpu',
                                               'balanced_cpu', 'balanced_score_cpu',
                                               'prefetch_event', 'prefetch_early', 'prefetch_early_cpu',
                                               'prefetch_early_rank_cpu', 'balanced_score_prefetch_early_cpu',
                                               'prefetch_early_fused_rank_cpu'],
               default=['highscore', 'cpu', 'minmax', 'prefetch', 'staged'])
p.add_argument('--mv', action='store_true')
p.add_argument('--profile', action='store_true')
p.add_argument('--profile-policy', choices=['legacy', 'prefetch'], default='legacy')
a = p.parse_args()
a.outdir = a.outdir.resolve()
if a.outdir.exists() and any(a.outdir.iterdir()):
    raise FileExistsError(a.outdir)
a.outdir.mkdir(parents=True, exist_ok=True)
if a.input_num < 1 or a.warmup_num < 0:
    raise ValueError('positive input count and nonnegative warmup required')
invocation = list(sys.argv)
os.environ.update(OMP_NUM_THREADS=str(a.cores-1), MKL_NUM_THREADS=str(a.cores-1),
                  OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
                  PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
                  SMOE_CPU_BF16_MV=str(int(a.mv)), SMOE_DECODE_MINMAX='0',
                  SMOE_DECODE_CPU_ONLY='0')
os.environ['SMOE_LOAD_HIGH_SCORE'] = '0'
os.environ['SMOE_PINNED_STAGING'] = '0'
os.environ['SMOE_STRICT_SCORE_ORDER'] = '0'
os.environ['SMOE_TRACK_DECODE_WORK'] = '1'
os.environ['SMOE_LAYER_CACHE_FLOOR'] = '0'
os.environ['SMOE_PREFETCH_EARLY'] = '0'
os.environ['SMOE_PREFETCH_EVENT_CHAIN'] = '0'
os.environ['SMOE_PREFETCH_SCORE_ORDER'] = '0'
os.environ['SMOE_PREFETCH_FUSED_NORMS'] = '0'
sys.setswitchinterval(0.0001)
config = json.loads((ROOT / 'configs/deepseekmoe_config.json').read_text())
assert config['if_replace'] is True
if not 0 <= config['replaceScoreRatio'] <= 0.35:
    raise ValueError('this optimization run requires replaceScoreRatio <= 0.35')
config['if_prefetch'] = False
config_path = a.outdir / 'base_config.json'
config_path.write_text(json.dumps(config, indent=2))
(a.outdir / 'source.patch').write_text(subprocess.check_output(['git', 'diff'], cwd=ROOT, text=True))
from gpt_output.optimization_3080.snapshot_source import snapshot
snapshot(ROOT, a.outdir)
(a.outdir / 'invocation.json').write_text(json.dumps({'argv': invocation, 'environment': {
    k: v for k, v in os.environ.items() if k.startswith(('SMOE_', 'OMP_', 'MKL_', 'PYTORCH_'))}}, indent=2))

sys.argv = [str(ROOT / 'main.py'), '--model_name', 'deepseekmoe',
            '--model_path', '/root/models/deepseekmoe', '--config_path', str(config_path),
            '--dataset_path', 'wic', '--input_num', str(a.input_num), '--output_len', '100',
            '--warmup_num', str(a.warmup_num), '--cpu_cores', str(a.cores), '--GPU_mem', '10']
print('[SWEEP] initial legacy baseline uses main.py; see raw prompt logs', flush=True)
# GPU validation runs in separate processes before this benchmark. CUDA must
# first initialize under main.py's CPU placement, as in acceptance runs.
if any('prefetch' in policy for policy in a.policies):
    subprocess.run([sys.executable, str(ROOT / 'gpt_output/optimization_3080/test_queue_drain.py')],
                   check=True, cwd=ROOT)
state = runpy.run_path(str(ROOT / 'main.py'), run_name='__main__')
model, tokenizer = state['model'], state['tokenizer']
prompts, torch = state['all_inputs'], state['torch']
ec, sb = state['expertcache'], state['_smoe_base']
layers = [layer.mlp for layer in model.model.layers if hasattr(layer.mlp, 'ExpertCache')]
cache = layers[0].ExpertCache
staging_checked = False


def configure(policy):
    global staging_checked
    cache.clear_queue()
    cache.wait_until_queue_empty()
    torch.cuda.synchronize()
    if 'staged' in policy and not staging_checked:
        from gpt_output.optimization_3080.test_pinned_staging import check_ring
        check_ring()
        staging_checked = True
    cache.pinned_staging = 'staged' in policy
    uses_score = policy.startswith('score') or policy.startswith('balanced_score')
    cache.cache_infos.strict_score_order = uses_score
    cache.cache_infos.layer_cache_floor = a.layer_cache_floor if policy.startswith('balanced') else 0
    cache.cache_window = a.score_window if uses_score else config['window_size']
    # Each score variant fills its own window during the separate warmups.
    cache._score_buf = cache._score_sum = cache._score_ptr = cache._score_cnt = None
    for layer in layers:
        layer._decode_cpu_only = policy == 'cpu' or policy.endswith('_cpu')
        layer._decode_minmax = policy == 'minmax'
        layer._load_high_score = 'highscore' in policy
        layer.if_prefetch = 'prefetch' in policy
        layer._prefetch_early = 'prefetch_early' in policy
        layer._prefetch_event_chain = policy == 'prefetch_event'
        layer._prefetch_score_order = 'rank' in policy
        layer._prefetch_fused_norms = 'fused' in policy
        layer.config.if_prefetch = layer.if_prefetch


def reset():
    ec.tokens = 0
    ec.decode_time = ec.prefill_time = 0.0
    ec.cache_hits_per_token = ec.cache_total_per_token = 0
    ec.prefetch_loaded_by_layer = {}
    ec.prefetch_start_time = {}
    sb.cpu_compute_ms_per_token.clear()
    sb.cpu_compute_token_indices.clear()
    sb._cpu_ms_cur_token_samples.clear()
    sb._cpu_ms_cur_token_idx = -1
    sb.cpu_activation_d2h_copies = sb.cpu_activation_d2h_bytes = 0
    sb.cpu_output_h2d_copies = sb.cpu_output_h2d_bytes = 0
    sb.decode_work_by_layer.clear()


def generate(i):
    reset()
    inputs = tokenizer(prompts[i % len(prompts)], return_tensors='pt', padding=True, truncation=True)
    inputs = {k: v.to('cuda:0') for k, v in inputs.items() if k != 'token_type_ids'}
    torch.cuda.synchronize()
    begin = time.perf_counter()
    with torch.no_grad():
        result = model.generate(**inputs, max_new_tokens=100)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - begin
    return dict(prompt=i, warmup=i < 0, decode_tokens=ec.tokens-1,
                decode_s=ec.decode_time/(ec.tokens-1) if ec.tokens > 1 else None,
                prefill_s=ec.prefill_time, wall_s=elapsed,
                peak_allocated=torch.cuda.max_memory_allocated(),
                peak_reserved=torch.cuda.max_memory_reserved(),
                decode_work_by_layer=dict(sb.decode_work_by_layer),
                token_ids=result.cpu().tolist())


summaries = []
for index, policy in enumerate(a.policies):
    configure(policy)
    records = []
    stem = f'{index:02d}_{policy}'
    for i in range(-a.warmup_num, a.input_num):
        record = generate(i)
        records.append(record)
        (a.outdir / f'{stem}.json').write_text(json.dumps(records, indent=2))
        print('[SWEEP]', policy, json.dumps({k: v for k, v in record.items() if k != 'token_ids'}), flush=True)
    measured = [r for r in records if not r['warmup']]
    summary = dict(policy=policy, run_index=index, prompts=len(measured),
                   mean_decode_s=statistics.fmean(r['decode_s'] for r in measured),
                   replace_score_ratio=config['replaceScoreRatio'],
                   if_prefetch=layers[0].if_prefetch, score_window=cache.cache_window,
                   layer_cache_floor=cache.cache_infos.layer_cache_floor)
    work = [w for r in measured for w in r['decode_work_by_layer'].values()]
    summary['gpu_cache_hit_rate'] = sum(w[1] for w in work) / sum(sum(w[1:]) for w in work)
    summaries.append(summary)
    (a.outdir / 'summary.json').write_text(json.dumps(summaries, indent=2))
    print('[SWEEP RESULT]', json.dumps(summary), flush=True)

if a.profile:
    # Capture a warmed path separately from the unprofiled measurements.
    configure(a.profile_policy)
    generate(-1)
    original = model.model.forward

    def on_trace(prof):
        prof.export_chrome_trace(str(a.outdir / 'warm_decode.trace.json'))
        (a.outdir / 'warm_operators.txt').write_text(
            prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=50))

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=10, warmup=1, active=5, repeat=1),
            record_shapes=True, with_stack=True, on_trace_ready=on_trace) as prof:
        def forward(*fa, **fk):
            result = original(*fa, **fk)
            prof.step()
            return result
        model.model.forward = forward
        try:
            generate(0)
        finally:
            model.model.forward = original
