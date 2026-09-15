"""Serial WiC acceptance/long runs; retain every warmup and measured prompt."""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
p = argparse.ArgumentParser()
p.add_argument('--outdir', type=Path, required=True)
p.add_argument('--cores', type=int, nargs='+', default=[3, 8, 16])
p.add_argument('--switch-interval', type=float, default=0.0001)
p.add_argument('--mv', action='store_true')
p.add_argument('--minmax', action='store_true')
p.add_argument('--cpu-only-misses', action='store_true')
p.add_argument('--load-high-score', action='store_true')
p.add_argument('--reserve-load-core', action='store_true')
p.add_argument('--pinned-staging', action='store_true')
p.add_argument('--strict-score-order', action='store_true')
p.add_argument('--score-window', type=int)
p.add_argument('--layer-cache-floor', type=int, default=0)
p.add_argument('--prefetch', action='store_true')
p.add_argument('--prefetch-early', action='store_true')
p.add_argument('--prefetch-event-chain', action='store_true')
p.add_argument('--prefetch-score-order', action='store_true')
p.add_argument('--prefetch-fused-norms', action='store_true')
p.add_argument('--gpu-mem', type=float, default=10)
p.add_argument('--warmup-num', type=int, default=2)
p.add_argument('--input-num', type=int, default=10)
p.add_argument('--summarize-only', action='store_true')
a = p.parse_args()
if (a.prefetch_early or a.prefetch_event_chain or a.prefetch_score_order
        or a.prefetch_fused_norms) and not a.prefetch:
    raise ValueError('prefetch optimizations require --prefetch')
a.outdir = a.outdir.resolve()
a.outdir.mkdir(parents=True, exist_ok=True)
config = json.loads((ROOT / 'configs/deepseekmoe_config.json').read_text())
config['if_prefetch'] = a.prefetch
if a.score_window is not None:
    if a.score_window < 1:
        raise ValueError('score window must be positive')
    config['window_size'] = a.score_window
if a.strict_score_order and config['window_size'] is None:
    raise ValueError('--strict-score-order requires --score-window')
assert config['if_replace'] is True
config_path = a.outdir / 'config.json'
if a.summarize_only and config_path.exists():
    config = json.loads(config_path.read_text())
if not 0 <= config['replaceScoreRatio'] <= 0.35:
    raise ValueError('this acceptance run requires replaceScoreRatio <= 0.35')
if not a.summarize_only:
    config_path.write_text(json.dumps(config, indent=2))
    (a.outdir / 'source.patch').write_text(subprocess.check_output(
        ['git', 'diff'], cwd=ROOT, text=True))
    from gpt_output.optimization_3080.snapshot_source import snapshot
    snapshot(ROOT, a.outdir)

pattern = re.compile(r'\[SMoE\] prompt=(-?\d+)\s+prefill=([\d.]+) s\s+'
                     r'avg_decode=([\d.]+) s\s+total=([\d.]+) s\s+decode_tokens=(\d+)')
summary = []
for cores in a.cores:
    log = a.outdir / f'cpu{cores}.log'
    rc_path = a.outdir / f'cpu{cores}.returncode'
    if not a.summarize_only:
        if log.exists() or rc_path.exists():
            raise FileExistsError(f'Refusing to overwrite an existing run: {log}')
        cmd = [sys.executable, str(ROOT / 'gpt_output/probe_deepseek_decode.py'),
               '--artifacts', str(a.outdir / f'cpu{cores}'),
               '--switch-interval', str(a.switch_interval), '--',
               '--model_name', 'deepseekmoe', '--model_path', '/root/models/deepseekmoe',
               '--config_path', str(config_path), '--dataset_path', 'wic',
               '--input_num', str(a.input_num), '--output_len', '100',
               '--warmup_num', str(a.warmup_num), '--cpu_cores', str(cores),
               '--GPU_mem', str(a.gpu_mem)]
        env = os.environ.copy()
        env.update(PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false',
                   PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
                   OMP_NUM_THREADS=str(cores-1), MKL_NUM_THREADS=str(cores-1),
                   OPENBLAS_NUM_THREADS='1', SMOE_CPU_BF16_MV=str(int(a.mv)),
                   SMOE_DECODE_MINMAX=str(int(a.minmax)),
                   SMOE_DECODE_CPU_ONLY=str(int(a.cpu_only_misses)),
                   SMOE_LOAD_HIGH_SCORE=str(int(a.load_high_score)),
                   SMOE_PINNED_STAGING=str(int(a.pinned_staging)),
                   SMOE_STRICT_SCORE_ORDER=str(int(a.strict_score_order)),
                   SMOE_TRACK_DECODE_WORK='1',
                   SMOE_LAYER_CACHE_FLOOR=str(a.layer_cache_floor),
                   SMOE_PREFETCH_EARLY=str(int(a.prefetch_early)),
                   SMOE_PREFETCH_EVENT_CHAIN=str(int(a.prefetch_event_chain)),
                   SMOE_PREFETCH_SCORE_ORDER=str(int(a.prefetch_score_order)),
                   SMOE_PREFETCH_FUSED_NORMS=str(int(a.prefetch_fused_norms)),
                   SMOE_RESERVE_LOAD_CORE=str(int(a.reserve_load_core)))
        print(f'START CPU_CORE={cores}: {log}', flush=True)
        with log.open('w') as f:
            f.write(json.dumps({'command': cmd}) + '\n')
            f.flush()
            child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
            with (a.outdir / f'cpu{cores}_resources.jsonl').open('w') as resource_log:
                while child.poll() is None:
                    sample = {'timestamp': time.time(), 'pid': child.pid}
                    try:
                        status = Path(f'/proc/{child.pid}/status').read_text()
                        sample['process_kib'] = dict((k, int(v)) for k, v in
                            re.findall(r'^(VmRSS|VmSwap|VmLck):\s+(\d+) kB', status, re.M))
                    except FileNotFoundError:
                        pass
                    try:
                        gpu = subprocess.run(['nvidia-smi', '-i', '0',
                            '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, timeout=5)
                        sample['gpu_memory_mib_util_percent'] = gpu.stdout.strip()
                    except (OSError, subprocess.TimeoutExpired) as error:
                        sample['gpu_query_error'] = str(error)
                    resource_log.write(json.dumps(sample) + '\n')
                    resource_log.flush()
                    time.sleep(2)
            rc = child.wait()
        rc_path.write_text(str(rc))
        if rc:
            raise RuntimeError(f'CPU_CORE={cores} failed: {log}')
    if not log.exists():
        continue
    content = log.read_text(errors='replace')
    rows = [dict(prompt=int(i), prefill_s=float(pr), decode_s=float(de),
                 total_s=float(t), decode_tokens=int(n)) for i, pr, de, t, n in
            pattern.findall(content)]
    from gpt_output.optimization_3080.benchmark_metrics import parse_prompt_hits
    prompt_hits = parse_prompt_hits(content)
    for row in rows:
        row.update(prompt_hits[row['prompt']])
        row['warmup'] = row['prompt'] < 0
    measured = [r for r in rows if r['prompt'] >= 0]
    complete = (rc_path.exists() and rc_path.read_text().strip() == '0'
                and len(measured) == a.input_num
                and {r['prompt'] for r in measured} == set(range(a.input_num)))
    (a.outdir / f'cpu{cores}_prompts.json').write_text(json.dumps(rows, indent=2))
    if rows:
        with (a.outdir / f'cpu{cores}_prompts.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    result = dict(cpu_cores=cores, complete=complete, measured_prompts=len(measured),
                  dataset='wic', output_len=100,
                  if_replace=config['if_replace'], replace_score_ratio=config['replaceScoreRatio'],
                  if_prefetch=config['if_prefetch'], score_window=config['window_size'],
                  affinity=next((s for s in content.splitlines() if s.startswith('[AFFINITY]')), ''),
                  warmup_prompts=len(rows)-len(measured),
                  mean_decode_s=statistics.fmean(r['decode_s'] for r in measured) if measured else None,
                  mean_prefill_s=statistics.fmean(r['prefill_s'] for r in measured) if measured else None,
                  mean_total_s=statistics.fmean(r['total_s'] for r in measured) if measured else None,
                  weighted_decode_s=(sum(r['decode_s']*r['decode_tokens'] for r in measured) /
                                     sum(r['decode_tokens'] for r in measured)) if measured else None,
                  mean_generate_minus_prefill_per_decode_s=statistics.fmean(
                      (r['total_s']-r['prefill_s'])/r['decode_tokens'] for r in measured
                      if r['decode_tokens'] > 0) if measured else None,
                  excluded_prompts=[])
    result['gpu_cache_hits'] = sum(r['gpu_cache_hits'] for r in measured)
    result['routed_expert_calls'] = sum(r['routed_expert_calls'] for r in measured)
    result['gpu_cache_hit_rate'] = (result['gpu_cache_hits'] / result['routed_expert_calls']
                                  if result['routed_expert_calls'] else None)
    resource_path = a.outdir / f'cpu{cores}_resources.jsonl'
    if resource_path.exists():
        samples = [json.loads(s) for s in resource_path.read_text().splitlines()]
        device_mib = []
        for sample in samples:
            try:
                device_mib.append(float(sample['gpu_memory_mib_util_percent'].split(',')[0]))
            except (ValueError, KeyError):
                pass
        result['sampled_driver_peak_mib'] = max(device_mib, default=None)
        result['sampled_process_peak_rss_kib'] = max(
            (s.get('process_kib', {}).get('VmRSS', 0) for s in samples), default=None)
    generation_path = a.outdir / f'cpu{cores}' / 'generations.json'
    invocation_path = a.outdir / f'cpu{cores}' / 'invocation.json'
    if invocation_path.exists():
        run_env = json.loads(invocation_path.read_text()).get('environment', {})
        result['omp_wait_policy'] = run_env.get('OMP_WAIT_POLICY', 'unset')
        for column, variable in {
            'cpu_bf16_mv': 'SMOE_CPU_BF16_MV',
            'cpu_only_misses': 'SMOE_DECODE_CPU_ONLY',
            'decode_minmax': 'SMOE_DECODE_MINMAX',
            'load_high_score': 'SMOE_LOAD_HIGH_SCORE',
            'reserve_load_core': 'SMOE_RESERVE_LOAD_CORE',
            'strict_score_order': 'SMOE_STRICT_SCORE_ORDER',
            'pinned_staging': 'SMOE_PINNED_STAGING',
            'prefetch_early': 'SMOE_PREFETCH_EARLY',
            'prefetch_event_chain': 'SMOE_PREFETCH_EVENT_CHAIN',
            'prefetch_score_order': 'SMOE_PREFETCH_SCORE_ORDER',
            'prefetch_fused_norms': 'SMOE_PREFETCH_FUSED_NORMS',
        }.items():
            result[column] = run_env.get(variable, '0') == '1'
        result['layer_cache_floor'] = int(run_env.get('SMOE_LAYER_CACHE_FLOOR', '0'))
    if generation_path.exists():
        generations = json.loads(generation_path.read_text())
        for key in ('peak_allocated', 'peak_reserved'):
            result[key] = max((g[key] for g in generations), default=None)
    summary.append(result)
    (a.outdir / 'summary.json').write_text(json.dumps(summary, indent=2))
    with (a.outdir / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for s in summary for k in s)))
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, list) else v for k, v in s.items()}
                         for s in summary)
    print(json.dumps(result), flush=True)
