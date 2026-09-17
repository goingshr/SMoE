#!/usr/bin/env python3
"""Reproducible WiC acceptance runner; preserve raw logs and direct counters."""
import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'gpt_output'))
from run_xverse_wic_4060ti import SMOE_RE,HIT_RE,CPU_RE,GPU_RE


def parse(text,warmup,n):
    rows={}
    for m in SMOE_RE.finditer(text):
        rows.setdefault(int(m[1]),{}).update(prompt=int(m[1]),prefill_s=float(m[2]),
            avg_decode_s=float(m[3]),total_s=float(m[4]),decode_tokens=int(m[5]))
    for m in HIT_RE.finditer(text):
        rows.setdefault(int(m[1]),{}).update(gpu_cache_hit_rate=float(m[2]),
            gpu_cache_hits=int(m[3]),routed_experts=int(m[4]))
    for m in CPU_RE.finditer(text):
        rows.setdefault(int(m[1]),{}).update(cpu_token_mean_expert_ms=float(m[2]))
    for m in GPU_RE.finditer(text):
        rows.setdefault(int(m[1]),{}).update(cache_slots=int(m[2]),
            peak_allocated_bytes=int(m[3]),peak_reserved_bytes=int(m[4]))
    for m in re.finditer(r'\[Decode metrics\] prompt=(\d+) (\{[^\n]+\})',text):
        rows.setdefault(int(m[1]),{}).update(json.loads(m[2]))
    measured=[rows[i] for i in range(warmup,warmup+n) if i in rows]
    required={'avg_decode_s','decode_tokens','cpu_forward_calls','gpu_cache_hits',
              'misses','miss_to_gpu','miss_to_cpu','gpu_forward_total_ms'}
    if len(measured)!=n or any(not required.issubset(r) for r in measured):
        raise ValueError('Incomplete prompt records; see raw log')
    for r in measured:
        assert r['routed_experts']==r['gpu_cache_hits']+r['misses']
        assert r['misses']==r['miss_to_gpu']+r['miss_to_cpu']
        assert r['miss_to_cpu']==r['cpu_forward_calls']
        assert r['gpu_cache_hits']+r['miss_to_gpu']==r['gpu_forward_calls']
    return measured


def aggregate(rows):
    def total(k): return sum(r[k] for r in rows)
    def avg(a,b): return total(a)/total(b) if total(b) else 0.
    out=dict(prompt_count=len(rows),mean_avg_decode_s=statistics.mean(r['avg_decode_s'] for r in rows),
        stdev_avg_decode_s=statistics.stdev(r['avg_decode_s'] for r in rows) if len(rows)>1 else 0.,
        weighted_avg_decode_s=sum(r['avg_decode_s']*r['decode_tokens'] for r in rows)/total('decode_tokens'),
        gpu_cache_hit_rate=avg('gpu_cache_hits','routed_experts'),
        avg_cpu_expert=avg('cpu_forward_total_ms','cpu_forward_calls'),
        avg_pcie_load=avg('pcie_load_total_ms','pcie_copies'),
        avg_gpu_forward=avg('gpu_forward_total_ms','gpu_forward_calls'),
        cpu_expert_forward_ms_per_decode_token=avg('cpu_forward_total_ms','decode_tokens'),
        miss_reload_gpu_fraction=avg('miss_to_gpu','misses'),
        miss_cpu_compute_fraction=avg('miss_to_cpu','misses'),
        mean_total_s=statistics.mean(r['total_s'] for r in rows),
        mean_prefill_s=statistics.mean(r['prefill_s'] for r in rows),
        timing_unit='ms per expert/copy; decode in s/token')
    for k in ('decode_tokens','gpu_cache_hits','routed_experts','misses','miss_to_gpu','miss_to_cpu',
              'cpu_forward_calls','pcie_copies','gpu_forward_calls','cpu_forward_total_ms',
              'pcie_load_total_ms','gpu_forward_total_ms'):
        out[k]=total(k)
    for k in ('peak_allocated_bytes','peak_reserved_bytes'):
        out[k]=max(r[k] for r in rows)
    if all('gpu_miss_forward_total_ms' in r for r in rows):
        out['avg_gpu_miss_forward_ms']=avg('gpu_miss_forward_total_ms','miss_to_gpu')
        out['avg_gpu_hit_forward_ms']=avg('gpu_hit_forward_total_ms','gpu_cache_hits')
    return out


def csv_write(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output-dir',required=True)
    p.add_argument('--cpus',type=int,nargs='+',default=[16,8,3])
    p.add_argument('--input-num',type=int,default=30)
    p.add_argument('--warmup-num',type=int,default=1)
    p.add_argument('--config',default=str(ROOT/'configs/xversemoe_config.json'))
    p.add_argument('--tag',default='optimized')
    p.add_argument('--parse-only',action='store_true')
    a=p.parse_args(); dest=Path(a.output_dir);dest.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(Path(a.config).read_text());assert cfg['replaceScoreRatio']==.25
    env=os.environ.copy();env.update(PYTHON_GIL='0',PYTHONPATH=str(ROOT),
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',SMOE_DECODE_METRICS='1')
    manifest=dict(base_commit='37a7d8b',config=cfg,seed=37,input_num=a.input_num,
        warmup_num=a.warmup_num,input_len=30,output_len=100,gpu_mem=14.,cpus=a.cpus,
        environment={k:v for k,v in env.items() if k.startswith(('SMOE_','PYTHON','OMP_','MKL_'))},
        dataset_sha256=hashlib.sha256((ROOT/'datasets/SuperGLUE/WiC/val.jsonl').read_bytes()).hexdigest(),commands=[])
    summaries=[];all_rows=[]
    for cpu in a.cpus:
        command=[sys.executable,'-u',str(ROOT/'main.py'),'--model_name','xversemoe',
            '--model_path','/root/models/xversemoe','--config_path',a.config,'--dataset_path','wic',
            '--input_num',str(a.input_num),'--warmup_num',str(a.warmup_num),'--input_len','30',
            '--output_len','100','--GPU_mem','14','--cpu_cores',str(cpu),'--seed','37']
        manifest['commands'].append(command)
        (dest/'manifest.json').write_text(json.dumps(manifest,indent=2))
        log_path=dest/f'cpu{cpu}.log'
        peak=0
        if not a.parse_only:
            started=time.time()
            with log_path.open('w') as f:
                f.write('[COMMAND] '+json.dumps(command)+'\n');f.flush()
                process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
                while process.poll() is None:
                    q=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],
                        capture_output=True,text=True)
                    if q.returncode==0: peak=max(peak,int(q.stdout.strip().splitlines()[0]))
                    time.sleep(2)
            if process.returncode: raise RuntimeError(f'CPU={cpu} failed: {log_path}')
            (dest/f'cpu{cpu}_resource.json').write_text(json.dumps(dict(driver_peak_mib=peak,
                wall_s=time.time()-started,returncode=process.returncode)))
        rows=parse(log_path.read_text(),a.warmup_num,a.input_num)
        summary=dict(tag=a.tag,cpu_cores=cpu,gpu_mem_gib=14.,replace=.25,prefetch=cfg['if_prefetch'],
            **aggregate(rows))
        resource=dest/f'cpu{cpu}_resource.json'
        if resource.exists():summary.update(json.loads(resource.read_text()))
        summary['target_met']=summary['mean_avg_decode_s']<=.07 if cpu==16 else ''
        summaries.append(summary)
        all_rows.extend(dict(cpu_cores=cpu,**r) for r in rows)
        csv_write(dest/'results.csv',summaries);csv_write(dest/'prompts.csv',all_rows)
        (dest/'summary.json').write_text(json.dumps(dict(manifest=manifest,runs=summaries,prompts=all_rows),indent=2))
        print(json.dumps(summary),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
