"""Resident-model CPU14 A/B and separate warmed torch.profiler capture."""
import argparse
import csv
import json
import os
from pathlib import Path
import runpy
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
p = argparse.ArgumentParser()
p.add_argument('--outdir', required=True, type=Path)
p.add_argument('--input-num', type=int, default=3)
p.add_argument('--policies', nargs='+', default=['baseline', 'avx2', 'triton', 'both', 'baseline'])
p.add_argument('--profile', action='store_true')
p.add_argument('--skip-initial-generation',action='store_true')
p.add_argument('--python-profile', action='store_true')
a = p.parse_args()
a.outdir.mkdir(parents=True, exist_ok=False)
os.environ.update(OMP_NUM_THREADS='13', MKL_NUM_THREADS='13', OPENBLAS_NUM_THREADS='1',
    TOKENIZERS_PARALLELISM='false', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
    SMOE_CPU_BF16_MV='1', SMOE_DECODE_CPU_ONLY='1', SMOE_STRICT_SCORE_ORDER='1',
    SMOE_TRACK_DECODE_WORK='1', SMOE_LAYER_CACHE_FLOOR='12', SMOE_RESERVE_LOAD_CORE='1',
    SMOE_CPU_AVX2_GEMV='0', SMOE_GPU_TRITON_EXPERT='0', SMOE_CPU_BATCH_FORWARD='0',
    SMOE_GPU_GROUPED_TRITON='0',SMOE_TRITON_NORM='0',SMOE_TRITON_ROPE='0',SMOE_GPU_INLINE_SUBMIT='0',SMOE_DECODE_LOAD_LIMIT='0')
os.environ.pop('OMP_WAIT_POLICY', None)
sys.setswitchinterval(.0001)
invocation = list(sys.argv)
config = json.loads((ROOT/'configs/deepseekmoe_config.json').read_text())
config.update(window_size=16, if_prefetch=False)
assert config['if_replace'] and config['replaceScoreRatio'] <= .35
cfg = a.outdir/'config.json'
cfg.write_text(json.dumps(config, indent=2))
from gpt_output.optimization_3080.snapshot_source import snapshot
snapshot(ROOT, a.outdir)
(a.outdir/'invocation.json').write_text(json.dumps({'argv':invocation,'environment':{
    k:v for k,v in os.environ.items() if k.startswith(('SMOE_','OMP_','MKL_','OPENBLAS_','PYTORCH_'))}},indent=2))

# Warm/reference generation uses the unmodified application entrypoint.
sys.argv = [str(ROOT/'main.py'), '--model_name','deepseekmoe','--model_path','/root/models/deepseekmoe',
    '--config_path',str(cfg),'--dataset_path','wic','--input_num',str(0 if a.skip_initial_generation else a.input_num),
    '--warmup_num',str(0 if a.skip_initial_generation else 2),'--output_len','100','--cpu_cores','14','--GPU_mem','10']
state = runpy.run_path(str(ROOT/'main.py'), run_name='__main__')
torch, model, tokenizer = state['torch'],state['model'],state['tokenizer']
ec,sb,prompts = state['expertcache'],state['_smoe_base'],state['all_inputs']
if a.skip_initial_generation:
    from utils.load_dataset import load_all
    prompts=load_all('wic',1,a.input_num)
import MoEModule.fused_gate_up as fg
from utils.cpu_bf16_kernel import load_kernel
fg._avx2_gemv = load_kernel()

def generate(i):
    ec.tokens=0
    ec.decode_time=ec.prefill_time=0.
    ec.cache_hits_per_token=ec.cache_total_per_token=0
    sb.cpu_compute_ms_per_token.clear()
    sb.cpu_compute_token_indices.clear()
    sb._cpu_ms_cur_token_samples.clear()
    sb._cpu_ms_cur_token_idx=-1
    sb.decode_work_by_layer.clear()
    sb.decode_balance_by_layer.clear()
    sb.cpu_decode_forward_ms.clear()
    sb.cpu_decode_stage_ms.clear()
    sb.cpu_batch_forward_boundary_ms.clear()
    inputs=tokenizer(prompts[i%len(prompts)],return_tensors='pt',padding=True,truncation=True)
    inputs={k:v.to('cuda:0') for k,v in inputs.items() if k!='token_type_ids'}
    torch.cuda.synchronize()
    t=time.perf_counter()
    with torch.no_grad(): out=model.generate(**inputs,max_new_tokens=100)
    torch.cuda.synchronize()
    wall=time.perf_counter()-t
    work=dict(sb.decode_work_by_layer)
    hits=sum(v[1] for v in work.values())
    calls=sum(sum(v[1:]) for v in work.values())
    return dict(prompt=i,warmup=i<0,decode_tokens=ec.tokens-1,
        decode_s=ec.decode_time/(ec.tokens-1),prefill_s=ec.prefill_time,wall_s=wall,
        gpu_cache_hits=hits,routed_expert_calls=calls,gpu_cache_hit_rate=hits/calls,
        decode_work_by_layer=work,balance_estimates_by_layer=dict(sb.decode_balance_by_layer),
        token_ids=out.cpu().tolist(),
        cpu_expert_forward_calls=len(sb.cpu_decode_forward_ms),
        cpu_expert_forward_total_ms=sum(sb.cpu_decode_forward_ms),
        cpu_expert_forward_mean_ms=statistics.fmean(sb.cpu_decode_forward_ms) if sb.cpu_decode_forward_ms else 0.,
        cpu_forward_scope="native_aten_forward" if sb.cpu_batch_forward_boundary_ms else "python_expert_forward",
        cpu_stage_total_ms=sum(sb.cpu_decode_stage_ms),
        cpu_batch_boundary_total_ms=sum(sb.cpu_batch_forward_boundary_ms),
        peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())

if a.profile:
    original=model.model.forward
    def on_trace(prof):
        prof.export_chrome_trace(str(a.outdir/'baseline_decode.trace.json'))
        (a.outdir/'baseline_operators.txt').write_text(prof.key_averages().table(
            sort_by='self_cpu_time_total',row_limit=50))
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=10,warmup=1,active=5,repeat=1),
        record_shapes=True,with_stack=True,on_trace_ready=on_trace) as prof:
        def traced(*fa,**fk):
            r=original(*fa,**fk)
            prof.step()
            return r
        model.model.forward=traced
        try: generate(0)
        finally: model.model.forward=original
    del prof
    import gc
    gc.collect()

summary=[]
if a.python_profile:
    import cProfile, pstats
    profiler = cProfile.Profile()
    profiler.enable()
    generate(0)
    profiler.disable()
    profiler.dump_stats(str(a.outdir/'python_profile.pstats'))
    with (a.outdir/'python_profile.txt').open('w') as f:
        pstats.Stats(profiler,stream=f).sort_stats('cumulative').print_stats(100)
for index,policy in enumerate(a.policies):
    assert policy in ('baseline','avx2','triton','both','grouped','norm','grouped_norm',
                      'grouped_norm_rope','grouped_norm_rope_prefetch','cpu_batch',
                      'grouped_cpu_batch','grouped_norm_rope_cpu_batch',
                      'grouped_norm_rope_cpu_batch_inline','grouped_norm_rope_cpu_batch_inline_load1',
                      'grouped_norm_rope_cpu_batch_prefetch','grouped_norm_rope_cpu_batch_load1',
                      'grouped_norm_rope_cpu_batch_balance','grouped_norm_rope_cpu_batch_balance_minmax')
    cache = model.model.layers[1].mlp.ExpertCache
    cache.clear_queue()
    cache.wait_until_queue_empty()
    torch.cuda.synchronize()
    fg.CPU_AVX2_GEMV=policy in ('avx2','both')
    fg.GPU_TRITON_EXPERT=policy in ('triton','both')
    from utils import decode_norm
    decode_norm.enabled='norm' in policy
    if decode_norm.enabled:
        decode_norm.install(model)
    from utils import decode_rope
    decode_rope.enabled='rope' in policy
    if decode_rope.enabled:
        decode_rope.install(model)
    for layer in model.model.layers:
        if hasattr(layer.mlp, '_grouped_decode_enabled'):
            if 'grouped' in policy and layer.mlp._grouped_decode is None:
                from utils.grouped_decode import GroupedDecode
                layer.mlp._grouped_decode=GroupedDecode(config['hidden_size'],
                    config['moe_intermediate_size'],config['num_experts_per_tok'],'cuda:0')
            layer.mlp._decode_cpu_only='balance' not in policy
            layer.mlp._decode_minmax='minmax' in policy
            layer.mlp._decode_load_limit=1 if 'load1' in policy else 0
            layer.mlp._gpu_inline_submit='inline' in policy
            layer.mlp._cpu_batch_forward='cpu_batch' in policy
            layer.mlp._grouped_decode_enabled='grouped' in policy
            layer.mlp.if_prefetch='prefetch' in policy
            layer.mlp.config.if_prefetch=layer.mlp.if_prefetch
            layer.mlp._prefetch_early=layer.mlp.if_prefetch
            layer.mlp._prefetch_event_chain=layer.mlp.if_prefetch
            layer.mlp._prefetch_fused_norms=layer.mlp.if_prefetch
            layer.mlp._prefetch_score_order=layer.mlp.if_prefetch
    print('[KERNEL POLICY]',index,policy,flush=True)
    records=[]
    for i in range(-2,a.input_num):
        record=generate(i)
        records.append(record)
        (a.outdir/f'{index:02d}_{policy}.json').write_text(json.dumps(records,indent=2))
        print('[KERNEL PROMPT]',policy,json.dumps({k:v for k,v in record.items()
            if k not in ('token_ids','decode_work_by_layer','balance_estimates_by_layer')}),flush=True)
    measured=[r for r in records if not r['warmup']]
    row=dict(index=index,policy=policy,cpu_cores=14,prompts=len(measured),
        if_prefetch='prefetch' in policy,cpu_only_misses='balance' not in policy,decode_minmax='minmax' in policy,
        decode_load_limit=1 if 'load1' in policy else 0,
        demand_loaded_experts=sum(v[2] for r in measured for v in r['decode_work_by_layer'].values()),
        mean_decode_s=statistics.fmean(r['decode_s'] for r in measured),
        weighted_decode_s=sum(r['decode_s']*r['decode_tokens'] for r in measured)/sum(r['decode_tokens'] for r in measured),
        gpu_cache_hit_rate=sum(r['gpu_cache_hits'] for r in measured)/sum(r['routed_expert_calls'] for r in measured))
    row['cpu_expert_forward_calls']=sum(r['cpu_expert_forward_calls'] for r in measured)
    row['cpu_expert_forward_mean_ms']=sum(r['cpu_expert_forward_total_ms'] for r in measured)/row['cpu_expert_forward_calls']
    row['cpu_expert_forward_ms_per_decode_token']=sum(r['cpu_expert_forward_total_ms'] for r in measured)/sum(r['decode_tokens'] for r in measured)
    row['cpu_forward_scope']=measured[0]['cpu_forward_scope']
    row['cpu_stage_ms_per_decode_token']=sum(r['cpu_stage_total_ms'] for r in measured)/sum(r['decode_tokens'] for r in measured)
    row['cpu_batch_boundary_ms_per_decode_token']=sum(r['cpu_batch_boundary_total_ms'] for r in measured)/sum(r['decode_tokens'] for r in measured)
    estimates=[v for r in measured for v in r['balance_estimates_by_layer'].values()]
    for index,key in [(1,'cpu_avg_estimate_ms'),(2,'pcie_load_estimate_ms')]:
        row[key]=sum(v[index] for v in estimates)/sum(v[0] for v in estimates)
    row['predicted_cpu_ms_per_decode_token']=sum(v[3] for v in estimates)/sum(r['decode_tokens'] for r in measured)
    row['predicted_pcie_ms_per_decode_token']=sum(v[4] for v in estimates)/sum(r['decode_tokens'] for r in measured)
    summary.append(row)
    (a.outdir/'summary.json').write_text(json.dumps(summary,indent=2))
    with (a.outdir/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(row)); writer.writeheader(); writer.writerows(summary)
    print('[KERNEL RESULT]',json.dumps(row),flush=True)
