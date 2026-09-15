"""CPU3/14/3 paired diagnosis, tagged traces and identical CPU-work replay.

Diagnostic runs are explicitly separate from acceptance. No output limits,
router decisions, expert assignments, or model timing boundaries are changed.
"""
import argparse
import contextlib
import functools
import json
import os
from pathlib import Path
import runpy
import statistics
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
p=argparse.ArgumentParser()
p.add_argument('--outdir',type=Path,required=True)
p.add_argument('--cores',type=int,nargs='+',default=[3,14,3])
p.add_argument('--policies',nargs='+',default=None)
p.add_argument('--input-num',type=int,default=3)
p.add_argument('--trace-all',action='store_true')
p.add_argument('--capture-on-load',action='store_true')
a=p.parse_args()
if a.outdir.exists() and any(a.outdir.iterdir()):
    raise FileExistsError(a.outdir)
a.outdir.mkdir(parents=True,exist_ok=True)
os.environ.update(OMP_NUM_THREADS='13',MKL_NUM_THREADS='13',OPENBLAS_NUM_THREADS='1',
    TOKENIZERS_PARALLELISM='false',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
    SMOE_CPU_BF16_MV='1',SMOE_DECODE_CPU_ONLY='1',SMOE_STRICT_SCORE_ORDER='1',
    SMOE_TRACK_DECODE_WORK='1',SMOE_LAYER_CACHE_FLOOR='12',SMOE_RESERVE_LOAD_CORE='1',
    SMOE_CPU_AVX2_GEMV='0',SMOE_GPU_TRITON_EXPERT='0',SMOE_GPU_GROUPED_TRITON='0',
    SMOE_TRITON_NORM='0',SMOE_TRITON_ROPE='0',SMOE_CPU_BATCH_FORWARD='0',SMOE_GPU_INLINE_SUBMIT='0')
os.environ.pop('OMP_WAIT_POLICY',None)
sys.setswitchinterval(.0001)
cfg=json.loads((ROOT/'configs/deepseekmoe_config.json').read_text())
cfg.update(window_size=16,if_prefetch=False)
config_path=a.outdir/'config.json'
config_path.write_text(json.dumps(cfg,indent=2))
from gpt_output.optimization_3080.snapshot_source import snapshot
snapshot(ROOT,a.outdir)
(a.outdir/'invocation.json').write_text(json.dumps({'argv':sys.argv,'environment':{
    k:v for k,v in os.environ.items() if k.startswith(('OMP_','SMOE_','MKL_','OPENBLAS_','PYTORCH_'))}},indent=2))
# Build through main.py without generating yet. We explicitly load the same
# first ten WiC inputs afterwards and use indices 8/9 for every warmup pair.
sys.argv=[str(ROOT/'main.py'),'--model_name','deepseekmoe','--model_path','/root/models/deepseekmoe',
    '--config_path',str(config_path),'--dataset_path','wic','--input_num','0',
    '--warmup_num','0','--output_len','100','--cpu_cores','14','--GPU_mem','10']
state=runpy.run_path(str(ROOT/'main.py'),run_name='__main__')
torch,model,tokenizer=state['torch'],state['model'],state['tokenizer']
ec,sb=state['expertcache'],state['_smoe_base']
compute_cpus=sorted(state['_compute_cores'])
from utils.cpu_bf16_kernel import load_kernel
load_kernel()
shared_cpu=state['_shared_core']
from utils.load_dataset import load_all
prompts=load_all('wic',1,10)
layers=[l.mlp for l in model.model.layers if hasattr(l.mlp,'ExpertCache')]
cache=layers[0].ExpertCache
mode='off'
events=[]
last_cpu_work={}

def wrap(cls,name,label,device_label=False):
    original=getattr(cls,name)
    @functools.wraps(original)
    def call(self,*args,**kwargs):
        if mode=='off': return original(self,*args,**kwargs)
        tag=label
        if device_label:
            tag += '_'+self.storage.device.type
        layer=getattr(self,'layerid',-1)
        start=time.perf_counter_ns()
        cm=torch.profiler.record_function(f'SMoE::{tag}/L{layer}') if mode=='trace' else contextlib.nullcontext()
        with cm:
            try: return original(self,*args,**kwargs)
            finally:
                events.append(dict(stage=tag,layer=layer,start_ns=start,
                    end_ns=time.perf_counter_ns(),thread=threading.get_native_id(),token=ec.tokens))
                if name=='_cpu_compute' and args[0] and self._staged_decode_input is not None:
                    # Snapshot diagnostic replay inputs after computation.
                    # This extra copy is never part of acceptance measurements.
                    last_cpu_work[layer]=(list(args[0]),self._staged_decode_input[0].clone())
    setattr(cls,name,call)

from utils.model_loader import ExpertWrapper
wrap(ExpertWrapper,'forward','expert_forward',True)
wrap(sb.AbstractMoELayer,'run_with_cache','moe_layer')
wrap(sb.AbstractMoELayer,'_cpu_compute','cpu_stage')
wrap(sb.AbstractMoELayer,'_stage_decode_input','activation_d2h_enqueue')
wrap(sb.AbstractMoELayer,'_work_cachehit_and_predict','gpu_hit_submit')
wrap(sb._PersistentBgThread,'submit','background_submit')
wrap(sb._PersistentBgThread,'wait','background_wait')
wrap(ec.ExpertCache,'wait_until_queue_empty','load_queue_wait')
wrap(ec.ExpertCache,'_swap','weight_copy_submit')
wrap(ec.ExpertCache,'get_compute_expert','expert_acquire')
wrap(ec.EvictionInfo,'choose_expert_to_evictbyScore','eviction_select')
from MoEModule.deepseek_moe import DeepseekMoEwithCache
wrap(DeepseekMoEwithCache,'compute_shared_expert','shared_gpu_submit')

def reset():
    ec.tokens=0;ec.decode_time=ec.prefill_time=0.
    ec.cache_hits_per_token=ec.cache_total_per_token=0
    sb.cpu_compute_ms_per_token.clear();sb.cpu_compute_token_indices.clear()
    sb._cpu_ms_cur_token_samples.clear();sb._cpu_ms_cur_token_idx=-1
    sb.cpu_decode_forward_ms.clear();sb.decode_work_by_layer.clear()
    sb.decode_balance_by_layer.clear()
    sb.cpu_decode_stage_ms.clear();sb.cpu_batch_forward_boundary_ms.clear()

def generate(index):
    reset()
    probes_before=cache.decode_probe_count
    inputs=tokenizer(prompts[index],return_tensors='pt',padding=True,truncation=True)
    inputs={k:v.to('cuda:0') for k,v in inputs.items() if k!='token_type_ids'}
    torch.cuda.synchronize();start=time.perf_counter()
    with torch.no_grad():output=model.generate(**inputs,max_new_tokens=100)
    torch.cuda.synchronize();wall=time.perf_counter()-start
    work=dict(sb.decode_work_by_layer)
    return dict(prompt=index,decode_s=ec.decode_time/(ec.tokens-1),decode_tokens=ec.tokens-1,
        prefill_s=ec.prefill_time,wall_s=wall,cpu_expert_forward_calls=len(sb.cpu_decode_forward_ms),
        cpu_expert_forward_total_ms=sum(sb.cpu_decode_forward_ms),
        cpu_expert_forward_mean_ms=statistics.fmean(sb.cpu_decode_forward_ms),
        cpu_forward_scope='native_aten_forward' if sb.cpu_batch_forward_boundary_ms else 'python_expert_forward',
        cpu_stage_total_ms=sum(sb.cpu_decode_stage_ms),
        cpu_batch_boundary_total_ms=sum(sb.cpu_batch_forward_boundary_ms),
        gpu_cache_hit_rate=sum(w[1] for w in work.values())/sum(sum(w[1:]) for w in work.values()),
        work=work,balance_estimates_by_layer=dict(sb.decode_balance_by_layer),
        decode_load_samples_s=list(cache.DecodeLoadTimeOneExpert),decode_probe_experts=cache.decode_probe_count-probes_before,
        token_ids=output.cpu().tolist())

def placement(cores):
    # One persistent GPU worker CPU remains isolated in every comparison.
    os.sched_setaffinity(0,set(compute_cpus))
    torch.set_num_threads(cores-1)
    active=compute_cpus[:cores-1]
    os.sched_setaffinity(0,set(active))
    # Set affinity inside each participating OpenMP worker, including the
    # caller, so recycled runtime threads cannot retain a wider mask.
    torch.ops.smoe_cpu.bind_threads(active)
    return active

policies=a.policies or ['baseline']*len(a.cores)
assert len(policies)==len(a.cores)
results=[]
for order,(cores,policy) in enumerate(zip(a.cores,policies)):
    assert policy in ('baseline','grouped_norm_rope','grouped_norm_rope_cpu_batch',
                      'grouped_norm_rope_cpu_batch_inline','grouped_norm_rope_cpu_batch_load1',
                      'grouped_norm_rope_cpu_batch_balance','grouped_norm_rope_cpu_batch_balance_minmax',
                      'grouped_norm_rope_cpu_batch_balance_minmax_pinned',
                      'grouped_norm_rope_cpu_batch_balance_minmax_pinned_calibrated')
    cache.clear_queue();cache.wait_until_queue_empty();torch.cuda.synchronize()
    cache.pinned_staging='pinned' in policy
    cache.DecodeLoadTimeOneExpert.clear()
    cache.decode_load_sample_age=0;cache.decode_probe_count=0
    from utils import decode_norm,decode_rope
    decode_norm.enabled='norm' in policy
    decode_rope.enabled='rope' in policy
    if decode_norm.enabled:decode_norm.install(model)
    if decode_rope.enabled:decode_rope.install(model)
    for layer in layers:
        layer._decode_cpu_only='balance' not in policy
        layer._decode_minmax='minmax' in policy
        layer._decode_cost_samples='calibrated' in policy
        layer._decode_load_limit=1 if 'load1' in policy else 0
        layer._cpu_batch_forward='cpu_batch' in policy
        layer._gpu_inline_submit='inline' in policy
        layer._grouped_decode_enabled='grouped' in policy
        if layer._grouped_decode_enabled and layer._grouped_decode is None:
            from utils.grouped_decode import GroupedDecode
            layer._grouped_decode=GroupedDecode(cfg['hidden_size'],cfg['moe_intermediate_size'],
                cfg['num_experts_per_tok'],'cuda:0')
    stem=a.outdir/f'{order:02d}_cpu{cores}'
    stem.mkdir()
    active=placement(cores)
    print('[TRACE CPU]',cores,active,'shared',shared_cpu,flush=True)
    for index in (8,9):
        row=generate(index)
        print('[TRACE WARMUP]',cores,index,row['decode_s'],flush=True)
    rows=[generate(index) for index in range(a.input_num)]
    (stem/'unprofiled_prompts.json').write_text(json.dumps(rows,indent=2))
    entry=dict(order=order,cpu_cores=cores,policy=policy,compute_cpus=active,shared_cpu=shared_cpu,
        cpu_forward_scope=rows[0]['cpu_forward_scope'],
        pinned_staging=cache.pinned_staging,decode_probe_experts=sum(r['decode_probe_experts'] for r in rows),
        demand_loads_per_decode_token=sum(v[2] for r in rows for v in r['work'].values())/sum(r['decode_tokens'] for r in rows),
        cpu_avg_estimate_ms=sum(v[1] for r in rows for v in r['balance_estimates_by_layer'].values())/sum(v[0] for r in rows for v in r['balance_estimates_by_layer'].values()),
        pcie_load_estimate_ms=sum(v[2] for r in rows for v in r['balance_estimates_by_layer'].values())/sum(v[0] for r in rows for v in r['balance_estimates_by_layer'].values()),
        cpu_stage_ms_per_decode_token=sum(r['cpu_stage_total_ms'] for r in rows)/sum(r['decode_tokens'] for r in rows),
        mean_decode_s=statistics.fmean(r['decode_s'] for r in rows),
        cpu_expert_forward_mean_ms=sum(r['cpu_expert_forward_total_ms'] for r in rows)/sum(r['cpu_expert_forward_calls'] for r in rows),
        gpu_cache_hit_rate=sum(w[1] for r in rows for w in r['work'].values())/sum(sum(w[1:]) for r in rows for w in r['work'].values()))
    results.append(entry)
    (a.outdir/'summary.json').write_text(json.dumps(results,indent=2))
    print('[TRACE UNPROFILED]',json.dumps(entry),flush=True)
    original=model.model.forward
    # Separate low-overhead host timestamps and full CPU/CUDA profiler. Both
    # capture forward steps 11..15 after prefill+ten decode steps.
    for capture in (('timers','trace') if order<2 or a.trace_all else ('timers',)):
        step=0;events.clear();last_cpu_work.clear()
        capture_start=None;last_step_had_dma=False
        tracer=None
        def forward(*fa,**fk):
            global step,mode,tracer,capture_start,last_step_had_dma
            nearing_refresh=cache.decode_load_sample_age>=16*len(layers)-len(layers)
            start_now=(step>=11 and capture_start is None and (not a.capture_on_load
                or last_step_had_dma or nearing_refresh or step>=40))
            if start_now:
                capture_start=step
                mode=capture
                if capture=='trace':
                    tracer=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],record_shapes=True,with_stack=True)
                    tracer.__enter__()
            cm=torch.profiler.record_function('SMoE::decode_step') if mode=='trace' else contextlib.nullcontext()
            copies_before=cache.measured_dma_copies
            with cm: result=original(*fa,**fk)
            last_step_had_dma=cache.measured_dma_copies>copies_before
            step+=1
            if capture_start is not None and step==capture_start+5:
                mode='off'
                if tracer is not None:
                    tracer.__exit__(None,None,None)
                    tracer.export_chrome_trace(str(stem/'decode.trace.json'))
                    (stem/'operators.txt').write_text(tracer.key_averages().table(sort_by='self_cpu_time_total',row_limit=70))
            return result
        model.model.forward=forward
        try: record=generate(0)
        finally: model.model.forward=original;mode='off'
        clock_samples=[]
        for _ in range(25):
            before=time.perf_counter_ns();wall=time.time_ns();after=time.perf_counter_ns()
            clock_samples.append((after-before,wall-(before+after)//2))
        width,offset=min(clock_samples)
        (stem/f'{capture}_clock_anchor.json').write_text(json.dumps(dict(sample_bracket_ns=width,clock_offset_ns=offset)))
        (stem/f'{capture}_capture_window.json').write_text(json.dumps(dict(start_forward_step=capture_start,steps=5,capture_on_load=a.capture_on_load)))
        (stem/f'{capture}_host_events.json').write_text(json.dumps(events))
        (stem/f'{capture}_prompt.json').write_text(json.dumps(record,indent=2))
        print('[TRACE CAPTURED]',cores,capture,len(events),flush=True)
        # Do not carry Kineto event/stack storage into the next unprofiled A/B.
        tracer=None
        import gc,ctypes
        gc.collect()
        ctypes.CDLL(None).malloc_trim(0)
    # Per-thread masks are evidence, not an assumption that n-1 means n-1 cores.
    masks={}
    for tid in Path('/proc/self/task').iterdir():
        try: masks[tid.name]=sorted(os.sched_getaffinity(int(tid.name)))
        except ProcessLookupError:pass
    (stem/'thread_affinities.json').write_text(json.dumps(masks,indent=2))

# Repeat exactly the same saved CPU expert calls at 2 and 13 intra-op workers.
# No GPU expert work and no router are active in this isolated mechanism test.
cases=[(cache.get_compute_expert(uid,offload=True),x) for _,(uids,x) in sorted(last_cpu_work.items()) for uid in uids]
micro=[]
for cores in (3,14,3,14):
    placement(cores)
    with torch.no_grad():
        for expert,x in cases:expert(x)
        for repeat in range(3):
            timings=[]
            for expert,x in cases:
                start=time.perf_counter_ns();expert(x);timings.append((time.perf_counter_ns()-start)/1e6)
            micro.append(dict(cpu_cores=cores,repeat=repeat,expert_calls=len(cases),
                mean_forward_ms=statistics.fmean(timings),total_forward_ms=sum(timings)))
(a.outdir/'identical_cpu_work_replay.json').write_text(json.dumps(micro,indent=2))
print('[IDENTICAL CPU REPLAY]',json.dumps(micro),flush=True)
