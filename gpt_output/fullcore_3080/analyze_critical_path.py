"""Render requested CPU/GPU/PCIe views using the supplied skill's trace helpers.

The skill's three tables remain separate and unmodified. This adapter selects
SMoE record_function annotations and uses its existing interval sweep for the
additional user-requested host/device overlap measurements.
"""
import argparse
from collections import Counter,defaultdict
import html
import json
from pathlib import Path
import sys
import time

SKILL=Path('/root/Codex-Skills/Byte_work/perf/llm-torch-profiler-analysis/scripts')
sys.path.insert(0,str(SKILL))
import profile_common as common
import triage_overlap_helpers as overlap

p=argparse.ArgumentParser()
p.add_argument('directory',type=Path)
a=p.parse_args()
trace=common.load_trace_json(a.directory/'decode.trace.json')
raw=common.extract_trace_events(trace)
kernels,gpu_pid=overlap.extract_kernel_events(trace,None)
annotations=[e for e in raw if common.is_complete_duration_event(e)
             and (e.get('cat')=='user_annotation' or (e.get('cat')=='cpu_op' and e.get('name')=='SMoE::cpu_expert_native'))
             and str(e.get('name','')).startswith('SMoE::')]
# Kineto's CPU annotations are thread-local, while CUPTI runtime/device
# activities include the persistent Python GPU worker. Align its independently
# recorded host spans using trace epoch metadata and the Linux clock offset;
# validate against every main-thread CPU expert range before using the overlay.
host=json.loads((a.directory/'trace_host_events.json').read_text())
anchors=[]
for _ in range(25):
    before=time.perf_counter_ns();wall=time.time_ns();after=time.perf_counter_ns()
    anchors.append((after-before,wall-(before+after)//2))
width,clock_offset=min(anchors)
anchor_file=a.directory/'trace_clock_anchor.json'
if anchor_file.exists():
    saved=json.loads(anchor_file.read_text())
    width,clock_offset=saved['sample_bracket_ns'],saved['clock_offset_ns']
base=int(trace['baseTimeNanoseconds'])
shift_us=(clock_offset-base)/1000
anchor_tag='cpu_stage'
cpu_recorded=sorted([e for e in annotations if e['name'].startswith('SMoE::'+anchor_tag+'/')],key=lambda e:e['ts'])
cpu_timed=sorted([e for e in host if e['stage']==anchor_tag],key=lambda e:e['start_ns'])
assert len(cpu_recorded)==len(cpu_timed)
entry_slack=[e['ts']-(h['start_ns']/1000+shift_us) for e,h in zip(cpu_recorded,cpu_timed)]
exit_slack=[e['ts']+e['dur']-(h['end_ns']/1000+shift_us) for e,h in zip(cpu_recorded,cpu_timed)]
if min(entry_slack+exit_slack)<-5:
    raise ValueError('Clock alignment failed validation; recapture with explicit clock anchors')
clock_evidence=dict(method='trace baseTimeNanoseconds + bracketed realtime/perf_counter offset',
    sample_bracket_ns=width,clock_offset_ns=clock_offset,trace_shift_us=shift_us,
    cpu_entry_slack_min_us=min(entry_slack),cpu_entry_slack_max_us=max(entry_slack),
    cpu_exit_slack_min_us=min(exit_slack),validated_cpu_calls=len(cpu_recorded),anchor_stage=anchor_tag)
for e in host:
    if e['stage'] not in ('expert_forward_cuda','gpu_hit_submit','weight_copy_submit','eviction_select','expert_acquire'):continue
    target_name=f"SMoE::{e['stage']}/L{e['layer']}"
    if any(v['name']==target_name and str(v['tid'])==str(e['thread']) and v.get('cat')=='user_annotation' for v in annotations):continue
    annotations.append(dict(name=f"SMoE::{e['stage']}/L{e['layer']}",
        cat='smoe_host_timer_overlay',pid=cpu_recorded[0]['pid'],tid=e['thread'],
        ts=e['start_ns']/1000+shift_us,dur=(e['end_ns']-e['start_ns'])/1000))
bythread=defaultdict(list)
for e in annotations:bythread[(str(e['pid']),str(e['tid']))].append(e)
contexts=overlap.extract_cpu_launch_contexts(raw,{k.external_id for k in kernels if k.external_id is not None})

def inside(ts,e):return float(e['ts'])<=ts<float(e['ts'])+float(e['dur'])
def interval(e):return (float(e['ts']),float(e['ts'])+float(e['dur']))
def named(prefix):return [e for e in annotations if e['name'].startswith('SMoE::'+prefix)]

gpu_expert=[]
expert_layer={}
runtime_by_correlation={e.get('args',{}).get('correlation'):e for e in raw
    if e.get('cat')=='cuda_runtime' and common.is_complete_duration_event(e)
    and e.get('args',{}).get('correlation') is not None}
for kernel in kernels:
    if kernel.name.startswith(('_group_gate','_group_down')):
        gpu_expert.append(kernel)
        owners=[e for e in annotations if e['name'].startswith('SMoE::moe_layer/') and inside(kernel.ts,e)]
        if len(owners)==1:expert_layer[kernel.idx]=int(owners[0]['name'].rsplit('L',1)[1])
        continue
    # Reuse skill CPU-op context mapping where available. Persistent worker
    # events additionally use CUPTI's exact launch/device correlation ID.
    launch=runtime_by_correlation.get(kernel.correlation)
    if launch is not None:
        spans=bythread.get((str(launch['pid']),str(launch['tid'])),[])
        if any(e['name'].startswith('SMoE::expert_forward_cuda') and inside(launch['ts'],e) for e in spans):
            gpu_expert.append(kernel)
            owners=[e for e in spans if e['name'].startswith('SMoE::gpu_hit_submit/') and inside(launch['ts'],e)]
            if owners:expert_layer[kernel.idx]=int(owners[0]['name'].rsplit('L',1)[1])
            continue
    for ctx in contexts.get(kernel.external_id,[]):
        spans=bythread.get((ctx.pid,ctx.tid),[])
        if any(e['name'].startswith('SMoE::expert_forward_cuda') and inside(ctx.ts,e) for e in spans):
            gpu_expert.append(kernel)
            owners=[e for e in spans if e['name'].startswith('SMoE::gpu_hit_submit/') and inside(ctx.ts,e)]
            if owners:expert_layer[kernel.idx]=int(owners[0]['name'].rsplit('L',1)[1])
            break

def group(k):
    n=k.name.lower()
    if 'htod' in n or 'host to device' in n:return 'h2d'
    if 'dtoh' in n or 'device to host' in n:return 'd2h'
    return 'gpu'

resources={
    'cpu_expert': [interval(e) for e in named('expert_forward_cpu')+named('cpu_expert_native')],
    'cpu_stage': [interval(e) for e in named('cpu_stage/')],
    'gpu_expert': [(k.ts,k.end) for k in gpu_expert],
    'gpu_all': [(k.ts,k.end) for k in kernels if group(k)=='gpu'],
    'h2d': [(k.ts,k.end) for k in kernels if group(k)=='h2d'],
    'd2h': [(k.ts,k.end) for k in kernels if group(k)=='d2h']}
resources['pcie']=resources['h2d']+resources['d2h']

def sweep(groups):
    events=[]
    for label,items in groups.items():
        for start,end in items:
            events.append(overlap.KernelEvent(idx=len(events),name=label,canonical_name=label,
                category='compute' if label!='pcie' else 'memory',pid=label,tid=label,stream=label,
                ts=start,dur=end-start,end=end))
    return overlap.analyze_overlap(events)

def clipped(items,start,end):
    return [(max(s,start),min(t,end)) for s,t in items if s<end and t>start]

steps=sorted(named('decode_step'),key=lambda e:e['ts'])
if len(steps)!=5:raise ValueError(f'Expected five tagged decode steps, got {len(steps)}')
start=steps[0]['ts'];end=steps[-1]['ts']+steps[-1]['dur']
resources={k:clipped(v,start,end) for k,v in resources.items()}
busy={k:sweep({k:v})['total_busy_us']/1000 for k,v in resources.items()}
pairs={}
for x,y in [('cpu_expert','gpu_expert'),('cpu_expert','gpu_all'),('cpu_expert','pcie'),('gpu_all','pcie')]:
    pairs[f'{x}_and_{y}_ms']=sweep({x:resources[x],y:resources[y]})['total_overlap_us']/1000
union=sweep({k:resources[k] for k in ('cpu_expert','gpu_all','pcie')})['total_busy_us']/1000
triple=union-sum(busy[k] for k in ('cpu_expert','gpu_all','pcie'))+sum(
    pairs[k] for k in ('cpu_expert_and_gpu_all_ms','cpu_expert_and_pcie_ms','gpu_all_and_pcie_ms'))
stats=dict(trace=str(a.directory/'decode.trace.json'),steps=5,window_ms=(end-start)/1000,
    clock_alignment=clock_evidence,
    cpu_forward_scope='native_aten_forward' if named('cpu_expert_native') else 'python_expert_forward',
    resource_union_ms=busy,overlap_ms=pairs,triple_cpu_gpu_pcie_ms=max(0.,triple),
    cpu_expert_ms_without_gpu_or_pcie=busy['cpu_expert']-pairs['cpu_expert_and_gpu_all_ms']-pairs['cpu_expert_and_pcie_ms']+triple,
    annotations=dict(Counter(e['name'].split('/')[0] for e in annotations)),
    gpu_expert_kernels_mapped=len(gpu_expert),gpu_events=len(kernels),gpu_pid=gpu_pid,
    interpretation='CPU expert is host call wall time, not CPU hardware busy cycles. Overlap is temporal coactivity, not automatically useful hiding.')
stats['loader_host_ms']={tag:sum(v['dur'] for v in named(tag))/1000 for tag in ('weight_copy_submit','eviction_select','expert_acquire')}
stats['pcie_events']=[dict(name=k.name,duration_us=k.dur) for k in kernels if group(k)!='gpu']
stats['pcie_bytes']={direction:sum(int(e.get('args',{}).get('bytes',0)) for e in raw
    if e.get('cat')=='gpu_memcpy' and direction in str(e.get('name',''))
    and start<=float(e.get('ts',-1))<end) for direction in ('HtoD','DtoH')}
(a.directory/'overlap_metrics.json').write_text(json.dumps(stats,indent=2))

# Pair last CPU expert and GPU expert completion inside every layer. Device
# copies of CPU outputs naturally follow CPU completion and are listed apart.
layer_finish=[]
cpu_events=named('expert_forward_cpu')+named('cpu_expert_native')
for layer in named('moe_layer/'):
    s,t=interval(layer);lid=int(layer['name'].rsplit('L',1)[1])
    c=[e for e in cpu_events if inside(e['ts'],layer)]
    g=[k for k in gpu_expert if expert_layer.get(k.idx)==lid and s<=k.ts<t]
    if c and g:
        ce=max(e['ts']+e['dur'] for e in c);ge=max(k.end for k in g)
        layer_finish.append(dict(layer=lid,layer_start_us=s,layer_ms=(t-s)/1000,
            cpu_experts=len(c),gpu_expert_kernels=len(g),cpu_end_minus_gpu_end_ms=(ce-ge)/1000))
(a.directory/'layer_completion.json').write_text(json.dumps(layer_finish,indent=2))

def render(window,title,file):
    left,right=window;duration=right-left
    lanes=[('cpu_expert','CPU expert forward','#278f74'),('gpu_expert','GPU expert kernels','#176ec3'),
           ('gpu_all','GPU all, incl. experts','#639ed3'),('h2d','PCIe H2D','#e69f26'),('d2h','PCIe D2H','#a15dbe')]
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1240" height="440" viewBox="0 0 1240 440">',
         '<style>text{font-family:DejaVu Sans,sans-serif;fill:#172d42;font-size:14px}</style>',
         '<rect width="1240" height="440" fill="white"/>',
         f'<text x="24" y="32" font-size="21">{html.escape(title)}</text>',
         f'<text x="24" y="58">Window: {duration/1000:.3f} ms. GPU bars are device execution, not host submission.</text>']
    def xpos(t):return 215+(t-left)/duration*990
    for j in range(11):
        x=215+j*99
        svg.append(f'<path d="M{x} 85 V350" stroke="#dfe5eb"/><text x="{x-13}" y="374">{j*duration/10000:.2f}</text>')
    for i,(key,label,color) in enumerate(lanes):
        y=98+i*49
        svg.append(f'<text x="14" y="{y+19}">{label}</text><rect x="215" y="{y}" width="990" height="28" fill="#f1f4f7"/>')
        for s,t in clipped(resources[key],left,right):
            svg.append(f'<rect x="{xpos(s):.3f}" y="{y}" width="{max(.25,(t-s)/duration*990):.3f}" height="28" fill="{color}"><title>{key}: {(s-left)/1000:.6f}..{(t-left)/1000:.6f} ms ({t-s:.3f} us)</title></rect>')
    svg+=['<text x="520" y="399">Time from window start (ms)</text>',
          '<text x="24" y="428">CPU bars include call/scheduling overhead. Tiny transfers have a 0.25 pixel minimum width; hover for actual duration.</text></svg>']
    (a.directory/file).write_text('\n'.join(svg))
third=steps[2];s,t=interval(third)
render((s,t),'Third captured decode token — CPU / GPU / PCIe','token_timeline.svg')
layer13=[e for e in named('moe_layer/L13') if s<=e['ts']<t]
if layer13:render(interval(layer13[0]),'Same token, MoE layer 13 — fixed zoom selection','layer13_timeline.svg')
print(json.dumps({k:v for k,v in stats.items() if k not in ('pcie_events','annotations')},indent=2))

# Also zoom the actual weight copy, rather than assuming the fixed third token
# contains one. This selects evidence without altering the inference workload.
weight_copies=[e for e in raw if e.get('cat')=='gpu_memcpy' and 'HtoD' in str(e.get('name',''))
               and int(e.get('args',{}).get('bytes',0))>=16*1024*1024 and start<=float(e.get('ts',-1))<end]
if weight_copies:
    first=min(weight_copies,key=lambda e:e['ts'])
    owners=[e for e in named('moe_layer/') if inside(first['ts'],e)]
    if owners:render(interval(owners[0]),'Actual expert-weight H2D — containing MoE layer','weight_transfer_timeline.svg')
    owner_steps=[e for e in steps if inside(first['ts'],e)]
    if owner_steps:render(interval(owner_steps[0]),'Actual expert-weight H2D — containing decode token','weight_transfer_token_timeline.svg')
