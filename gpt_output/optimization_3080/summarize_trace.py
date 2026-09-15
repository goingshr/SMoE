"""Framework-neutral Chrome trace summary for SMoE; no serving-framework impersonation."""
import argparse,collections,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('trace',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
es=json.loads(a.trace.read_text())['traceEvents']
def merge(intervals):
 out=[]
 for lo,hi in sorted(intervals):
  if out and lo<=out[-1][1]:out[-1][1]=max(out[-1][1],hi)
  else:out.append([lo,hi])
 return out
def overlap(xs,ys):
 i=j=0;out=0
 while i<len(xs) and j<len(ys):
  out+=max(0,min(xs[i][1],ys[j][1])-max(xs[i][0],ys[j][0]))
  if xs[i][1]<ys[j][1]:i+=1
  else:j+=1
 return out
cats={cat:[e for e in es if e.get('cat')==cat and e.get('ph')=='X'] for cat in ['kernel','gpu_memcpy','cuda_runtime']}
kernels=collections.defaultdict(lambda:[0,0.0]);copies=collections.defaultdict(lambda:[0,0.0,0])
for e in cats['kernel']:
 x=kernels[e['name']];x[0]+=1;x[1]+=e['dur']
for e in cats['gpu_memcpy']:
 x=copies[e['name']];x[0]+=1;x[1]+=e['dur'];x[2]+=e.get('args',{}).get('bytes',0)
ku=merge([(e['ts'],e['ts']+e['dur']) for e in cats['kernel']]);hu=merge([(e['ts'],e['ts']+e['dur']) for e in cats['gpu_memcpy'] if 'HtoD' in e['name']])
ks=sum(hi-lo for lo,hi in ku);hs=sum(hi-lo for lo,hi in hu);ov=overlap(ku,hu)
lines=['# SMoE trace triage','',f'Trace: `{a.trace}`','',
 'Framework: SMoE (custom Hugging Face model). Single-trace analysis; no mapping/formal pair.',
 'The shared skill CLI rejects this framework. These tables use generic Chrome events, without labeling SMoE as SGLang/vLLM.',
 '', '## Kernel table','', '| Kernel (truncated label) | Calls | Total ms | Kernel share |','|---|---:|---:|---:|']
total=sum(x[1] for x in kernels.values())
for name,(n,d) in sorted(kernels.items(),key=lambda x:-x[1][1]):
 if d/total>=0.01:lines.append(f'| `{name[:150]}` | {n} | {d/1000:.3f} | {100*d/total:.2f}% |')
lines+=['','## Transfer / overlap table','','| Transfer | Calls | Observed duration ms | Bytes |','|---|---:|---:|---:|']
for name,(n,d,b) in copies.items():lines.append(f'| {name} | {n} | {d/1000:.3f} | {b} |')
lines+=['',f'Kernel union: {ks/1000:.3f} ms; H2D union: {hs/1000:.3f} ms; intersection: {ov/1000:.3f} ms.',
 'CUDA copy activity is an observed interval, not proof of physical PCIe bandwidth. Cold-page faults and profiler overhead must be separated with a warmed unprofiled run.',
 '', '## Source-backed fusion table','','| Pattern | Local source | Interpretation |','|---|---|---|',
 '| BF16 gate/up projection | MoEModule/fused_gate_up.py | Existing one-token fused linear path. CPU MV is a separately gated candidate. |',
 '| Flash attention | model checkpoint DeepseekSdpaAttention.forward | SDPA already calls fused attention; enabling another attention implementation is not a new opportunity here. |',
 '| CPU activation/output transfers | MoEModule/SMoE_base.py::_cpu_compute | Existing per-layer pinned activation/output batching; inspect large expert weight H2D separately. |',
 '| Expert weight H2D | utils/expertcache.py::_swap | Pageable backing is sent through cudaMemcpyAsync. A bounded pinned staging buffer is a candidate only if warmed transfer tail is material. |',
 '', 'No fuzzy kernel-to-source matches are treated as exact attribution. Kernel and memcpy totals can overlap and must not be added as end-to-end latency.']
a.output.write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:8]));print('kernel_ms',ks/1000,'h2d_ms',hs/1000,'overlap_ms',ov/1000)
