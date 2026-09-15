"""Verify native batch uses the same BF16 ATen math, and measure boundary cost."""
import json, os, sys, time, statistics
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from utils.cpu_bf16_kernel import load_kernel
os.sched_setaffinity(0,set(range(13)))
torch.set_num_threads(13)
load_kernel()
torch.manual_seed(42)
with torch.no_grad():
    for h,i in [(65,97),(2048,1408)]:
        packs=[torch.randn(3*h*i,dtype=torch.bfloat16)*.02 for _ in range(32)]
        x=torch.randn(h,dtype=torch.bfloat16)
        def reference(ps):
            out=[]
            for w in ps:
                g,u=torch.mv(w[:2*h*i].view(2*i,h),x).split(i)
                out.append(torch.mv(w[2*h*i:].view(h,i),torch.nn.functional.silu(g)*u))
            return torch.stack(out) if out else torch.empty(0,h,dtype=x.dtype)
        for n in [0,1,3,6]:
            out,ts=torch.ops.smoe_cpu.bf16_experts(packs[:n],x,i)
            assert torch.equal(out,reference(packs[:n]))
            assert ts.shape==(n,) and (ts>=0).all()
            print(json.dumps(dict(test='exact_reference',h=h,i=i,experts=n,passed=True)),flush=True)
        if h==2048:
            for rep in range(3):
                for name,fn in [('python',reference),('native',lambda ps:torch.ops.smoe_cpu.bf16_experts(ps,x,i)[0])]:
                    ts=[]
                    for k in range(36):
                        ps=[packs[(3*k+j)%32] for j in range(3)]
                        t=time.perf_counter();fn(ps);ts.append((time.perf_counter()-t)*1000/3)
                    print(json.dumps(dict(test='boundary_benchmark',mode=name,repeat=rep,mean_ms=statistics.fmean(ts))),flush=True)
