"""Local BF16 expert diagnostic; conversions widen BF16 exactly, never quantize."""
import os
import json
import time
import statistics
import argparse
p=argparse.ArgumentParser()
p.add_argument('--threads',type=int,default=2)
p.add_argument('--cpus',default='10,11')
a=p.parse_args()
os.sched_setaffinity(0,{int(x) for x in a.cpus.split(',')})
import torch
import torch.nn.functional as F
torch.set_num_threads(a.threads)
torch.set_num_interop_threads(1)
torch.manual_seed(19)
x=torch.randn(1,2048,dtype=torch.bfloat16)
w=torch.randn(2816,2048,dtype=torch.bfloat16)*0.02
d=torch.randn(2048,1408,dtype=torch.bfloat16)*0.02

def linear(x,w,mode):
 if mode=='widen': return F.linear(x.float(),w.float()).bfloat16()
 if mode=='mv': return torch.mv(w,x[0]).unsqueeze(0)
 return F.linear(x,w)

def forward(mode):
 g,u=linear(x,w,mode).split(1408,-1)
 return linear(F.silu(g)*u,d,mode)
with torch.no_grad():
 ref=forward('native')
 for mode in ['native','widen','mv']:
  for _ in range(3): forward(mode)
  ts=[]
  for _ in range(20):
   t=time.perf_counter(); y=forward(mode); ts.append((time.perf_counter()-t)*1000)
  print(json.dumps(dict(mode=mode,threads=a.threads,median_ms=statistics.median(ts),max_abs=(y.float()-ref.float()).abs().max().item(),rms=(y.float()-ref.float()).square().mean().sqrt().item(),equal=torch.equal(y,ref))),flush=True)
