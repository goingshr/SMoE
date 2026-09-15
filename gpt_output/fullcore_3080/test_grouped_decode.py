"""Dynamic weight addresses, counts, score slots, replay and storage reuse."""
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
import torch.nn.functional as F
from utils.grouped_decode import GroupedDecode
torch.set_num_threads(1)
torch.manual_seed(32)
h,i,k=2048,1408,6
packs=[torch.randn((3*h*i,),dtype=torch.bfloat16,device='cuda')*.02 for _ in range(8)]
def expert(x,p):
    g,u=F.linear(x,p[:2*h*i].view(2*i,h)).split(i,-1)
    return F.linear(F.silu(g)*u,p[2*h*i:].view(h,i))
with torch.no_grad():
    grouped=GroupedDecode(h,i,k,'cuda:0')
    errors=[]
    for step in range(35):
        count=step%7
        ids=[(step+j)%8 for j in range(count)]
        slots=list(reversed(range(count)))
        x=torch.randn((1,h),dtype=torch.bfloat16,device='cuda')
        scores=torch.softmax(torch.randn(k,device='cuda'),0).bfloat16()
        if step%5==0:
            # Same slot address, new weight contents: no stale-weight capture.
            packs[step%8].mul_(.9)
        output=grouped(x,scores,[packs[j].data_ptr() for j in ids],slots)
        for n,j in enumerate(ids):
            ref=expert(x,packs[j])*scores[slots[n]]
            err=output[n:n+1].float()-ref.float()
            rel=(err.square().mean()/ref.float().square().mean()).sqrt().item()
            assert rel < .01, (step,n,rel)
            errors.append(rel)
    print(json.dumps(dict(test='grouped_replay',steps=35,counts=list(range(7)),
        max_relative_rms=max(errors),allocated=torch.cuda.max_memory_allocated(),
        reserved=torch.cuda.max_memory_reserved())),flush=True)
    for count in (1,3,6):
        ids=list(range(count));slots=ids
        for mode in ('torch','grouped'):
            def run():
                if mode=='torch': return [expert(x,packs[j])*scores[j] for j in ids]
                return grouped(x,scores,[packs[j].data_ptr() for j in ids],slots)
            for _ in range(10): run()
            torch.cuda.synchronize()
            start=time.perf_counter()
            for _ in range(100): run()
            torch.cuda.synchronize()
            print(json.dumps(dict(test='benchmark',count=count,mode=mode,
                mean_ms=(time.perf_counter()-start)*10)),flush=True)
