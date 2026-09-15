"""Diagnostic GPU expert A/B, retaining all BF16 intermediate casts."""
import json
from pathlib import Path
import statistics
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import torch.nn.functional as F
from utils.triton_bf16_expert import bf16_expert
torch.set_num_threads(1)
torch.manual_seed(19)
with torch.no_grad():
    for h, i in [(2048,1408), (97,65)]:
        x = torch.randn((1,h), dtype=torch.bfloat16, device='cuda')
        w = torch.randn((2*i,h), dtype=x.dtype, device=x.device)*.02
        d = torch.randn((h,i), dtype=x.dtype, device=x.device)*.02
        def reference():
            g,u = F.linear(x,w).split(i,-1)
            return F.linear(F.silu(g)*u,d)
        ref = reference()
        for rows in (1,2,4,8):
            out = bf16_expert(x,w,d,rows)
            err = (out.float()-ref.float())
            rel_rms = (err.square().mean()/ref.float().square().mean()).sqrt().item()
            assert rel_rms < .01
            print(json.dumps(dict(test='correctness', h=h,i=i,rows=rows,
                max_abs=err.abs().max().item(),relative_rms=rel_rms)), flush=True)
        if h != 2048: continue
        for name,fn in [('torch',reference)]+[(f'triton_r{r}',lambda r=r:bf16_expert(x,w,d,r)) for r in (1,2,4,8)]:
            for _ in range(10): fn()
            torch.cuda.synchronize()
            times=[]
            for _ in range(50):
                t=time.perf_counter()
                fn()
                torch.cuda.synchronize()
                times.append((time.perf_counter()-t)*1e3)
            print(json.dumps(dict(test='benchmark', mode=name,
                mean_ms=statistics.fmean(times), median_ms=statistics.median(times))),flush=True)
