import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from utils.decode_rope import rope
torch.manual_seed(21)
with torch.no_grad():
    for nq,nk,d in [(16,16,128),(8,2,64),(3,1,10)]:
        for pos in (0,137):
            q=torch.randn(1,nq,1,d,dtype=torch.bfloat16,device='cuda')
            k=torch.randn(1,nk,1,d,dtype=torch.bfloat16,device='cuda')
            angles=torch.randn(200,d,device='cuda')
            c,s=angles.cos().bfloat16(),angles.sin().bfloat16()
            p=torch.tensor([[pos]],device='cuda')
            def reference(x):
                half=torch.cat((-x[...,d//2:],x[...,:d//2]),-1)
                return x*c[p].unsqueeze(1)+half*s[p].unsqueeze(1)
            oq,ok=rope(q,k,c,s,p)
            assert torch.equal(oq,reference(q))
            assert torch.equal(ok,reference(k))
            print(json.dumps(dict(heads=[nq,nk],head_dim=d,position=pos,exact=True)),flush=True)
