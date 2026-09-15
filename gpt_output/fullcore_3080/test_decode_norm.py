import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from utils.decode_norm import rms_norm
torch.manual_seed(17)
torch.set_num_threads(1)
with torch.no_grad():
    errors=[]
    for h in (97,2048):
        for scale in (0.,1e-4,1.,100.):
            x=torch.randn((1,1,h),dtype=torch.bfloat16,device='cuda')*scale
            w=torch.randn(h,dtype=torch.bfloat16,device='cuda')
            def ref():
                f=x.float()
                return w*(f*torch.rsqrt(f.square().mean(-1,keepdim=True)+1e-6)).bfloat16()
            expected=ref()
            out=rms_norm(x,w,1e-6)
            delta=(out.float()-expected.float()).abs()
            torch.testing.assert_close(out,expected,atol=.03125,rtol=.008)
            errors.append(dict(h=h,scale=scale,max_abs=delta.max().item()))
    print(json.dumps(dict(test='rmsnorm',errors=errors)),flush=True)
    for mode,fn in [('torch',ref),('triton',lambda:rms_norm(x,w,1e-6))]:
        for _ in range(10): fn()
        torch.cuda.synchronize()
        t=time.perf_counter()
        for _ in range(200):fn()
        torch.cuda.synchronize()
        print(json.dumps(dict(mode=mode,mean_us=(time.perf_counter()-t)*1e6/200)),flush=True)
