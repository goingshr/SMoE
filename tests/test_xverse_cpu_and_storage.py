"""Real-shape CPU equivalence and compact pinned pack ownership."""
import copy
import os
import torch
from configs.configuration_xverse import XverseConfig
from MoEModule.xverse_moe import XverseMLP
from utils.model_loader import ExpertWrapper

torch.set_num_threads(15)
torch.manual_seed(37)
config=XverseConfig(hidden_size=2560,intermediate_size=1728,device='cpu')
with torch.no_grad():
    m=XverseMLP(config)
    weight=torch.cat((m.gate_proj.weight,m.up_proj.weight))
    m.configure_cpu_bf16_gate_up(weight)
    for scale in (.01,1.,10.):
        x=torch.randn(1,2560,dtype=torch.bfloat16)*scale
        m._cpu_bf16_mv=False;ref=m(x)
        m._cpu_bf16_mv=True;out=m(x)
        err=(out.float()-ref.float())
        relative=(err.square().mean()/ref.float().square().mean().clamp_min(1e-20)).sqrt().item()
        assert relative < .01
        print('CPU linear/mv',scale,'max_abs',err.abs().max().item(),'relative_rms',relative)
    wrappers=[]
    for n in range(7):
        wrapped=ExpertWrapper(copy.deepcopy(m),'xversemoe',torch.device('cpu'),tocpu=True)
        wrappers.append(wrapped)
        pack=torch.as_tensor(wrapped.storage,dtype=torch.uint8)
        assert pack.is_pinned()
        if n:assert torch.equal(pack,torch.as_tensor(wrappers[0].storage,dtype=torch.uint8))
    spans=sorted((w.storage.data_ptr(),w.storage.data_ptr()+w.storage.nbytes()) for w in wrappers)
    assert all(a[1]<=b[0] for a,b in zip(spans,spans[1:]))
    for wrapped in wrappers:torch.testing.assert_close(wrapped(x),out,atol=.004,rtol=.02)
print('PASS compact pinned packs: exact contents, non-overlap, retained storage lifetime')
