"""Exact mask and SDPA equivalence, including left/right padding and cache rows."""
import torch
import torch.nn.functional as F
from utils.xverse_causal_mask import compact_mask

torch.manual_seed(37)
for device in ('cpu','cuda'):
    binary=torch.triu(torch.ones(257,257,device=device),diagonal=1)
    for length in (1,30,129,257):
        for batch in (1,2):
            for padding in ('none','left','right','all'):
                for dtype in (torch.bfloat16,torch.float32):
                    attention=torch.ones(batch,length,device=device)
                    if padding=='left':attention[:,:min(3,length)]=0
                    elif padding=='right':attention[:,-min(3,length):]=0
                    elif padding=='all':attention[:]=0
                    x=torch.zeros(batch,1,80,device=device,dtype=dtype)
                    minimum=torch.finfo(dtype).min
                    full=binary[None,None].repeat(batch,1,1,1).to(dtype)*minimum
                    mask=full[...,:length].eq(0)*attention[:,None,None,:].eq(0)
                    full[...,:length]=full[...,:length].masked_fill(mask,minimum)
                    if torch.any(attention!=1):
                        full=full.mul(~torch.all(full==minimum,dim=-1,keepdim=True)).to(dtype)
                    actual=compact_mask(binary,attention,x,True)
                    ref=full[:,:,:length,:length]
                    assert torch.equal(actual,ref),(device,length,batch,padding,dtype)
                    for rows in ([0],[length-1],list(range(length))):
                        q=torch.randn(batch,2,len(rows),80,device=device,dtype=dtype)
                        k=torch.randn(batch,2,length,80,device=device,dtype=dtype)
                        v=torch.randn_like(k)
                        a=F.scaled_dot_product_attention(q,k,v,attn_mask=actual[:,:,rows,:])
                        b=F.scaled_dot_product_attention(q,k,v,attn_mask=full[:,:,rows,:length])
                        assert torch.equal(a,b)
print('PASS compact mask and SDPA exact: CPU/CUDA, BF16/FP32, lengths 1/30/129/257, batch 1/2, padding and absolute cache rows')
