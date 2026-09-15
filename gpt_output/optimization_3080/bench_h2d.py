"""Short, isolated pageable/pinned/staged expert-sized H2D measurement."""
import argparse,json,time,statistics,os
p=argparse.ArgumentParser();p.add_argument('--iterations',type=int,default=20);p.add_argument('--cpu',type=int,default=0);a=p.parse_args()
os.sched_setaffinity(0,{a.cpu})
import torch
torch.set_num_threads(1)
n=3*2048*1408*2
pageable=torch.empty(n,dtype=torch.uint8).fill_(71)
pinned=torch.empty(n,dtype=torch.uint8,pin_memory=True).fill_(71)
dst=torch.empty(n,dtype=torch.uint8,device='cuda:0')
stream=torch.cuda.Stream()
for mode in ['pageable','pinned','staged']:
 times=[];gpu=[]
 for i in range(a.iterations+5):
  stream.synchronize();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
  t=time.perf_counter()
  if mode=='staged':pinned.copy_(pageable)
  with torch.cuda.stream(stream):
   start.record();dst.copy_(pageable if mode=='pageable' else pinned,non_blocking=True);end.record()
  end.synchronize();elapsed=time.perf_counter()-t
  if i>=5:times.append(elapsed);gpu.append(start.elapsed_time(end))
 assert torch.equal(dst.cpu(),pageable)
 m=statistics.median(times)
 print(json.dumps(dict(mode=mode,bytes=n,direction='H2D',device=0,cpu=a.cpu,iterations=a.iterations,median_wall_ms=1000*m,median_event_ms=statistics.median(gpu),wall_payload_GBps=n/m/1e9)),flush=True)
