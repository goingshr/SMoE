"""A pending copy must not be sampled, and prefill cannot pollute decode cost."""
from collections import deque
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from utils.expertcache import ExpertCache
class Event:
    def __init__(self,ms=0,ready=True):self.ms,self.ready=ms,ready
    def query(self):return self.ready
    def elapsed_time(self,end):return end.ms
cache=ExpertCache.__new__(ExpertCache)
cache.LoadTimeOneExpert=[.002]
cache.DecodeLoadTimeOneExpert=[]
cache.measured_dma_copies=0
cache.decode_load_sample_age=700
pending=Event(2,False)
cache._dma_timings=deque([(Event(),Event(9),False),(Event(),Event(3),True),(Event(),pending,True)])
cache.consume_dma_timings()
assert cache.LoadTimeOneExpert==[.002,.009,.003]
assert cache.DecodeLoadTimeOneExpert==[.003]
assert cache.decode_load_sample_age==0
assert cache.measured_dma_copies==2 and len(cache._dma_timings)==1
pending.ready=True
cache.consume_dma_timings()
assert cache.DecodeLoadTimeOneExpert==[.003,.002]
for _ in range(20):cache._dma_timings.append((Event(),Event(50),False))
cache.consume_dma_timings()
assert cache.DecodeLoadTimeOneExpert==[.003,.002]
assert cache.LoadTimeOneExpert==[.05]*10
print('PASS: complete-event sampling, pending prefix, separate decode history, bounded general history')
