"""Conservation and aggregation tests for the CSV's direct observations."""
import importlib.util
import math
import os
from pathlib import Path
import random
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.expertcache import remove_outliers_and_average

random.seed(37)
for n in range(1,11):
    for _ in range(20):
        values=[random.random()*.002 for _ in range(n)]
        os.environ['SMOE_FAST_COST_AVERAGE']='0'; ref=remove_outliers_and_average(values)
        os.environ['SMOE_FAST_COST_AVERAGE']='1'; out=remove_outliers_and_average(values)
        assert math.isclose(ref,out,rel_tol=1e-12,abs_tol=1e-15),(values,ref,out)
print('PASS scalar cost estimator against NumPy population-std reference')

from utils import decode_metrics as dm
class Event:
    def elapsed_time(self,end):return end
saved=dm.torch.cuda.synchronize
dm.torch.cuda.synchronize=lambda:None
try:
    dm.reset()
    dm.counts.update(hits=2,misses=3,miss_to_gpu=1,miss_to_cpu=2)
    dm.cpu_ms.extend([1.,3.]);dm.pcie_ms.append(2.)
    dm.gpu_events.extend([(Event(),.4,2,'hit'),(Event(),.3,1,'miss')])
    r=dm.summary(2)
    assert r['avg_cpu_expert_ms']==2.
    assert r['avg_pcie_load_ms']==2.
    assert r['avg_gpu_miss_forward_ms']==.3
    assert r['cpu_expert_forward_ms_per_decode_token']==2.
    dm.counts['miss_to_cpu']+=1
    try:dm.summary(2)
    except AssertionError:pass
    else:raise AssertionError('Inconsistent destinations were accepted')
finally:
    dm.torch.cuda.synchronize=saved;dm.reset()
print('PASS direct-count conservation and service-time denominators')
