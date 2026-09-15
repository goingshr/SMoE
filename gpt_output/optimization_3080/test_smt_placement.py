"""Exercise a 14-core/28-thread host without requiring idle GPU hardware."""
from pathlib import Path
import os
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import utils.cpu_affinity as affinity

cpus = [affinity._LogicalCPU(i, 0, i % 14, 0.0) for i in range(28)]
affinity._logical_cpus = lambda interval: cpus
os.environ['SMOE_RESERVE_LOAD_CORE'] = '1'
for n in (3, 8, 14, 15, 16, 27, 28):
    placement = affinity.select_cpu_placement(n)
    assert len(placement.compute_cores) == n-1
    assert len(set(placement.compute_cores)) == n-1
    assert placement.shared_core not in placement.compute_cores
    if n <= 27:
        assert placement.shared_core % 14 not in {i % 14 for i in placement.compute_cores}
    assert set(placement.compute_cores) | {placement.shared_core} <= set(range(28))
    print(n, placement)
print('PASS: physical loading-core isolation when feasible; full-budget fallback')
