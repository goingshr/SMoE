"""Check work conservation and predicted finish time across small expert sets."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.expertcache import CPU_load_management, CPU_load_management_decode

for n in range(7):
    jobs = {(1, i): 1 for i in range(n)}
    for cpu_ms in (0.3, 1.0, 3.0, 12.0):
        for dma_ms in (0.5, 1.7, 3.0, 8.0):
            gpu, cpu = CPU_load_management_decode(jobs, cpu_ms, dma_ms)
            assert len(gpu) + len(cpu) == n
            assert set(gpu).isdisjoint(cpu)
            assert set(gpu) | set(cpu) == set(jobs)
            # Every legal split for equal-cost decode work reduces to a count.
            expected = min(max(c * cpu_ms, (n-c) * dma_ms) for c in range(n+1))
            assert max(len(cpu)*cpu_ms, len(gpu)*dma_ms) == expected
assert CPU_load_management_decode({(1, 0): 1}, 3, 1.7) == ([(1, 0)], [])
assert CPU_load_management_decode({(1, 0): 1}, 0.5, 1.7) == ([], [(1, 0)])
prefill = {(1, 0): 5, (1, 1): 2}
assert CPU_load_management_decode(prefill, 3, 1.7) == CPU_load_management(prefill, 3, 1.7)
print('PASS: equal-cost decode partition minimizes predicted finish time; all experts retained; prefill fallback')
