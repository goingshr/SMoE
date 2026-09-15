"""Verify score-primary eviction, priority protection and exact LRU ties."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from utils.expertcache import EvictionInfo, ExpertInfo, FixedSizeQueueForScore

cache = EvictionInfo(strict_score_order=True)
for i in range(4):
    item = ExpertInfo(uid=(0, i), offloaded=False, priority=0, loading=False,
                      scores=FixedSizeQueueForScore(1), index=i, offload_index=i)
    cache.add(item)
sums = np.array([[0.1, 0.02, 0.02, 0.001]])
counts = np.array([1])
cache.main_infos[(0, 3)].priority = 2
assert cache.choose_expert_to_evictbyScore(sums, counts).uid == (0, 1)
cache.mark_used(cache.main_infos[(0, 1)])
assert cache.choose_expert_to_evictbyScore(sums, counts).uid == (0, 2)
cache.main_infos[(0, 2)].priority = 1
assert cache.choose_expert_to_evictbyScore(sums, counts).uid == (0, 1)
cache.strict_score_order = False
assert cache.choose_expert_to_evictbyScore(sums, counts).uid == (0, 0)
print('score eviction: priority protection, score order, LRU ties and legacy fallback passed')
