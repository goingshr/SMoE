"""Soft per-layer residency must preserve priority and prefill progress."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from utils.expertcache import EvictionInfo, ExpertInfo, FixedSizeQueueForScore

cache = EvictionInfo(strict_score_order=True, layer_cache_floor=1)
for i, uid in enumerate([(1, 0), (0, 0), (0, 1), (0, 2)]):
    cache.add(ExpertInfo(uid, False, 0, False, FixedSizeQueueForScore(1), i, i))
scores = np.array([[0.2, 0.1, 0.3], [0.001, 0.0, 0.0]])
counts = np.ones(2, dtype=np.int32)
assert cache.choose_expert_to_evictbyLRU().uid == (0, 0)
assert cache.choose_expert_to_evictbyScore(scores, counts).uid == (0, 1)
cache.main_infos[(0, 1)].priority = 2
assert cache.choose_expert_to_evictbyScore(scores, counts).uid == (0, 0)
cache.layer_cache_floor = 3  # Every layer at/below floor: retain progress.
assert cache.choose_expert_to_evictbyLRU().uid == (1, 0)
assert cache.choose_expert_to_evictbyScore(scores, counts).uid == (1, 0)
cache.layer_cache_floor = 1
for uid in [(0, 0), (0, 2)]:
    cache.main_infos[uid].priority = 1
assert cache.choose_expert_to_evictbyLRU().uid == (1, 0)
assert cache.choose_expert_to_evictbyScore(scores, counts).uid == (1, 0)
print('soft layer floor: residency preference, priority, ties and capacity fallback passed')
