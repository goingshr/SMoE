"""Clearing speculative queue entries must not hide a load already running."""
from collections import deque
from pathlib import Path
import sys
import threading
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.expertcache import ExpertCache

cache = ExpertCache.__new__(ExpertCache)
cache.cv = threading.Condition()
cache.load_queue = deque([(0, 0)])
cache._queue_generation = 0
cache.pending_callbacks = 0
cache.active_loads = 1
cache.clear_queue()
done = threading.Event()
def wait():
    cache.wait_until_queue_empty()
    done.set()
thread = threading.Thread(target=wait)
thread.start()
assert not done.wait(0.05), 'clear_queue hid an active load'
with cache.cv:
    cache.active_loads = 0
    cache.cv.notify_all()
assert done.wait(1), 'drain did not wake after active load finished'
thread.join()
print('queue drain: active load remains visible after speculative queue clear')
