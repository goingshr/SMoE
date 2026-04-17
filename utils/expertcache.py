from dataclasses import dataclass,field
from typing import Dict, Optional, Tuple
from collections import deque, OrderedDict
import threading
import torch
from torch import nn
import copy
import time
from tqdm import tqdm
import psutil
import torch.profiler
import numpy as np
import scipy.stats as stats
import ctypes

# ---------------------------------------------------------------------------
# Truly async PCIe HtoD via cudaMemcpyAsync
# ---------------------------------------------------------------------------
try:
    _libcudart = ctypes.CDLL('libcudart.so')
    _cudaMemcpyAsync = _libcudart.cudaMemcpyAsync
    _cudaMemcpyAsync.restype  = ctypes.c_int
    _cudaMemcpyAsync.argtypes = [
        ctypes.c_void_p,   # dst (GPU)
        ctypes.c_void_p,   # src (CPU pinned)
        ctypes.c_size_t,   # count (bytes)
        ctypes.c_int,      # kind (1 = HtoD)
        ctypes.c_void_p,   # stream handle
    ]
    _CUDART_AVAILABLE = True
except OSError:
    _CUDART_AVAILABLE = False

_cudaMemcpyHostToDevice = 1
ExpertUID = Tuple[int,int]
import logging
import os


logger = logging.getLogger(__name__)

# CPU affinity: set by main.py before model init
_shared_core   = None   # core for loading thread + bg_workers
_compute_cores = None   # cores for intra-op matmul

tokens = 0        # incremented by patcher: 0 = prefill, >0 = decode
decode_time  = 0.0  # cumulative decode wall time (seconds)
prefill_time = 0.0  # prefill wall time (seconds)

# per-token GPU cache hit accumulators (reset after each token's log)
cache_hits_per_token  = 0
cache_total_per_token = 0

# per-layer prefetch loaded set: layer_id -> set of expert_ids queued for prefetch
prefetch_loaded_by_layer: dict = {}   # {layer_id: set of expert_ids}
prefetch_start_time: dict = {}        # {layer_id: timestamp when prefetch add_to_queue ran}

class FixedSizeQueueForScore():
    """Sliding-window score queue backed by a fixed-size float array.

    Replaces deque to eliminate dynamic memory allocation on every add().
    Uses a circular buffer (ptr wraps mod k) and tracks count for the
    cold-start phase (fewer than k entries seen).
    """
    __slots__ = ('k', '_buf', '_ptr', '_cnt', '_sum')

    def __init__(self, k):
        self.k    = k
        self._buf = [0.0] * k if k else []  # circular buffer (empty if k=None)
        self._ptr = 0            # next write position
        self._cnt = 0            # entries filled (saturates at k)
        self._sum = 0.0

    def add(self, value):
        if self._cnt == self.k:
            self._sum -= self._buf[self._ptr]
        else:
            self._cnt += 1
        self._buf[self._ptr] = value
        self._sum += value
        self._ptr = (self._ptr + 1) % self.k

    def get_average(self):
        if self._cnt == 0:
            return 0.0
        return self._sum / self._cnt



@dataclass(frozen=False)
class ExpertInfo:
    uid: ExpertUID
    offloaded: bool
    priority: int
    loading: bool
    scores: FixedSizeQueueForScore
    index: int
    offload_index:int

    access_count: int = 0

    load_sequence: int = 0


@dataclass
class EvictionInfo():
    # infos in main and offload devices; ordered from least recently used to most
    main_infos: OrderedDict[ExpertUID, ExpertInfo] = field(default_factory=OrderedDict)
    offloaded_infos: OrderedDict[ExpertUID, ExpertInfo] = field(default_factory=OrderedDict)
    hits: int = field(default=0)
    misses: int = field(default=0)
    _global_load_counter: int = field(default=0)

    def add(self, info: ExpertInfo):
        infos_odict = self.offloaded_infos if info.offloaded else self.main_infos
        assert info.uid not in infos_odict, f"expert {info.uid} already exists"
        infos_odict[info.uid] = info
        if not info.offloaded:
            info.load_sequence = self._global_load_counter
            self._global_load_counter += 1
            
        infos_odict[info.uid] = info
        

    def choose_expert_to_evictbyScore(self, score_sum=None, score_cnt=None) -> ExpertInfo:
        """Find the best expert to evict using score-based policy.

        If score_sum / score_cnt (numpy arrays from ExpertCache) are provided,
        reads scores directly from those arrays instead of per-expert Python objects.
        Tiebreaker: LRU position in main_infos OrderedDict (earlier = LRU).
        """
        infos = list(self.main_infos.items())   # [(uid, ExpertInfo), ...]
        if not infos:
            raise ValueError("No evictable experts")

        n = len(infos)
        priorities = np.empty(n, dtype=np.int32)
        averages   = np.empty(n, dtype=np.float64)

        buf_ready = (score_sum is not None) and (score_cnt is not None)

        for i, (uid, info) in enumerate(infos):
            priorities[i] = info.priority
            if buf_ready:
                layer, expert = uid
                cnt = int(score_cnt[layer])
                averages[i] = (float(score_sum[layer, expert]) / cnt) if cnt else 0.0
            else:
                s = info.scores
                averages[i] = (s._sum / s._cnt) if s._cnt else 0.0

        min_prio = int(priorities.min())
        assert min_prio < 2, "Cache size is too small to support normal system operation."

        mask     = (priorities == min_prio)
        lru_pos  = np.arange(n, dtype=np.float64)
        combined = averages * (n + 1) + lru_pos
        combined[~mask] = np.inf

        best_i     = int(np.argmin(combined))
        evict_info = infos[best_i][1]
        return evict_info

    def choose_expert_to_evictbyLRU(self):
        min_priority = min(info.priority for info in self.main_infos.values())
        assert min_priority <2, "Cache size is too small to support normal system operation."
        for uid, info in self.main_infos.items():
            if info.priority == min_priority:
                return info  # least recently used
        raise ValueError("No evictable experts")

    def choose_expert_to_evictbyFCFS(self) -> ExpertInfo:
        min_priority = min(info.priority for info in self.main_infos.values())
        assert min_priority < 2, "Cache size is too small to support normal system operation."
        
        evict_info = None
        min_load_seq = float('inf')
        
        for uid, info in self.main_infos.items():

            if info.priority == min_priority and info.load_sequence < min_load_seq:
                min_load_seq = info.load_sequence
                evict_info = info
                
        if evict_info is None:
            raise ValueError("No evictable experts")
            
        return evict_info


    def choose_expert_to_evictbyLFU(self) -> ExpertInfo:
        min_priority = min(info.priority for info in self.main_infos.values())
        assert min_priority < 2, "Cache size is too small to support normal system operation."
        
        evict_info = None
        min_access_count = float('inf')
        
        for uid, info in self.main_infos.items():

            if info.priority == min_priority and info.access_count < min_access_count:
                min_access_count = info.access_count
                evict_info = info
                
        if evict_info is None:
            raise ValueError("No evictable experts")
            
        return evict_info
    def swap(self, info_to_load: ExpertInfo, info_to_evict: ExpertInfo):
        assert info_to_load.uid in self.offloaded_infos and info_to_evict.uid in self.main_infos
        self.main_infos[info_to_load.uid] = self.offloaded_infos.pop(info_to_load.uid)
        self.main_infos.move_to_end(info_to_load.uid, last=True)
        self.offloaded_infos[info_to_evict.uid] = self.main_infos.pop(info_to_evict.uid)

    def mark_pro(self,info: ExpertInfo,priority:int):
        assert info.uid in self.main_infos
        self.main_infos[info.uid].priority = priority
    def mark_used(self, info: ExpertInfo):
        if info.uid in self.main_infos:
            self.main_infos.move_to_end(info.uid, last=True)
            self.hits += 1
        elif info.uid in self.offloaded_infos:
            self.offloaded_infos.move_to_end(info.uid, last=True)
            self.misses += 1
        else:
            raise ValueError(f"Expert {info} not in group")


class ExpertCache:
    def __init__(self, config,make_module_cuda: callable, make_module_cpu: callable, main_size: int, offload_size: int,window_size:int ,state_dict_00,model_type,model_path):
        """Dynamically loads an array of modules with identical hyperparameters"""
        self.module_type = self.module_size = self.device = None
        self.active = False

        self.registered_experts: Dict[ExpertUID, ExpertInfo] = dict()
        self.main_modules = []
        # import psutil
        for _ in tqdm(range(main_size),desc = "init cache space for experts in GPU memory"):
            self.main_modules.append(self._check_module(make_module_cuda(model_path,model_type,config.device,state_dict_00)))
        # logger.debug(f"Current CPU memory usage: {psutil.Process().memory_info().rss / (1024 ** 2):.2f} MB")
        self.main_infos = [0 for _ in range(main_size)]

        assert self.module_size is not None
        self.offloaded_storages = []
        for _ in tqdm(range(offload_size),desc = "init offloading space in CPU memory"):
            self.offloaded_storages.append(make_module_cpu(model_path,model_type,config.device,state_dict_00))
        # logger.debug(f"Current CPU memory usage: {psutil.Process().memory_info().rss / (1024 ** 2):.2f} MB")
        self.offloaded_infos = [0 for _ in range(offload_size)]
        self.cache_window = window_size
        self.cache_infos = EvictionInfo()

        # ── Vectorised score buffer (only used when cache_window is not None) ──
        # Layout: _score_buf[layer, expert, slot]  float32
        # _score_ptr[layer]  – next write slot (circular)
        # _score_cnt[layer]  – filled slots so far (saturates at window_size)
        # _score_sum[layer, expert] – rolling sum for O(1) average
        # num_layers / num_experts are inferred lazily at first update_scores call.
        self._score_buf  = None   # shape (L, E, W)  np.float32
        self._score_sum  = None   # shape (L, E)     np.float64
        self._score_ptr  = None   # shape (L,)       int32
        self._score_cnt  = None   # shape (L,)       int32
        self._score_L    = 0
        self._score_E    = 0
        
        self.load_queue = deque()
        self.capacity_1 = main_size//2
        self.priority_one_queue = deque()   # ordered: popleft evicts oldest
        self.priority_one_set   = set()     # O(1) membership / removal mirror
        self.mtx = threading.Lock()
        self.cv = threading.Condition(self.mtx)
        # Generation counter: incremented by clear_queue().
        # loading() thread snapshots this before phase2; if it changed by phase3,
        # the swap is discarded (eviction is rolled back).  This ensures clear_queue()
        # can truly abort prefetch loads even after cudaMemcpyAsync was submitted.
        self._queue_generation = 0
        self.loading_thread = threading.Thread(target=self.loading,daemon=True)
        self.loading_thread.start()
        self.load_stream = torch.cuda.Stream(device=config.device)
        self.compute_stream = torch.cuda.Stream(device=config.device)
        self.predict_stream = torch.cuda.Stream(device=config.device)
        self.on_expert_loaded = None
        self.pending_callbacks = 0
        # Initial estimate: ~2ms per expert actual PCIe DMA time (52.5MB @ ~26GB/s effective).
        # This is updated in-flight from real B8-elapsed / n_pcie_experts measurements.
        # DO NOT use _swap() wall time here — that only measures DMA submission (~0.2ms),
        # not DMA completion. The balancer needs completion time to compare with cpucost.
        self.LoadTimeOneExpert = [0.002]
        # Per-expert CUDA timing events set by loading thread, consumed by B8.
        # Key: ExpertUID  →  (start_event, done_event)
        self._pcie_timing: dict = {}
    def _check_module(self, module: nn.Module):
        assert isinstance(module.storage, torch.UntypedStorage)
        if self.module_type is None:
            self.module_type = type(module)
            self.module_size = len(module.storage)
            self.device = module.storage.device
        else:
            assert isinstance(module, self.module_type)
            assert len(module.storage) == self.module_size
            assert module.storage.device == self.device
        return module
    def query_expert(self,uid: ExpertUID):
        with self.mtx:
            if uid in self.cache_infos.main_infos and not self.cache_infos.main_infos[uid].offloaded:
                return True
            return False
    def query_expert_inload(self,uid: ExpertUID):
        if uid in self.cache_infos.main_infos and not self.cache_infos.main_infos[uid].offloaded:
            return True
        return False
    def add_expert(self, uid: ExpertUID, module: nn.Module, offload: Optional[bool] = None):
        # with self.mtx:
        assert self.module_type is not None
        assert isinstance(module, self.module_type)
        return self.add_expert_storage(uid, module.storage, offload=offload)

    def add_expert_storage(self, uid: ExpertUID, storage: torch.UntypedStorage, offload: Optional[bool] = None):
        assert uid not in self.registered_experts, f"expert {uid} already registered"
        assert isinstance(storage, torch.UntypedStorage)
        assert len(storage) == self.module_size

        if offload is None or not offload:  # False or None
            for i in range(len(self.main_modules)):
                if self.main_infos[i] == 0:
                    # logger.debug("----------")
                    # logger.debug(f"Current CPU memory usage: {psutil.Process().memory_info().rss / (1024 ** 2):.2f} MB")
                    self.main_modules[i].storage.copy_(storage)
                    # logger.debug(f"Current CPU memory usage: {psutil.Process().memory_info().rss / (1024 ** 2):.2f} MB")
                    info = ExpertInfo(uid, False,0,False, scores = FixedSizeQueueForScore(self.cache_window),index=i,offload_index=0)
                    self.registered_experts[uid] = info
                    self.cache_infos.add(info)
                    self.main_infos[i] =1
                    break
        for i in range(len(self.offloaded_storages)):
            if self.offloaded_infos[i] == 0:
                self.offloaded_storages[i].storage.copy_(storage)
                if offload:
                    info = ExpertInfo(uid, True, 0,False,scores = FixedSizeQueueForScore(self.cache_window),index=i,offload_index=i)
                    self.registered_experts[uid] = info
                    self.cache_infos.add(info)
                else:
                    self.registered_experts[uid].offload_index=i
                self.offloaded_infos[i] = 1
                return  # done allocating; found an offloaded spot
        raise ValueError("Cache is full")


    def _swap(self, info_to_load_index, info_to_evict_index):
        """Submit PCIe HtoD copy on load_stream via cudaMemcpyAsync (truly non-blocking).

        Uses libcudart.cudaMemcpyAsync directly so the CPU returns immediately after
        dispatching the DMA — no synchronize() in the loading thread.

        Returns a CUDA Event recorded on load_stream after the copy is submitted.
        compute_stream.wait_event(copy_done_event) in the callback will stall the GPU
        until the DMA finishes, without ever blocking the CPU.

        This means:
          - Loading thread: dispatches copy + records event + fires callback in ~0.5ms total
          - Main thread: can run B8/B9/B10 etc. completely unblocked by PCIe
          - GPU compute_stream: waits for copy_done_event before reading expert weights
            (GPU-level fence, zero CPU stall)
        """
        start = time.time()
        dst_ptr = self.main_modules[info_to_evict_index].storage.data_ptr()
        src_ptr = self.offloaded_storages[info_to_load_index].storage.data_ptr()
        nbytes  = self.offloaded_storages[info_to_load_index].storage.nbytes()
        stream_handle = self.load_stream.cuda_stream

        # Record start event BEFORE the DMA is submitted.
        # CUDA stream is ordered, so elapsed_time(copy_start_event, copy_done_event)
        # after load_stream.synchronize() gives the true per-expert DMA duration,
        # completely independent of CPU activity on the main thread.
        copy_start_event = torch.cuda.Event(enable_timing=True)
        copy_start_event.record(self.load_stream)

        if _CUDART_AVAILABLE:
            err = _cudaMemcpyAsync(
                ctypes.c_void_p(dst_ptr),
                ctypes.c_void_p(src_ptr),
                ctypes.c_size_t(nbytes),
                ctypes.c_int(_cudaMemcpyHostToDevice),
                ctypes.c_void_p(stream_handle),
            )
            if err != 0:
                # Fallback to synchronous copy if cudaMemcpyAsync fails
                with torch.cuda.stream(self.load_stream):
                    self.main_modules[info_to_evict_index].storage.copy_(
                        self.offloaded_storages[info_to_load_index].storage)
        else:
            # Fallback: synchronous copy (old behavior)
            with torch.cuda.stream(self.load_stream):
                self.main_modules[info_to_evict_index].storage.copy_(
                    self.offloaded_storages[info_to_load_index].storage)

        # Record done event AFTER the DMA is submitted on the stream.
        # GPU stream ordering guarantees this event fires only after the copy completes.
        copy_done_event = torch.cuda.Event(enable_timing=True)
        copy_done_event.record(self.load_stream)

        return copy_start_event, copy_done_event

    def get_compute_expert(self,uid:ExpertUID,offload=False):
        with self.mtx:
            info = self.registered_experts[uid]
            if not offload:
                # logger.debug("self.main_modules[info.index]",uid,info.index)
                return self.main_modules[info.index]
            else:
                return self.offloaded_storages[info.offload_index]

    def ready_compute(self, uid: ExpertUID):
        self.registered_experts[uid].priority = 2
        if uid in self.priority_one_set:
            self.priority_one_queue.remove(uid)
            self.priority_one_set.discard(uid)

    def predict_compute(self, uid: ExpertUID):
        if self.registered_experts[uid].priority == 2:
            return
        self.registered_experts[uid].priority = 1
        if uid not in self.priority_one_set:
            self.priority_one_queue.append(uid)
            self.priority_one_set.add(uid)
        self._check_priority_capacity()

    def end_compute(self, uid: ExpertUID):
        self.registered_experts[uid].priority = 0
        if uid in self.priority_one_set:
            self.priority_one_queue.remove(uid)
            self.priority_one_set.discard(uid)

    def _check_priority_capacity(self):
        while len(self.priority_one_queue) > self.capacity_1:
            oldest_uid = self.priority_one_queue.popleft()
            self.priority_one_set.discard(oldest_uid)
            if oldest_uid in self.registered_experts:
                self.registered_experts[oldest_uid].priority = 0
    def update_scores(self, layerid, scores_lst):
        """Update sliding-window scores for all experts in one layer.

        Fully vectorised: single numpy column-write replaces 64 Python loop
        iterations. ExpertInfo.scores is NOT updated here — eviction reads
        directly from the numpy buffer via choose_expert_to_evictbyScore().
        """
        num_experts = len(scores_lst)

        # ── lazy init of numpy score buffer ──────────────────────────────────
        if self._score_buf is None:
            all_layers = {uid[0] for uid in self.registered_experts}
            L = max(all_layers) + 1 if all_layers else (layerid + 1)
            E = num_experts
            W = self.cache_window
            self._score_L   = L
            self._score_E   = E
            self._score_buf = np.zeros((L, E, W), dtype=np.float32)
            self._score_sum = np.zeros((L, E),    dtype=np.float64)
            self._score_ptr = np.zeros(L,          dtype=np.int32)
            self._score_cnt = np.zeros(L,          dtype=np.int32)

        W   = self.cache_window
        ptr = int(self._score_ptr[layerid])
        cnt = int(self._score_cnt[layerid])

        new_col = np.asarray(scores_lst, dtype=np.float32)  # shape (E,)

        if cnt == W:
            self._score_sum[layerid] -= self._score_buf[layerid, :, ptr]
        else:
            self._score_cnt[layerid] = cnt + 1

        self._score_buf[layerid, :, ptr] = new_col
        self._score_sum[layerid]        += new_col
        self._score_ptr[layerid]         = (ptr + 1) % W
    def loading(self):
        try:
            if _shared_core is not None:
                os.sched_setaffinity(0, {_shared_core})
        except Exception:
            pass
        while True:
            # ── phase 1: pick next uid, snapshot generation (under lock) ─────────
            already_loaded = False
            cb_for_already = None
            uid = None
            gen_snapshot = None
            with self.cv:
                self.cv.wait_for(lambda: len(self.load_queue) > 0)
                uid = self.load_queue[0]
                gen_snapshot = self._queue_generation   # snapshot HERE, while locked

                if self.query_expert_inload(uid):
                    self.load_queue.popleft()
                    already_loaded = True
                    cb_for_already = self.on_expert_loaded
                    if cb_for_already is not None:
                        self.pending_callbacks += 1
                    self.cv.notify_all()

            if already_loaded:
                if cb_for_already is not None:
                    try:
                        cb_for_already(uid, None)
                    except TypeError:
                        cb_for_already(uid)
                    with self.cv:
                        self.pending_callbacks -= 1
                        self.cv.notify_all()
                continue

            # ── phase 2: choose eviction (under lock), check generation first ─────
            with self.cv:
                # If clear_queue() fired between phase1 unlock and phase2 lock,
                # generation will have changed — skip this uid entirely (no DMA).
                if self._queue_generation != gen_snapshot:
                    if len(self.load_queue) > 0 and self.load_queue[0] == uid:
                        self.load_queue.popleft()
                    self.cv.notify_all()
                    continue

                if self.query_expert_inload(uid):
                    self.load_queue.popleft()
                    self.cv.notify_all()
                    continue

                assert uid in self.cache_infos.offloaded_infos
                self.registered_experts[uid].loading = True
                info_to_load = self.registered_experts[uid]
                info_to_evict = self.cache_infos.choose_expert_to_evictbyScore(
                    score_sum=self._score_sum, score_cnt=self._score_cnt
                ) if self.cache_window != None else self.cache_infos.choose_expert_to_evictbyLRU()
                assert info_to_load.offloaded and not info_to_evict.offloaded
                self.registered_experts[info_to_evict.uid].offloaded = True
                evictindex = info_to_evict.index
                loadindex  = info_to_load.offload_index
                # gen_snapshot is still valid (we just confirmed it above)

            # ── phase 3: async PCIe copy (outside lock) ──────────────────────────
            copy_start_event, copy_done_event = self._swap(loadindex, evictindex)

            # ── phase 4: commit metadata; suppress callback if generation changed ─
            with self.cv:
                # DMA is in flight or done — always commit metadata for consistency.
                self.registered_experts[uid].index = evictindex
                self.registered_experts[info_to_evict.uid].index = info_to_evict.offload_index
                if len(self.load_queue) > 0 and uid == self.load_queue[0]:
                    self.load_queue.popleft()
                self.registered_experts[uid].loading = False
                self.registered_experts[uid].offloaded = False
                self.cache_infos.swap(info_to_load, info_to_evict)

                # Store CUDA timing events for B8 to read after synchronize().
                self._pcie_timing[uid] = (copy_start_event, copy_done_event)

                # Only fire callback if clear_queue() was NOT called after our snapshot.
                # If generation changed, the layer that requested this prefetch is gone;
                # the expert is now silently in GPU cache (bonus hit for next layer).
                fired_by_cleared_layer = (self._queue_generation != gen_snapshot)
                cb = None if fired_by_cleared_layer else self.on_expert_loaded
                if cb is not None:
                    self.pending_callbacks += 1
                    self._last_copy_event = copy_done_event
                self.cv.notify_all()

            if cb is not None:
                try:
                    cb(uid, copy_done_event)
                except TypeError:
                    cb(uid)
                with self.cv:
                    self.pending_callbacks -= 1
                    self.cv.notify_all()
    def add_to_queue(self,uid:ExpertUID):
        with self.cv:
            if uid not in self.load_queue and not self.query_expert_inload(uid):

                self.load_queue.append(uid)
                self.cv.notify_all()
    def count_uids_in_queue(self, uids: set) -> int:
        """Return how many of the given uids are still pending in load_queue."""
        with self.cv:
            return sum(1 for uid in self.load_queue if uid in uids)
    def clear_queue(self):
        with self.cv:
            self.load_queue = deque()
            self._queue_generation += 1   # signal loading thread to abort current prefetch
    def wait_pending_callbacks(self):
        """Wait until all in-flight on_expert_loaded callbacks have completed.

        Unlike wait_until_queue_empty(), this does NOT wait for queued but not-yet-
        started loads — it only drains callbacks for loads already in progress.
        Used by B4 before clear_queue() to avoid a race where a non_blocking async
        copy completes after B4 clears the queue, then tries to fire a callback
        against the wrong (next-layer) expert_token_dic.
        """
        with self.cv:
            self.cv.wait_for(lambda: self.pending_callbacks == 0)

    def wait_until_queue_empty(self):
        with self.cv:
            self.cv.wait_for(lambda: len(self.load_queue) == 0 and self.pending_callbacks == 0)
def replaceset_between_tokens(scores:list,a:float,topk):
    replaceset = set()
    allset = set()
    n_tokens = len(scores)
    n_experts = len(scores[0])
    sort_index = [sorted(range(len(input_list)),key=lambda i:input_list[i],reverse=True) for input_list in scores]
    sort_scores = [[scores[j][i] for i in sort_index[j]] for j in range(n_tokens)]
    for token_id in range(n_tokens):
        midscore = sort_scores[token_id][topk]
        # lscore = find_smallest_max_outlier(sort_scores[token_id][:topk+6])
        lscore = midscore+a*midscore
        for expert_i in range(n_experts):
            if sort_scores[token_id][expert_i] >= lscore and expert_i<topk:
                replaceset.add(sort_index[token_id][expert_i])
            if expert_i <topk:
                allset.add(sort_index[token_id][expert_i])
    return list(replaceset),list(allset)

def cache_router(scores:list,cache:ExpertCache,a:float,topk:int,replaceset:list,layer_id:int):
    n_tokens =len(scores)
    n_experts = len(scores[0])
    cacherouter_experts = [[None for i in range(topk)] for j in range(n_tokens)]
    top_uid = [[] for j in range(n_tokens)]
    sort_index = [sorted(range(len(input_list)),key=lambda i:input_list[i],reverse=True) for input_list in scores]
    sort_scores = [[scores[j][i] for i in sort_index[j]] for j in range(n_tokens)]
    expertdic_batch = dict()
    tokendict_alter = dict()
    tokendict_highnum = dict()
    
    for token_id in range(n_tokens):
        midscore = sort_scores[token_id][topk]
        lscore = midscore+a*midscore
        rscore = midscore-a*midscore
        high_num=0
        canreplaceset = set()
        for expert_i in range(n_experts):
            expertid = sort_index[token_id][expert_i]
            if sort_scores[token_id][expert_i]>=lscore and expert_i<topk:
                cache.ready_compute((layer_id,expertid))
                cacherouter_experts[token_id][expert_i] =expertid
                top_uid[token_id].append((layer_id,expertid))
                high_num+=1
                uid = (layer_id,expertid)
                expertdic_batch[uid]=expertdic_batch.get(uid, 0) + 1
            elif rscore < sort_scores[token_id][expert_i] < midscore:
                canreplaceset.add(expertid)
                tokendict_alter[token_id] = tokendict_alter.get(token_id,[])
                replaceuid = (layer_id,expertid)
                tokendict_alter[token_id].append(replaceuid)

        low_score_experts_needload=0
        tokendict_highnum[token_id] = high_num
        for expert_i in range(high_num,topk):
            expertid = sort_index[token_id][expert_i]
            uid = (layer_id,expertid)
            if cache.query_expert(uid) or expertid in replaceset:
                cacherouter_experts[token_id][expert_i] =expertid
                cache.ready_compute((layer_id,expertid))
                expertdic_batch[uid]=expertdic_batch.get(uid, 0) + 1
                continue
            flag = 0
            low_score_experts_needload+=1
            for replaceexpertid in canreplaceset:
                replaceuid = (layer_id,replaceexpertid)
                if (cache.query_expert(replaceuid) or replaceexpertid in replaceset) and replaceexpertid not in cacherouter_experts[token_id]:

                    cacherouter_experts[token_id][expert_i] = replaceexpertid
        
                    expertdic_batch[replaceuid]=expertdic_batch.get(replaceuid, 0) + 1
                    canreplaceset.remove(replaceexpertid)
                    cache.ready_compute(replaceuid)
                    flag=1
                    break
            if flag==1:
                continue
            else:
                cacherouter_experts[token_id][expert_i]=expertid
                expertdic_batch[uid]=expertdic_batch.get(uid, 0) + 1
                cache.ready_compute(uid)

    return cacherouter_experts,top_uid
def remove_outliers_and_average(raw):
    numbers = raw[:]
    if len(numbers) == 0:
        raise ValueError("list can not be empty")
    
    if len(numbers) <= 2:
        return np.mean(numbers)
    
    mean = np.mean(numbers)
    std_dev = np.std(numbers)
    

    outliers = [x for x in numbers if abs(x - mean) > std_dev]
    

    filtered_numbers = [x for x in numbers if x not in outliers]
    

    if len(filtered_numbers) == 0:
        return np.mean(numbers)
    

    average = np.mean(filtered_numbers)
    
    return average


def CPU_load_management(uid_batch, cpucost, loadcost, prefetch_pcie_budget=0.0):
    """
    Split miss experts between PCIe-load and CPU-compute to minimise
    max(total_pcie_time, total_cpu_time)  — i.e. balance B8 ≈ B7.

    PCIe DMA transfers are serialised by the loading thread, so:
        total_pcie_time = n_pcie * loadcost
    CPU compute is also serialised on the main thread:
        total_cpu_time  = n_cpu  * cpucost

    We greedily assign each expert to whichever side finishes first,
    breaking ties in favour of CPU (avoids over-loading PCIe which has
    fixed per-expert DMA time that can't be hidden by parallelism).

    uid_batch: {uid: token_count}  — sorted large→small so high-load
               experts go to the faster resource.
    """
    uids = sorted(uid_batch.keys(), key=lambda x: uid_batch[x], reverse=True)

    all_cpucost  = 0.0
    all_loadcost = prefetch_pcie_budget

    cpulst  = []
    loadlst = []
    for uid in uids:
        # TODO实现：如果当前两条流水线完全空闲（耗时均为0），优先分配给单次开销较小的资源
        if all_cpucost == 0.0 and all_loadcost == 0.0:
            if loadcost <= cpucost:
                loadlst.append(uid)
                all_loadcost += loadcost
            else:
                cpulst.append(uid)
                all_cpucost += cpucost
        # 常规贪心逻辑：分配给当前累计开销较小（先执行完）的流水线
        else:
            if all_loadcost <= all_cpucost:
                loadlst.append(uid)
                all_loadcost += loadcost
            else:
                cpulst.append(uid)
                all_cpucost += cpucost

    if tokens > 0:  # decode 阶段才打印
        print('CPU_load_management: cpucost={}'.format(cpucost))
        print('CPU_load_management: loadcost={}'.format(loadcost))
        print('CPU_load_management: prefetch_pcie_budget={}'.format(prefetch_pcie_budget))
        print('CPU_load_management: final cpulst={}'.format(cpulst))
        print('CPU_load_management: final loadlst={}'.format(loadlst),flush=True)
                
    return loadlst, cpulst 

            
import numpy as np

def find_smallest_max_outlier(data, threshold=3):


    data = np.asarray(data)
    mean = np.mean(data)
    std_dev = np.std(data)


    z_scores = (data - mean) / std_dev


    outliers = data[np.abs(z_scores) > threshold]

    if outliers.size > 0:

        min_of_max_outliers = np.min(outliers)
        return min_of_max_outliers
    else:
        return mean

