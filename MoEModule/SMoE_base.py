"""
SMoE_base.py — Abstract MoE Layer for SMoEaligned.

AbstractMoELayer provides the complete B0–B14 inference pipeline and delegates
model-specific details to subclass implementations of the abstract methods.

Users can subclass AbstractMoELayer to plug any MoE layer into the SMoEaligned
cache infrastructure without modifying the core expertcache/loading logic.

Concrete subclasses:
  - DeepseekMoEwithCache  (MoEModule/deepseek_moe.py)
  - Qwen2MoeSparseMoeBlockwithCache  (MoEModule/qwen_moe.py)
  - XverseMoEMLPwithCache  (MoEModule/xverse_moe.py)

"""

import time
import threading
import logging
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, List, Optional, Tuple

from utils.expertcache import (
    ExpertCache,
    cache_router,
    remove_outliers_and_average,
    CPU_load_management,
    replaceset_between_tokens,
)
import utils.expertcache as expertcache_module

logger = logging.getLogger(__name__)

ExpertUID = Tuple[int, int]

# ---------------------------------------------------------------------------
# Module-level CPU compute-time tracking (per generated token)
# ---------------------------------------------------------------------------
_cpu_ms_cur_token_idx: int = -1           # token index currently being accumulated
_cpu_ms_cur_token_samples: List[float] = []    # compute_ms of each CPU expert this token
cpu_compute_ms_per_token: List[float] = []     # average compute_ms flushed per token


# ---------------------------------------------------------------------------
# Persistent background worker thread
# ---------------------------------------------------------------------------

class _PersistentBgThread:
    """
    Single persistent thread that replaces per-layer threading.Thread creation.

    Protocol:
      1. Main thread calls submit(fn, args) to dispatch work.
      2. Main thread calls wait() to block until work is done.
      3. Thread is alive for the lifetime of the module (daemon).

    Eliminates N thread create/join cycles per generated token (N = num layers).
    """
    def __init__(self):
        self._work_fn   = None
        self._work_args = None
        self._ready     = threading.Event()
        self._done      = threading.Event()
        self._done.set()   # starts "done" (no work pending)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            from utils.expertcache import _shared_core
            if _shared_core is not None:
                import os
                os.sched_setaffinity(0, {_shared_core})
        except Exception:
            pass
        while True:
            self._ready.wait()
            self._ready.clear()
            try:
                self._work_fn(*self._work_args)
            except Exception as e:
                logger.error("BgThread error: %s", e, exc_info=True)
            finally:
                self._done.set()

    def submit(self, fn, args=()):
        """Submit work; caller must call wait() before reading results."""
        self._done.clear()
        self._work_fn   = fn
        self._work_args = args
        self._ready.set()

    def wait(self):
        """Block until submitted work is complete."""
        self._done.wait()


# ---------------------------------------------------------------------------
# Abstract MoE layer base class
# ---------------------------------------------------------------------------

class AbstractMoELayer(nn.Module, ABC):
    """
    Abstract base class for SMoEaligned MoE layers.

    Provides the complete inference pipeline (B0–B14) and delegates
    model-specific details to subclass implementations of the abstract methods.

    Subclasses must implement:
      - get_gate()              → nn.Module (routing gate)
      - get_num_experts()       → int
      - get_top_k()             → int
      - get_norm_topk_prob()    → bool
      - compute_shared_expert() → Tensor (return zeros if no shared expert)

    Optionally override:
      - predict_next_layer_experts() → list[int] | None  (default: None)
    """

    def __init__(self, config, expertcache: ExpertCache, layerid: int):
        super().__init__()
        self.config      = config
        self.ExpertCache = expertcache
        self.layerid     = layerid

        # Flags from config (with safe fallbacks)
        self.if_usecpu        = getattr(config, 'if_usecpu',        False)
        self.if_prefetch      = getattr(config, 'if_prefetch',      False)
        self.if_replace       = getattr(config, 'if_replace',       False)
        self.replaceScoreRatio = getattr(config, 'replaceScoreRatio', None)

        # Rolling window for CPU-compute time estimator
        self.CPUComputeTimeOneExpertOneBatch = [0.05]

        # Persistent background thread (one per MoE layer, reused across tokens)
        self._bg_worker = _PersistentBgThread()

        # Predicted next-layer expert IDs (set by background worker)
        self.next_experts: Optional[List[int]] = None

    # ------------------------------------------------------------------
    # Abstract interface — subclasses must implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def get_gate(self) -> nn.Module:
        """Return the routing gate (nn.Linear or equivalent)."""
        ...

    @abstractmethod
    def get_num_experts(self) -> int:
        """Total number of routed experts in this layer."""
        ...

    @abstractmethod
    def get_top_k(self) -> int:
        """Number of experts selected per token."""
        ...

    @abstractmethod
    def get_norm_topk_prob(self) -> bool:
        """Whether to normalize top-k routing probabilities to sum to 1."""
        ...

    @abstractmethod
    def compute_shared_expert(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute the shared expert output for this layer.
        Return zeros_like(hidden_states) if there is no shared expert.
        """
        ...

    # ------------------------------------------------------------------
    # Optional override — next-layer prefetch prediction
    # ------------------------------------------------------------------

    def predict_next_layer_experts(self, *args, **kwargs) -> Optional[List[int]]:
        """
        Predict top expert IDs for the next layer to prefetch.
        Return None to disable prefetch (default).
        Override in subclasses that support look-ahead prediction.
        """
        return None

    # ------------------------------------------------------------------
    # Core inference pipeline (B0 – B14)
    # ------------------------------------------------------------------

    def run_with_cache(
        self,
        hidden_states: torch.Tensor,
        residual=None,
        attn_weights=None,
        present_key_value=None,
        attention_mask=None,
        position_ids=None,
        output_attentions: bool = False,
        cache_position=None,
        position_embeddings=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full B0–B14 inference pipeline using expertcache.

        Returns:
            (final_hidden_states, router_logits)

        Subclasses call this from their forward() after setting up
        any model-specific state.
        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states  = hidden_states.view(-1, hidden_dim)
        num_tokens     = batch_size * sequence_length
        num_experts    = self.get_num_experts()
        top_k          = self.get_top_k()
        gate           = self.get_gate()

        # ── B0: gate + softmax + score tracking + topk ──────────────────
        router_logits   = gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)

        # Compute .tolist() once — reused in B1/B2 to avoid duplicate GPU→CPU syncs
        routing_weights_list = routing_weights.tolist()

        if self.ExpertCache.cache_window is not None:
            for score_l in routing_weights_list:
                self.ExpertCache.update_scores(self.layerid, score_l)

        topk_weight, topk_idx = torch.topk(
            routing_weights, top_k, dim=-1, sorted=True)

        # ── B1: replaceset + cache_router ────────────────────────────────
        replaceset = []
        if self.replaceScoreRatio is not None and expertcache_module.tokens > 0:
            replaceset, allset = replaceset_between_tokens(
                routing_weights_list, self.replaceScoreRatio, top_k)

            if self.if_replace:
                cacherouter_experts, _ = cache_router(
                    routing_weights_list, self.ExpertCache,
                    self.replaceScoreRatio, top_k,
                    replaceset, self.layerid)
                topk_idx    = torch.tensor(cacherouter_experts,
                                           device=topk_weight.device)
                topk_weight = routing_weights[
                    torch.arange(routing_weights.size(0)).unsqueeze(1),
                    topk_idx]

        # Prefetch hit accounting
        if self.if_prefetch and expertcache_module.tokens > 0:
            loaded_set = expertcache_module.prefetch_loaded_by_layer.get(
                self.layerid, set())

        # ── B2: build expert_token_dic (optimized: no one_hot/nonzero) ──
        topk_idx_cpu = topk_idx.tolist()

        # Notify cache for all chosen experts
        for tok_experts in topk_idx_cpu:
            for eid in tok_experts:
                self.ExpertCache.ready_compute((self.layerid, eid))

        if self.get_norm_topk_prob():
            topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)

        final_hidden_states = torch.zeros(
            (num_tokens, hidden_dim),
            dtype=hidden_states.dtype, device=hidden_states.device)

        # Direct scatter from topk_idx — avoids 64 × num_tokens GPU kernel calls
        expert_token_map: Dict = {}
        for tok_i, tok_experts in enumerate(topk_idx_cpu):
            for slot, eid in enumerate(tok_experts):
                if eid not in expert_token_map:
                    expert_token_map[eid] = ([], [])
                expert_token_map[eid][0].append(tok_i)
                expert_token_map[eid][1].append(slot)

        expert_token_dic = {}
        for eid, (tok_indices, slot_indices) in expert_token_map.items():
            uid   = (self.layerid, eid)
            top_x = torch.tensor(tok_indices,  dtype=torch.long,
                                 device=hidden_states.device)
            slots = torch.tensor(slot_indices, dtype=torch.long,
                                 device=hidden_states.device)
            expert_token_dic[uid] = [
                hidden_states[top_x],
                topk_weight[top_x, slots, None],
                top_x,
            ]

        # ── B3: shared expert (GPU default stream) ───────────────────────
        shared_expert_output = self.compute_shared_expert(hidden_states)

        # ── MoE inference (B4 – B13) ─────────────────────────────────────
        self._moe_infer(
            replaceset, final_hidden_states, expert_token_dic,
            hidden_states.dtype, residual, attn_weights, present_key_value,
            hidden_states, attention_mask, position_ids, output_attentions,
            cache_position, position_embeddings,
            (batch_size, sequence_length, hidden_dim), shared_expert_output)

        # ── B14: end_compute + combine + reshape ─────────────────────────
        for tok_experts in topk_idx_cpu:
            for eid in tok_experts:
                self.ExpertCache.end_compute((self.layerid, eid))
        final_hidden_states = (
            (final_hidden_states + shared_expert_output)
            .reshape(batch_size, sequence_length, hidden_dim)
        )

        return final_hidden_states, router_logits

    # ------------------------------------------------------------------
    # Background worker: cache-hit GPU compute + optional prefetch predict
    # ------------------------------------------------------------------

    def _work_cachehit_and_predict(
            self, hit_uids, expert_token_dic, expert_out_dict,
            miss_count, residual_cur, identity, bsh, shared_expert_output):
        """
        Called by the persistent bg-worker thread (B6), parallel with B7 + PCIe.
        1. Compute all cache-hit experts on GPU (default stream).
        2. Optionally run predict to get next-layer top experts.
        """
        self.next_experts = None

        for uid in hit_uids:
            expert = self.ExpertCache.get_compute_expert(uid)
            out    = expert(expert_token_dic[uid][0])
            out.mul_(expert_token_dic[uid][1])
            expert_out_dict[uid] = out

        if self.if_prefetch and self.layerid < 27 and miss_count > 0:
            self.next_experts = self.predict_next_layer_experts(
                residual_cur, identity, bsh, shared_expert_output)

    # ------------------------------------------------------------------
    # Main MoE inference loop (B4 – B13)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _moe_infer(self, replaceset, final_hidden_states, expert_token_dic,
                   hdtype, residual_cur, attn_weights_cur, present_key_value_cur,
                   identity, attention_mask, position_ids, output_attentions,
                   cache_position, position_embeddings, bsh, shared_expert_output):

        # ── B4: drop stale prefetch queue ────────────────────────────────
        self.ExpertCache.clear_queue()

        # ── Token boundary: flush CPU-ms accumulator ─────────────────────
        global _cpu_ms_cur_token_idx, _cpu_ms_cur_token_samples, cpu_compute_ms_per_token
        cur_tok = expertcache_module.tokens
        if cur_tok != _cpu_ms_cur_token_idx:
            if _cpu_ms_cur_token_samples:
                cpu_compute_ms_per_token.append(
                    sum(_cpu_ms_cur_token_samples) / len(_cpu_ms_cur_token_samples))
            _cpu_ms_cur_token_samples = []
            _cpu_ms_cur_token_idx = cur_tok

        # ── B5: classify experts → hit / PCIe-load / CPU-compute ────────
        hit_uids  = []
        uid_batch = {}
        newreplace = {(self.layerid, i) for i in replaceset}

        for uid in expert_token_dic:
            if self.ExpertCache.query_expert(uid):
                hit_uids.append(uid)
            else:
                uid_batch[uid] = expert_token_dic[uid][0].size(0)

        expertcache_module.cache_hits_per_token  += len(hit_uids)
        expertcache_module.cache_total_per_token += len(expert_token_dic)

        cpu_avg  = remove_outliers_and_average(self.CPUComputeTimeOneExpertOneBatch)
        load_avg = (remove_outliers_and_average(self.ExpertCache.LoadTimeOneExpert)
                    if len(self.CPUComputeTimeOneExpertOneBatch) > 2 else cpu_avg)

        if self.if_usecpu:
            pcie_uids, cpu_uids = CPU_load_management(uid_batch, cpu_avg, load_avg)
        else:
            pcie_uids = list(uid_batch.keys())
            cpu_uids  = []

        expert_out_dict = {}

        # ── B6: submit bg-thread work + enqueue PCIe loads (parallel) ───
        self._bg_worker.submit(
            self._work_cachehit_and_predict,
            args=(hit_uids, expert_token_dic, expert_out_dict,
                  len(uid_batch), residual_cur, identity,
                  bsh, shared_expert_output))
        for uid in pcie_uids:
            self.ExpertCache.add_to_queue(uid)

        # ── B7: CPU compute miss experts (main thread, parallel) ─────────
        self._cpu_compute(cpu_uids, expert_token_dic, expert_out_dict)

        # ── B8: wait for all PCIe loads + DMA ────────────────────────────
        _tb8 = time.time()
        self.ExpertCache.wait_until_queue_empty()
        self.ExpertCache.load_stream.synchronize()
        _b8_elapsed = time.time() - _tb8

        if pcie_uids and _b8_elapsed > 0:
            actual_per_expert = _b8_elapsed / len(pcie_uids)
            lst = self.ExpertCache.LoadTimeOneExpert
            lst.append(actual_per_expert)
            if len(lst) > 10:
                self.ExpertCache.LoadTimeOneExpert = lst[-10:]

        # ── B9: wait for background thread ───────────────────────────────
        self._bg_worker.wait()

        # ── B11: compute PCIe-loaded miss experts on GPU ─────────────────
        for uid in pcie_uids:
            expert = self.ExpertCache.get_compute_expert(uid)
            out    = expert(expert_token_dic[uid][0])
            out.mul_(expert_token_dic[uid][1])
            expert_out_dict[uid] = out

        # ── B12: sync GPU + scatter all expert outputs ───────────────────
        torch.cuda.synchronize()
        for uid in expert_token_dic:
            final_hidden_states.index_add_(
                0, expert_token_dic[uid][2],
                expert_out_dict[uid].to(hdtype))

        # ── B13: prefetch exactly 1 miss expert for next layer ───────────
        if self.next_experts is not None and self.layerid < 27:
            loaded_ids = set()
            for eid in self.next_experts:
                uid = (self.layerid + 1, eid)
                if not self.ExpertCache.query_expert(uid):
                    loaded_ids.add(eid)
                    self.ExpertCache.add_to_queue(uid)
                    break  # only enqueue the first miss expert
            expertcache_module.prefetch_loaded_by_layer[self.layerid + 1] = loaded_ids
            expertcache_module.prefetch_start_time[self.layerid + 1]      = time.time()

    # ------------------------------------------------------------------
    # CPU compute for miss experts assigned to CPU
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _cpu_compute(self, cpu_uids, expert_token_dic, expert_out_dict):
        for uid in cpu_uids:
            expert = self.ExpertCache.get_compute_expert(uid, offload=True)

            t_to_cpu_0  = time.time()
            tokens_cpu  = expert_token_dic[uid][0].to("cpu")
            t_compute_0 = time.time()
            out_cpu     = expert(tokens_cpu)
            t_to_gpu_0  = time.time()
            out         = out_cpu.to(self.config.device)
            t_end       = time.time()

            to_cpu_ms  = (t_compute_0 - t_to_cpu_0)  * 1000
            compute_ms = (t_to_gpu_0  - t_compute_0) * 1000
            to_gpu_ms  = (t_end       - t_to_gpu_0)  * 1000
            elapsed    = t_to_gpu_0   - t_compute_0   # compute-only for balancer

            out.mul_(expert_token_dic[uid][1])
            expert_out_dict[uid] = out

            self.CPUComputeTimeOneExpertOneBatch.append(elapsed)
            self.CPUComputeTimeOneExpertOneBatch = \
                self.CPUComputeTimeOneExpertOneBatch[-10:]

            # Accumulate compute_ms for per-token average
            global _cpu_ms_cur_token_samples
            _cpu_ms_cur_token_samples.append(compute_ms)
