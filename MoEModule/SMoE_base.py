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
import os
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
from utils import decode_metrics

logger = logging.getLogger(__name__)

ExpertUID = Tuple[int, int]

# ---------------------------------------------------------------------------
# Module-level CPU compute-time tracking (per generated token)
# ---------------------------------------------------------------------------
_cpu_ms_cur_token_idx: int = -1           # token index currently being accumulated
_cpu_ms_cur_token_samples: List[float] = []    # compute_ms of each CPU expert this token
cpu_compute_ms_per_token: List[float] = []     # average compute_ms flushed per token
cpu_compute_token_indices: List[int] = []      # matching token index (0 = prefill)
cpu_activation_d2h_copies: int = 0
cpu_activation_d2h_bytes: int = 0
cpu_output_h2d_copies: int = 0
cpu_output_h2d_bytes: int = 0


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
        self._error = None
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
                self._error = e
                logger.error("BgThread error: %s", e, exc_info=True)
            finally:
                self._done.set()

    def submit(self, fn, args=()):
        """Submit work; caller must call wait() before reading results."""
        self._error = None
        self._done.clear()
        self._work_fn   = fn
        self._work_args = args
        self._ready.set()

    def wait(self):
        """Block until submitted work is complete."""
        self._done.wait()
        if self._error is not None:
            raise RuntimeError('MoE background work failed') from self._error


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
        self._batch_cpu_transfers = os.environ.get(
            "SMOE_CPU_BATCH_TRANSFERS", "1"
        ).strip().lower() not in {"0", "false", "off", "no"}
        self._decode_direct = os.environ.get(
            "SMOE_DECODE_DIRECT", "1"
        ).strip().lower() not in {"0", "false", "off", "no"}
        self._decode_direct_logged = False
        self._decode_stream_chain = os.environ.get(
            "SMOE_DECODE_STREAM_CHAIN", "1"
        ).strip().lower() not in {"0", "false", "off", "no"}
        self._pinned_decode = os.environ.get(
            "SMOE_PINNED_DECODE", "1"
        ).strip().lower() not in {"0", "false", "off", "no"}
        self._decode_arena = None
        self._staged_decode_input = None
        if layerid == 0:
            logger.info(
                "[CPU transfer] batch_repeated_activations_and_outputs=%s "
                "(override with SMOE_CPU_BATCH_TRANSFERS=0|1)",
                self._batch_cpu_transfers,
            )

        # Rolling window for CPU-compute time estimator
        self.CPUComputeTimeOneExpertOneBatch = [0.05]
        self._decode_minmax = os.environ.get('SMOE_DECODE_MINMAX', '0') == '1'
        self._decode_cpu_only = os.environ.get('SMOE_DECODE_CPU_ONLY', '0') == '1'
        self._gpu_inline_submit = os.environ.get('SMOE_GPU_INLINE_SUBMIT', '0') == '1'
        self._grouped_decode = None
        self._validate_grouped = os.environ.get('SMOE_VALIDATE_GROUPED', '0') == '1'
        if (os.environ.get('SMOE_GPU_GROUPED_TRITON', '0') == '1'
                and getattr(config, 'model_type', '') == 'xverse'
                and config.hidden_act == 'silu' and config.pretraining_tp == 1):
            from utils.grouped_decode import GroupedDecode
            self._grouped_decode = GroupedDecode(config.hidden_size,
                config.intermediate_size, config.moe_top_k, config.device)

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

        # ── B0: gate + softmax + score tracking ─────────────────────────
        router_logits   = gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)

        # Compute .tolist() once — reused in B1/B2 to avoid duplicate GPU→CPU syncs
        routing_weights_list = routing_weights.tolist()

        if self.ExpertCache.cache_window is not None:
            for score_l in routing_weights_list:
                self.ExpertCache.update_scores(self.layerid, score_l)

        # ── B1: replaceset + cache_router/top-k ──────────────────────────
        replaceset = []
        topk_idx_cpu = None
        cache_selection_pinned = False
        if self.replaceScoreRatio is not None and expertcache_module.tokens > 0:
            replaceset, allset, sorted_indices, sorted_scores = \
                replaceset_between_tokens(
                    routing_weights_list, self.replaceScoreRatio, top_k,
                    return_sorted=True)

            if self.if_replace:
                cacherouter_experts, _ = cache_router(
                    routing_weights_list, self.ExpertCache,
                    self.replaceScoreRatio, top_k,
                    replaceset, self.layerid,
                    sorted_indices=sorted_indices,
                    sorted_scores=sorted_scores)
                cache_selection_pinned = True
                topk_idx_cpu = cacherouter_experts
                topk_idx = torch.tensor(
                    topk_idx_cpu, dtype=torch.long,
                    device=routing_weights.device)
                topk_weight = routing_weights.gather(1, topk_idx)

        if topk_idx_cpu is None:
            topk_weight, topk_idx = torch.topk(
                routing_weights, top_k, dim=-1, sorted=True)
            topk_idx_cpu = topk_idx.tolist()

        # Prefetch hit accounting
        if self.if_prefetch and expertcache_module.tokens > 0:
            loaded_set = expertcache_module.prefetch_loaded_by_layer.get(
                self.layerid, set())

        # ── B2: build expert_token_dic (optimized: no one_hot/nonzero) ──

        # cache_router pins replacement selections immediately.  The regular
        # top-k path still needs to pin its selections here.
        if not cache_selection_pinned:
            for tok_experts in topk_idx_cpu:
                for eid in tok_experts:
                    self.ExpertCache.ready_compute((self.layerid, eid))

        if self.get_norm_topk_prob():
            topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)
        self._current_topk_weight = topk_weight

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
            if self._decode_direct and num_tokens == 1 and len(tok_indices) == 1:
                # A top-k expert sees the same sole token. Views preserve the
                # FP32 gate and avoid tiny index transfers and gather launches.
                slot = slot_indices[0]
                expert_token_dic[uid] = [
                    hidden_states, topk_weight[:, slot:slot + 1], None,
                    tok_indices,
                ]
                if not self._decode_direct_logged:
                    logger.info("[decode direct] layer=%d engaged", self.layerid)
                    self._decode_direct_logged = True
                continue
            top_x = torch.tensor(tok_indices,  dtype=torch.long,
                                 device=hidden_states.device)
            slots = torch.tensor(slot_indices, dtype=torch.long,
                                 device=hidden_states.device)
            expert_token_dic[uid] = [
                hidden_states[top_x],
                topk_weight[top_x, slots, None],
                top_x,
                tok_indices,
            ]

        # ── B3: shared expert (GPU default stream) ───────────────────────
        self._staged_decode_input = None
        if (self._pinned_decode and self._batch_cpu_transfers and self.if_usecpu
                and num_tokens == 1 and hidden_states.is_cuda
                and all(len(v[3]) == 1 for v in expert_token_dic.values())):
            self._stage_decode_input(hidden_states, top_k)
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

    @torch.no_grad()
    def _work_cachehit_and_predict(
            self, hit_uids, expert_token_dic, expert_out_dict,
            miss_count, residual_cur, identity, bsh, shared_expert_output):
        """
        Called by the persistent bg-worker thread (B6), parallel with B7 + PCIe.
        1. Compute all cache-hit experts on GPU (default stream).
        2. Optionally run predict to get next-layer top experts.
        """
        self.next_experts = None

        timing = decode_metrics.gpu_begin() if expertcache_module.tokens > 0 and hit_uids else None
        grouped = (self._grouped_decode is not None and bsh[0] * bsh[1] == 1
                   and bool(hit_uids) and self._decode_direct
                   and identity.dtype == torch.bfloat16
                   and self._current_topk_weight.dtype == self._grouped_decode.weights.dtype)
        if grouped:
            experts = [self.ExpertCache.get_compute_expert(uid) for uid in hit_uids]
            base = self._current_topk_weight.storage_offset()
            slots = [expert_token_dic[uid][1].storage_offset() - base for uid in hit_uids]
            out = self._grouped_decode(identity, self._current_topk_weight,
                [expert.storage.data_ptr() for expert in experts], slots)
            if self._validate_grouped:
                for row, (uid, expert) in enumerate(zip(hit_uids, experts)):
                    reference = expert(expert_token_dic[uid][0])
                    reference.mul_(expert_token_dic[uid][1])
                    actual = out[row:row+1]
                    delta = actual.float() - reference.float()
                    relative_rms = (delta.square().mean() /
                        reference.float().square().mean().clamp_min(1e-20)).sqrt().item()
                    if not torch.isfinite(actual).all() or relative_rms >= .01:
                        raise AssertionError(f'Grouped expert {uid} relative RMS={relative_rms}')
                    logger.info('[Grouped validation] uid=%s max_abs=%.8g relative_rms=%.8g',
                                uid, delta.abs().max().item(), relative_rms)
                self._validate_grouped = False
            for row, uid in enumerate(hit_uids):
                expert_out_dict[uid] = out[row:row+1]
        for uid in ([] if grouped else hit_uids):
            expert = self.ExpertCache.get_compute_expert(uid)
            out    = expert(expert_token_dic[uid][0])
            out.mul_(expert_token_dic[uid][1])
            expert_out_dict[uid] = out
        decode_metrics.gpu_end(timing, len(hit_uids))

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
        global _cpu_ms_cur_token_idx, _cpu_ms_cur_token_samples
        global cpu_compute_ms_per_token, cpu_compute_token_indices
        cur_tok = expertcache_module.tokens
        if cur_tok != _cpu_ms_cur_token_idx:
            if _cpu_ms_cur_token_samples:
                cpu_compute_ms_per_token.append(
                    sum(_cpu_ms_cur_token_samples) / len(_cpu_ms_cur_token_samples))
                cpu_compute_token_indices.append(_cpu_ms_cur_token_idx)
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
            if cur_tok > 0 and bsh[0] * bsh[1] == 1:
                if self._decode_cpu_only:
                    pcie_uids, cpu_uids = [], list(uid_batch)
                elif self._decode_minmax:
                    uids = list(uid_batch)
                    n = len(uids)
                    cpu_count = min(range(n + 1), key=lambda c: (
                        max(c * cpu_avg, (n - c) * load_avg), n - c))
                    pcie_uids, cpu_uids = uids[cpu_count:], uids[:cpu_count]
        else:
            pcie_uids = list(uid_batch.keys())
            cpu_uids  = []

        if decode_metrics.enabled and cur_tok > 0:
            decode_metrics.counts.update(hits=len(hit_uids), misses=len(uid_batch),
                                        miss_to_gpu=len(pcie_uids), miss_to_cpu=len(cpu_uids))
        expert_out_dict = {}

        # ── B6: submit bg-thread work + enqueue PCIe loads (parallel) ───
        inline = self._gpu_inline_submit and bsh[0] * bsh[1] == 1 and not self.if_prefetch
        work_args = (hit_uids, expert_token_dic, expert_out_dict,
                     len(uid_batch), residual_cur, identity, bsh, shared_expert_output)
        if inline:
            self._work_cachehit_and_predict(*work_args)
        else:
            self._bg_worker.submit(self._work_cachehit_and_predict, args=work_args)
        for uid in pcie_uids:
            self.ExpertCache.add_to_queue(uid)

        # ── B7: CPU compute miss experts (main thread, parallel) ─────────
        self._cpu_compute(cpu_uids, expert_token_dic, expert_out_dict)

        # ── B8: wait for all PCIe loads + DMA ────────────────────────────
        _tb8 = time.time()
        self.ExpertCache.wait_until_queue_empty()
        self.ExpertCache.load_stream.synchronize()
        _b8_elapsed = time.time() - _tb8

        if getattr(self.ExpertCache, "measure_dma", False):
            self.ExpertCache.consume_dma_timings()
        elif pcie_uids and _b8_elapsed > 0:
            actual_per_expert = _b8_elapsed / len(pcie_uids)
            lst = self.ExpertCache.LoadTimeOneExpert
            lst.append(actual_per_expert)
            if len(lst) > 10:
                self.ExpertCache.LoadTimeOneExpert = lst[-10:]

        # ── B9: wait for background thread ───────────────────────────────
        if not inline:
            self._bg_worker.wait()

        # ── B11: compute PCIe-loaded miss experts on GPU ─────────────────
        timing = decode_metrics.gpu_begin() if cur_tok > 0 and pcie_uids else None
        for uid in pcie_uids:
            expert = self.ExpertCache.get_compute_expert(uid)
            out    = expert(expert_token_dic[uid][0])
            out.mul_(expert_token_dic[uid][1])
            expert_out_dict[uid] = out
        decode_metrics.gpu_end(timing, len(pcie_uids), 'miss')

        # ── B12: sync GPU + scatter all expert outputs ───────────────────
        # With prefetch disabled, the current load queue has been drained and
        # no new eviction can be submitted until the next layer's B0 tolist()
        # completes. That mandatory readback drains the default stream before
        # it can reuse any released weight slot. Keep the final-layer fence so
        # the original model-forward timing boundary is not weakened.
        chain_default_stream = (
            self._decode_stream_chain and not self.if_prefetch
            and bsh[0] * bsh[1] == 1
            and self.layerid < getattr(self.config, "num_hidden_layers", 1) - 1
            and torch.cuda.current_stream(identity.device)
                == torch.cuda.default_stream(identity.device)
        )
        if not chain_default_stream:
            torch.cuda.synchronize()
        for uid in expert_token_dic:
            if expert_token_dic[uid][2] is None:
                # Same expert order and one BF16 rounding per addition as the
                # single-destination index_add_ reference.
                final_hidden_states.add_(expert_out_dict[uid].to(hdtype))
                continue
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

    def _stage_decode_input(self, hidden_states, top_k):
        """Queue the sole D2H before shared/GPU experts, wait only on its event.

        All transfers use the caller's stream. The CPU can start as soon as
        this small copy completes, without waiting for subsequent GPU MLPs.
        Each layer owns its buffers until the next invocation of that layer.
        """
        key = (hidden_states.shape[1], hidden_states.dtype, hidden_states.device, top_k)
        if self._decode_arena is None or self._decode_arena[0] != key:
            if self._decode_arena is not None:
                self._decode_arena[3].synchronize()
                self._decode_arena[4].synchronize()
            host_input = torch.empty_like(hidden_states, device='cpu', pin_memory=True)
            host_output = torch.empty((top_k, hidden_states.shape[1]),
                                      dtype=hidden_states.dtype, device='cpu', pin_memory=True)
            self._decode_arena = (key, host_input, host_output,
                                  torch.cuda.Event(), torch.cuda.Event())
            logger.info("[pinned decode] layer=%d arena_bytes=%d", self.layerid,
                        host_input.nbytes + host_output.nbytes)
        _, host_input, _, input_ready, _ = self._decode_arena
        host_input.copy_(hidden_states, non_blocking=True)
        input_ready.record(torch.cuda.current_stream(hidden_states.device))
        self._staged_decode_input = (host_input, input_ready)
        self._record_cpu_transfer(host_input, d2h=True)

    @torch.no_grad()
    def _cpu_compute(self, cpu_uids, expert_token_dic, expert_out_dict):
        if not cpu_uids:
            return
        if not self._batch_cpu_transfers:
            self._cpu_compute_reference(
                cpu_uids, expert_token_dic, expert_out_dict)
            return

        # Decode normally routes one token to several CPU experts.  Reuse the
        # D2H activation for identical token slices, then concatenate all CPU
        # expert outputs for one H2D transfer.
        cpu_inputs = {}
        cpu_results = []
        staged = self._staged_decode_input
        if staged is not None:
            staged[1].synchronize()
            cpu_inputs[(0,)] = staged[0]
        for uid in cpu_uids:
            expert = self.ExpertCache.get_compute_expert(uid, offload=True)
            token_key = tuple(expert_token_dic[uid][3])
            tokens_cpu = cpu_inputs.get(token_key)
            if tokens_cpu is None:
                tokens_cpu = expert_token_dic[uid][0].to("cpu")
                cpu_inputs[token_key] = tokens_cpu
                self._record_cpu_transfer(tokens_cpu, d2h=True)

            t_compute_0 = time.perf_counter()
            out_cpu = expert(tokens_cpu)
            compute_ms = (time.perf_counter() - t_compute_0) * 1000
            cpu_results.append((uid, out_cpu, compute_ms))
            self._record_cpu_compute(compute_ms)

        if staged is not None:
            _, _, host_output, _, output_copied = self._decode_arena
            # Previous H2D must finish before the pinned source is overwritten.
            output_copied.synchronize()
            for row, (_, result, _) in enumerate(cpu_results):
                host_output[row:row + 1].copy_(result)
            output_batch_cpu = host_output[:len(cpu_results)]
        elif len(cpu_results) == 1:
            output_batch_cpu = cpu_results[0][1]
        else:
            output_batch_cpu = torch.cat(
                [result[1] for result in cpu_results], dim=0)
        output_batch = output_batch_cpu.to(self.config.device, non_blocking=staged is not None)
        if staged is not None:
            output_copied.record(torch.cuda.current_stream(output_batch.device))
        self._record_cpu_transfer(output_batch_cpu, d2h=False)

        row_offset = 0
        for uid, out_cpu, _ in cpu_results:
            rows = out_cpu.size(0)
            out = output_batch.narrow(0, row_offset, rows)
            out.mul_(expert_token_dic[uid][1])
            expert_out_dict[uid] = out
            row_offset += rows

    def _record_cpu_compute(self, compute_ms):
        if decode_metrics.enabled and expertcache_module.tokens > 0:
            decode_metrics.cpu_ms.append(compute_ms)
        elapsed = compute_ms / 1000.0
        self.CPUComputeTimeOneExpertOneBatch.append(elapsed)
        self.CPUComputeTimeOneExpertOneBatch = \
            self.CPUComputeTimeOneExpertOneBatch[-10:]

        global _cpu_ms_cur_token_samples
        _cpu_ms_cur_token_samples.append(compute_ms)

    @staticmethod
    def _record_cpu_transfer(tensor, d2h):
        global cpu_activation_d2h_copies, cpu_activation_d2h_bytes
        global cpu_output_h2d_copies, cpu_output_h2d_bytes
        nbytes = tensor.numel() * tensor.element_size()
        if d2h:
            cpu_activation_d2h_copies += 1
            cpu_activation_d2h_bytes += nbytes
        else:
            cpu_output_h2d_copies += 1
            cpu_output_h2d_bytes += nbytes

    @torch.no_grad()
    def _cpu_compute_reference(self, cpu_uids, expert_token_dic, expert_out_dict):
        for uid in cpu_uids:
            expert = self.ExpertCache.get_compute_expert(uid, offload=True)

            tokens_cpu  = expert_token_dic[uid][0].to("cpu")
            self._record_cpu_transfer(tokens_cpu, d2h=True)
            t_compute_0 = time.perf_counter()
            out_cpu     = expert(tokens_cpu)
            compute_ms  = (time.perf_counter() - t_compute_0) * 1000
            out         = out_cpu.to(self.config.device)
            self._record_cpu_transfer(out_cpu, d2h=False)

            out.mul_(expert_token_dic[uid][1])
            expert_out_dict[uid] = out
            self._record_cpu_compute(compute_ms)
