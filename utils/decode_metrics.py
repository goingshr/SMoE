"""Decode-only, directly observed expert destinations and service times.

CUDA events are resolved after generation, outside the decode timing window.
Times are service time sums (overlap means they must not be added to latency).
"""
import os
from collections import Counter

import torch

enabled = os.environ.get('SMOE_DECODE_METRICS', '0') == '1'
counts = Counter()
cpu_ms = []
pcie_ms = []
gpu_events = []


def reset():
    counts.clear()
    cpu_ms.clear()
    pcie_ms.clear()
    gpu_events.clear()


def gpu_begin():
    if not enabled:
        return None
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    return event


def gpu_end(begin, experts, kind='hit'):
    if begin is not None:
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        gpu_events.append((begin, end, experts, kind))


def summary(decode_tokens):
    torch.cuda.synchronize()
    gpu_ms = sum(begin.elapsed_time(end) for begin, end, _, _ in gpu_events)
    gpu_calls = sum(n for _, _, n, _ in gpu_events)
    result = dict(counts)
    result.update(cpu_forward_calls=len(cpu_ms), pcie_copies=len(pcie_ms),
                  gpu_forward_calls=gpu_calls, cpu_forward_total_ms=sum(cpu_ms),
                  pcie_load_total_ms=sum(pcie_ms), gpu_forward_total_ms=gpu_ms)
    result['avg_cpu_expert_ms'] = sum(cpu_ms) / len(cpu_ms) if cpu_ms else 0.
    result['avg_pcie_load_ms'] = sum(pcie_ms) / len(pcie_ms) if pcie_ms else 0.
    result['avg_gpu_forward_ms'] = gpu_ms / gpu_calls if gpu_calls else 0.
    for kind in ('hit', 'miss'):
        elapsed = sum(b.elapsed_time(e) for b, e, _, k in gpu_events if k == kind)
        calls = sum(n for _, _, n, k in gpu_events if k == kind)
        result[f'gpu_{kind}_forward_total_ms'] = elapsed
        result[f'avg_gpu_{kind}_forward_ms'] = elapsed / calls if calls else 0.
    result['cpu_expert_forward_ms_per_decode_token'] = sum(cpu_ms) / decode_tokens if decode_tokens else None
    assert counts['misses'] == counts['miss_to_gpu'] + counts['miss_to_cpu']
    assert counts['miss_to_cpu'] == len(cpu_ms)
    assert counts['hits'] + counts['miss_to_gpu'] == gpu_calls
    return result
