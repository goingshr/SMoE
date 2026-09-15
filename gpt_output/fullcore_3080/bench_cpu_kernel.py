"""Compare complete BF16 experts and GEMV errors, including cold weight sets."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
p = argparse.ArgumentParser()
p.add_argument('--threads', type=int, default=13)
p.add_argument('--experts', type=int, default=32)
p.add_argument('--iterations', type=int, default=96)
a = p.parse_args()
os.sched_setaffinity(0, set(range(a.threads)))
import torch
import torch.nn.functional as F
from utils.cpu_bf16_kernel import load_kernel
torch.set_num_threads(a.threads)
torch.set_num_interop_threads(1)
torch.manual_seed(19)
gemv = load_kernel()

with torch.no_grad():
    # A double reference distinguishes changed FP32 sum order from extra
    # BF16 rounding. Near-zero entries use an absolute rather than relative bound.
    for m, n in [(0, 17), (7, 0), (7, 13), (31, 65), (2816, 2048), (2048, 1408)]:
        x = torch.randn(n, dtype=torch.bfloat16)
        w = torch.randn(m, n, dtype=torch.bfloat16) * .02
        out = gemv(w, x)
        ref = (w.double() @ x.double()).bfloat16()
        native = torch.mv(w, x)
        max_abs = (out.float()-ref.float()).abs().max().item() if m else 0
        torch.testing.assert_close(out, ref, rtol=.008, atol=2e-5)
        print(json.dumps(dict(test='gemv', shape=[m,n], max_abs_to_fp64=max_abs,
                              native_equal=torch.equal(out, native))), flush=True)
    weights = [(torch.randn(2816, 2048, dtype=torch.bfloat16)*.02,
                torch.randn(2048, 1408, dtype=torch.bfloat16)*.02)
               for _ in range(a.experts)]
    x = torch.randn(2048, dtype=torch.bfloat16)
    def forward(pair, fn):
        w, d = pair
        g, u = fn(w, x).split(1408)
        return fn(d, F.silu(g) * u)
    errors = []
    for pair in weights[:8]:
        ref, out = forward(pair, torch.mv), forward(pair, gemv)
        err = (out.float()-ref.float())
        rel_rms = err.square().mean().sqrt() / ref.float().square().mean().sqrt()
        assert rel_rms.item() < .01
        errors.append(dict(max_abs=err.abs().max().item(), relative_rms=rel_rms.item()))
    print(json.dumps(dict(test='expert_correctness', results=errors)), flush=True)
    for rep in range(3):
        for name, fn in [('torch_mv', torch.mv), ('avx2', gemv)]:
            for k in range(8): forward(weights[k % a.experts], fn)
            ts = []
            for k in range(a.iterations):
                t = time.perf_counter()
                forward(weights[k % a.experts], fn)
                ts.append((time.perf_counter()-t)*1000)
            print(json.dumps(dict(test='expert_benchmark', mode=name,
                threads=a.threads, experts=a.experts, repeat=rep,
                mean_ms=statistics.fmean(ts), median_ms=statistics.median(ts),
                p95_ms=sorted(ts)[int(.95*len(ts))])), flush=True)
