"""Run the real main.py workload with optional, explicitly separated profiling.

Example: python gpt_output/probe_deepseek_decode.py --artifacts /tmp/probe
--switch-interval 0.0001 --profile -- --model_name deepseekmoe ...
Profiling timings are diagnostic, never acceptance benchmark results.
"""
import argparse
import cProfile
import functools
import json
import os
from pathlib import Path
import pstats
import runpy
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
p = argparse.ArgumentParser()
p.add_argument('--artifacts', type=Path, required=True)
p.add_argument('--switch-interval', type=float)
p.add_argument('--profile', action='store_true')
p.add_argument('--python-profile', action='store_true',
               help='Also enable cProfile; use a separate diagnostic run because it perturbs thread scheduling')
p.add_argument('--profile-prompt', type=int, default=0)
p.add_argument('--trace-start', type=int, default=10)
p.add_argument('--trace-steps', type=int, default=5)
p.add_argument('main_args', nargs=argparse.REMAINDER)
args = p.parse_args()
args.artifacts.mkdir(parents=True, exist_ok=True)
if args.switch_interval is not None:
    sys.setswitchinterval(args.switch_interval)

import torch
import utils.model_loader as loader

original_build = loader.build_model
profile = cProfile.Profile()
records = []
warmup_count = (int(args.main_args[args.main_args.index('--warmup_num') + 1])
                if '--warmup_num' in args.main_args else 0)


@functools.wraps(original_build)
def build(*a, **kw):
    model = original_build(*a, **kw)
    original_generate = model.generate
    original_forward = model.model.forward
    step = 0
    tracer = None
    call_index = -warmup_count

    def finish_trace():
        nonlocal tracer
        if args.python_profile:
            profile.disable()
        tracer.__exit__(None, None, None)
        tracer.export_chrome_trace(str(args.artifacts / 'decode.trace.json'))
        (args.artifacts / 'operators.txt').write_text(
            tracer.key_averages().table(sort_by='self_cpu_time_total', row_limit=50))
        if args.python_profile:
            profile.dump_stats(str(args.artifacts / 'decode.pstats'))
            with (args.artifacts / 'python_profile.txt').open('w') as f:
                pstats.Stats(profile, stream=f).sort_stats('cumulative').print_stats(70)
        tracer = None

    def forward(*fa, **fk):
        nonlocal step, tracer
        if args.profile and call_index == args.profile_prompt and step == args.trace_start:
            tracer = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True, with_stack=True)
            tracer.__enter__()
            if args.python_profile:
                profile.enable()
        result = original_forward(*fa, **fk)
        step += 1
        if tracer is not None and step == args.trace_start + args.trace_steps:
            finish_trace()
        return result

    def generate(*ga, **gk):
        nonlocal step, call_index
        step = 0
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = original_generate(*ga, **gk)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if tracer is not None:
            finish_trace()
        import MoEModule.SMoE_base as sb
        records.append({'prompt': call_index, 'warmup': call_index < 0,
                        'wall_s': elapsed, 'token_ids': result.cpu().tolist(),
                        'peak_allocated': torch.cuda.max_memory_allocated(),
                        'decode_work_by_layer': dict(sb.decode_work_by_layer),
                        'peak_reserved': torch.cuda.max_memory_reserved()})
        call_index += 1
        (args.artifacts / 'generations.json').write_text(json.dumps(records, indent=2))
        return result

    model.model.forward = forward
    model.generate = generate
    return model


loader.build_model = build
main_args = args.main_args
if main_args and main_args[0] == '--':
    main_args = main_args[1:]
sys.argv = [str(ROOT / 'main.py'), *main_args]
(args.artifacts / 'invocation.json').write_text(json.dumps({
    'argv': sys.argv, 'switch_interval': sys.getswitchinterval(),
    'profile': args.profile, 'trace_start': args.trace_start,
    'python_profile': args.python_profile, 'profile_prompt': args.profile_prompt,
    'trace_steps': args.trace_steps, 'python': sys.version,
    'torch': torch.__version__, 'environment': {
        k: v for k, v in os.environ.items() if k.startswith(
            ('SMOE_', 'OMP_', 'MKL_', 'PYTORCH_', 'PYTHON_GIL', 'OPENBLAS_'))}}, indent=2))
runpy.run_path(str(ROOT / 'main.py'), run_name='__main__')
