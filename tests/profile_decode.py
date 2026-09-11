"""Capture five real decode forwards after ten warmup tokens (diagnostic only).

Run from the repository with PYTHONPATH=. and SMOE_PROFILE_TRACE set, passing
the usual main.py CLI arguments. Profiled latency is not an acceptance result.
"""
import os
from pathlib import Path
import runpy
import torch
import utils.model_loader as loader
import utils.expertcache as ec

build_model = loader.build_model


def build_profiled_model(*args, **kwargs):
    model = build_model(*args, **kwargs)
    target = Path(os.environ['SMOE_PROFILE_TRACE'])
    state = dict(profiler=None, done=False)

    def before(module, inputs):
        if not state['done'] and ec.tokens == 10:
            torch.cuda.synchronize()
            state['profiler'] = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True)
            state['profiler'].__enter__()

    def after(module, inputs, output):
        if state['profiler'] is not None and ec.tokens == 15:
            torch.cuda.synchronize()
            prof = state['profiler']
            prof.__exit__(None, None, None)
            target.parent.mkdir(parents=True, exist_ok=True)
            prof.export_chrome_trace(str(target))
            target.with_suffix('.operators.txt').write_text(
                prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=50))
            state.update(profiler=None, done=True)
            print(f'[profile] trace={target}', flush=True)

    model.model.register_forward_pre_hook(before)
    model.model.register_forward_hook(after)
    return model


loader.build_model = build_profiled_model
runpy.run_path(str(Path(__file__).resolve().parents[1] / 'main.py'), run_name='__main__')
