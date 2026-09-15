"""Reproduce the measured BF16 decode configuration on this machine.

The default uses eight CPU cores, which was the fastest 10-prompt setting in
the validation sweep.  Pass ``--cores`` explicitly for other core budgets.
"""
import os
from pathlib import Path
import sys

root=Path(__file__).resolve().parents[2]
os.environ.update(SMOE_CPU_AVX2_GEMV='0',SMOE_GPU_TRITON_EXPERT='0',
    SMOE_GPU_GROUPED_TRITON='1',SMOE_TRITON_NORM='1',SMOE_TRITON_ROPE='1',
    SMOE_CPU_BATCH_FORWARD='1',SMOE_GPU_INLINE_SUBMIT='0',SMOE_DECODE_LOAD_LIMIT='0',
    SMOE_DECODE_COST_SAMPLES='0',SMOE_MEASURE_DMA='1')
os.environ.pop('OMP_WAIT_POLICY',None)
os.environ.setdefault('TORCH_EXTENSIONS_DIR','/tmp/smoe_fullcore_extensions')
os.environ.setdefault('TRITON_CACHE_DIR','/tmp/smoe_fullcore_triton')
runner=root/'gpt_output/optimization_3080/run_wic_acceptance.py'
os.execv(sys.executable,[sys.executable,str(runner),'--cores','8','--mv',
    '--cpu-only-misses','--strict-score-order','--score-window','16',
    '--layer-cache-floor','12','--reserve-load-core',*sys.argv[1:]])
