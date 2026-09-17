# Source before invoking main.py or run_xverse_optimized.py.
# All weights remain BF16; replaceScoreRatio stays at 0.25 in the config.
export PYTHON_GIL=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SMOE_COMPACT_PINNED=1
export SMOE_DECODE_METRICS=1
export SMOE_MEASURE_DMA=1
export SMOE_GPU_GROUPED_TRITON=1
export SMOE_GPU_INLINE_SUBMIT=1
export SMOE_DECODE_MINMAX=1
export SMOE_CPU_BF16_MV=1
export SMOE_TRITON_DECODE_NORM=1
export SMOE_FAST_COST_AVERAGE=1
export SMOE_TRITON_DECODE_ROPE=1
export SMOE_XVERSE_SHARED_GRAPH=1
export SMOE_GPU_TRITON_EXPERT=1
export SMOE_COMPACT_CAUSAL_MASK=1
