#!/usr/bin/env bash
# =============================================================================
# run.sh — SMoEaligned one-click inference script
# Usage: bash run.sh
#
# Examples:
#   # Qwen2-MoE (default)
#   bash run.sh --model_name qwenmoe --model_path /path/to/qwen2_moe
#
#   # DeepSeek-MoE
#   bash run.sh --model_name deepseekmoe --model_path /path/to/deepseekmoe
#
#   # Xverse-MoE
#   bash run.sh --model_name xversemoe --model_path /path/to/xversemoe
# =============================================================================
set -euo pipefail

# ── User-configurable parameters ──────────────────────────────────────────────

MODEL_NAME="${MODEL_NAME:-qwenmoe}"           # deepseekmoe | qwenmoe | xversemoe
MODEL_PATH="${MODEL_PATH:-}"                   # Model weights directory (empty = auto-download)
CONFIG_PATH="${CONFIG_PATH:-}"                 # SMoE config.json path (empty = use default in model dir)
DATASET_PATH="${DATASET_PATH:-wic}"            # Dataset name or path
INPUT_NUM="${INPUT_NUM:-20}"                   # Number of inference samples
BATCH_SIZE="${BATCH_SIZE:-1}"                  # Batch size
OUTPUT_LEN="${OUTPUT_LEN:-100}"                # Max new tokens per prompt
GPU_MEM="${GPU_MEM:-24}"                       # GPU memory in GB, affects cache_size
CPU_CORES="${CPU_CORES:-16}"                   # Number of CPU cores allocated to inference

# ── Logging settings ───────────────────────────────────────────────────────────
# LOG_LEVEL: DEBUG | INFO | WARNING | ERROR
# INFO: shows expert hit rate, prefill time, decoding time (logger.info level)
# WARNING: shows final results only (quieter)
LOG_LEVEL="${LOG_LEVEL:-INFO}"

# Log output: written to both terminal and file
LOG_DIR="${LOG_DIR:-./logs}"
mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${MODEL_NAME}_${TIMESTAMP}.log"

# ── Conda environment ──────────────────────────────────────────────────────────
CONDA_ENV="${CONDA_ENV:-SMoE}"
# Skip activation if already in the correct environment
if [[ "$(conda info --envs 2>/dev/null | grep '*' | awk '{print $1}')" != "${CONDA_ENV}" ]]; then
    echo "[run.sh] Activating conda env: ${CONDA_ENV}"
    # shellcheck disable=SC1090
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
fi

# ── Working directory ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ── Build command ──────────────────────────────────────────────────────────────
CMD=(python main.py
    --model_name  "${MODEL_NAME}"
    --input_num   "${INPUT_NUM}"
    --batch_size  "${BATCH_SIZE}"
    --output_len  "${OUTPUT_LEN}"
    --GPU_mem     "${GPU_MEM}"
    --cpu_cores   "${CPU_CORES}"
    --dataset_path "${DATASET_PATH}"
)

if [[ -n "${MODEL_PATH}" ]]; then
    CMD+=(--model_path "${MODEL_PATH}")
fi
if [[ -n "${CONFIG_PATH}" ]]; then
    CMD+=(--config_path "${CONFIG_PATH}")
fi

# ── Log level (passed to main.py via environment variable) ─────────────────────
export SMOE_LOG_LEVEL="${LOG_LEVEL}"

# ── Print run info ─────────────────────────────────────────────────────────────
echo "============================================================"
echo "  SMoEaligned Inference"
echo "  Model:      ${MODEL_NAME}"
echo "  Weights:    ${MODEL_PATH:-(auto-download to parameters/)}"
echo "  Log level:  ${LOG_LEVEL}"
echo "  Log file:   ${LOG_FILE}"
echo "  Command:    ${CMD[*]}"
echo "============================================================"

# ── Execute, tee stdout+stderr to terminal and log file ────────────────────────
"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"

EXIT_CODE="${PIPESTATUS[0]}"
if [[ "${EXIT_CODE}" -eq 0 ]]; then
    echo ""
    echo "[run.sh] Done. Log saved to: ${LOG_FILE}"
else
    echo ""
    echo "[run.sh] Exit code: ${EXIT_CODE}. See log: ${LOG_FILE}"
    exit "${EXIT_CODE}"
fi
