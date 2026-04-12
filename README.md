# SMoE

SMoE inference acceleration framework with GPU/CPU expert caching for Mixture-of-Experts LLMs. SMoE leverages expert importance to guide decisions, substituting low-importance active experts with functionally similar ones already cached in GPU memory, thereby preserving accuracy. 

Supports three models: **deepseek-ai/deepseek-moe-16b-chat**, **Qwen/Qwen2-57B-A14B-Instruct**, **xverse/XVERSE-MoE-A4.2B**.

---

## Requirements

```bash
# 1. Create a Python 3.13 free-threading (no-GIL) environment
conda create -n SMoE python=3.13 python-freethreading -c conda-forge
conda activate SMoE

# 2. Install Rust toolchain and build tokenizers from source (requires sudo)
cd ./SMoE && bash dependency.sh

# 3. Install remaining Python dependencies
pip install -r requirements.txt
```

> **Why `dependency.sh`?**
> The standard `tokenizers` wheel does not support Python 3.13's free-threading (no-GIL) build.
> `dependency.sh` compiles and installs the `tokenizers` library directly from source using the local Rust toolchain, ensuring full compatibility.
>
> - **Root privileges** — `dependency.sh` must be run with `sudo`, as it installs system packages and manages the Rust toolchain via `snap`.
> - **Ubuntu only** — `dependency.sh` relies on `apt` and `snap` for package management. Users on other Linux distributions can easily adapt the script with minor modifications.
---

## Quick Start

### Run with `run.sh`

```bash
# Qwen2-MoE
MODEL_NAME=qwenmoe \
MODEL_PATH=parameters/qwenmoe \
CONFIG_PATH=configs/qwen2moe_config.json \
bash run.sh

# DeepSeek-MoE
MODEL_NAME=deepseekmoe \
MODEL_PATH=parameters/deepseekmoe \
CONFIG_PATH=configs/deepseekmoe_config.json \
bash run.sh

# Xverse-MoE
MODEL_NAME=xversemoe \
MODEL_PATH=parameters/xversemoe \
CONFIG_PATH=configs/xversemoe_config.json \
bash run.sh
```

### Run with `main.py` directly

```bash
python main.py \
    --model_name   qwenmoe \
    --model_path   parameters/qwenmoe \
    --config_path  configs/qwen2moe_config.json \
    --dataset_path gaokao_math_ii \
    --input_num    20 \
    --output_len   100 \
    --cpu_cores    3 \
    --GPU_mem      24
```

---

## All Arguments

| Argument | Default | Description |
|---|---|---|
| `--model_name` | `qwenmoe` | Model to run: `qwenmoe` \| `deepseekmoe` \| `xversemoe` |
| `--model_path` | *(empty)* | Path to model weights directory. If empty, uses the hardcoded default in `main.py` |
| `--config_path` | *(empty)* | Path to SMoE config JSON (see `configs/`). If empty, falls back to `config.json` inside model directory |
| `--dataset_path` | `gaokao_math_ii` | Dataset name or path passed to `utils/load_dataset.py` |
| `--input_num` | `20` | Number of prompts to run |
| `--batch_size` | `1` | Batch size per forward pass |
| `--output_len` | `100` | Max new tokens to generate per prompt |
| `--GPU_mem` | `45` | GPU memory in GB, used to compute expert cache offload size |
| `--cpu_cores` | `3` | Number of CPU cores allocated to inference (n-1 for compute, 1 for loading/bg worker) |
| `--debug` | `False` | Enable debugpy remote debugger on port 9501 |

## Auto-Download

**Models and datasets are downloaded automatically** if not found locally.

- Model weights are saved to `parameters/<model_name>/` inside the project directory.
- Datasets are saved to `datasets/` inside the project directory.

Simply set `--model_path` to the corresponding folder under `parameters/`:

| Model | `--model_path` |
|---|---|
| Qwen2-MoE | `parameters/qwenmoe` |
| DeepSeek-MoE | `parameters/deepseekmoe` |
| Xverse-MoE | `parameters/xversemoe` |

If the folder is empty or does not exist, the weights will be downloaded automatically from HuggingFace before inference starts.

> **Note:** Model weights are large (16B+ parameters). Make sure you have sufficient disk space and a stable network connection before the first run.

---
## SMoE Config JSON

Each model has a config JSON under `configs/`:

Key SMoE-specific fields:

| Field | Description |
|---|---|
| `replaceScoreRatio` | Ratio about experts replaced|
| `window_size` | SCore-eviction cache method window size (`null` = LRU) |
| `if_prefetch` | Enable background prefetch prediction |
| `if_usecpu` | Enable CPU fallback for cache-miss experts |
| `if_replace` | Enable expert cache replacement |

The expert cache size (number of experts kept on GPU) is set via `--GPU_mem` in `main.py` / `GPU_MEM` in `run.sh` and is computed automatically inside `build_model()`.

---

### `run.sh` environment variables

All `run.sh` arguments are controlled via environment variables (same names, uppercase):

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `qwenmoe` | Same as `--model_name` |
| `MODEL_PATH` | *(empty)* | Same as `--model_path` |
| `CONFIG_PATH` | *(empty)* | Same as `--config_path` |
| `DATASET_PATH` | `gaokao_math_ii` | Same as `--dataset_path` |
| `INPUT_NUM` | `20` | Same as `--input_num` |
| `BATCH_SIZE` | `1` | Same as `--batch_size` |
| `OUTPUT_LEN` | `100` | Same as `--output_len` |
| `GPU_MEM` | `45` | Same as `--GPU_mem` |
| `CPU_CORES` | `3` | Same as `--cpu_cores` |
| `LOG_LEVEL` | `INFO` | Logging level: `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` |
| `LOG_DIR` | `./logs` | Directory for log files |
| `CONDA_ENV` | `SMoE` | Conda environment to activate |

---

## Accuracy Evaluation

We provide OpenCompass-based benchmark evaluation for DeepSeek MoE, Xverse MoE, and Qwen MoE models.

Supported benchmarks: GaokaoBench, GSM8K, RACE, TriviaQA, WiC.

See [`opencompass_test/README.md`](opencompass_test/README.md) for full setup and usage instructions.

