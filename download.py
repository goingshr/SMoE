import argparse
import os
import json
import sys

ROOT        = os.path.dirname(os.path.abspath(__file__))
PARAMS_DIR  = os.path.join(ROOT, "parameters")
DATASET_DIR = os.path.join(ROOT, "datasets")

# ── Model configs ──────────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "deepseekmoe": {
        "repo_id":   "deepseek-ai/deepseek-moe-16b-chat",
        "local_dir": os.path.join(PARAMS_DIR, "deepseekmoe"),
    },
    "qwenmoe": {
        "repo_id":   "Qwen/Qwen2-57B-A14B-Instruct",
        "local_dir": os.path.join(PARAMS_DIR, "qwenmoe"),
    },
    "xversemoe": {
        "repo_id":   "xverse/XVERSE-MoE-A4.2B",
        "local_dir": os.path.join(PARAMS_DIR, "xversemoe"),
    },
}

# ── Dataset configs ────────────────────────────────────────────────────────────
# keyword: matched against --dataset_path (consistent with load_dataset.py path.lower() logic)
DATASET_CONFIGS = {
    "wic": {
        "local_path": os.path.join(DATASET_DIR, "SuperGLUE", "WiC", "val.jsonl"),
        "hf_dataset": ("super_glue", "wic", "validation"),
        "hf_fields":  {"sentence1": "sentence1", "sentence2": "sentence2", "word": "word"},
    },
    "gsm8k": {
        "local_path": os.path.join(DATASET_DIR, "gsm8k", "train.jsonl"),
        "hf_dataset": ("gsm8k", "main", "train"),
        "hf_fields":  {"question": "question", "answer": "answer"},
    },
    "triviaqa": {
        "local_path": os.path.join(DATASET_DIR, "triviaqa", "triviaqa-train.jsonl"),
        "hf_dataset": ("trivia_qa", "rc", "train"),
        "hf_fields":  {"question": "question"},
    },
    "race_middle": {
        "local_path": os.path.join(DATASET_DIR, "race", "validation", "middle.jsonl"),
        "hf_dataset": ("race", "middle", "validation"),
        "hf_fields":  {"article": "article", "question": "question",
                       "options": "options", "answer": "answer"},
    },
    "race_high": {
        "local_path": os.path.join(DATASET_DIR, "race", "validation", "high.jsonl"),
        "hf_dataset": ("race", "high", "validation"),
        "hf_fields":  {"article": "article", "question": "question",
                       "options": "options", "answer": "answer"},
    },
}


# ── Internal helpers ───────────────────────────────────────────────────────────

def _model_exists(local_dir: str) -> bool:
    """Return True if the model directory is non-empty (already downloaded)."""
    return os.path.isdir(local_dir) and bool(os.listdir(local_dir))


def _dataset_exists(local_path: str) -> bool:
    return os.path.isfile(local_path)


# ── Model download ─────────────────────────────────────────────────────────────

def download_model(name: str) -> str:
    """Download the specified model to parameters/<name> and return the local path."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    cfg       = MODEL_CONFIGS[name]
    repo_id   = cfg["repo_id"]
    local_dir = cfg["local_dir"]

    if _model_exists(local_dir):
        print(f"[skip] {name} already exists at {local_dir}")
        return local_dir

    os.makedirs(local_dir, exist_ok=True)
    print(f"[download] {name}  repo={repo_id}  ->  {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
    )
    print(f"[done] {name}")
    return local_dir


def ensure_model(name: str, user_path: str = "") -> str:
    """
    Called by main.py.
    If user_path exists locally, return it directly;
    otherwise auto-download to parameters/<name> and return the path.
    """
    if user_path and _model_exists(user_path):
        return user_path
    if name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(MODEL_CONFIGS)}")
    local_dir = MODEL_CONFIGS[name]["local_dir"]
    if _model_exists(local_dir):
        return local_dir
    print(f"[auto-download] model '{name}' not found locally, downloading from HuggingFace...")
    return download_model(name)


# ── Dataset download ───────────────────────────────────────────────────────────

def download_dataset(name: str) -> str:
    """Download the specified dataset to datasets/ and return the local file path."""
    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        print("ERROR: 'datasets' library not installed. Run: pip install datasets")
        sys.exit(1)

    cfg        = DATASET_CONFIGS[name]
    local_path = cfg["local_path"]

    if _dataset_exists(local_path):
        print(f"[skip] {name} already exists at {local_path}")
        return local_path

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    ds_name, ds_config, ds_split = cfg["hf_dataset"]
    fields = cfg["hf_fields"]

    print(f"[download] dataset={name}  hf={ds_name}/{ds_config}/{ds_split}  ->  {local_path}")
    ds = hf_load(ds_name, ds_config, split=ds_split)

    with open(local_path, "w", encoding="utf-8") as f:
        for row in ds:
            record = {k: row[v] for k, v in fields.items() if v in row}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[done] {name}  ({len(ds)} records -> {local_path})")
    return local_path


def ensure_dataset(keyword: str) -> str:
    """
    Called by load_dataset.py.
    Find the dataset config matching keyword (e.g. 'wic', 'gsm8k'),
    auto-download if not present locally, and return the local file path.
    """
    kw = keyword.lower()
    for name, cfg in DATASET_CONFIGS.items():
        if kw in name.lower() or kw in cfg["local_path"].lower():
            local_path = cfg["local_path"]
            if not _dataset_exists(local_path):
                print(f"[auto-download] dataset '{keyword}' not found locally, downloading...")
                download_dataset(name)
            return local_path
    raise FileNotFoundError(
        f"Dataset '{keyword}' not found in DATASET_CONFIGS and cannot be auto-downloaded. "
        f"Available: {list(DATASET_CONFIGS)}"
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download models/datasets for SMoEaligned.")
    parser.add_argument("--all",      action="store_true",
                        help="Download all models and datasets")
    parser.add_argument("--models",   nargs="+", metavar="NAME",
                        help=f"Models: all | {' | '.join(MODEL_CONFIGS)}")
    parser.add_argument("--datasets", nargs="+", metavar="NAME",
                        help=f"Datasets: all | {' | '.join(DATASET_CONFIGS)}")
    args = parser.parse_args()

    if not any([args.all, args.models, args.datasets]):
        parser.print_help()
        sys.exit(0)

    model_targets = (list(MODEL_CONFIGS) if (args.all or (args.models and "all" in args.models))
                     else (args.models or []))
    for name in model_targets:
        if name not in MODEL_CONFIGS:
            print(f"ERROR: unknown model '{name}'. Choose from: {list(MODEL_CONFIGS)}")
            sys.exit(1)
        download_model(name)

    dataset_targets = (list(DATASET_CONFIGS) if (args.all or (args.datasets and "all" in args.datasets))
                       else (args.datasets or []))
    for name in dataset_targets:
        if name not in DATASET_CONFIGS:
            print(f"ERROR: unknown dataset '{name}'. Choose from: {list(DATASET_CONFIGS)}")
            sys.exit(1)
        download_dataset(name)


if __name__ == "__main__":
    main()
