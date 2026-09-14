#!/usr/bin/env python3
"""Run the qwen.csv workload matrix on local DeepSeekMoE and summarize logs."""

import argparse
import csv
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
from datetime import datetime


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = Path("/root/SMoE_3060_docs/reference/qwen.csv")
RESULTS = Path("/root/SMoE_3060_docs/results")
MODEL = Path("/root/models/parameters/deepseekmoe")
CONFIG = ROOT / "configs/deepseekmoe_config.json"
INPUT_NUM = 20
FIELDS = (
    "dataset", "cpu_cores", "input_num", "output_len", "prefetch",
    "complete", "returncode", "prompt_count", "mean_avg_decode_s",
    "stdev_avg_decode_s", "mean_prefill_s", "mean_ttft_model_forward_s",
    "mean_total_s", "mean_cpu_expert_ms", "affinity",
)
E2E = re.compile(
    r"\[SMoE\] prompt=(\d+)\s+prefill=([0-9.]+) s\s+"
    r"avg_decode=([0-9.]+) s\s+total=([0-9.]+) s\s+decode_tokens=(\d+)"
)
CPU = re.compile(r"\[CPU expert\] prompt=(\d+) decode_forward_mean=([0-9.]+) ms")


def matrix():
    with REFERENCE.open(newline="") as f:
        records = list(csv.DictReader(f, delimiter="\t"))
    pairs = [(r["dataset"], int(r["cpu_cores"])) for r in records]
    if len(pairs) != 24 or len(set(pairs)) != 24:
        raise ValueError("Expected 24 distinct dataset/core pairs in qwen.csv")
    if {core for _, core in pairs} != {3, 8, 16}:
        raise ValueError("Expected CPU core settings 3, 8, 16")
    # The repository also supports race_high, which is absent from the Qwen
    # reference. Include every named dataset while retaining its CSV schema.
    from utils.load_dataset import _DATASET_MAP
    for core in (3, 8, 16):
        for dataset in _DATASET_MAP:
            if (dataset, core) not in pairs:
                pairs.append((dataset, core))
    if len(pairs) != 3 * len(_DATASET_MAP):
        raise ValueError("Dataset/core matrix does not cover all named datasets")
    return pairs


def metrics(dataset, cores, log, returncode):
    content = log.read_text(errors="replace") if log.exists() else ""
    e2e = E2E.findall(content)
    cpu = CPU.findall(content)
    prompt_ids = {int(row[0]) for row in e2e}
    config_ok = "'if_prefetch': False" in content
    affinity = next((line.strip() for line in content.splitlines()
                     if line.startswith("[AFFINITY]")), "")
    complete = returncode == 0 and len(prompt_ids) == INPUT_NUM and config_ok
    def mean(values):
        return round(statistics.fmean(values), 9) if values else ""
    decode = [float(row[2]) for row in e2e]
    prefill = [float(row[1]) for row in e2e]
    total = [float(row[3]) for row in e2e]
    cpu_ms = [float(row[1]) for row in cpu]
    return {
        "dataset": dataset, "cpu_cores": cores, "input_num": INPUT_NUM,
        "output_len": 100, "prefetch": False, "complete": complete,
        "returncode": returncode if returncode is not None else "",
        "prompt_count": len(prompt_ids), "mean_avg_decode_s": mean(decode),
        "stdev_avg_decode_s": round(statistics.stdev(decode), 9)
        if len(decode) > 1 else "", "mean_prefill_s": mean(prefill),
        "mean_ttft_model_forward_s": mean(prefill), "mean_total_s": mean(total),
        "mean_cpu_expert_ms": mean(cpu_ms), "affinity": affinity,
    }


def write_outputs(outdir, pairs):
    rows = []
    for dataset, cores in pairs:
        stem = f"deepseekmoe_{dataset}_cpu{cores}"
        rc_path = outdir / "logs" / f"{stem}.returncode"
        rc = int(rc_path.read_text().strip()) if rc_path.exists() else None
        rows.append(metrics(dataset, cores, outdir / "logs" / f"{stem}.log", rc))
    csv_path = outdir / "deepseek.csv"
    tmp = csv_path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(csv_path)

    finished = [r for r in rows if r["complete"]]
    failed = [r for r in rows if r["returncode"] != "" and not r["complete"]]
    with REFERENCE.open(newline="") as f:
        qwen = list(csv.DictReader(f, delimiter="\t"))
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    lines = [
        "# DeepSeekMoE experiment results", "",
        f"Updated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Revision: `{revision}`", "",
        "Model: `/root/models/parameters/deepseekmoe`; GPU budget: 10 GiB; "
        f"input_num={INPUT_NUM}; output_len=100; prefetch=False; SMoE environment.",
        "The reference CSV provides eight datasets; race_high is added "
        "to cover every dataset supported by this repository.", "",
        f"Completed: {len(finished)}/{len(pairs)}; "
        f"failed: {len(failed)}/{len(pairs)}.", "",
        "| CPU cores | DeepSeek completed | DeepSeek decode (s/token) | "
        "Qwen reference decode (s/token) | DeepSeek prefill (s) | DeepSeek total (s) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cores in (3, 8, 16):
        group = [r for r in finished if r["cpu_cores"] == cores]
        qwen_group = [r for r in qwen if int(r["cpu_cores"]) == cores]
        def avg(key):
            return f"{statistics.fmean(float(r[key]) for r in group):.6f}" if group else "—"
        qwen_decode = statistics.fmean(float(r["mean_avg_decode_s"])
                                        for r in qwen_group)
        lines.append(f"| {cores} | {len(group)}/{len(pairs)//3} | "
                     f"{avg('mean_avg_decode_s')} | "
                     f"{qwen_decode:.6f} | {avg('mean_prefill_s')} | "
                     f"{avg('mean_total_s')} |")
    lines += ["", "Qwen figures come from another machine and use 100 prompts "
              "per dataset. They are workload context, not a controlled "
              "speed comparison."]
    if failed:
        lines += ["", "Failed runs (inspect corresponding raw logs):", ""]
        lines += [f"- {r['dataset']}, {r['cpu_cores']} cores: returncode "
                  f"{r['returncode']}, {r['prompt_count']}/{INPUT_NUM} prompts"
                  for r in failed]
    lines += ["", "Per-run metrics: `deepseek.csv`; raw logs: `logs/`.", ""]
    (outdir / "summary.md").write_text("\n".join(lines))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--only", help="Dataset for a focused run")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    if config.get("if_prefetch") is not False:
        raise ValueError("DeepSeek config must disable prefetch")
    if not MODEL.joinpath("model.safetensors.index.json").exists():
        raise FileNotFoundError(MODEL)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run from a GPU-enabled environment")

    pairs = matrix()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "logs").mkdir(exist_ok=True)
    write_outputs(args.results_dir, pairs)
    with REFERENCE.open(newline="") as f:
        reference_rows = list(csv.DictReader(f, delimiter="\t"))
    # Shorter reference workloads finish first, giving an early complete row
    # while preserving qwen.csv's original row order in the output matrix.
    reference_total = {(r["dataset"], int(r["cpu_cores"])):
                       float(r["mean_total_s"]) for r in reference_rows}
    run_order = sorted(pairs, key=lambda pair: reference_total.get(pair, float("inf")))
    for dataset, cores in run_order:
        if args.only and dataset != args.only:
            continue
        stem = f"deepseekmoe_{dataset}_cpu{cores}"
        log = args.results_dir / "logs" / f"{stem}.log"
        rc_path = args.results_dir / "logs" / f"{stem}.returncode"
        if rc_path.exists() and metrics(dataset, cores, log,
                                        int(rc_path.read_text().strip()))["complete"]:
            continue
        cmd = [sys.executable, "main.py", "--model_name", "deepseekmoe",
               "--model_path", str(MODEL), "--config_path", str(CONFIG),
               "--dataset_path", dataset, "--input_num", str(INPUT_NUM),
               "--batch_size", "1", "--output_len", "100",
               "--GPU_mem", "10", "--cpu_cores", str(cores)]
        env = os.environ.copy()
        env.update(SMOE_LOG_LEVEL="INFO", PYTHONUNBUFFERED="1")
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        print(f"START {stem}", flush=True)
        rc_path.unlink(missing_ok=True)
        with log.open("w") as f:
            f.write("[EXPERIMENT] command=" + " ".join(cmd) + "\n")
            f.flush()
            rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f,
                                stderr=subprocess.STDOUT).returncode
        rc_path.write_text(str(rc) + "\n")
        rows = write_outputs(args.results_dir, pairs)
        row = next(r for r in rows if r["dataset"] == dataset and r["cpu_cores"] == cores)
        print(f"DONE {stem}: rc={rc} prompts={row['prompt_count']} "
              f"complete={row['complete']}", flush=True)
        if not row["complete"]:
            raise RuntimeError(f"Incomplete run: {stem}; see {log}")


if __name__ == "__main__":
    main()
