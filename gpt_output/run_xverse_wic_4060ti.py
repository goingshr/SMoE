#!/usr/bin/env python3
"""Run and summarize the requested Xverse/WiC 4060 Ti CPU matrix."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys


CPU_RE = re.compile(
    r"\[CPU expert\] prompt=(\d+) decode_forward_mean=([0-9.]+) ms "
    r"median=([0-9.]+) ms p95=([0-9.]+) ms sampled_tokens=(\d+)"
)
SMOE_RE = re.compile(
    r"\[SMoE\] prompt=(\d+)\s+prefill=([0-9.]+) s\s+"
    r"avg_decode=([0-9.]+) s\s+total=([0-9.]+) s\s+decode_tokens=(\d+)"
)
HIT_RE = re.compile(
    r"\[GPU cache\] prompt=(\d+) decode_hit_rate=([0-9.]+) "
    r"hits=(\d+) total=(\d+)"
)
GPU_RE = re.compile(
    r"\[GPU decode\] prompt=(\d+) cache_slots=(\d+).*"
    r"peak_allocated=(\d+) peak_reserved=(\d+)"
)
INPUT_RE = re.compile(
    r"\[INPUT\] phase=(\w+) sample_id=(\d+) input_tokens=(\d+)"
)


def run_capture(command: list[str], cwd: Path) -> str:
    result = subprocess.run(
        command, cwd=cwd, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    return result.stdout


def mean_sd_cv(values: list[float]) -> dict[str, float | None]:
    avg = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "mean": avg,
        "stdev": sd,
        "cv": sd / avg if avg else None,
        "min": min(values),
        "max": max(values),
    }


def parse_log(text: str, cpu_cores: int, warmup_num: int, input_num: int) -> dict:
    records: dict[int, dict] = {}
    for match in CPU_RE.finditer(text):
        prompt = int(match[1])
        records.setdefault(prompt, {}).update({
            "cpu_forward_mean_ms": float(match[2]),
            "cpu_forward_median_ms": float(match[3]),
            "cpu_forward_p95_ms": float(match[4]),
            "cpu_sampled_tokens": int(match[5]),
        })
    for match in SMOE_RE.finditer(text):
        prompt = int(match[1])
        records.setdefault(prompt, {}).update({
            "prefill_s": float(match[2]),
            "decode_s_per_token": float(match[3]),
            "total_s": float(match[4]),
            "decode_tokens": int(match[5]),
        })
    for match in HIT_RE.finditer(text):
        prompt = int(match[1])
        records.setdefault(prompt, {}).update({
            "gpu_hit_rate": float(match[2]),
            "gpu_hits": int(match[3]),
            "gpu_total": int(match[4]),
        })
    for match in GPU_RE.finditer(text):
        prompt = int(match[1])
        records.setdefault(prompt, {}).update({
            "gpu_cache_slots": int(match[2]),
            "peak_allocated_bytes": int(match[3]),
            "peak_reserved_bytes": int(match[4]),
        })

    input_rows = [
        {"phase": match[1], "sample_id": int(match[2]), "tokens": int(match[3])}
        for match in INPUT_RE.finditer(text)
    ]
    measured_ids = list(range(warmup_num, warmup_num + input_num))
    measured = [records[prompt] | {"prompt": prompt} for prompt in measured_ids
                if prompt in records]
    complete = len(measured) == input_num and all(
        all(key in row for key in (
            "cpu_forward_mean_ms", "decode_s_per_token", "gpu_hit_rate",
            "decode_tokens", "gpu_cache_slots", "peak_allocated_bytes",
        ))
        for row in measured
    )
    summary = {
        "cpu_cores": cpu_cores,
        "complete": complete,
        "measured_prompt_count": len(measured),
        "input_token_rows": input_rows,
        "prompts": measured,
    }
    if complete:
        summary["metrics"] = {
            "decode_s_per_token": mean_sd_cv(
                [row["decode_s_per_token"] for row in measured]),
            "cpu_forward_mean_ms": mean_sd_cv(
                [row["cpu_forward_mean_ms"] for row in measured]),
            "gpu_hit_rate": mean_sd_cv(
                [row["gpu_hit_rate"] for row in measured]),
            "prefill_s": mean_sd_cv([row["prefill_s"] for row in measured]),
            "total_s": mean_sd_cv([row["total_s"] for row in measured]),
            "decode_tokens": sorted({row["decode_tokens"] for row in measured}),
            "gpu_cache_slots": sorted({row["gpu_cache_slots"] for row in measured}),
            "peak_allocated_bytes": max(
                row["peak_allocated_bytes"] for row in measured),
            "peak_reserved_bytes": max(
                row["peak_reserved_bytes"] for row in measured),
            "aggregate_gpu_hits": sum(row["gpu_hits"] for row in measured),
            "aggregate_gpu_total": sum(row["gpu_total"] for row in measured),
        }
        totals = summary["metrics"]
        totals["aggregate_gpu_hit_rate"] = (
            totals["aggregate_gpu_hits"] / totals["aggregate_gpu_total"]
        )
    return summary


def render_markdown(manifest: dict, runs: list[dict]) -> str:
    lines = [
        "# Xverse-MoE / WiC — RTX 4060 Ti 首批结果",
        "",
        f"- 基线提交：`{manifest['base_commit']}`",
        f"- 实验分支：`{manifest['branch']}`",
        f"- 配置：replaceScoreRatio={manifest['replace_score_ratio']}, "
        f"input_len={manifest['input_len']}, output_len={manifest['output_len']}, "
        f"if_prefetch={str(manifest['if_prefetch']).lower()}, GPU_MEM={manifest['gpu_mem_gb']} GiB",
        f"- 口径：{manifest['warmup_num']} 条独立 warmup，随后 "
        f"{manifest['input_num']} 条正式 WiC prompt；三组使用相同数据顺序",
        "- CPU cores 是总配额，其中 1 core 给 loading/background worker，余下用于 CPU expert compute",
        "- GPU hit-rate 仅统计 decode，已排除 prefill",
        "",
        "| CPU cores | 完成 | Decode (s/token) | CPU expert forward (ms) | GPU cache hit | Prefill (s) | Total/prompt (s) | GPU cache slots | Peak allocated (GiB) |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        if not run.get("complete"):
            lines.append(
                f"| {run['cpu_cores']} | 否 | - | - | - | - | - | - | - |"
            )
            continue
        metrics = run["metrics"]
        lines.append(
            f"| {run['cpu_cores']} | 是 | "
            f"{metrics['decode_s_per_token']['mean']:.6f} ± {metrics['decode_s_per_token']['stdev']:.6f} | "
            f"{metrics['cpu_forward_mean_ms']['mean']:.3f} ± {metrics['cpu_forward_mean_ms']['stdev']:.3f} | "
            f"{metrics['aggregate_gpu_hit_rate']:.3%} | "
            f"{metrics['prefill_s']['mean']:.4f} | {metrics['total_s']['mean']:.4f} | "
            f"{metrics['gpu_cache_slots'][0]} | "
            f"{metrics['peak_allocated_bytes'] / (1024 ** 3):.3f} |"
        )
    lines += [
        "",
        "## 文件",
        "",
        "- `logs/cpu3.log`, `logs/cpu8.log`, `logs/cpu16.log`：完整 stdout/stderr",
        "- `summary.json`：逐 prompt 原始指标与聚合值",
        "- `run_manifest.json`：命令、环境和实验参数",
        "- `source_diff.patch`：相对基线提交的代码/config 修改",
        "- `environment.txt`：GPU、CPU、Python/PyTorch 与 Git 快照",
        "",
    ]
    incomplete = [run for run in runs if not run.get("complete")]
    if incomplete:
        lines += [
            "## 异常",
            "",
            "以下组未产生完整的 5 条正式数据，请查看对应日志："
            + ", ".join(str(run["cpu_cores"]) for run in incomplete),
            "",
        ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="/root/SMoE_4060_results/origin/4060ti_697e81c",
    )
    parser.add_argument("--gpu-mem", type=float, default=14.0)
    parser.add_argument("--input-num", type=int, default=5)
    parser.add_argument("--warmup-num", type=int, default=1)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    config_path = root / "configs/xversemoe_config.json"
    config = json.loads(config_path.read_text())
    expected = {"replaceScoreRatio": 0.25, "if_prefetch": False}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"{key} must be {value!r}, got {config.get(key)!r}")

    base_commit = run_capture(["git", "rev-parse", "HEAD"], root).strip()
    branch = run_capture(["git", "branch", "--show-current"], root).strip()
    manifest = {
        "created_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
        "base_commit": base_commit,
        "requested_commit": "697e81c67a76842a4d4be1b60a33782ca0502f7d",
        "branch": branch,
        "model": "xversemoe",
        "model_path": "/root/models/xversemoe",
        "dataset": "wic",
        "replace_score_ratio": config["replaceScoreRatio"],
        "if_prefetch": config["if_prefetch"],
        "input_len": 30,
        "output_len": 100,
        "input_num": args.input_num,
        "warmup_num": args.warmup_num,
        "gpu_mem_gb": args.gpu_mem,
        "cpu_cores": [3, 8, 16],
        "commands": [],
    }

    shutil.copy2(config_path, output_dir / "xversemoe_config.json")
    (output_dir / "source_diff.patch").write_text(
        run_capture(["git", "diff", "--binary", manifest["requested_commit"]], root)
    )
    environment = "\n".join([
        "# nvidia-smi",
        run_capture(["nvidia-smi"], root),
        "# lscpu",
        run_capture(["lscpu"], root),
        "# Python / torch",
        run_capture([
            sys.executable, "-c",
            "import sys,torch,transformers; "
            "print(sys.version); print(torch.__version__); "
            "print(transformers.__version__); print(torch.cuda.get_device_name(0))",
        ], root),
        "# git status",
        run_capture(["git", "status", "--short", "--branch"], root),
    ])
    (output_dir / "environment.txt").write_text(environment)

    runs = []
    for cpu_cores in manifest["cpu_cores"]:
        command = [
            sys.executable, "main.py",
            "--model_name", "xversemoe",
            "--model_path", manifest["model_path"],
            "--config_path", str(config_path),
            "--dataset_path", "wic",
            "--input_num", str(args.input_num),
            "--warmup_num", str(args.warmup_num),
            "--input_len", "30",
            "--output_len", "100",
            "--GPU_mem", str(args.gpu_mem),
            "--cpu_cores", str(cpu_cores),
        ]
        manifest["commands"].append(command)
        log_path = logs_dir / f"cpu{cpu_cores}.log"
        env = os.environ.copy()
        env.update({
            "PYTHON_GIL": "0",
            "SMOE_LOG_LEVEL": "INFO",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        })
        with log_path.open("w", buffering=1) as log:
            log.write(f"[MATRIX] started={dt.datetime.now().astimezone().isoformat()}\n")
            log.write(f"[MATRIX] command={' '.join(command)}\n")
            process = subprocess.Popen(
                command, cwd=root, env=env, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
            returncode = process.wait()
            log.write(f"[MATRIX] returncode={returncode}\n")
            log.write(f"[MATRIX] finished={dt.datetime.now().astimezone().isoformat()}\n")

        text = log_path.read_text(errors="replace")
        result = parse_log(text, cpu_cores, args.warmup_num, args.input_num)
        result.update({"returncode": returncode, "log": str(log_path)})
        runs.append(result)
        (output_dir / "summary.json").write_text(
            json.dumps({"manifest": manifest, "runs": runs}, indent=2, ensure_ascii=False)
        )
        (output_dir / "SUMMARY.md").write_text(render_markdown(manifest, runs))
        if returncode != 0:
            break

    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )
    return 0 if len(runs) == 3 and all(run["complete"] for run in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
