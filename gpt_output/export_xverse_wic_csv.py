#!/usr/bin/env python3
"""Export the completed Xverse/WiC matrix using the formal result schema."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import re
import statistics
import subprocess


FIELDS = (
    "dataset cpu_cores input_num output_len prefetch complete returncode "
    "prompt_count mean_avg_decode_s stdev_avg_decode_s mean_prefill_s "
    "mean_ttft_model_forward_s mean_total_s mean_cpu_expert_ms affinity "
    "model commit warmup_num warmup_prompt_count gpu_mem_gib input_loader_mode "
    "max_input_tokens skipped_long_prompts decode_metric_prompt_count "
    "zero_decode_token_prompt_count zero_decode_token_prompt_ids "
    "weighted_avg_decode_s mean_generate_minus_prefill_per_decode_s decode_tokens "
    "cpu_expert_forward_scope cpu_expert_forward_mean_ms "
    "cpu_expert_forward_total_ms cpu_expert_forward_calls "
    "cpu_expert_forward_ms_per_decode_token cpu_stage_ms_per_decode_token "
    "cpu_batch_boundary_ms_per_decode_token gpu_cache_hit_rate gpu_cache_hits "
    "routed_expert_calls demand_loads if_replace replace_score_ratio score_window "
    "layer_cache_floor cpu_only_misses gpu_grouped_triton cpu_batch_forward "
    "triton_decode_norm triton_decode_rope sampled_driver_peak_mib "
    "sampled_process_peak_rss_kib peak_allocated_bytes peak_reserved_bytes "
    "excluded_prompts elapsed_wall_s started_at finished_at"
).split()

AFFINITY_RE = re.compile(r"^(\[AFFINITY\].*)$", re.MULTILINE)
DMA_RE = re.compile(r"\[DMA timing\] completed_copies=(\d+)")
START_RE = re.compile(r"^\[MATRIX\] started=(.+)$", re.MULTILINE)
FINISH_RE = re.compile(r"^\[MATRIX\] finished=(.+)$", re.MULTILINE)
WARMUP_RE = re.compile(r"^\[RUN\] phase=warmup ", re.MULTILINE)


def bool_csv(value: bool) -> str:
    return "TRUE" if value else "FALSE"


def timestamp(value: str) -> tuple[datetime, str]:
    parsed = datetime.fromisoformat(value)
    return parsed, parsed.strftime("%Y-%m-%dT%H:%M:%S%z")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        default="/root/SMoE_4060_results/origin/4060ti_697e81c",
    )
    parser.add_argument("--output-name", default="xverse_wic_results.csv")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    payload = json.loads((result_dir / "summary.json").read_text())
    manifest = payload["manifest"]
    config = json.loads((result_dir / "xversemoe_config.json").read_text())
    rows = []

    for run in payload["runs"]:
        cpu_cores = run["cpu_cores"]
        metrics = run.get("metrics", {})
        prompts = run.get("prompts", [])
        log_text = (result_dir / "logs" / f"cpu{cpu_cores}.log").read_text(
            errors="replace"
        )

        affinity_match = AFFINITY_RE.search(log_text)
        start_match = START_RE.search(log_text)
        finish_match = FINISH_RE.search(log_text)
        if not (affinity_match and start_match and finish_match):
            raise ValueError(f"Missing run metadata in CPU={cpu_cores} log")
        started, started_text = timestamp(start_match[1])
        finished, finished_text = timestamp(finish_match[1])

        decode_tokens = sum(row["decode_tokens"] for row in prompts)
        weighted_decode = (
            sum(row["decode_s_per_token"] * row["decode_tokens"] for row in prompts)
            / decode_tokens
        )
        generate_minus_prefill = statistics.fmean(
            (row["total_s"] - row["prefill_s"]) / row["decode_tokens"]
            for row in prompts if row["decode_tokens"] > 0
        )
        zero_prompt_ids = [
            row["prompt"] for row in prompts if row["decode_tokens"] == 0
        ]

        dma_counts = [int(value) for value in DMA_RE.findall(log_text)]
        # The first prompt is warmup. With prefetch disabled, every subsequent
        # expert DMA is a measured demand load.
        demand_loads = dma_counts[-1] - dma_counts[0] if len(dma_counts) >= 2 else ""
        routed_calls = metrics.get("aggregate_gpu_total", "")
        gpu_hits = metrics.get("aggregate_gpu_hits", "")
        cpu_forward_calls = ""
        if all(isinstance(value, int) for value in (routed_calls, gpu_hits, demand_loads)):
            cpu_forward_calls = routed_calls - gpu_hits - demand_loads
            if cpu_forward_calls < 0:
                raise ValueError(f"Negative CPU forward calls for CPU={cpu_cores}")

        row = {field: "" for field in FIELDS}
        row.update({
            "dataset": manifest["dataset"],
            "cpu_cores": cpu_cores,
            "input_num": manifest["input_num"],
            "output_len": manifest["output_len"],
            "prefetch": bool_csv(manifest["if_prefetch"]),
            "complete": bool_csv(run["complete"]),
            "returncode": run["returncode"],
            "prompt_count": run["measured_prompt_count"],
            "mean_avg_decode_s": metrics["decode_s_per_token"]["mean"],
            "stdev_avg_decode_s": metrics["decode_s_per_token"]["stdev"],
            "mean_prefill_s": metrics["prefill_s"]["mean"],
            "mean_ttft_model_forward_s": metrics["prefill_s"]["mean"],
            "mean_total_s": metrics["total_s"]["mean"],
            "mean_cpu_expert_ms": metrics["cpu_forward_mean_ms"]["mean"],
            "affinity": affinity_match[1],
            "model": manifest["model"],
            "commit": manifest["requested_commit"],
            "warmup_num": manifest["warmup_num"],
            "warmup_prompt_count": len(WARMUP_RE.findall(log_text)),
            "gpu_mem_gib": manifest["gpu_mem_gb"],
            "input_loader_mode": "load_all",
            "max_input_tokens": manifest["input_len"],
            "skipped_long_prompts": 0,
            "decode_metric_prompt_count": len(prompts) - len(zero_prompt_ids),
            "zero_decode_token_prompt_count": len(zero_prompt_ids),
            "zero_decode_token_prompt_ids": json.dumps(zero_prompt_ids),
            "weighted_avg_decode_s": weighted_decode,
            "mean_generate_minus_prefill_per_decode_s": generate_minus_prefill,
            "decode_tokens": decode_tokens,
            "cpu_expert_forward_scope": "per_decode_token_mean_of_cpu_expert_forwards",
            "cpu_expert_forward_mean_ms": metrics["cpu_forward_mean_ms"]["mean"],
            # The current branch logs a per-token mean but not every call's
            # duration, so total_ms and ms_per_decode_token stay intentionally blank.
            "cpu_expert_forward_calls": cpu_forward_calls,
            "gpu_cache_hit_rate": metrics["aggregate_gpu_hit_rate"],
            "gpu_cache_hits": gpu_hits,
            "routed_expert_calls": routed_calls,
            "demand_loads": demand_loads,
            "if_replace": bool_csv(config["if_replace"]),
            "replace_score_ratio": config["replaceScoreRatio"],
            "score_window": "" if config["window_size"] is None else config["window_size"],
            # This implementation has one global cache rather than a per-layer floor.
            "layer_cache_floor": "",
            "cpu_only_misses": bool_csv(config["if_usecpu"]),
            "gpu_grouped_triton": "FALSE",
            "cpu_batch_forward": "FALSE",
            "triton_decode_norm": "FALSE",
            "triton_decode_rope": "FALSE",
            "peak_allocated_bytes": metrics["peak_allocated_bytes"],
            "peak_reserved_bytes": metrics["peak_reserved_bytes"],
            "excluded_prompts": "[]",
            "elapsed_wall_s": (finished - started).total_seconds(),
            "started_at": started_text,
            "finished_at": finished_text,
        })
        rows.append(row)

    output_path = result_dir / args.output_name
    with output_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    # Keep the archived diff aligned with all post-processing code used.
    root = Path(__file__).resolve().parents[1]
    diff = subprocess.run(
        ["git", "diff", "--binary", manifest["requested_commit"]],
        cwd=root, check=True, text=True, stdout=subprocess.PIPE,
    ).stdout
    (result_dir / "source_diff.patch").write_text(diff)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
