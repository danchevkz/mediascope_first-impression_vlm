#!/usr/bin/env python3
"""Consolidate masked-test API, base-model, and fine-tuned metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_config import write_json


def metric(path: Path, task: str) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["tasks"][task]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-dir", default="artifacts/evaluations/api-masked-test-20261009")
    parser.add_argument("--corrected-api-dir", default="artifacts/evaluations/api-masked-test-gpt56-opus5-20261010")
    parser.add_argument("--qwen-recovery-dir", default="artifacts/evaluations/api-masked-test-qwen-recovery-20261009")
    parser.add_argument("--local-dir", default="artifacts/evaluations/masked-local-20261009")
    parser.add_argument("--output-dir", default="artifacts/audits/masked_test_comparison_20261009")
    args = parser.parse_args()

    api = json.loads((Path(args.api_dir) / "summary.json").read_text(encoding="utf-8"))["models"]
    corrected = json.loads((Path(args.corrected_api_dir) / "summary.json").read_text(encoding="utf-8"))["models"]
    recovery_path = Path(args.qwen_recovery_dir) / "summary.json"
    qwen_metrics = (json.loads(recovery_path.read_text(encoding="utf-8"))["models"]["qwen"]["metrics"]
                    if recovery_path.is_file() else api["qwen"]["metrics"])
    local = Path(args.local_dir)
    methods = {
        "GPT-5.6 Sol": corrected["gpt56"]["metrics"],
        "Claude Opus 5": corrected["opus5"]["metrics"],
        "Kimi K3": api["kimi"]["metrics"],
        "Qwen 3.8 Max": qwen_metrics,
        "Qwen3.5-4B base": {
            task: metric(local / "qwen35_base.audit/metrics.json", task) for task in ("factuality", "bias")
        },
        "Qwen3.5-4B fine-tuned": {
            "factuality": metric(local / "qwen35_finetuned_factuality.audit/metrics.json", "factuality"),
            "bias": metric(local / "qwen35_finetuned_bias.audit/metrics.json", "bias"),
        },
        "Ministral-3-8B base": {
            task: metric(local / "ministral_base.audit/metrics.json", task) for task in ("factuality", "bias")
        },
        "Ministral-3-8B fine-tuned": {
            "factuality": metric(local / "ministral_finetuned_factuality.audit/metrics.json", "factuality"),
            "bias": metric(local / "ministral_finetuned_bias.audit/metrics.json", "bias"),
        },
    }
    result = {"dataset_records": 133, "methods": methods,
              "warning": "Qwen 3.8 Max recovery is complete, but Alibaba content filtering reduced valid output coverage."}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "analysis.json", result)

    lines = [
        "# Masked test comparison",
        "",
        "| Method | Factuality accuracy | Bias accuracy | Fact valid | Bias valid |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, tasks in methods.items():
        fact, bias = tasks["factuality"], tasks["bias"]
        lines.append(
            f"| {name} | {100*fact['accuracy']:.2f} | {100*bias['accuracy']:.2f} | "
            f"{100*fact['valid_output_rate']:.1f}% | {100*bias['valid_output_rate']:.1f}% |"
        )
    lines += [
        "",
        "Qwen 3.8 Max recovery is complete, but provider filtering leaves lower valid coverage; its metrics count filtered outputs as incorrect.",
        "The masked set is a provided 133-image test-only subset. Without a mapping to the exact unmasked counterparts, cross-dataset score differences are not causal masking effects.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
