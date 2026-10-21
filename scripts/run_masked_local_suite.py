#!/usr/bin/env python3
"""Run base and task-specific adapter evaluations on the masked test set."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


MODELS = {
    "qwen35": {
        "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "factuality_adapter": "artifacts/adapters/qwen35_lora_factuality_hf_fullpage_20261001/final_adapter",
        "bias_adapter": "artifacts/adapters/qwen35_lora_bias_runpod_20261002",
    },
    "ministral": {
        "revision": "f6fae9795746f63c9be8344932f01275f3c63734",
        "factuality_adapter": "artifacts/adapters/ministral_lora_factuality_runpod_20261002",
        "bias_adapter": "artifacts/adapters/ministral_lora_bias_hf_fullpage_20261002/final_adapter",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--dataset-root", default="data/masked_data/masked_dataset_filtered")
    parser.add_argument("--output-dir", default="artifacts/evaluations/masked-local-20261009")
    args = parser.parse_args()

    root = Path(args.dataset_root).resolve()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = MODELS[args.model]
    common = [
        sys.executable,
        "scripts/mediascope_eval.py",
        "--csv", str(root / "test.csv"),
        "--image-root", str(root),
        "--predictor", "local",
        "--model", args.model,
        "--quant", "4bit",
        "--thinking", "off",
        "--tile-height", "1400",
        "--max-tiles", "6",
        "--max-width", "1024",
        "--max-new-tokens", "16",
        "--parser", "strict",
    ]
    runs = [
        ("base", common + [
            "--base-only", "--processor-dir", config["factuality_adapter"],
            "--revision", config["revision"], "--task", "both",
            "--out", str(output / f"{args.model}_base.csv"),
        ]),
        ("factuality", common + [
            "--adapter-dir", config["factuality_adapter"], "--task", "factuality",
            "--out", str(output / f"{args.model}_finetuned_factuality.csv"),
        ]),
        ("bias", common + [
            "--adapter-dir", config["bias_adapter"], "--task", "bias",
            "--out", str(output / f"{args.model}_finetuned_bias.csv"),
        ]),
    ]
    for name, command in runs:
        audit = Path(command[-1]).with_suffix(".audit")
        if (audit / "manifest.json").is_file():
            command.append("--resume")
        print(f"[{args.model}] starting {name}", flush=True)
        subprocess.run(command, check=True)
        print(f"[{args.model}] completed {name}", flush=True)


if __name__ == "__main__":
    main()
