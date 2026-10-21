#!/usr/bin/env python3
"""Evaluate uniform-random, majority, and ordinal-midpoint baselines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from mediascope_eval import TASKS, evaluate


MIDPOINT = {"factuality": "MIXED", "bias": "LEAST BIASED"}
METRICS = ("macro_precision", "macro_recall", "macro_f1", "mae", "mse", "accuracy")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_labels(path: Path, task: str) -> list[str]:
    allowed = TASKS[task]["order"]
    key = f"true_label_{task}"
    with path.open(newline="", encoding="utf-8") as handle:
        return [row[key] for row in csv.DictReader(handle) if row[key] in allowed]


def selected(metrics: dict) -> dict:
    return {key: metrics[key] for key in METRICS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        default="artifacts/datasets/mediascope-clustered-prepared-ea1f7ec37e90",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-repetitions", type=int, default=10_000)
    parser.add_argument("--out", default="results/naive_baselines_seed42.json")
    args = parser.parse_args()
    if args.random_repetitions <= 0:
        parser.error("--random-repetitions must be positive")

    root = Path(args.dataset_root).resolve()
    train_path, test_path = root / "train.csv", root / "test.csv"
    rng = random.Random(args.seed)
    output = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "random_repetitions": args.random_repetitions,
        "random_protocol": "independent uniform label per test example",
        "dataset": {
            "root": str(root),
            "train_csv_sha256": sha256(train_path),
            "test_csv_sha256": sha256(test_path),
        },
        "tasks": {},
    }

    for task in ("factuality", "bias"):
        labels = TASKS[task]["order"]
        train = load_labels(train_path, task)
        test = load_labels(test_path, task)
        counts = Counter(train)
        majority = max(labels, key=lambda label: counts[label])
        deterministic = {
            "majority": {
                "predicted_label": majority,
                "metrics": selected(evaluate(task, test, [majority] * len(test), False)),
            },
            "ordinal_midpoint": {
                "predicted_label": MIDPOINT[task],
                "metrics": selected(evaluate(task, test, [MIDPOINT[task]] * len(test), False)),
            },
        }
        random_runs = []
        for _ in range(args.random_repetitions):
            predictions = [rng.choice(labels) for _ in test]
            random_runs.append(selected(evaluate(task, test, predictions, False)))
        random_summary = {
            key: {
                "mean": statistics.fmean(run[key] for run in random_runs),
                "stddev": statistics.stdev(run[key] for run in random_runs),
            }
            for key in METRICS
        }
        output["tasks"][task] = {
            "train_label_counts": dict(counts),
            "test_examples": len(test),
            **deterministic,
            "uniform_random": {"metrics": random_summary},
        }

    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {destination}")
    for task, result in output["tasks"].items():
        print(f"\n{task}")
        for baseline in ("majority", "ordinal_midpoint"):
            metrics = result[baseline]["metrics"]
            print(f"  {baseline:16} P={metrics['macro_precision']:.6f} "
                  f"R={metrics['macro_recall']:.6f} F1={metrics['macro_f1']:.6f} "
                  f"MAE={metrics['mae']:.6f} MSE={metrics['mse']:.6f}")
        metrics = result["uniform_random"]["metrics"]
        print(f"  {'uniform_random':16} P={metrics['macro_precision']['mean']:.6f} "
              f"R={metrics['macro_recall']['mean']:.6f} F1={metrics['macro_f1']['mean']:.6f} "
              f"MAE={metrics['mae']['mean']:.6f} MSE={metrics['mse']['mean']:.6f}")


if __name__ == "__main__":
    main()
