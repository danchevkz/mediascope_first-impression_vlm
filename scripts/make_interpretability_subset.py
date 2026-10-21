#!/usr/bin/env python3
"""Create a deterministic, proportionally stratified interpretability subset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from run_config import sha256_file, write_json


def counts(rows, key):
    return dict(sorted(Counter(key(row) for row in rows).items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="artifacts/datasets/mediascope-clustered-main")
    parser.add_argument("--output-dir", default="artifacts/audits/interpretability_subset_seed42_50pct_20261010")
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.fraction < 1:
        parser.error("--fraction must be between zero and one")

    root = Path(args.dataset_root).resolve()
    with (root / "test.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with (root / "clusters_13.csv").open(newline="", encoding="utf-8") as handle:
        cluster_map = {
            (row["media_name"], row["url"]): row["cluster"]
            for row in csv.DictReader(handle) if row["split"] == "test"
        }
    for row in rows:
        row["cluster_group"] = cluster_map.get((row["media_name"], row["url"]), "unclustered")
        row["stratum"] = "|".join((row["cluster_group"], row["true_label_factuality"], row["true_label_bias"]))

    target = round(len(rows) * args.fraction)
    strata = defaultdict(list)
    for row in rows:
        strata[row["stratum"]].append(row)
    allocations = {name: math.floor(len(group) * target / len(rows)) for name, group in strata.items()}
    remainder = target - sum(allocations.values())
    fractions = sorted(
        strata,
        key=lambda name: (-(len(strata[name]) * target / len(rows) - allocations[name]), name),
    )
    for name in fractions[:remainder]:
        allocations[name] += 1

    rng = random.Random(args.seed)
    selected = []
    for name in sorted(strata):
        group = sorted(strata[name], key=lambda row: (row["media_name"], row["url"]))
        rng.shuffle(group)
        selected.extend(group[:allocations[name]])
    selected.sort(key=lambda row: (row["media_name"], row["url"]))
    if len(selected) != target:
        raise RuntimeError(f"selected {len(selected)} rows, expected {target}")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    fields = ["media_name", "url", "true_label_factuality", "true_label_bias", "cluster_group", "stratum"]
    with (output / "subset.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in selected)

    summarize = lambda values: {
        "factuality": counts(values, lambda row: row["true_label_factuality"]),
        "bias": counts(values, lambda row: row["true_label_bias"]),
        "cluster_group": counts(values, lambda row: row["cluster_group"]),
        "joint_stratum": counts(values, lambda row: row["stratum"]),
    }
    manifest = {
        "protocol": "proportional-stratified-outlet-sample-v1",
        "seed": args.seed,
        "requested_fraction": args.fraction,
        "population_records": len(rows),
        "selected_records": len(selected),
        "realized_fraction": len(selected) / len(rows),
        "stratification": "recorded test cluster membership x factuality label x bias label",
        "test_csv_sha256": sha256_file(root / "test.csv"),
        "clusters_csv_sha256": sha256_file(root / "clusters_13.csv"),
        "subset_csv_sha256": sha256_file(output / "subset.csv"),
        "population_counts": summarize(rows),
        "selected_counts": summarize(selected),
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
