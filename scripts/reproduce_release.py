#!/usr/bin/env python3
"""Recompute released scores offline using only Python's standard library."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from mediascope_eval import TASKS, evaluate


ROOT = Path(__file__).resolve().parents[1]
METRICS = ("macro_precision", "macro_recall", "macro_f1", "accuracy", "mae", "mse")
PRIMARY = {"asserted", "uncertain_recall"}


def load(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def equal(actual, expected, label):
    if actual is None or expected is None:
        if actual is not expected:
            raise ValueError(f"Mismatch: {label}")
    elif not math.isclose(float(actual), float(expected), abs_tol=1e-12, rel_tol=1e-12):
        raise ValueError(f"Mismatch: {label}: {actual} != {expected}")


def verify_manifest(root):
    path = root / "release_manifest.json"
    if not path.exists():
        raise ValueError("Missing release_manifest.json")
    files = load(path)["files"]
    for relative, expected in files.items():
        path = root / relative
        if not path.is_file():
            raise ValueError(f"Missing release file: {relative}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
            raise ValueError(f"Changed release file: {relative}")
    return len(files)


def score_predictions(root):
    targets = {r["media_name"]: r for r in read_csv(root / "data/main/test.csv")}
    images = load(root / "data/main/image_inventory.json")["splits"]["test"]["images"]
    hashes = {r["path"]: r["sha256"] for r in images}
    pairs = {r["pair_id"]: r for r in load(root / "data/paired/manifest.json")["pairs"]}
    expected = {(r["condition"], r["method"], r["task"]): r
                for r in read_csv(root / "results/expected_metrics.csv")}
    expected_paired = load(root / "results/expected_paired_analysis.json")
    result, covered = [], set()
    for path in sorted((root / "results/predictions").glob("*/*.jsonl")):
        condition, method = path.parent.name, path.stem
        records = read_jsonl(path)
        keys = [(r["media_name"], r["task"]) for r in records]
        if len(keys) != len(set(keys)):
            raise ValueError(f"Duplicate outcome: {condition}/{method}")
        for record in records:
            id, task = record["media_name"], record["task"]
            if condition == "main":
                true = targets[id]["true_label_" + task]
                digest = hashes[targets[id]["image_path"]]
            else:
                true = pairs[id]["labels"][task]
                digest = pairs[id][condition]["image_sha256"]
            if true != record["true_label"] or digest != record["image_sha256"]:
                raise ValueError(f"Dataset identity mismatch: {condition}/{method}/{id}/{task}")
            correct = record["status"] == "ok" and record["predicted_label"] == true
            if correct != record["correct"]:
                raise ValueError("Incorrect correctness flag")
            if record["status"] == "ok" and record["predicted_label"] not in TASKS[task]["order"]:
                raise ValueError("Invalid label marked successful")
        subsets = {"all": set(targets if condition == "main" else pairs)}
        if condition != "main":
            subsets.update({
                "training_seen_outlets": {id for id, p in pairs.items() if p["training_seen_outlet"]},
                "pixel_identical_training_images": {id for id, p in pairs.items() if p["pixel_identical_training_image"]},
                "unseen_outlets": {id for id, p in pairs.items() if not p["training_seen_outlet"]},
            })
        for task in sorted({r["task"] for r in records}):
            covered.add((condition, method, task))
            for subset, ids in subsets.items():
                chosen = [r for r in records if r["task"] == task and r["media_name"] in ids]
                if len(chosen) != len(ids):
                    raise ValueError(f"Incomplete outcomes: {condition}/{method}/{task}/{subset}")
                scores = evaluate(task, [r["true_label"] for r in chosen],
                                  [r["predicted_label"] if r["status"] == "ok" else None for r in chosen], False)
                if subset == "all":
                    stored = expected[(condition, method, task)]
                    equal(scores["n"], stored["n"], "sample count")
                    for metric in METRICS:
                        equal(scores[metric], stored[metric], f"{condition}/{method}/{task}/{metric}")
                if condition != "main":
                    stored = expected_paired["models"][method][condition][subset][task]
                    for metric in METRICS:
                        equal(scores[metric], stored[metric], f"paired/{method}/{subset}/{task}/{metric}")
                result.append({"condition": condition, "method": method, "task": task, "subset": subset,
                               "n": len(chosen), **{m: scores[m] for m in METRICS},
                               "valid_output_rate": sum(r["status"] == "ok" for r in chosen) / len(chosen)})
    if covered != set(expected):
        raise ValueError("Released prediction coverage differs from expected metrics")
    return result


def reference_annotations(root):
    """Rebuild reviewed ranges, including default-none only inside reviewed ranges."""
    result = {}
    for directory in sorted((root / "results/reference_audits").iterdir()):
        if not directory.is_dir():
            continue
        snapshot = load(directory / "snapshot.json")
        source = {r["id"]: r for r in snapshot["records"]}
        codes = {}
        for path in sorted((directory / "annotations").glob("*.json")):
            batch = load(path)
            reviewed = set()
            for start, end in batch["reviewed_ranges"]:
                for i in range(start, end + 1):
                    reviewed.add(str(i))
                    prefix = batch["model"] + ":" if "model" in batch else ""
                    id = f"{prefix}{batch['task']}:{i:03d}"
                    if id not in source or id in codes:
                        raise ValueError(f"Unknown or duplicate review: {id}")
                    record = source[id]
                    code = batch.get("decisions", {}).get(str(i), "none")
                    if not record.get("observed", True):
                        code = "unobserved"
                    if isinstance(code, str):
                        code = {"category": code, "evidence": []}
                    text = "\n".join(str(v or "") for v in (record.get("structured") or {}).values())
                    if any(not quote or quote not in text for quote in code.get("evidence", [])):
                        raise ValueError(f"Evidence not found: {id}")
                    codes[id] = code
            if not set(batch.get("decisions", {})).issubset(reviewed):
                raise ValueError("Annotation outside reviewed ranges")
        for id, record in source.items():
            if id not in codes:
                if record.get("observed", True):
                    raise ValueError(f"Unreviewed valid report: {id}")
                codes[id] = {"category": "unobserved", "evidence": []}
        result[directory.name] = (source, codes)
    return result


def score_references(root):
    audits = reference_annotations(root)
    records = read_jsonl(root / "results/references.jsonl")
    subset = {r["media_name"]: r for r in read_csv(root / "data/recognition/subset.csv")}
    keys = [(r["model"], r["task"], r["media_name"]) for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate reference case")
    for row in records:
        original, codes = audits[row["audit"]]
        frozen = original[row["id"]]
        if row["category"] != codes[row["id"]]["category"]:
            raise ValueError("Derived reference category differs from reviewed annotations")
        for field in ("source_sha256", "structured", "provider_summary", "correct"):
            if row[field] != frozen[field]:
                raise ValueError(f"Derived reference changed: {field}")
        truth = subset[row["media_name"]]["true_label_" + row["task"]]
        if row["true_label"] != truth or row["correct"] != (row["observed"] and row["predicted_label"] == truth):
            raise ValueError("Reference target/correctness mismatch")
    results = []
    for model in ("gpt56", "opus5", "kimi", "qwen"):
        for task in TASKS:
            chosen = [r for r in records if r["model"] == model and r["task"] == task]
            if len(chosen) != 212 or {r["media_name"] for r in chosen} != set(subset):
                raise ValueError("Reference subset mismatch")
            observed = [r for r in chosen if r["observed"]]
            refs = [r for r in observed if r["category"] in PRIMARY]
            correct = sum(r["correct"] for r in chosen)
            correct_refs = sum(r["correct"] for r in refs)
            results.append({"model": model, "task": task, "rubric": chosen[0]["rubric"],
                            "n": 212, "valid_reports": len(observed), "references": len(refs),
                            "correct": correct, "correct_references": correct_refs,
                            "reference_rate_valid_reports": len(refs) / len(observed),
                            "accuracy_among_references": correct_refs / len(refs) if refs else None,
                            "overall_accuracy": correct / 212,
                            "reference_lower_bound_all_attempts": len(refs) / 212,
                            "correct_references_fraction_all_attempts": correct_refs / 212,
                            "reference_share_of_correct_predictions": correct_refs / correct if correct else None})
    return results


def verify_baselines(root, random_check=False):
    saved = load(root / "results/naive_baselines.json")
    for task, details in saved["tasks"].items():
        train = [r["true_label_" + task] for r in read_csv(root / "data/main/train.csv")]
        truth = [r["true_label_" + task] for r in read_csv(root / "data/main/test.csv")]
        if Counter(train) != details["train_label_counts"] or len(truth) != details["test_examples"]:
            raise ValueError("Baseline split mismatch")
        for baseline in ("majority", "ordinal_midpoint"):
            prediction = details[baseline]["predicted_label"]
            metrics = evaluate(task, truth, [prediction] * len(truth), False)
            for metric in METRICS:
                equal(metrics[metric], details[baseline]["metrics"][metric], "baseline/" + metric)
    if random_check:
        with tempfile.TemporaryDirectory(prefix="mediascope-baselines-") as directory:
            destination = Path(directory) / "baselines.json"
            subprocess.run([sys.executable, str(root / "scripts/run_naive_baselines.py"),
                            "--dataset-root", str(root / "data/main"), "--seed", str(saved["seed"]),
                            "--random-repetitions", str(saved["random_repetitions"]), "--out", str(destination)], check=True)
            current = load(destination)
            for task in TASKS:
                for metric in METRICS:
                    for statistic in ("mean", "stddev"):
                        equal(current["tasks"][task]["uniform_random"]["metrics"][metric][statistic],
                              saved["tasks"][task]["uniform_random"]["metrics"][metric][statistic], "random baseline")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-random", action="store_true", help="also regenerate the 10,000 uniform-random baseline repetitions")
    parser.add_argument("--output", type=Path, default=ROOT / "results/reproduced")
    args = parser.parse_args()
    checked = verify_manifest(ROOT)
    metrics = score_predictions(ROOT)
    references = score_references(ROOT)
    verify_baselines(ROOT, args.verify_random)
    write_csv(args.output / "metrics.csv", metrics)
    write_csv(args.output / "reference_metrics.csv", references)
    # Figure-ready data for the six systems shown in the paper's masking plot.
    figure3 = []
    for row in metrics:
        if row["condition"] == "main" or row["subset"] != "all" or row["method"].endswith("_base"):
            continue
        figure3.append({"method": row["method"], "task": row["task"], "condition": row["condition"],
                        "macro_f1_percent": 100 * row["macro_f1"]})
    write_csv(args.output / "figure3_data.csv", figure3)
    missing = load(ROOT / "provenance/coverage.json")["missing_main_prediction_artifacts"]
    lines = ["# Reproduced results", "",
             "All metrics below are recomputed from released per-example outcomes.",
             "Macro P/R/F1 and accuracy are fractions in CSV; MAE/MSE are unnormalized ordinal errors.", "",
             "## Paired VLM results", "",
             "| Method | Condition | Task | P (%) | R (%) | F1 (%) | Acc (%) | MAE | MSE |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in metrics:
        if row["condition"] == "main" or row["subset"] != "all":
            continue
        values = [f"{row[m] * (100 if m in METRICS[:4] else 1):.3f}" for m in METRICS]
        lines.append(f"| {row['method']} | {row['condition']} | {row['task']} | " + " | ".join(values) + " |")
    coverage = (f"Main-test model/task groups without included prediction artifacts: {len(missing)}." if missing else
                "All eight VLM methods are included for both tasks on the main test and both paired conditions.")
    lines += ["", "## Coverage", "", coverage,
              "See provenance/coverage.json and docs/PROTOCOLS.md for source details.",
              "Reference metrics preserve V1 Opus/GPT and V2 Kimi/Qwen coding; comparisons remain rubric-dependent.",
              "Full 133-pair adapter scores include 43 training-seen outlets (42 identical unmasked training images).",
              "The 90 unseen-outlet scores are included separately in metrics.csv.", ""]
    (args.output / "REPORT.md").write_text("\n".join(lines))
    print(f"Verified {len(metrics)} metric groups, {len(references)} reference groups, and deterministic baselines.")
    print(f"Release files verified: {checked}. Main-test missing model/task artifacts: {len(missing)}.")


if __name__ == "__main__":
    main()
