#!/usr/bin/env python3
"""Consolidate the eight paired unmasked tests and matched historical masked outcomes."""
import argparse
import csv
import json
from pathlib import Path

from mediascope_eval import TASKS, evaluate
from run_config import sha256_file, write_json


METHODS = {
    "gpt56": "GPT-5.6 Sol", "opus5": "Claude Opus 5", "kimi": "Kimi K3", "qwen": "Qwen3.8 Max Prime",
    "qwen35_base": "Qwen3.5-4B base", "qwen35_finetuned": "Qwen3.5-4B fine-tuned",
    "ministral_base": "Ministral-3-8B base", "ministral_finetuned": "Ministral-3-8B fine-tuned",
}


def journal(directory):
    if not (directory / "manifest.json").exists():
        return {}, []
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["identity"]["parser"]["mode"] != "strict":
        raise ValueError(f"Unexpected parser contract: {directory}")
    path = directory / "predictions.jsonl"
    rows = {}
    if path.exists():
        for raw in path.read_bytes().splitlines(keepends=True):
            if raw.endswith(b"\n") and raw.strip():
                r = json.loads(raw)
                key = (r["media_name"], r["task"])
                rows[key] = r
    return rows, [{"path": str(path), "sha256": sha256_file(path)}] if path.exists() else []


def locations(key, api, local, masked=False):
    if key in ("gpt56", "opus5", "kimi", "qwen"):
        if not masked:
            return [api / key / "predictions.audit"]
        if key in ("gpt56", "opus5"):
            return [Path("artifacts/evaluations/api-masked-test-gpt56-opus5-20261010") / f"{key}.audit"]
        if key == "qwen":
            return [Path("artifacts/evaluations/api-masked-test-qwen-recovery-20261009/qwen.audit")]
        return [Path("artifacts/evaluations/api-masked-test-20261009/kimi.audit")]
    root = Path("artifacts/evaluations/masked-local-20261009") if masked else local
    if key.endswith("_base"):
        return [root / f"{key}.audit"]
    return [root / f"{key}_{task}.audit" for task in TASKS]


def score_rows(rows, ids, labels, task):
    actual = [rows[(id, task)] for id in sorted(ids) if (id, task) in rows]
    for r in actual:
        if r["true_label"] != labels[r["media_name"]][task]:
            raise ValueError(f"Target mismatch: {r['media_name']} {task}")
        if r["status"] == "ok" and (r["predicted_label"] not in TASKS[task]["order"]
                                      or r["correct"] != (r["predicted_label"] == r["true_label"])):
            raise ValueError("Invalid saved prediction/correctness metadata")
    scores = evaluate(task, [r["true_label"] for r in actual],
                      [r["predicted_label"] if r["status"] == "ok" else None for r in actual], print_report=False)
    scores["expected"] = len(ids)
    scores["complete"] = len(actual) == len(ids)
    scores["valid_output_rate"] = sum(r["status"] == "ok" for r in actual) / len(actual) if actual else None
    scores["status_counts"] = {status: sum(r["status"] == status for r in actual)
                               for status in sorted({r["status"] for r in actual})}
    return scores


def markdown(report):
    lines = ["# Paired unmasked test — eight methods", "",
             "| Model | Factuality accuracy | Factuality macro-F1 | Bias accuracy | Bias macro-F1 |",
             "|---|---:|---:|---:|---:|"]
    for key, name in METHODS.items():
        values = []
        for task in TASKS:
            scores = report["models"][key]["unmasked"]["all"][task]
            for metric in ("accuracy", "macro_f1"):
                values.append(f"{100*scores[metric]:.2f}%" if scores["complete"] else f"Pending ({scores['n']}/{scores['expected']})")
        lines.append(f"| {name} | {' | '.join(values)} |")
    lines.extend(["", "## Unseen-outlet subset", "",
                  "| Model | Factuality accuracy | Bias accuracy |", "|---|---:|---:|"])
    for key, name in METHODS.items():
        values = []
        for task in TASKS:
            scores = report["models"][key]["unmasked"]["unseen_outlets"][task]
            values.append(f"{100*scores['accuracy']:.2f}%" if scores["complete"] and scores["accuracy"] is not None else "Pending")
        lines.append(f"| {name} | {' | '.join(values)} |")
    lines.extend(["", "## Interpretation", "",
                  "- Label-only classification, matched to the historical masked protocol: full-page six-tile inputs, 1024 max width; frontier medium reasoning/8192 cap; locals thinking off/4-bit/16-token cap.",
                  "- All 133 pairs are evaluated as requested. 43 outlet filenames appeared in adapter training, including 42 pixel-identical training images; this full set is not a pure held-out fine-tuning test.",
                  "- JSON includes all, training-seen-outlet, pixel-identical-training-image, and unseen-outlet breakdowns with class distributions and macro precision/recall/F1, accuracy, MAE/MSE, coverage, and confusion matrices.",
                  "- Earlier masked images and labels match exactly. Paired masked/unmasked differences are computed on the same site IDs, with conditional datasets and model versions kept explicit.",
                  "- Failed/unparseable outputs count as incorrect; their ordinal errors follow the existing farthest-class penalty. No retraining or success resampling is performed.", ""])
    return "\n".join(lines)


def summarize(dataset, api, local, output):
    manifest = json.loads((dataset / "manifest.json").read_text())
    pairs = manifest["pairs"]
    labels = {p["pair_id"]: p["labels"] for p in pairs}
    all_ids = set(labels)
    overlap = manifest["adapter_training_overlap"]
    seen_sets = [set(v["filename_overlap"]) for v in overlap.values()]
    pixels = [set(v["pixel_identical_training_images"]) for v in overlap.values()]
    if any(s != seen_sets[0] for s in seen_sets) or any(s != pixels[0] for s in pixels):
        raise ValueError("Adapter training populations differ; use model-specific subset reports")
    groups = {"all": all_ids, "training_seen_outlets": seen_sets[0], "pixel_identical_training_images": pixels[0],
              "unseen_outlets": all_ids - seen_sets[0]}
    report = {"dataset_manifest_sha256": sha256_file(dataset / "manifest.json"), "n": len(pairs),
              "subsets": {name: {"n": len(ids), "class_counts": {t: {label: sum(labels[id][t] == label for id in ids)
                                                                      for label in TASKS[t]["order"]} for t in TASKS}}
                          for name, ids in groups.items()}, "models": {}}
    for key in METHODS:
        model = {"name": METHODS[key], "sources": []}
        for condition in ("unmasked", "masked"):
            records = {}
            for directory in locations(key, api, local, condition == "masked"):
                rows, hashes = journal(directory)
                if set(records) & set(rows):
                    raise ValueError("Duplicate task outcomes across local adapters")
                records.update(rows)
                model["sources"].extend(hashes)
            if any(id not in all_ids or task not in TASKS for id, task in records):
                raise ValueError("Unexpected test site/task")
            model[condition] = {name: {t: score_rows(records, ids, labels, t) for t in TASKS} for name, ids in groups.items()}
        model["paired_changes"] = {}
        for name in groups:
            model["paired_changes"][name] = {}
            for task in TASKS:
                u, m = model["unmasked"][name][task], model["masked"][name][task]
                model["paired_changes"][name][task] = {metric: u[metric]-m[metric]
                    if u["complete"] and m["complete"] and u.get(metric) is not None and m.get(metric) is not None
                    else None for metric in ("accuracy", "macro_f1", "macro_precision", "macro_recall", "mae", "mse")}
        report["models"][key] = model
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "analysis.json", report)
    fields = ["method", "condition", "subset", "task", "n", "expected", "complete", "macro_precision",
              "macro_recall", "macro_f1", "accuracy", "mae", "mse", "valid_output_rate"]
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, model in report["models"].items():
            for condition in ("unmasked", "masked"):
                for subset, tasks in model[condition].items():
                    for task, scores in tasks.items():
                        writer.writerow({"method": METHODS[key], "condition": condition, "subset": subset, "task": task,
                                         **{field: scores.get(field) for field in fields[4:]}})
    text = markdown(report)
    (output / "REPORT.md").write_text(text, encoding="utf-8")
    print(text)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("artifacts/datasets/paired-test-133-20261011"))
    parser.add_argument("--api-root", type=Path, default=Path("artifacts/evaluations/paired-unmasked-133-api-20261011"))
    parser.add_argument("--local-root", type=Path, default=Path("artifacts/evaluations/paired-unmasked-133-local-20261011"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/audits/paired-unmasked-133-comparison-20261011"))
    args = parser.parse_args()
    summarize(args.dataset_root, args.api_root, args.local_root, args.output_dir)
