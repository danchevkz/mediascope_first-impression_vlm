#!/usr/bin/env python3
"""Quantify benchmark concentration and explanation evidence without API calls.

Cluster sensitivity is not an estimate of causal contamination. Bootstrap
intervals condition on fixed predictions and the explicitly stated grouping.
"""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from export_api_explanations import extract
from review_api_reasoning import PATTERNS


LABELS = {"factuality": ["LOW", "MIXED", "HIGH"],
          "bias": ["LEFT", "LEFT-CENTER", "LEAST BIASED", "RIGHT-CENTER", "RIGHT"]}
MODELS = ("opus", "openai", "kimi")


def csv_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def journal_snapshot(path):
    data = Path(path).read_bytes()
    # A concurrently running evaluator can be in the middle of its last write.
    complete = data[:data.rfind(b"\n") + 1]
    latest = {}
    for number, line in enumerate(complete.splitlines(), 1):
        row = json.loads(line)
        row["journal_line"] = number
        latest[(row["media_name"], row["url"], row["task"])] = row
    return latest, hashlib.sha256(complete).hexdigest()


def score_matrices(matrices):
    k = matrices.shape[-2]
    true_positive = np.diagonal(matrices[..., :k], axis1=-2, axis2=-1)
    supports = matrices.sum(axis=-1)
    predicted = matrices[..., :k].sum(axis=-2)
    denominator = supports + predicted
    f1 = np.divide(2 * true_positive, denominator, out=np.zeros_like(denominator, dtype=float), where=denominator > 0)
    return {"accuracy": true_positive.sum(axis=-1) / matrices.sum(axis=(-2, -1)),
            "macro_f1": f1.mean(axis=-1)}


def paired_bootstrap(matrices, draws, rng):
    """Each model uses identical resampled units; units may have unequal sizes."""
    n = matrices["opus"].shape[0]
    weights = rng.multinomial(n, np.full(n, 1 / n), size=draws)
    result, samples = {}, {}
    for model, blocks in matrices.items():
        aggregate = blocks.sum(axis=0)
        simulated = (weights @ blocks.reshape(n, -1)).reshape((draws,) + aggregate.shape)
        samples[model] = score_matrices(simulated)
        point = score_matrices(aggregate)
        result[model] = {metric: {"point": float(value),
                                 "percentile_95_interval": np.quantile(samples[model][metric], [.025, .975]).tolist()}
                         for metric, value in point.items()}
    difference = {}
    for metric in ("accuracy", "macro_f1"):
        delta = samples["opus"][metric] - samples["openai"][metric]
        difference[metric] = {
            "point": result["opus"][metric]["point"] - result["openai"][metric]["point"],
            "percentile_95_interval": np.quantile(delta, [.025, .975]).tolist(),
        }
    return {"resampling_units": n, "models": result, "opus_minus_openai": difference}


def model_blocks(keys, task, records, group_for, equal_group_weight=False):
    labels = LABELS[task]
    groups = list(dict.fromkeys(group_for[key] for key in keys))
    indexes = {group: index for index, group in enumerate(groups)}
    matrices = {}
    for model in MODELS:
        blocks = np.zeros((len(groups), len(labels), len(labels) + 1), dtype=float)
        for key in keys:
            row = records[model][(*key, task)]
            actual = labels.index(row["true_label"])
            prediction = labels.index(row["predicted_label"]) if row["predicted_label"] in labels else len(labels)
            blocks[indexes[group_for[key]], actual, prediction] += 1
        if equal_group_weight:
            blocks /= blocks.sum(axis=(1, 2))[:, None, None]
        matrices[model] = blocks
    return matrices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--opus-rerun")
    parser.add_argument("--rerun-outlets", type=int, default=60)
    parser.add_argument("--bootstrap-draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.bootstrap_draws < 1000 or args.rerun_outlets < 1:
        parser.error("Use at least 1000 bootstrap draws and a positive prefix size")
    run, root, output = Path(args.run_dir), Path(args.dataset_root), Path(args.output_dir)
    if output.exists():
        parser.error("Choose a new analysis directory")
    output.mkdir(parents=True)
    created = datetime.now(timezone.utc).isoformat()
    source = csv_rows(root / "test.csv")
    keys = [(r["media_name"], r["url"]) for r in source]
    assert len(keys) == len(set(keys)) == 423
    raw = dict(zip(keys, source))
    cluster_rows = [r for r in csv_rows(root / "clusters_13.csv") if r["split"] == "test"]
    cluster_map = {(r["media_name"], r["url"]): r["cluster"] for r in cluster_rows}
    group_for = {key: cluster_map.get(key, f"outlet:{key!r}") for key in keys}
    singleton_groups = {key: f"outlet:{key!r}" for key in keys}
    cluster_counts = Counter(cluster_map.values())
    largest = cluster_counts.most_common(1)[0][0]
    records, hashes = {}, {}
    expected = {(*key, task) for key in keys for task in LABELS}
    for model in MODELS:
        records[model], hashes[model] = journal_snapshot(run / f"{model}.audit/predictions.jsonl")
        assert set(records[model]) == expected
        assert all(r["status"] == "ok" and r["true_label"] == raw[(r["media_name"], r["url"])][f"true_label_{r['task']}"]
                   for r in records[model].values())
    views = {
        "all_outlet_iid": (keys, singleton_groups, False),
        "all_known_cluster_bootstrap": (keys, group_for, False),
        "equal_known_cluster_weight": (keys, group_for, True),
        "without_largest_cluster": ([key for key in keys if cluster_map.get(key) != largest], group_for, False),
        "unclustered_only": ([key for key in keys if key not in cluster_map], singleton_groups, False),
    }
    rng = np.random.default_rng(args.seed)
    estimates = {}
    for task in LABELS:
        estimates[task] = {}
        for view, (selected, groups, balanced) in views.items():
            matrices = model_blocks(selected, task, records, groups, balanced)
            estimate = paired_bootstrap(matrices, args.bootstrap_draws, rng)
            estimate["outlets"] = len(selected)
            estimates[task][view] = estimate
        for model in MODELS:
            existing = json.loads((run / f"{model}.audit/metrics.json").read_text())["tasks"][task]
            for metric in ("accuracy", "macro_f1"):
                assert abs(estimates[task]["all_outlet_iid"]["models"][model][metric]["point"] - existing[metric]) < 1e-12
        print(f"Computed paired score sensitivity and bootstrap intervals: {task}", flush=True)
    contributions = {}
    for task in LABELS:
        contributions[task] = {}
        for group in [*cluster_counts, "unclustered"]:
            selected = [key for key in keys if cluster_map.get(key, "unclustered") == group]
            correct = {model: sum(records[model][(*key, task)]["correct"] for key in selected) for model in MODELS}
            contributions[task][group] = {"n": len(selected), "correct": correct,
                                        "opus_minus_openai_correct": correct["opus"] - correct["openai"],
                                        "true_labels": dict(Counter(raw[key][f"true_label_{task}"] for key in selected))}
    evidence, evidence_rows = {}, []
    for model in (*MODELS, "qwen"):
        model_records = records.get(model)
        if model_records is None:
            model_records, hashes[model] = journal_snapshot(run / f"{model}.audit/predictions.jsonl")
        counts, by_task, flags = Counter(), defaultdict(Counter), {}
        for key, row in model_records.items():
            text, _, _ = extract(model, row.get("details", {}).get("response") or {})
            flag = {name: bool(re.search(pattern, text, re.I)) for name, pattern in PATTERNS.items()}
            flags[key] = (bool(text), flag)
            for counter in (counts, by_task[row["task"]]):
                counter["records"] += 1
                counter["valid_records"] += row["status"] == "ok"
                counter["readable_valid"] += bool(text) and row["status"] == "ok"
                if text and row["status"] == "ok":
                    counter.update({name: int(value) for name, value in flag.items()})
            evidence_rows.append({"model": model, "media_name": row["media_name"], "url": row["url"],
                                  "task": row["task"], "status": row["status"], "correct": row["correct"],
                                  "readable": bool(text), **flag, "source_line": row["journal_line"]})
        associations = {}
        for task in LABELS:
            associations[task] = {}
            for name in ("readable", "direct_rating_reference", "metric_media_family"):
                bins = defaultdict(list)
                for key, row in model_records.items():
                    if row["task"] != task or row["status"] != "ok":
                        continue
                    readable, flag = flags[key]
                    if name != "readable" and not readable:
                        continue
                    value = readable if name == "readable" else flag[name]
                    bins["present" if value else "absent"].append(row["correct"])
                associations[task][name] = {category: {"n": len(values), "accuracy": sum(values) / len(values)}
                                           for category, values in bins.items()}
        evidence[model] = {"counts": dict(counts), "by_task": {k: dict(v) for k, v in by_task.items()},
                           "descriptive_associations_not_causal": associations}

    rerun_info = None
    if args.opus_rerun:
        rerun, rerun_hash = journal_snapshot(Path(args.opus_rerun) / "opus.audit/predictions.jsonl")
        selected = [key for key in keys if all((*key, task) in rerun for task in LABELS)][:args.rerun_outlets]
        snapshot = []
        for key in selected:
            for task in LABELS:
                row = rerun[(*key, task)]
                original = records["opus"][(*key, task)]
                assert row["true_label"] == original["true_label"] and row["image_sha256"] == original["image_sha256"]
                assert row["details"]["prompt"] == original["details"]["prompt"]
                if row["status"] == "ok":
                    assert row["details"]["submitted_png_sha256"] == original["details"]["submitted_png_sha256"]
                response = row.get("details", {}).get("response") or {}
                text, _, _ = extract("opus", response)
                snapshot.append({"media_name": key[0], "url": key[1], "task": task,
                                 "status": row["status"], "true_label": row["true_label"],
                                 "predicted_label": row["predicted_label"], "correct": row["correct"],
                                 "original_prediction": original["predicted_label"],
                                 "same_prediction_as_original": row["predicted_label"] == original["predicted_label"],
                                 "cluster": cluster_map.get(key, "unclustered"), "response_id": response.get("id"),
                                 "source_line": row["journal_line"], "provider_summary": text})
        with (output / "opus_rerun_prefix_snapshot.jsonl").open("w", encoding="utf-8") as stream:
            for row in snapshot:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        rerun_info = {"source": args.opus_rerun, "observed_prefix_sha256": rerun_hash,
                      "selection": f"First {args.rerun_outlets} outlets in original test order with both tasks completed; not random.",
                      "outlets": len(selected), "records": len(snapshot),
                      "valid": sum(r["status"] == "ok" for r in snapshot),
                      "readable": sum(bool(r["provider_summary"]) for r in snapshot),
                      "same_prediction_as_original": sum(r["same_prediction_as_original"] for r in snapshot),
                      "last_outlet": selected[-1][0] if selected else None,
                      "observed_records_in_live_run": len(rerun),
                      "complete_live_run": len(rerun) == 846}

    report = {
        "created_at": created, "run_dir": str(run.resolve()), "source_journal_sha256": hashes,
        "bootstrap_draws": args.bootstrap_draws, "seed": args.seed,
        "cluster_sizes": dict(cluster_counts), "largest_cluster": largest,
        "known_groups_including_unclustered_singletons": len(set(group_for.values())),
        "class_counts": {task: dict(Counter(r[f"true_label_{task}"] for r in source)) for task in LABELS},
        "score_estimates": estimates, "correct_prediction_contributions": contributions,
        "explanation_evidence": evidence, "wording_patterns": PATTERNS,
        "opus_rerun_prefix": rerun_info,
        "limitations": [
            "Sensitivity differences are not causal estimates of contamination or clean-data scores.",
            "Cluster-balanced scoring changes the target weighting; it is diagnostic, not the official score.",
            "Bootstrap intervals resample paired predictions, condition on recorded grouping, and omit API repeat variability and label uncertainty.",
            "Unlisted outlets are treated as singleton groups, not proven independent networks.",
            "Wording matches describe returned explanations, not verified memory use; missing text is not evidence of absence.",
            "The live Opus prefix is alphabetically selected and cannot estimate full-run explanation prevalence.",
        ],
    }
    (output / "quantification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output / "explanation_flags.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(evidence_rows[0]))
        writer.writeheader()
        writer.writerows(evidence_rows)
    print(json.dumps({"cluster_sizes": report["cluster_sizes"], "opus_rerun_prefix": rerun_info,
                      "factuality_opus_minus_gpt": {k: v["opus_minus_openai"] for k, v in estimates["factuality"].items()},
                      "output_dir": str(output)}, indent=2))


if __name__ == "__main__":
    main()
