#!/usr/bin/env python3
"""Quantify identity-risk diagnostics from a completed API run without new calls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from quantify_api_risk import LABELS, score_matrices
from review_api_reasoning import PATTERNS


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_journal(path: Path) -> tuple[list[dict], str]:
    digest = hashlib.sha256()
    rows = []
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            digest.update(raw)
            if not raw.strip():
                continue
            row = json.loads(raw)
            row["source_line"] = line_number
            rows.append(row)
    return rows, digest.hexdigest()


def extract_reasoning(response: dict) -> tuple[str, str, int]:
    """Extract provider-returned summaries by response shape, not model tag."""
    if isinstance(response.get("output"), list):
        items = [item for item in response["output"] if item.get("type") == "reasoning"]
        text = "\n\n".join(
            block.get("text", "")
            for item in items
            for block in item.get("summary", [])
            if isinstance(block, dict)
        )
        return text.strip(), "openai_reasoning_summary", len(items)
    if isinstance(response.get("content"), list):
        blocks = [block for block in response["content"] if block.get("type") == "thinking"]
        text = "\n\n".join(block.get("thinking", "") for block in blocks)
        return text.strip(), "anthropic_thinking", len(blocks)
    choices = response.get("choices") or []
    message = choices[0].get("message", {}) if choices else {}
    if isinstance(message.get("reasoning"), str) and message["reasoning"].strip():
        return message["reasoning"].strip(), "openrouter_reasoning", 1
    details = message.get("reasoning_details") or []
    text = "\n\n".join(
        item.get("text", "") for item in details
        if isinstance(item, dict) and isinstance(item.get("text"), str)
    )
    return text.strip(), "openrouter_reasoning_details", len(details)


def reasoning_tokens(response: dict) -> int:
    usage = response.get("usage") or {}
    details = usage.get("output_tokens_details") or usage.get("completion_tokens_details") or {}
    return details.get("reasoning_tokens", details.get("thinking_tokens", 0)) or 0


def matrix_for(rows: list[dict], task: str, group_for: dict, equal_group_weight=False) -> np.ndarray:
    labels = LABELS[task]
    groups = list(dict.fromkeys(group_for[(row["media_name"], row["url"])] for row in rows))
    indexes = {group: index for index, group in enumerate(groups)}
    matrix = np.zeros((len(groups), len(labels), len(labels) + 1), dtype=float)
    for row in rows:
        key = (row["media_name"], row["url"])
        actual = labels.index(row["true_label"])
        predicted = labels.index(row["predicted_label"]) if row["predicted_label"] in labels else len(labels)
        matrix[indexes[group_for[key]], actual, predicted] += 1
    if equal_group_weight:
        sizes = matrix.sum(axis=(1, 2))
        matrix /= sizes[:, None, None]
    return matrix


def score_view(rows: list[dict], task: str, group_for: dict, equal=False) -> dict:
    score = score_matrices(matrix_for(rows, task, group_for, equal).sum(axis=0))
    groups = {group_for[(row["media_name"], row["url"])] for row in rows}
    return {"records": len(rows), "groups": len(groups),
            "accuracy": float(score["accuracy"]), "macro_f1": float(score["macro_f1"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset-root", default="artifacts/datasets/mediascope-clustered-main")
    parser.add_argument("--models", nargs="+", help="optional audit-directory tags to analyze")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run, root, output = map(Path, (args.run_dir, args.dataset_root, args.output_dir))
    if output.exists():
        parser.error("Choose a new output directory")
    journals = sorted(run.glob("*.audit/predictions.jsonl"))
    if args.models:
        requested = set(args.models)
        journals = [path for path in journals if path.parent.name.removesuffix(".audit") in requested]
        found = {path.parent.name.removesuffix(".audit") for path in journals}
        if found != requested:
            parser.error("Missing requested model audits: " + ", ".join(sorted(requested - found)))
    if not journals:
        parser.error("No *.audit/predictions.jsonl journals found")
    output.mkdir(parents=True)

    source = read_csv(root / "test.csv")
    source_keys = {(row["media_name"], row["url"]) for row in source}
    cluster_rows = [row for row in read_csv(root / "clusters_13.csv") if row["split"] == "test"]
    cluster_map = {(row["media_name"], row["url"]): row["cluster"] for row in cluster_rows}
    cluster_counts = Counter(cluster_map.values())
    largest = cluster_counts.most_common(1)[0][0]
    grouped = {key: cluster_map.get(key, f"outlet:{key!r}") for key in source_keys}
    singleton = {key: f"outlet:{key!r}" for key in source_keys}

    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_run": str(run.resolve()),
        "dataset_root": str(root.resolve()),
        "cluster_sizes": dict(cluster_counts),
        "largest_cluster": largest,
        "patterns": PATTERNS,
        "models": {},
        "limitations": [
            "Lexical matches describe returned summaries, not verified causal memory use.",
            "Missing summary text is unobserved and cannot be counted as absence of prior knowledge.",
            "Cluster-balanced metrics change target weighting and are sensitivity diagnostics.",
            "No saved-output analysis can estimate the counterfactual score with publisher identity removed.",
        ],
    }
    detail_rows = []
    for journal in journals:
        tag = journal.parent.name.removesuffix(".audit")
        rows, digest = read_journal(journal)
        if {(row["media_name"], row["url"]) for row in rows} != source_keys:
            raise ValueError(f"{tag}: journal outlets do not match the test split")
        saved_metrics = json.loads((journal.parent / "metrics.json").read_text())["tasks"]
        model = {"journal_sha256": digest, "records": len(rows), "tasks": {}}
        for task in LABELS:
            task_rows = [row for row in rows if row["task"] == task]
            if not task_rows:
                continue
            all_groups = grouped
            without_largest = [row for row in task_rows
                               if cluster_map.get((row["media_name"], row["url"])) != largest]
            unclustered = [row for row in task_rows
                           if (row["media_name"], row["url"]) not in cluster_map]
            views = {
                "official_equal_outlet": score_view(task_rows, task, singleton),
                "equal_known_group_weight": score_view(task_rows, task, all_groups, True),
                "without_largest_cluster": score_view(without_largest, task, all_groups),
                "unclustered_only": score_view(unclustered, task, singleton),
            }
            for metric in ("accuracy", "macro_f1"):
                if abs(views["official_equal_outlet"][metric] - saved_metrics[task][metric]) > 1e-12:
                    raise ValueError(f"{tag}/{task}: reconstructed {metric} does not match saved metrics")
            counts = Counter()
            associations = {name: defaultdict(list) for name in ("readable", *PATTERNS)}
            for row in task_rows:
                response = row.get("details", {}).get("response") or {}
                text, kind, blocks = extract_reasoning(response)
                readable = bool(text)
                flags = {name: bool(re.search(pattern, text, re.I)) for name, pattern in PATTERNS.items()}
                valid = row["status"] == "ok"
                counts["records"] += 1
                counts["valid"] += valid
                counts["readable"] += readable
                counts["readable_valid"] += readable and valid
                counts["reasoning_blocks"] += blocks
                counts["reasoning_tokens"] += reasoning_tokens(response)
                for name, present in {"readable": readable, **flags}.items():
                    if valid and (name == "readable" or readable):
                        associations[name]["present" if present else "absent"].append(bool(row["correct"]))
                if readable and valid:
                    counts.update({name: int(value) for name, value in flags.items()})
                detail_rows.append({
                    "model": tag, "task": task, "media_name": row["media_name"], "url": row["url"],
                    "status": row["status"], "correct": row["correct"], "readable": readable,
                    "explanation_kind": kind, "reasoning_tokens": reasoning_tokens(response),
                    **flags, "source_line": row["source_line"],
                })
            model["tasks"][task] = {
                "score_sensitivity": views,
                "explanation_counts": dict(counts),
                "lexical_match_fraction_of_readable_valid": {
                    name: counts[name] / counts["readable_valid"] if counts["readable_valid"] else None
                    for name in PATTERNS
                },
                "accuracy_stratification_not_causal": {
                    name: {bucket: {"n": len(values), "accuracy": sum(values) / len(values)}
                           for bucket, values in bins.items()}
                    for name, bins in associations.items()
                },
            }
        report["models"][tag] = model

    (output / "analysis.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output / "record_flags.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)
    print(json.dumps({"output": str(output), "models": {
        tag: {task: data["explanation_counts"] for task, data in model["tasks"].items()}
        for tag, model in report["models"].items()
    }}, indent=2))


if __name__ == "__main__":
    main()
