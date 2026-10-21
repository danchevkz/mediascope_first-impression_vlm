#!/usr/bin/env python3
"""Offline snapshot of visible rationales and separately saved provider summaries.

Regex matches are review candidates, never verified contamination or recall rates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from analyze_saved_run import extract_reasoning
from review_api_reasoning import PATTERNS
from run_config import write_json


REVIEW_PATTERN = re.compile(
    r"\bMBFC\b|media bias[ /-]*fact[ /-]*check|"
    r"\b(?:rated|classified|listed)\s+by\b|\b(?:I recall|I remember|prior knowledge)\b",
    re.I,
)

ORIGINAL_RUNS = {
    "gpt56": "artifacts/evaluations/api-full-hf-gpt56-opus5-20261006/gpt56.audit/predictions.jsonl",
    "opus5": "artifacts/evaluations/api-full-hf-gpt56-opus5-20261006/opus5.audit/predictions.jsonl",
    "kimi": "artifacts/evaluations/api-full-hf-20261001/kimi.audit/predictions.jsonl",
}


def paired_original_accuracy(model: str, rows: list[dict]) -> dict | None:
    """Compare only exact screenshot/target matches; make no new model calls."""
    path = Path(ORIGINAL_RUNS.get(model, ""))
    if not path.is_file():
        return None
    previous = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            previous[(row["task"], row["media_name"], row["url"])] = row
    pairs = []
    for current in rows:
        key = (current["task"], current["media_name"], current["url"])
        old = previous[key]
        if old["image_sha256"] != current["image_sha256"] or old["true_label"] != current["true_label"]:
            raise ValueError(f"Paired input mismatch: {model}, {key}")
        pairs.append((old, current))
    n = len(pairs)
    return {"source_journal": str(path), "paired_records": n,
            "original_correct": sum(bool(old["correct"]) for old, _ in pairs),
            "rationale_correct": sum(bool(new["correct"]) for _, new in pairs),
            "same_predicted_labels": sum(old["predicted_label"] == new["predicted_label"] for old, new in pairs),
            "original_accuracy": sum(bool(old["correct"]) for old, _ in pairs) / n if n else None,
            "rationale_accuracy": sum(bool(new["correct"]) for _, new in pairs) / n if n else None}


def flags(text: str) -> dict[str, bool]:
    return {name: bool(re.search(pattern, text, re.I)) for name, pattern in PATTERNS.items()}


def texts(row: dict, local: bool = False) -> tuple[str, str]:
    if local:
        return row.get("raw_output") or "", ""
    details = row.get("details") or {}
    return details.get("visible_rationale") or "", extract_reasoning(details.get("response") or {})[0]


def summary(rows: list[dict], expected: int = 212, local: bool = False) -> dict:
    visible_rows, provider_rows, either_rows = [], [], []
    counts = Counter()
    for row in rows:
        visible, provider = texts(row, local)
        vf, pf = flags(visible), flags(provider)
        counts["visible_text"] += bool(visible.strip())
        counts["provider_text"] += bool(provider.strip())
        counts["visible_non_label_text"] += bool(re.sub(
            r"(?im)^\s*(?:\*\*)?LABEL\s*:[^\n]*$", "", visible
        ).strip() and visible.strip().upper() not in ("LOW", "MIXED", "HIGH", "LEFT", "LEFT-CENTER", "LEAST BIASED", "RIGHT-CENTER", "RIGHT"))
        counts["reached_token_budget"] += bool(row.get("reached_token_budget") or (row.get("details") or {}).get("reached_token_budget"))
        counts["visible_network_wording"] += vf["metric_media_family"]
        counts["provider_network_wording"] += pf["metric_media_family"]
        if vf["direct_rating_reference"]:
            visible_rows.append(row)
        if pf["direct_rating_reference"]:
            provider_rows.append(row)
        if vf["direct_rating_reference"] or pf["direct_rating_reference"]:
            either_rows.append(row)

    def screen(values: list[dict]) -> dict:
        correct = sum(bool(row["correct"]) for row in values)
        return {"matched_records": len(values), "correct_predictions": correct,
                "matched_fraction_of_expected": len(values) / expected,
                "prediction_accuracy_among_matches": correct / len(values) if values else None}

    correct = sum(bool(row["correct"]) for row in rows)
    return {
        "expected": expected, "recorded": len(rows), "complete": len(rows) == expected,
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "correct_predictions": correct,
        "accuracy_on_recorded": correct / len(rows) if rows else None,
        "overall_subset_accuracy": correct / expected if len(rows) == expected else None,
        **dict(counts),
        "strict_lexical_candidates_visible": screen(visible_rows),
        "strict_lexical_candidates_provider": screen(provider_rows),
        "strict_lexical_candidates_either_deduplicated": screen(either_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-dir", default="artifacts/evaluations/api-visible-rationales-stratified50-20261010")
    parser.add_argument("--local-prefix", default="artifacts/evaluations/visible-rationales-stratified50-20261010")
    parser.add_argument("--subset-csv", default="artifacts/audits/interpretability_subset_seed42_50pct_20261010/subset.csv")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    with open(args.subset_csv, newline="", encoding="utf-8") as handle:
        subset = list(csv.DictReader(handle))
    expected = {(task, row["media_name"], row["url"]): row[f"true_label_{task}"]
                for row in subset for task in ("factuality", "bias")}
    sources = [(tag, Path(args.api_dir) / f"{tag}.audit/predictions.jsonl", False)
               for tag in ("gpt56", "opus5", "kimi", "qwen")]
    sources += [(f"{family}_{mode}", Path(f"{args.local_prefix}-{family}-{mode}-{task}/predictions.jsonl"), True)
                for family in ("qwen35", "ministral") for mode in ("base", "finetuned")
                for task in ("factuality", "bias")]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "subset_sha256": hashlib.sha256(Path(args.subset_csv).read_bytes()).hexdigest(),
        "outlets": len(subset), "sources": [], "models": {},
        "interpretation": "Strict regex candidate rates are not manually verified rating-recall or contamination rates.",
    }
    records = []
    grouped = {}
    for model, path, local in sources:
        if not path.is_file():
            continue
        data = path.read_bytes()
        latest = {}
        for number, line in enumerate(data.splitlines(keepends=True), 1):
            if not line.endswith(b"\n"):
                continue  # Snapshot may overlap an ongoing journal append.
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["task"], row["media_name"], row["url"])
            if key not in expected or row["true_label"] != expected[key]:
                raise ValueError(f"Source/label mismatch at {path}:{number}")
            row["source_line"] = number
            latest[key] = row
        report["sources"].append({"model": model, "path": str(path),
                                  "sha256_snapshot": hashlib.sha256(data).hexdigest(), "records": len(latest)})
        for row in latest.values():
            task = row["task"]
            grouped.setdefault((model, task, local), []).append(row)
            visible, provider = texts(row, local)
            records.append({
                "model": model, "task": task, "media_name": row["media_name"], "url": row["url"],
                "true_label": row["true_label"], "predicted_label": row["predicted_label"],
                "correct": row["correct"], "status": row["status"],
                "visible_text": visible, "provider_text": provider,
                "visible_flags": flags(visible), "provider_flags": flags(provider),
                "review_candidate": bool(REVIEW_PATTERN.search(visible + "\n" + provider)),
                "source_journal": str(path), "source_line": row["source_line"],
            })
    for (model, task, local), rows in grouped.items():
        result = summary(rows, len(subset), local)
        if not local and model in ORIGINAL_RUNS:
            result["paired_original_predictions"] = paired_original_accuracy(model, rows)
        report["models"].setdefault(model, {})[task] = result
    write_json(output / "analysis.json", report)
    with (output / "texts.jsonl").open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output / "review_candidates.jsonl").open("w", encoding="utf-8") as stream:
        for row in records:
            if row["review_candidate"]:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = ["# Saved interpretability output audit", "", report["interpretation"], "",
             "212 unmasked outlets, stratified by cluster and joint labels, seed 42. No new inference.", "",
             "| Model | Task | Recorded | Visible text | Provider text | Visible regex matches | Provider regex matches | Either (deduplicated) | Overall accuracy |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for model, tasks in report["models"].items():
        for task, m in tasks.items():
            accuracy = f"{100*m['overall_subset_accuracy']:.2f}%" if m["complete"] else "pending"
            lines.append(f"| {model} | {task} | {m['recorded']}/{m['expected']} | {m.get('visible_text', 0)} | "
                         f"{m.get('provider_text', 0)} | {m['strict_lexical_candidates_visible']['matched_records']} | "
                         f"{m['strict_lexical_candidates_provider']['matched_records']} | "
                         f"{m['strict_lexical_candidates_either_deduplicated']['matched_records']} | {accuracy} |")
    lines += ["", "## What can be used", "",
              "- Visible explanations: a common-format spontaneous-disclosure analysis across completed methods.",
              "- Separate provider text: additional observed rating-recall evidence, with unequal/missing coverage explicitly reported.",
              "- Completed classification predictions: overall subset accuracy with invalid outputs counted as incorrect.",
              "- Original label-only predictions: paired accuracy on these same screenshots, checked against image hashes and target labels.",
              "- Network wording and saved excerpts: evidence of source/template recognition, separately from rating recall.",
              "", "## Limits", "",
              "- The old regex admits false positives (e.g. applying MBFC methodology) and misses reverse/possessive rating wording. Review candidates must be coded manually before reporting explicit-reference percentages.",
              "- A recalled label and a correct classification are distinct: prediction correctness does not validate the recalled external rating.",
              "- Missing provider summaries cannot be treated as no rating knowledge.",
              "- Opus has no readable provider summaries in this run; its visible network/rating statements can still be reviewed.",
              "- Partial Qwen Max and ongoing local tasks are snapshots, not full-subset results.",
              "- This is evidence of stated prior knowledge, not a verified training-contamination prevalence or causal dependence estimate."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "models": {
        model: {task: {"recorded": m["recorded"], "visible": m.get("visible_text", 0),
                       "provider": m.get("provider_text", 0),
                       "candidate_matches_either": m["strict_lexical_candidates_either_deduplicated"]["matched_records"]}
                for task, m in tasks.items()}
        for model, tasks in report["models"].items()
    }}, indent=2))


if __name__ == "__main__":
    main()
