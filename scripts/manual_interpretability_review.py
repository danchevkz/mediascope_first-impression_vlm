#!/usr/bin/env python3
"""Freeze, display, and aggregate manually judged model text. No regex judging."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from analyze_saved_run import extract_reasoning
from run_config import write_json


API_ROOT = Path("artifacts/evaluations/api-visible-rationales-stratified50-20261010")
LOCAL_PREFIX = "artifacts/evaluations/visible-rationales-stratified50-20261010"
SUBSET = Path("artifacts/audits/interpretability_subset_seed42_50pct_20261010/subset.csv")
MODELS = ("gpt56", "opus5", "kimi", "qwen", "qwen35_base", "qwen35_finetuned", "ministral_base", "ministral_finetuned")


def snapshot(output: Path) -> None:
    with SUBSET.open(newline="", encoding="utf-8") as stream:
        population = list(csv.DictReader(stream))
    index = {(r["media_name"], r["url"]): i for i, r in enumerate(population, 1)}
    targets = {(r["media_name"], r["url"], t): r[f"true_label_{t}"]
               for r in population for t in ("factuality", "bias")}
    output.mkdir(parents=True, exist_ok=True)
    snapshot_path = output / "snapshot.json"
    previous = json.loads(snapshot_path.read_text(encoding="utf-8")) if snapshot_path.is_file() else None
    records, hashes = [], []
    for model in MODELS:
        for task in ("factuality", "bias"):
            local = model not in MODELS[:4]
            if local:
                family, mode = model.split("_", 1)
                path = Path(f"{LOCAL_PREFIX}-{family}-{mode}-{task}/predictions.jsonl")
            else:
                path = API_ROOT / f"{model}.audit/predictions.jsonl"
            if not path.is_file():
                continue
            data = path.read_bytes()
            latest = {}
            for line_number, line in enumerate(data.splitlines(keepends=True), 1):
                if not line.endswith(b"\n") or not line.strip():
                    continue
                row = json.loads(line)
                if row["task"] != task:
                    continue
                key = (row["media_name"], row["url"])
                if key not in index or row["true_label"] != targets[(*key, task)]:
                    raise ValueError(f"Unexpected source/target at {path}:{line_number}")
                row["source_line"] = line_number
                latest[key] = row
            hashes.append({"model": model, "task": task, "path": str(path),
                           "sha256": hashlib.sha256(data).hexdigest(), "records": len(latest)})
            for key, row in sorted(latest.items(), key=lambda item: index[item[0]]):
                details = row.get("details") or {}
                visible = row.get("raw_output") or "" if local else details.get("visible_rationale") or ""
                provider = "" if local else extract_reasoning(details.get("response") or {})[0]
                records.append({"id": f"{model}:{task}:{index[key]:03d}", "model": model, "task": task,
                                "media_name": key[0], "url": key[1], "visible": visible, "provider": provider,
                                "true_label": row["true_label"], "predicted_label": row["predicted_label"],
                                "correct": row["correct"], "status": row["status"],
                                "token_limit": bool(row.get("reached_token_budget") or details.get("reached_token_budget")),
                                "source_journal": str(path), "source_line": row["source_line"]})
    if previous:
        current = {r["id"]: r for r in records}
        for row in previous["records"]:
            if current.get(row["id"]) != row:
                raise ValueError(f"Existing frozen record changed: {row['id']}")
    write_json(snapshot_path, {"subset_csv": str(SUBSET),
               "subset_sha256": hashlib.sha256(SUBSET.read_bytes()).hexdigest(), "expected_per_task": len(population),
               "sources": hashes, "records": records})
    print(json.dumps({"output": str(output), "sources": hashes}, indent=2))


def display(root: Path, model: str, task: str, offset: int, limit: int, channel: str) -> None:
    saved = json.loads((root / "snapshot.json").read_text(encoding="utf-8"))
    rows = [r for r in saved["records"] if r["model"] == model and r["task"] == task]
    selected = rows[offset:offset + limit]
    print(f"Review {offset+1}-{offset+len(selected)} of {len(rows)} recorded {model}/{task}")
    for row in selected:
        print(f"\n[{row['id']}] {row['media_name']}")
        for name in ("provider", "visible") if channel == "both" else (channel,):
            print(f"{name.upper()}:\n{row[name] if row[name] else '[no returned text]'}")


def aggregate(root: Path, models: tuple[str, ...] = MODELS) -> dict:
    saved = json.loads((root / "snapshot.json").read_text(encoding="utf-8"))
    rows = {r["id"]: r for r in saved["records"]}
    annotations = {}
    for path in sorted((root / "annotations").glob("*.json")):
        batch = json.loads(path.read_text(encoding="utf-8"))
        reviewed_ids = batch.get("reviewed_ids", [])
        for start, end in batch.get("reviewed_ranges", []):
            reviewed_ids.extend(f"{batch['model']}:{batch['task']}:{i:03d}" for i in range(start, end + 1))
        for id in reviewed_ids:
            if id not in rows or id in annotations:
                raise ValueError(f"Unknown/duplicate reviewed id: {id}")
            annotations[id] = batch.get("decisions", {}).get(id, {"category": "none", "channel": "none", "evidence": "", "note": "Manually read; no existing external-rating claim or recall attempt."})
        if not set(batch.get("decisions", {})).issubset(reviewed_ids):
            raise ValueError(f"Unreviewed decisions in {path}")
    allowed = {"asserted", "uncertain_recall", "ambiguous", "generic", "none", "unobserved"}
    for id, annotation in annotations.items():
        if annotation["category"] not in allowed:
            raise ValueError(f"Invalid category: {id}")
        if annotation["category"] in {"asserted", "uncertain_recall", "ambiguous", "generic"}:
            if not annotation.get("evidence") or not annotation.get("note"):
                raise ValueError(f"Evidence and note required: {id}")
            if annotation["evidence"] not in rows[id]["provider"] + "\n" + rows[id]["visible"]:
                raise ValueError(f"Evidence is not a saved excerpt: {id}")
    summaries = {}
    for model in models:
        for task in ("factuality", "bias"):
            actual = [r for r in rows.values() if r["model"] == model and r["task"] == task]
            coded = [r for r in actual if r["id"] in annotations]
            refs = [r for r in coded if annotations[r["id"]]["category"] in {"asserted", "uncertain_recall"}]
            correct_refs = sum(bool(r["correct"]) for r in refs)
            n = len(actual)
            complete_review = len(coded) == n
            counts = {cat: sum(annotations[r["id"]]["category"] == cat for r in coded) for cat in allowed}
            asserted = [r for r in coded if annotations[r["id"]]["category"] == "asserted"]
            summaries.setdefault(model, {})[task] = {
                "expected": saved["expected_per_task"], "recorded": n, "reviewed": len(coded),
                "complete_subset": n == saved["expected_per_task"], "complete_review": complete_review,
                "visible_text": sum(bool(r["visible"].strip()) for r in actual),
                "provider_text": sum(bool(r["provider"].strip()) for r in actual),
                "token_limited": sum(r["token_limit"] for r in actual),
                "categories": counts,
                "explicit_references": len(refs) if complete_review else None,
                "reference_rate": len(refs) / n if n and complete_review else None,
                "correct_reference_predictions": correct_refs if complete_review else None,
                "accuracy_among_references": correct_refs / len(refs) if refs and complete_review else None,
                "asserted_reference_rate": len(asserted) / n if n and complete_review else None,
                "correct_asserted_predictions": sum(bool(r["correct"]) for r in asserted) if complete_review else None,
                "accuracy_among_asserted_references": sum(bool(r["correct"]) for r in asserted) / len(asserted) if asserted and complete_review else None,
                "visible_only_references": sum(annotations[r["id"]]["channel"] in {"visible", "both"} for r in refs) if complete_review else None,
                "provider_only_references": sum(annotations[r["id"]]["channel"] == "provider" for r in refs) if complete_review else None,
                "overall_accuracy": sum(bool(r["correct"]) for r in actual) / n if n else None,
            }
    filename = "summary" if models == MODELS else "focused_summary"
    focused_rows = [r for r in rows.values() if r["model"] in models]
    result = {"models": summaries, "reviewed_records": sum(r["id"] in annotations for r in focused_rows),
              "snapshot_records": len(focused_rows), "judge": "single OpenCode LLM manual reviewer",
              "primary_reference_categories": ["asserted", "uncertain_recall"]}
    write_json(root / f"{filename}.json", result)
    with (root / f"{filename}_coded_records.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["id", "model", "task", "media_name", "true_label", "predicted_label", "status", "correct",
                  "reviewed", "category", "explicit_reference", "channel", "evidence", "note",
                  "visible_text_available", "provider_text_available", "token_limit", "source_journal", "source_line"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in focused_rows:
            annotation = annotations.get(row["id"], {})
            writer.writerow({
                **{key: row.get(key, "") for key in fields if key in row},
                "reviewed": row["id"] in annotations,
                "category": annotation.get("category", "unreviewed"),
                "explicit_reference": annotation.get("category") in {"asserted", "uncertain_recall"},
                **{key: annotation.get(key, "") for key in ("channel", "evidence", "note")},
                "visible_text_available": bool(row["visible"].strip()),
                "provider_text_available": bool(row["provider"].strip()),
            })
    display_names = {"opus5": "Claude Opus 5", "gpt56": "GPT-5.6 Sol"}
    percent = lambda value: f"{100*value:.2f}%" if value is not None else "N/A"
    lines = ["# Manual external-rating-reference analysis", "",
             "Dataset: the same 212 unmasked test outlets, seed-42 joint-label/cluster-stratified sample.",
             "Reviewer: one LLM judge who read every recorded response in the reported model/task rows.",
             "Rubric: JUDGE_PROMPT.md. Primary references include asserted existing ratings and uncertain recollections; generic analogies and ambiguous/hypothetical judgments are excluded.", "",
             "| Method | Task | Analyzed | Asserted | Uncertain recall | Explicit references | Correct among references | Overall accuracy |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for model, tasks in summaries.items():
        for task, m in tasks.items():
            nr = m["explicit_references"]
            refs_text = f"{nr}/{m['recorded']} ({percent(m['reference_rate'])})" if nr is not None else "not fully reviewed"
            correct_text = f"{m['correct_reference_predictions']}/{nr} ({percent(m['accuracy_among_references'])})" if nr else "N/A"
            total_correct = sum(bool(r["correct"]) for r in focused_rows if r["model"] == model and r["task"] == task)
            lines.append(f"| {display_names.get(model, model)} | {task} | {m['reviewed']}/{m['recorded']} | "
                         f"{m['categories']['asserted']} | {m['categories']['uncertain_recall']} | {refs_text} | "
                         f"{correct_text} | {total_correct}/{m['recorded']} ({percent(m['overall_accuracy'])}) |")
    lines += ["", "## Observation coverage and exclusions", "",
              "| Method | Task | Visible text | Provider text | Visible-only primary references | Provider-only primary references | Ambiguous | Generic analogies |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for model, tasks in summaries.items():
        for task, m in tasks.items():
            lines.append(f"| {display_names.get(model, model)} | {task} | {m['visible_text']} | {m['provider_text']} | "
                         f"{m['visible_only_references']} | {m['provider_only_references']} | "
                         f"{m['categories']['ambiguous']} | {m['categories']['generic']} |")
    lines += ["", "## Interpretation", "",
              "- Percentages measure observed external-rating statements, not training-contamination prevalence.",
              "- Provider text is included when returned, but availability differs by provider. Zero Opus references applies to its returned visible explanations; hidden reasoning is unobserved.",
              "- Correct among references means the final classification matches the dataset target. It does not validate the recalled rating itself.",
              "- Missing/invalid final labels are counted as incorrect. Opus has one unparseable bias label in this run.",
              "- Asserted-only rates are saved separately in JSON so tentative recall need not be merged into the headline definition.",
              "- Every positive/ambiguous/generic label has an exact excerpt checked against the frozen response text. No regex is used to assign manual categories."]
    (root / f"{filename}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("snapshot", "read", "aggregate"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--task", choices=("factuality", "bias"))
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--channel", choices=("both", "provider", "visible"), default="both")
    parser.add_argument("--models", nargs="+", choices=MODELS, help="optional reporting scope for aggregate")
    args = parser.parse_args()
    if args.action == "snapshot":
        snapshot(Path(args.root))
    elif args.action == "read":
        if not args.model or not args.task:
            parser.error("read requires --model and --task")
        display(Path(args.root), args.model, args.task, args.offset, args.limit, args.channel)
    else:
        aggregate(Path(args.root), tuple(args.models) if args.models else MODELS)


if __name__ == "__main__":
    main()
