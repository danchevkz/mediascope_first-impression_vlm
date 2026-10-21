#!/usr/bin/env python3
"""Freeze comparable probe self-reports for manual semantic review, no API calls."""
import argparse
import csv
import json
import shutil
from pathlib import Path

from run_config import sha256_file, write_json


RUNS = {"opus5": "opus5-summary-probe-parallel-20261010", "gpt56": "gpt56-stream-summary-probe-20261010"}
SUBSET = Path("artifacts/audits/interpretability_subset_seed42_50pct_20261010/subset.csv")
CATEGORIES = ("asserted", "uncertain_recall", "ambiguous", "generic", "none")
PRIMARY = ("asserted", "uncertain_recall")
MODEL_NAMES = {"opus5": "Claude Opus 5", "gpt56": "GPT-5.6 Sol"}


def freeze(root, previous_review=None):
    with SUBSET.open(newline="", encoding="utf-8") as handle:
        subset = list(csv.DictReader(handle))
    idx = {(r["media_name"], r["url"]): i for i, r in enumerate(subset, 1)}
    targets = {(r["media_name"], r["url"], task): r[f"true_label_{task}"] for r in subset for task in ("factuality", "bias")}
    rows, sources = [], []
    for model, directory in RUNS.items():
        for path in sorted((Path("artifacts/evaluations") / directory / "records").glob("*.json")):
            r = json.loads(path.read_text(encoding="utf-8"))
            if r["status"] not in ("ok", "ok_without_provider_summary") or not r["output_valid"]:
                raise ValueError(f"Invalid core structured report: {path}")
            key = (r["media_name"], r["url"])
            if key not in idx or r["true_label"] != targets[(*key, r["task"])]:
                raise ValueError(f"Subset mismatch: {path}")
            a = r["structured_assessment"]
            correct = a["label"] == r["true_label"]
            if correct != r["correct"]:
                raise ValueError(f"Correctness mismatch: {path}")
            id = f"{model}:{r['task']}:{idx[key]:03d}"
            rows.append({"id": id, "model": model, "task": r["task"], "media_name": r["media_name"],
                         "structured": a, "provider_summary": r["provider_summary"],
                         "true_label": r["true_label"], "predicted_label": a["label"], "correct": correct,
                         "source": str(path), "source_sha256": sha256_file(path)})
            sources.append({"id": id, "path": str(path), "sha256": sha256_file(path)})
    rows.sort(key=lambda r: (r["model"], r["task"], r["id"]))
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate source decisions")
    seed = None
    if previous_review is not None:
        seed = _load_review(previous_review)
        current = {r["id"]: r for r in rows}
        for id, prior in seed[1].items():
            if current.get(id) != prior:
                raise ValueError(f"Cannot inherit review of changed or missing record: {id}")
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "snapshot.json", {"expected_per_task": 212, "subset_sha256": sha256_file(SUBSET), "records": rows, "sources": sources})
    if seed is not None:
        (root / "annotations").mkdir()
        for source in seed[3]:
            path = Path(source["path"])
            destination = root / "annotations" / path.name
            shutil.copyfile(path, destination)
            if sha256_file(destination) != source["sha256"]:
                raise ValueError(f"Inherited annotation hash mismatch: {path}")
        rubric = (previous_review / "JUDGE_PROMPT.md").read_text(encoding="utf-8")
        rubric = rubric.replace(
            "Opus is partial and its recorded\nprefix is not representative of the complete fixed random/stratified subset.",
            "If a run is partial, its recorded\nprefix must not be treated as representative of the complete fixed subset.")
        (root / "JUDGE_PROMPT.md").write_text(rubric, encoding="utf-8")
        write_json(root / "review_provenance.json", {
            "previous_review": str(previous_review), "reused_decisions": len(seed[2]),
            "previous_snapshot_sha256": sha256_file(previous_review / "snapshot.json"),
            "previous_rubric_sha256": sha256_file(previous_review / "JUDGE_PROMPT.md"),
            "previous_annotation_sources": seed[3], "saved_responses_unchanged": True,
            "rubric_change": "Coverage wording made conditional; all semantic coding rules unchanged.",
        })
        print(f"Inherited {len(seed[2])} manually coded decisions with unchanged saved responses")
    print(f"Frozen {len(rows)} records")


def read(root, model, task, offset, limit):
    saved = json.loads((root / "snapshot.json").read_text())
    rows = [r for r in saved["records"] if r["model"] == model and r["task"] == task][offset:offset+limit]
    for r in rows:
        a = r["structured"]
        print(f"\n[{r['id']}] {r['media_name']}")
        print('SOURCE:',a['recognized_outlet'],'NETWORK:',a['recognized_network'])
        print('RATER:',a['external_rating_source'],'FACT RECALL:',a['recalled_factuality_rating'],'BIAS RECALL:',a['recalled_bias_rating'])
        print('BASIS:',a['self_report_basis'])
        print('EXPLANATION:',a['explanation_summary'])


def _load_review(root):
    saved = json.loads((root / "snapshot.json").read_text(encoding="utf-8"))
    expected = saved["expected_per_task"]
    if type(expected) is not int or expected < 1:
        raise ValueError("Invalid expected decision count")
    if saved["subset_sha256"] != sha256_file(SUBSET):
        raise ValueError("Subset hash mismatch")
    rows = {}
    for r in saved["records"]:
        id = r["id"]
        model, task, index = id.split(":")
        if (model not in RUNS or task not in ("factuality", "bias")
                or model != r["model"] or task != r["task"]
                or not 1 <= int(index) <= expected
                or id != f"{model}:{task}:{int(index):03d}"):
            raise ValueError(f"Invalid record identity: {id}")
        if id in rows:
            raise ValueError(f"Duplicate source decision: {id}")
        source = Path(r["source"])
        if sha256_file(source) != r["source_sha256"]:
            raise ValueError(f"Source hash mismatch: {id}")
        original = json.loads(source.read_text(encoding="utf-8"))
        if (original["status"] not in ("ok", "ok_without_provider_summary")
                or not original["output_valid"]
                or any(original[field] != r[field] for field in
                       ("task", "media_name", "true_label", "correct", "provider_summary"))
                or original["structured_assessment"] != r["structured"]):
            raise ValueError(f"Snapshot differs from source: {id}")
        correct = r["predicted_label"] == r["true_label"]
        if (r["predicted_label"] != r["structured"]["label"]
                or type(r["correct"]) is not bool or r["correct"] != correct):
            raise ValueError(f"Correctness mismatch: {id}")
        rows[id] = r

    annotations, annotation_sources = {}, []
    for file in sorted((root / "annotations").glob("*.json")):
        batch = json.loads(file.read_text(encoding="utf-8"))
        reviewed = set()
        for start, end in batch["reviewed_ranges"]:
            if type(start) is not int or type(end) is not int or not 1 <= start <= end <= expected:
                raise ValueError(f"Invalid reviewed range: {file}")
            for i in range(start, end + 1):
                id = f"{batch['model']}:{batch['task']}:{i:03d}"
                if id not in rows or id in annotations:
                    raise ValueError(f"Unknown/duplicate annotation: {id}")
                category = batch.get("decisions", {}).get(str(i), "none")
                if category not in CATEGORIES:
                    raise ValueError(f"Invalid category: {id}")
                annotations[id] = category
                reviewed.add(str(i))
        if not set(batch.get("decisions", {})).issubset(reviewed):
            raise ValueError("Decision outside reviewed range")
        annotation_sources.append({"path": str(file), "sha256": sha256_file(file)})
    if set(rows) != set(annotations):
        missing = sorted(set(rows) - set(annotations))
        raise ValueError(f"Incomplete review: {len(missing)} unreviewed decisions")
    return saved, rows, annotations, annotation_sources


def _summarize(actual, annotations, expected):
    refs = [r for r in actual if annotations[r["id"]] in PRIMARY]
    n, k = len(actual), len(refs)
    correct = sum(r["correct"] for r in actual)
    correct_refs = sum(r["correct"] for r in refs)
    return {
        "n": n, "expected": expected, "complete": n == expected,
        "categories": {c: sum(annotations[r["id"]] == c for r in actual) for c in CATEGORIES},
        "explicit_references": k, "reference_rate": k / n if n else None,
        "correct_reference_predictions": correct_refs,
        "accuracy_among_references": correct_refs / k if k else None,
        "correct_predictions": correct, "overall_accuracy": correct / n if n else None,
        "provider_summaries": sum(bool(r["provider_summary"].strip()) for r in actual),
        "structured_explanations": n,
    }


def _table(models):
    lines = ["| Model | Task | n | References | Correct among references | Overall accuracy |",
             "|---|---|---:|---:|---:|---:|"]
    for model in RUNS:
        for task in ("factuality", "bias"):
            m = models[model][task]
            n, k = m["n"], m["explicit_references"]
            refs = f"{k}/{n} ({100 * m['reference_rate']:.2f}%)" if n else "N/A"
            conditional = f"{m['correct_reference_predictions']}/{k} ({100 * m['accuracy_among_references']:.2f}%)" if k else "N/A"
            overall = f"{m['correct_predictions']}/{n} ({100 * m['overall_accuracy']:.2f}%)" if n else "N/A"
            lines.append(f"| {MODEL_NAMES[model]} | {task.capitalize()} | {n} | {refs} | {conditional} | {overall} |")
    return "\n".join(lines)


def _report_markdown(report):
    complete = all(m["complete"] for tasks in report["models"].values() for m in tasks.values())
    coverage = ("Both models completed the same fixed stratified subset. All recorded structured reports were reviewed."
                if complete else
                "Opus stopped early: its recorded alphabetical prefix is not representative of the full stratified subset. GPT completed the subset. All recorded structured reports were reviewed.")
    matching = ("The matched comparison includes the complete fixed stratified subset."
                if complete else
                "This comparison controls the outlet set, but the shared prefix is still not the full stratified sample.")
    lines = ["# Current explicit-recognition probe: manual review", "",
             "Primary references are asserted or uncertain recollections of an external factuality/bias rating for the outlet or identified family. Either rating type qualifies in either task, once per decision.", "",
             "## All recorded results", "", _table(report["models"]), "",
             coverage, "",
             f"## Matched comparison: {report['matched']['outlet_count']} shared outlets", "",
             _table(report["matched"]["models"]), "",
             matching, "",
             "## Coding breakdown and native-summary coverage", "",
             "| Model | Task | Asserted | Uncertain recall | Ambiguous | Generic | None | Native summaries |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for model in RUNS:
        for task in ("factuality", "bias"):
            m = report["models"][model][task]
            counts = " | ".join(str(m["categories"][c]) for c in CATEGORIES)
            lines.append(f"| {MODEL_NAMES[model]} | {task.capitalize()} | {counts} | {m['provider_summaries']}/{m['n']} |")
    lines.extend(["", "## Interpretation and audit", "",
                  "- Common structured final self-report text was coded; unequal provider-only summaries were retained but excluded from the comparative reference rate.",
                  "- This is explicitly elicited recall, not spontaneous reference prevalence or a verified training-contamination estimate.",
                  "- Correct among references means the final classification matched the dataset label; it does not validate the recalled external rating or prove causal influence.",
                  "- One LLM reviewer read the saved reports with targets and correctness hidden; no inter-rater reliability is claimed. Ambiguous and generic cases are excluded from primary references.",
                  "- Missing native summaries stay missing; all metrics use actual recorded decisions, with no imputation for unrecorded Opus cases.",
                  "- Manual review and aggregation are offline and make no API calls. Source hashes, snapshot hash, rubric hash, and annotation hashes are verified or retained in summary.json; coded_records.csv contains per-decision codes.", ""])
    return "\n".join(lines)


def aggregate(root):
    saved, rows, annotations, annotation_sources = _load_review(root)
    report, indexed = {}, {}
    for model in RUNS:
        indexed[model] = {}
        for task in ("factuality", "bias"):
            actual = [r for r in rows.values() if r["model"] == model and r["task"] == task]
            if not actual:
                raise ValueError(f"No recorded decisions: {model} {task}")
            report.setdefault(model, {})[task] = _summarize(actual, annotations, saved["expected_per_task"])
            indexed[model][task] = {int(r["id"].split(":")[-1]): r for r in actual}
    common = set.intersection(*(set(indexed[model][task]) for model in RUNS for task in ("factuality", "bias")))
    if not common:
        raise ValueError("No shared outlets across models and tasks")
    for i in common:
        names = {indexed[model][task][i]["media_name"] for model in RUNS for task in ("factuality", "bias")}
        if len(names) != 1:
            raise ValueError(f"Matched outlet identity mismatch: {i}")
        for task in ("factuality", "bias"):
            if len({indexed[model][task][i]["true_label"] for model in RUNS}) != 1:
                raise ValueError(f"Matched target mismatch: {i} {task}")
    matched = {model: {task: _summarize([indexed[model][task][i] for i in sorted(common)], annotations, len(common))
                       for task in ("factuality", "bias")} for model in RUNS}
    result = {
        "scope": "Manual coding of common structured self-report text, not unequal provider-only summaries",
        "record_count": len(rows), "reviewed_decisions": len(annotations),
        "subset_sha256": saved["subset_sha256"], "snapshot_sha256": sha256_file(root / "snapshot.json"),
        "rubric_sha256": sha256_file(root / "JUDGE_PROMPT.md"), "annotation_sources": annotation_sources,
        "models": report,
        "matched": {"outlet_count": len(common), "outlet_indices": sorted(common), "models": matched},
    }
    provenance = root / "review_provenance.json"
    if provenance.exists():
        result["review_provenance"] = json.loads(provenance.read_text(encoding="utf-8"))
    write_json(root / "summary.json", result)
    with (root / "coded_records.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["id", "model", "task", "media_name", "category", "explicit_reference", "correct", "source", "source_sha256"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for id, r in sorted(rows.items()):
            writer.writerow({**{f: r[f] for f in fields if f in r}, "category": annotations[id],
                             "explicit_reference": annotations[id] in PRIMARY})
    markdown = _report_markdown(result)
    (root / "REPORT.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('freeze','read','aggregate'));p.add_argument('--root',required=True)
    p.add_argument('--from-review',type=Path,help='inherit annotations only when every previously reviewed response is unchanged')
    p.add_argument('--model',choices=RUNS);p.add_argument('--task',choices=('factuality','bias'));p.add_argument('--offset',type=int,default=0);p.add_argument('--limit',type=int,default=30)
    a=p.parse_args();root=Path(a.root)
    if a.action=='freeze':freeze(root,a.from_review)
    elif a.action=='read':read(root,a.model,a.task,a.offset,a.limit)
    else:aggregate(root)
