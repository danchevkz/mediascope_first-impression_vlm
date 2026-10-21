#!/usr/bin/env python3
"""Freeze, display, and aggregate evidence-backed manual recognition-probe codes."""
import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

from run_config import sha256_file, write_json


SUBSET = Path("artifacts/audits/interpretability_subset_seed42_50pct_20261010/subset.csv")
RUBRIC = Path("scripts/recognition_probe_judge_v2.md")
FIELDS = ("recognized_outlet", "recognized_network", "external_rating_source",
          "recalled_factuality_rating", "recalled_bias_rating", "self_report_basis", "explanation_summary")
CATEGORIES = ("asserted", "uncertain_recall", "ambiguous", "generic", "none")
PRIMARY = ("asserted", "uncertain_recall")


def freeze(root, run):
    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    if plan["subset_csv_sha256"] != sha256_file(SUBSET):
        raise ValueError("Subset hash changed")
    with SUBSET.open(newline="", encoding="utf-8") as handle:
        selected = list(csv.DictReader(handle))
    indices = {(r["media_name"], r["url"]): i for i, r in enumerate(selected, 1)}
    targets = {(r["media_name"], r["url"], t): r[f"true_label_{t}"] for r in selected for t in ("factuality", "bias")}
    cases = {r["case_id"]: r for r in plan["cases"]}
    records = []
    for path in sorted((run / "records").glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        case = cases[row["case_id"]]
        for field in ("task", "media_name", "url", "true_label", "image_sha256"):
            if row[field] != case[field]:
                raise ValueError(f"Case identity changed: {path}")
        key = (row["media_name"], row["url"])
        if row["true_label"] != targets[(*key, row["task"])]:
            raise ValueError("Subset target mismatch")
        observed = row["status"] in ("ok", "ok_without_provider_summary") and row["output_valid"]
        assessment = row["structured_assessment"] if observed else None
        if observed and row["correct"] != (assessment["label"] == row["true_label"]):
            raise ValueError("Incorrect saved correctness flag")
        records.append({"id": f"{row['task']}:{indices[key]:03d}", "task": row["task"],
                        "media_name": row["media_name"], "observed": observed, "structured": assessment,
                        "provider_summary": row["provider_summary"], "correct": row["correct"],
                        "status": row["status"], "source": str(path), "source_sha256": sha256_file(path)})
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Duplicate saved decision")
    records.sort(key=lambda r: r["id"])
    root.mkdir(parents=True, exist_ok=False)
    (root / "annotations").mkdir()
    shutil.copyfile(RUBRIC, root / "JUDGE_PROMPT.md")
    write_json(root / "snapshot.json", {"model": plan["model"], "run": str(run), "expected_per_task": len(selected),
                                       "subset_sha256": sha256_file(SUBSET), "records": records})
    print(f"Frozen {len(records)} decisions; {sum(r['observed'] for r in records)} valid structured reports.")


def read(root, task, offset, limit):
    snapshot = json.loads((root / "snapshot.json").read_text(encoding="utf-8"))
    rows = [r for r in snapshot["records"] if r["task"] == task][offset:offset+limit]
    for row in rows:
        print(f"\n[{row['id']}] {row['media_name']}")
        if not row["observed"]:
            print(f"UNOBSERVED: {row['status']}")
            continue
        a = row["structured"]
        print(f"Source={a['recognized_outlet']}; Network={a['recognized_network']}")
        print(f"Rater={a['external_rating_source']}; F={a['recalled_factuality_rating']}; B={a['recalled_bias_rating']}")
        print(f"Basis: {a['self_report_basis']}")
        print(f"Explanation: {a['explanation_summary']}")


def aggregate(root):
    snapshot = json.loads((root / "snapshot.json").read_text(encoding="utf-8"))
    rows = {r["id"]: r for r in snapshot["records"]}
    codes, hashes = {}, []
    for path in sorted((root / "annotations").glob("*.json")):
        batch = json.loads(path.read_text(encoding="utf-8"))
        reviewed = set()
        for start, end in batch["reviewed_ranges"]:
            if not 1 <= start <= end <= snapshot["expected_per_task"]:
                raise ValueError("Invalid reviewed range")
            for i in range(start, end + 1):
                id = f"{batch['task']}:{i:03d}"
                reviewed.add(str(i))
                if id not in rows or id in codes:
                    raise ValueError(f"Unknown/duplicate reviewed decision: {id}")
                if not rows[id]["observed"]:
                    codes[id] = {"category": "unobserved", "rating_scope": "none", "family_link": "not_applicable", "evidence": []}
                    continue
                code = batch.get("decisions", {}).get(str(i), {"category": "none", "rating_scope": "none",
                                                              "family_link": "not_applicable", "evidence": []})
                category, scope, link = code["category"], code["rating_scope"], code["family_link"]
                if category not in CATEGORIES or scope not in ("outlet", "named_family", "generic_type", "unclear", "none"):
                    raise ValueError(f"Invalid manual code: {id}")
                if link not in ("identified_by_response", "tentative_or_inferred", "not_applicable"):
                    raise ValueError(f"Invalid family-link code: {id}")
                if category in PRIMARY and scope not in ("outlet", "named_family"):
                    raise ValueError(f"Primary reference has no anchored attribution: {id}")
                if scope != "named_family" and link != "not_applicable":
                    raise ValueError(f"Family-link code without family rating: {id}")
                evidence = code.get("evidence", [])
                text = "\n".join(str(rows[id]["structured"].get(f) or "") for f in FIELDS)
                if category != "none" and (not evidence or any(not quote or quote not in text for quote in evidence)):
                    raise ValueError(f"Evidence is not an exact saved excerpt: {id}")
                codes[id] = code
        if not set(batch.get("decisions", {})).issubset(reviewed):
            raise ValueError("Decision outside reviewed ranges")
        hashes.append({"path": str(path), "sha256": sha256_file(path)})
    missing = [id for id, r in rows.items() if r["observed"] and id not in codes]
    if missing:
        raise ValueError(f"Incomplete manual review: {len(missing)} valid reports remain")
    for id, row in rows.items():
        if sha256_file(row["source"]) != row["source_sha256"]:
            raise ValueError(f"Saved response changed: {id}")
        if not row["observed"]:
            codes.setdefault(id, {"category": "unobserved", "rating_scope": "none", "family_link": "not_applicable", "evidence": []})
    tasks = {}
    for task in ("factuality", "bias"):
        actual = [r for r in rows.values() if r["task"] == task]
        observed = [r for r in actual if r["observed"]]
        n = len(observed)
        refs = [r for r in observed if codes[r["id"]]["category"] in PRIMARY]
        categories = Counter(codes[r["id"]]["category"] for r in observed)
        correct_refs = sum(r["correct"] for r in refs)
        correct_all = sum(r["correct"] for r in observed)
        tasks[task] = {
            "recorded": len(actual), "valid_reports": n, "expected": snapshot["expected_per_task"],
            "categories": {c: categories[c] for c in CATEGORIES}, "references": len(refs),
            "reference_rate": len(refs) / n if n else None,
            "correct_reference_predictions": correct_refs,
            "accuracy_among_references": correct_refs / len(refs) if refs else None,
            "correct_predictions": correct_all,
            "overall_accuracy": correct_all / snapshot["expected_per_task"]
            if len(actual) == snapshot["expected_per_task"] else None,
            "observed_reference_lower_bound": len(refs) / len(actual) if actual else None,
            "asserted_only_rate": categories["asserted"] / n if n else None,
            "outlet_references": sum(codes[r["id"]]["rating_scope"] == "outlet" for r in refs),
            "family_references": sum(codes[r["id"]]["rating_scope"] == "named_family" for r in refs),
            "tentative_family_references": sum(codes[r["id"]]["family_link"] == "tentative_or_inferred" for r in refs),
            "prior_rating_influence": dict(Counter(r["structured"]["prior_rating_influence"] for r in observed)),
            "possible_contamination": dict(Counter(r["structured"].get("possible_contamination", "not_reported") for r in observed)),
            "provider_reasoning_available": sum(bool(r["provider_summary"].strip()) for r in actual),
        }
    report = {"model": snapshot["model"], "coding_version": 2, "tasks": tasks,
              "snapshot_sha256": sha256_file(root / "snapshot.json"), "rubric_sha256": sha256_file(root / "JUDGE_PROMPT.md"),
              "annotation_sources": hashes, "scope": "Manually judged common structured reports; native-only claims excluded"}
    write_json(root / "summary.json", report)
    with (root / "coded_records.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["id", "task", "media_name", "observed", "category", "rating_scope", "family_link", "evidence", "note", "source_sha256"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for id, row in sorted(rows.items()):
            code = codes[id]
            writer.writerow({**{f: row[f] for f in fields if f in row}, **{f: code.get(f, "") for f in ("category", "rating_scope", "family_link", "note")},
                             "evidence": json.dumps(code.get("evidence", []), ensure_ascii=False)})
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "read", "aggregate"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--task", choices=("factuality", "bias"))
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=25)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.root, args.run)
    elif args.action == "read":
        read(args.root, args.task, args.offset, args.limit)
    else:
        aggregate(args.root)
