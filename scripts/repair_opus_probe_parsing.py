#!/usr/bin/env python3
"""Reparse saved Opus outputs under an audited parser policy, without API calls."""
from __future__ import annotations

import argparse
import json
import shutil
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from filelock import FileLock

import eval_opus_summary_parallel as runner
from run_config import sha256_file, write_json


def validate_unchanged_request(plan: dict) -> None:
    controls = runner.request_body("factuality", [], plan["output_config"]["effort"], plan["max_output_tokens"])
    expected = {"model": runner.MODEL, "thinking": controls["thinking"], "output_config": controls["output_config"],
                "system": runner.BASE_SYSTEM,
                "prompts": {task: runner.probe_prompt(task) for task in ("factuality", "bias")}}
    for field, value in expected.items():
        if plan[field] != value:
            raise ValueError(f"Request protocol changed at {field}; refusing parser-only repair")


def reparse(row: dict, allow_missing: bool = False) -> dict:
    parsed = runner.decode_response(row["response"], row["task"], allow_missing)
    if not runner.acceptable(parsed):
        raise ValueError(f"Saved case still fails validation: {row['case_id']}: {parsed['validation_error']}")
    return {**row, **parsed, "correct": parsed["predicted_label"] == row["true_label"],
            "parser_recovery_provenance": {"type": "parser-and-observation-policy-v3", "response_reused": True,
                "previous_status": row["status"], "previous_validation_error": row.get("validation_error"),
                "reparsed_at": datetime.now(timezone.utc).isoformat()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--record-missing-summaries", action="store_true")
    args = parser.parse_args()
    root = Path(args.run_dir)
    with FileLock(str(root / "run.lock"), timeout=0):
        plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
        validate_unchanged_request(plan)
        cases = {row["case_id"]: row for row in plan["cases"]}
        records = runner.load_records(root / "records", cases)
        eligible = {"invalid_output", "missing_provider_summary"} if args.record_missing_summaries else {"invalid_output"}
        repairs = {id: reparse(row, args.record_missing_summaries) for id, row in records.items() if row["status"] in eligible}
        # Validate the entire repair before changing any active record file.
        if any(not runner.acceptable(row) for id, row in records.items() if id not in repairs):
            raise ValueError("Non-parser failures exist; refusing to declare smoke ready")
        backup = root / "parser_backup"
        backup.mkdir(exist_ok=True)
        if not (backup / "plan.original.json").exists():
            shutil.copyfile(root / "plan.json", backup / "plan.original.json")
        lock = threading.Lock()
        provenance = []
        for id, repaired in repairs.items():
            path = root / "records" / f"{id}.json"
            original_hash = sha256_file(path)
            destination = backup / f"{id}.original.json"
            if destination.exists():
                raise ValueError(f"Original record backup already exists: {destination}")
            shutil.copyfile(path, destination)
            if sha256_file(destination) != original_hash:
                raise ValueError("Backup verification failed")
            write_json(path, repaired)
            runner.append_event(root / "predictions.jsonl", repaired, lock)
            records[id] = repaired
            provenance.append({"case_id": id, "original_sha256": original_hash,
                               "reparsed_sha256": sha256_file(path), "backup": str(destination)})
        updated_plan = {**plan, "implementation_sha256": sha256_file(runner.__file__),
                        "mandatory_native_summary": not args.record_missing_summaries}
        runner.append_event(root / "parser_repairs.jsonl", {"type": "parser-and-observation-policy-upgrade",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "original_runner_sha256": plan["implementation_sha256"],
            "new_runner_sha256": updated_plan["implementation_sha256"],
            "request_protocol_unchanged": True,
            "missing_possible_contamination_policy": "Preserve missing field; count not_reported; never infer model yes/no/unknown.",
            "missing_provider_summary_policy": "preserved as unavailable, never generated again" if args.record_missing_summaries else "mandatory",
            "additional_api_calls": 0, "records": provenance}, lock)
        write_json(root / "plan.json", updated_plan)
        old_summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        task_summary = {}
        for task in ("factuality", "bias"):
            actual = [row for row in records.values() if row["task"] == task]
            reports = [row["structured_assessment"] for row in actual if row["output_valid"]]
            task_summary[task] = {"expected": 212, "recorded": len(actual),
                "status_counts": dict(Counter(row["status"] for row in actual)),
                "provider_summaries": sum(bool(row["provider_summary"]) for row in actual),
                "missing_provider_summaries": sum(not bool(row["provider_summary"]) for row in actual),
                "valid_structured_outputs": len(reports), "correct_predictions": sum(row["correct"] for row in actual),
                "overall_accuracy": sum(row["correct"] for row in actual) / 212 if len(actual) == 212 else None,
                "prior_rating_influence_self_reports": dict(Counter(row["prior_rating_influence"] for row in reports)),
                "possible_contamination_self_reports": dict(Counter(row.get("possible_contamination", "not_reported") for row in reports)),
                "missing_contamination_self_report": sum("possible_contamination" not in row for row in reports)}
        write_json(root / "summary.json", {**old_summary, "status": "smoke_ready_to_resume", "tasks": task_summary,
                   "parser_repaired_records": len(repairs), "parser_repair_api_calls": 0})
        print(f"Repaired {len(repairs)} saved outputs; all {len(records)} saved cases now validate. Additional API calls: 0.")


if __name__ == "__main__":
    main()
