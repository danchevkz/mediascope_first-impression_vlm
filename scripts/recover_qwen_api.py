#!/usr/bin/env python3
"""Recover Qwen transport failures into a derived audit, retaining original results.

Imports successful predictions verbatim in substance, including wrong labels.
Content filters, token-budget exhaustion and genuine invalid labels are recorded
as terminal model/provider outcomes, not repeatedly sampled for a better answer.
"""

import argparse
import copy
import json
import math
import os
import shutil
import signal
import threading
import time
import uuid
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from filelock import FileLock

from eval_api import DiskImages, EvaluationStopped, RateGate, usage_summary
from eval_audit import EvaluationAudit
from mediascope_eval import Sample, TASKS, audit_metrics, normalize_to_label
from qwen_predictor import PROMPTS
from resilient_api import APIResponseError, ResilientAPIPredictor, classify_record
from run_config import environment_info, json_hash, sha256_file, source_info, write_json


MODEL = "qwen/qwen3.8-max-prime"
SOURCE_FILES = ("plan.json", "inputs.json", "qwen.audit/manifest.json", "qwen.audit/predictions.jsonl", "qwen.requests.jsonl")


def read_source(root):
    plan = json.loads((root / "plan.json").read_text())
    manifest = json.loads((root / "qwen.audit/manifest.json").read_text())
    identity = manifest["identity"]
    if json_hash(identity) != manifest["identity_sha256"]:
        raise ValueError("Source manifest identity hash is invalid")
    predictor = identity["predictor"]
    if predictor["backend"] != "openrouter" or predictor["model_id"] != MODEL:
        raise ValueError("Expected the original OpenRouter Qwen3.8 Max Prime run")
    if identity["parser"] != {"mode": "strict", "version": 1}:
        raise ValueError("Only the original strict parser contract can be recovered")
    if predictor["prompts"] != PROMPTS:
        raise ValueError("Prompts differ from the original run")
    controls = predictor["request_controls"]
    expected = {"model": MODEL, "reasoning": {"enabled": True, "effort": controls["reasoning"]["effort"]},
                "max_tokens": controls["max_tokens"], "provider": {"require_parameters": True}}
    if controls != expected:
        raise ValueError("Unsupported source generation controls; refusing to silently change them")
    if not identity["tasks"] or any(t not in TASKS for t in identity["tasks"]):
        raise ValueError("Invalid source tasks")
    samples = {s["sample_id"]: s for s in identity["samples"]}
    if len(samples) != len(identity["samples"]):
        raise ValueError("Duplicate source samples")
    if any(s["labels"][t] not in TASKS[t]["order"] for s in samples.values() for t in identity["tasks"]):
        raise ValueError("Recovery requires valid source targets for every selected task")
    expected_keys = {(sample_id, task) for sample_id in samples for task in identity["tasks"]}
    latest = {}
    with (root / "qwen.audit/predictions.jsonl").open("rb") as stream:
        for line in stream:
            if not line.endswith(b"\n"):
                raise ValueError("Source journal has a torn final write; original history was not modified")
            record = json.loads(line)
            key = (record["sample_id"], record["task"])
            if key not in expected_keys:
                raise ValueError("Unknown source sample/task")
            sample = samples[key[0]]
            if (record["true_label"] != sample["labels"][key[1]]
                    or any(record[k] != sample[k] for k in ("media_name", "url", "image_path", "image_sha256"))):
                raise ValueError("Source record does not match its sample/label contract")
            if record["status"] == "ok":
                parsed = normalize_to_label(record["raw_text"], TASKS[key[1]]["order"])
                if parsed is None or parsed != record["predicted_label"] or classify_record(record):
                    raise ValueError("A source success has inconsistent response/label metadata")
                if record["correct"] != (parsed == record["true_label"]):
                    raise ValueError("Source correctness flag is inconsistent")
            latest[key] = record
    return {"plan": plan, "identity": identity, "records": latest,
            "inputs": json.loads((root / "inputs.json").read_text())}


def verify_inputs(source, asset_root, dataset_root=None):
    old_root = Path(source["plan"]["dataset_root"])
    data = Path(dataset_root).resolve() if dataset_root else old_root
    if sha256_file(data / "test.csv") != source["plan"]["test_csv_sha256"]:
        raise ValueError("The source test CSV changed")
    for sample in source["identity"]["samples"]:
        raw = data / Path(sample["image_path"]).relative_to(old_root)
        if sha256_file(raw) != sample["image_sha256"]:
            raise ValueError(f"Source screenshot changed: {sample['media_name']}")
        entry = source["inputs"][sample["image_path"]]
        image = entry["image"]
        if image["truncated_px"] or image["retained_height_fraction"] != 1:
            raise ValueError("Source inputs do not cover the full screenshot")
        if not len(entry["files"]) == len(entry["png_sha256"]) == image["tile_count"]:
            raise ValueError("Source tile inventory is inconsistent")
        for filename, expected in zip(entry["files"], entry["png_sha256"]):
            path = (asset_root / filename).resolve()
            if not path.is_relative_to(asset_root.resolve()) or sha256_file(path) != expected:
                raise ValueError(f"Source input tile changed or has an invalid path: {filename}")
    for record in source["records"].values():
        submitted = record.get("details", {}).get("submitted_png_sha256")
        if submitted is not None and submitted != source["inputs"][record["image_path"]]["png_sha256"]:
            raise ValueError("Recorded submitted images disagree with the saved input index")


def source_counts(source):
    counts = Counter()
    for record in source["records"].values():
        failure = classify_record(record)
        counts["retained_valid" if failure is None and record["status"] == "ok"
               else failure.kind if failure else record["status"]] += 1
        counts["eligible_transport_recovery"] += bool(failure and (failure.retryable or failure.fatal))
    expected = len(source["identity"]["samples"]) * len(source["identity"]["tasks"])
    counts["missing_source_records"] = expected - len(source["records"])
    return dict(counts)


def seed_records(audit, source):
    for key, original in source["records"].items():
        if key in audit.records:
            continue
        record = copy.deepcopy(original)
        failure = classify_record(record)
        record["recovery_provenance"] = {
            "origin": "source", "source_record_sha256": json_hash(original),
            "source_status": original["status"], "source_attempt": original.get("attempt"),
            "source_recorded_at": original.get("recorded_at"),
            "failure_classification": asdict(failure) if failure else None,
        }
        if failure and original["status"] != "ok":
            record.update(status="error", predicted_label=None, correct=False,
                          error={"type": "RecordedAPIFailure", **asdict(failure)})
        audit.append(record)


def eligible(record, retry_errors=False):
    if record is None:
        return True
    if record["status"] == "ok":
        return False
    provenance = record.get("recovery_provenance") or {}
    classification = provenance.get("failure_classification")
    if not classification or not (classification["retryable"] or classification["fatal"]):
        return False
    return provenance.get("origin") == "source" or retry_errors


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-dir", default="artifacts/evaluations/api-full-hf-20261001")
    ap.add_argument("--output-dir", required=True, help="new derived run; the original is preserved")
    ap.add_argument("--dataset-root", help="optional relocated copy of the identical HF dataset")
    ap.add_argument("--min-interval", type=float, default=2)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--backoff-base", type=float, default=5)
    ap.add_argument("--backoff-max", type=float, default=120)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--retry-errors", action="store_true", help="retry exhausted transient failures on resume")
    ap.add_argument("--dry-run", action="store_true", help="validate and count; no writes or API calls")
    args = ap.parse_args()
    if not 0 <= args.retries <= 10 or not all(math.isfinite(v) for v in (args.min_interval, args.backoff_base, args.backoff_max)):
        ap.error("Invalid bounded retry settings")
    if args.min_interval < 0 or not 0 <= args.backoff_base <= args.backoff_max or args.backoff_max <= 0:
        ap.error("Invalid pacing/backoff settings")
    if args.retry_errors and not args.resume:
        ap.error("--retry-errors requires --resume")
    origin, output = Path(args.source_dir).resolve(), Path(args.output_dir).resolve()
    if output == origin or output.is_relative_to(origin) or origin.is_relative_to(output):
        ap.error("Source and recovery output directories must be disjoint")
    snapshot = output / "source_snapshot"
    if args.resume:
        if not (output / "plan.json").is_file():
            ap.error("Resume requires the recovery run's plan.json")
        source = read_source(snapshot)
        hashes = {name: sha256_file(snapshot / name) for name in SOURCE_FILES}
    else:
        if output.exists():
            ap.error("Output already exists; use --resume or a new directory")
        with FileLock(str(origin / "run.lock"), timeout=0), FileLock(str(origin / "qwen.audit/writer.lock"), timeout=0):
            source = read_source(origin)
            verify_inputs(source, origin, args.dataset_root)
            hashes = {name: sha256_file(origin / name) for name in SOURCE_FILES}
            if not args.dry_run:
                if not os.environ.get("OPENROUTER_API_KEY", "").strip():
                    ap.error("Export OPENROUTER_API_KEY in this terminal")
                snapshot.mkdir(parents=True, exist_ok=False)
                for name in SOURCE_FILES:
                    target = snapshot / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(origin / name, target)
                    if sha256_file(target) != hashes[name]:
                        raise ValueError("Source snapshot copy failed verification")
    if args.resume:
        verify_inputs(source, origin, args.dataset_root)
        saved = json.loads((output / "plan.json").read_text())
        expected = {
            "protocol": "qwen-transient-recovery-v1", "source_dir": str(origin),
            "source_files_sha256": hashes,
            "dataset_root_override": str(Path(args.dataset_root).resolve()) if args.dataset_root else None,
            "min_interval": args.min_interval, "runner_sha256": sha256_file(__file__),
            "source": source_info(), "environment": environment_info(),
        }
        policy = saved["predictor"]["retry_policy"]
        code = Path(__file__).resolve().parent
        if (any(saved.get(k) != v for k, v in expected.items())
                or policy["retries"] != args.retries or policy["backoff_base"] != args.backoff_base
                or policy["backoff_max"] != args.backoff_max
                or saved["predictor"]["resilient_implementation_sha256"] != sha256_file(code / "resilient_api.py")
                or saved["predictor"]["implementation_sha256"] != sha256_file(code / "api_predictor.py")
                or saved["predictor"]["request_controls"] != source["identity"]["predictor"]["request_controls"]):
            ap.error("Recovery code/settings/source snapshot changed; refusing incompatible resume")
    counts = source_counts(source)
    print("Original Qwen outcome classification:", json.dumps(counts), flush=True)
    print("Preserved request controls:", json.dumps(source["identity"]["predictor"]["request_controls"]), flush=True)
    if args.dry_run:
        print("Dry run passed. No inference requests or result writes.")
        return 0
    if not os.environ.get("OPENROUTER_API_KEY", "").strip():
        ap.error("Export OPENROUTER_API_KEY in this terminal")
    stop, session_id = threading.Event(), uuid.uuid4().hex
    controls = source["identity"]["predictor"]["request_controls"]
    predictor = ResilientAPIPredictor(
        "openrouter", MODEL, DiskImages(origin, source["inputs"]), source["identity"]["predictor"]["preprocessing"],
        max_output_tokens=controls["max_tokens"], effort=controls["reasoning"]["effort"],
        timeout=source["identity"]["predictor"]["timeout_seconds"], gate=RateGate(args.min_interval, stop),
        request_log=output / "qwen.requests.jsonl", session_id=session_id, retries=args.retries,
        backoff_base=args.backoff_base, backoff_max=args.backoff_max, recover=args.resume,
    )
    metadata = predictor.run_metadata()
    if metadata["request_controls"] != controls:
        ap.error("Recovery request controls differ from the source")
    plan = {"protocol": "qwen-transient-recovery-v1", "source_dir": str(origin), "source_files_sha256": hashes,
            "dataset_root_override": str(Path(args.dataset_root).resolve()) if args.dataset_root else None,
            "predictor": metadata, "min_interval": args.min_interval,
            "runner_sha256": sha256_file(__file__), "source": source_info(), "environment": environment_info()}
    identity = {"tasks": source["identity"]["tasks"], "samples": source["identity"]["samples"],
                "parser": source["identity"]["parser"], "predictor": metadata, "context": plan,
                "source": source_info(), "environment": environment_info()}
    with FileLock(str(output / "run.lock"), timeout=0):
        if args.resume and json.loads((output / "plan.json").read_text()) != plan:
            ap.error("Recovery code/settings/source snapshot changed; refusing incompatible resume")
        if not args.resume:
            write_json(output / "plan.json", plan)
        with (output / "sessions.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"id": session_id, "started_at": datetime.now(timezone.utc).isoformat(),
                                     "retry_errors": args.retry_errors}) + "\n")

        def stop_requests(signum, frame):
            stop.set()
            print("Stopping after the current request; recovery history will be saved.", flush=True)

        previous = {sig: signal.signal(sig, stop_requests) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            with EvaluationAudit(output / "qwen.csv", identity,
                                 resume=(output / "qwen.audit/manifest.json").exists()) as audit:
                seed_records(audit, source)
                work = [(sample, task) for sample in identity["samples"] for task in identity["tasks"]
                        if eligible(audit.records.get((sample["sample_id"], task)), args.retry_errors)]
                print(f"Recovery requests selected: {len(work)}; valid source predictions will not be resampled.", flush=True)
                audit.save_metrics(audit_metrics(audit), "recovering")
                processed = 0
                try:
                    for sample, task in work:
                        if stop.is_set():
                            break
                        item = Sample(sample["media_name"], sample["url"], sample["image_path"], "",
                                      sample["labels"].get("factuality"), sample["labels"].get("bias"))
                        record = {k: v for k, v in sample.items() if k != "labels"}
                        record.update(task=task, true_label=sample["labels"][task], raw_text=None,
                                      predicted_label=None, correct=False, status="error", error=None,
                                      details={}, elapsed_seconds=0.0)
                        failure = None
                        started = time.perf_counter()
                        try:
                            answer = predictor.predict(item, task)
                            label = normalize_to_label(answer, TASKS[task]["order"])
                            record.update(raw_text=answer, predicted_label=label,
                                          correct=label == record["true_label"], status="ok" if label else "unparseable")
                        except APIResponseError as exc:
                            failure = asdict(exc.failure)
                            record["error"] = {"type": "APIResponseError", **failure}
                        except EvaluationStopped:
                            break
                        record["details"] = copy.deepcopy(predictor.last_details)
                        record["elapsed_seconds"] = time.perf_counter() - started
                        record["recovery_provenance"] = {
                            "origin": "recovery", "session_id": session_id,
                            "source_record_sha256": json_hash(source["records"][(sample["sample_id"], task)])
                            if (sample["sample_id"], task) in source["records"] else None,
                            "failure_classification": failure,
                        }
                        audit.append(record)
                        processed += 1
                        if processed % 10 == 0:
                            audit.export_csv()
                            audit.save_metrics(audit_metrics(audit), "recovering")
                            print(f"Recovered/attempted {processed}/{len(work)} selected records", flush=True)
                finally:
                    scores = audit_metrics(audit)
                    classifications = Counter()
                    for record in audit.records.values():
                        classification = (record.get("recovery_provenance") or {}).get("failure_classification")
                        classifications[classification["kind"] if classification else record["status"]] += 1
                    failures = any(r["status"] not in ("ok", "not_applicable") for r in audit.records.values())
                    status = "blocked" if predictor.fatal_error else "interrupted" if stop.is_set() or processed < len(work) else "complete_with_failures" if failures else "complete"
                    audit.export_csv()
                    audit.save_metrics(scores, status)
                    result = {"status": status, "source_counts": counts,
                              "selected_this_session": len(work), "processed_this_session": processed,
                              "recovery_records_total": sum((r.get("recovery_provenance") or {}).get("origin") == "recovery"
                                                            for r in audit.records.values()),
                              "retryable_failures_remaining": sum(eligible(r, True) for r in audit.records.values()),
                              "remaining_initial_recovery": sum(eligible(r, False) for r in audit.records.values())
                              + len(identity["samples"]) * len(identity["tasks"]) - len(audit.records),
                              "outcome_counts": dict(classifications), "metrics": scores,
                              "requests": {"source": usage_summary(snapshot / "qwen.requests.jsonl"),
                                           "recovery": usage_summary(output / "qwen.requests.jsonl")}}
                    write_json(output / "summary.json", {"models": {"qwen": result}})
                    print("Recovery outcomes:", json.dumps(result["outcome_counts"]), flush=True)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    return 130 if stop.is_set() and not predictor.fatal_error else int(status != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
