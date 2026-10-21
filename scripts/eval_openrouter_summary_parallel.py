#!/usr/bin/env python3
"""Comparable Kimi/Qwen recognition probes, with durable provider-response capture."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import requests
from filelock import FileLock

import eval_opus_summary_parallel as common
from eval_api import DiskImages
from eval_gpt56_summary_parallel import assessment_schema
from mediascope_eval import TASKS
from resilient_api import classify_failure
from run_config import json_hash, sha256_file, write_json


MODELS = {"kimi": "moonshotai/kimi-k3", "qwen": "qwen/qwen3.8-max-prime"}
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
SUBSET = "artifacts/audits/interpretability_subset_seed42_50pct_20261010/subset.csv"
REFERENCE = "artifacts/evaluations/opus5-summary-probe-parallel-20261010/plan.json"


def request_body(model, task, urls, effort, max_tokens):
    return {
        "model": model, "reasoning": {"enabled": True, "effort": effort, "exclude": False},
        "max_tokens": max_tokens, "provider": {"require_parameters": True},
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "source_rating_self_report", "strict": True, "schema": assessment_schema(task)}},
        "messages": [
            {"role": "system", "content": common.BASE_SYSTEM},
            {"role": "user", "content": [{"type": "text", "text": common.probe_prompt(task)}]
             + [{"type": "image_url", "image_url": {"url": url, "detail": "high"}} for url in urls]},
        ],
    }


def reasoning_text(message):
    for field in ("reasoning", "reasoning_content"):
        value = message.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip(), f"openrouter_{field}"
    parts, kinds = [], []
    for detail in message.get("reasoning_details") or []:
        if not isinstance(detail, dict):
            continue
        kind = detail.get("type")
        field = "summary" if kind == "reasoning.summary" else "text" if kind == "reasoning.text" else None
        value = detail.get(field) if field else None
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
            kinds.append(kind)
    return "\n\n".join(parts), "+".join(dict.fromkeys(kinds)) or "unavailable"


def decode_response(payload, task, model, allow_missing=False, http_status=200, error=""):
    choices = payload.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    summary, kind = reasoning_text(message)
    content = message.get("content")
    final = content if isinstance(content, str) else "\n".join(
        b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text"
        and isinstance(b.get("text"), str)) if isinstance(content, list) else ""
    result = {
        "provider_summary": summary, "provider_summary_kind": kind,
        "provider_summary_available": bool(summary), "provider_summary_origin": kind,
        "final_text": final, "structured_assessment": None, "output_valid": False,
        "predicted_label": None, "status": "invalid_output", "validation_error": None,
        "validation_warnings": [], "finish_reason": choice.get("finish_reason"),
        "response_model": payload.get("model"), "response_provider": payload.get("provider"),
        "failure_classification": None,
    }
    if payload.get("model") is not None and payload["model"] != model:
        result.update(status="model_mismatch", validation_error=f"Expected {model}, received {payload.get('model')}")
        return result
    failure = classify_failure({"response": payload, "http_status": http_status, "request_sent": True}, error)
    if failure:
        result.update(status="content_filtered" if failure.kind == "provider_content_filter" else "http_error",
                      validation_error=failure.message, failure_classification=asdict(failure))
        if failure.kind == "output_limit":
            result["status"] = "incomplete_response"
        return result
    if payload.get("model") != model:
        result.update(status="model_mismatch", validation_error=f"Expected {model}, received {payload.get('model')}")
        return result
    if len(choices) != 1 or choice.get("finish_reason") != "stop":
        result.update(status="incomplete_response", validation_error="Expected one completed stop-finish choice")
        return result
    try:
        data = common.parse_assessment(final)
        if not isinstance(data, dict) or data.get("label") not in TASKS[task]["order"]:
            raise ValueError("Invalid or missing task label")
        for field in ("explanation_summary", "self_report_basis"):
            if not isinstance(data.get(field), str) or not data[field].strip():
                raise ValueError(f"Nonempty {field} required")
        if data.get("prior_rating_influence") not in common.SELF_REPORT_VALUES:
            raise ValueError("prior_rating_influence must be yes/no/unknown")
        if "possible_contamination" in data:
            if data["possible_contamination"] not in common.SELF_REPORT_VALUES:
                raise ValueError("possible_contamination must be yes/no/unknown")
        else:
            result["validation_warnings"].append("possible_contamination_not_reported")
        for field in ("recognized_outlet", "recognized_network", "external_rating_source",
                      "recalled_factuality_rating", "recalled_bias_rating"):
            if field not in data or (data[field] is not None and
                    (not isinstance(data[field], str) or not data[field].strip())):
                raise ValueError(f"{field} must be a nonempty string or null")
        result.update(structured_assessment=data, predicted_label=data["label"], output_valid=True)
    except (ValueError, TypeError) as exc:
        result["validation_error"] = str(exc)
        return result
    if summary:
        result["status"] = "ok"
    elif allow_missing:
        result["status"] = "ok_without_provider_summary"
        result["validation_warnings"].append("native_reasoning_unavailable")
    else:
        result.update(status="missing_provider_summary", validation_error="No readable provider reasoning returned")
    return result


def output_limited(row):
    return (row["status"] == "incomplete_response"
            and (row.get("failure_classification") or {}).get("kind") == "output_limit")


def completed_invalid_report(row):
    payload = row.get("response") or {}
    choices = payload.get("choices") or []
    return (row["status"] == "invalid_output" and row.get("http_status") == 200
            and row.get("finish_reason") == "stop" and len(choices) == 1
            and payload.get("model") == row.get("response_model")
            and row.get("response_model") in MODELS.values())


def acceptable(row, record_refusals=False, record_output_limits=False, record_invalid_reports=False):
    return (common.acceptable(row) or (record_refusals and row["status"] == "content_filtered")
            or (record_output_limits and output_limited(row))
            or (record_invalid_reports and completed_invalid_report(row)))


def load_retry_plan(root, path, records, expected):
    """One explicit generation retry per archived failed case; never requeue it on resume."""
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("generation_attempts_authorized") != 1:
        raise ValueError("Retry authorization must specify one generation attempt")
    pending, seen = {}, set()
    digest = sha256_file(path)
    for item in data["cases"]:
        id = item["case_id"]
        if id not in expected or id not in records or id in seen:
            raise ValueError(f"Unknown/duplicate retry case: {id}")
        seen.add(id)
        backup = (root / item["backup"]).resolve()
        if not backup.is_relative_to(root.resolve()) or sha256_file(backup) != item["original_record_sha256"]:
            raise ValueError(f"Retry response backup changed: {id}")
        original = json.loads(backup.read_text(encoding="utf-8"))
        if not output_limited(original):
            raise ValueError(f"Only the authorized output-limit failures may be rerun: {id}")
        for field in ("case_id", "task", "media_name", "url", "image_sha256", "true_label"):
            if original[field] != expected[id][field]:
                raise ValueError(f"Retry input identity changed: {id}")
        current = root / "records" / f"{id}.json"
        if sha256_file(current) == item["original_record_sha256"]:
            pending[id] = {**item, "retry_plan": str(path), "retry_plan_sha256": digest, "user_requested": True}
        else:
            provenance = records[id].get("retry_provenance") or {}
            if (provenance.get("original_record_sha256") != item["original_record_sha256"]
                    or provenance.get("retry_plan_sha256") != digest
                    or records[id].get("generation_attempt") != 2):
                raise ValueError(f"Retry case changed without authorized-rerun provenance: {id}")
    return pending


def execute(cases, worker, workers, stop, on_record, record_refusals=False, record_output_limits=False, record_invalid_reports=False):
    pending = iter(cases)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        active = {}
        def submit():
            while len(active) < workers and not stop.is_set():
                case = next(pending, None)
                if case is None:
                    break
                active[pool.submit(worker, case)] = case
        submit()
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                active.pop(future)
                row = future.result()
                if row is not None:
                    on_record(row)
                    if not acceptable(row, record_refusals, record_output_limits, record_invalid_reports):
                        stop.set()
            submit()


def prepare_inputs(reference_path, subset_path):
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if reference["model"] != common.MODEL or reference["system"] != common.BASE_SYSTEM:
        raise ValueError("Reference model/system differs from the fixed Opus probe")
    if reference["prompts"] != {t: common.probe_prompt(t) for t in ("factuality", "bias")}:
        raise ValueError("Reference prompts differ")
    if sha256_file(subset_path) != reference["subset_csv_sha256"]:
        raise ValueError("Subset hash differs")
    with subset_path.open(newline="", encoding="utf-8") as stream:
        subset = list(csv.DictReader(stream))
    selected = {(r["media_name"], r["url"]): r for r in subset}
    if len(subset) != 212 or len(selected) != 212:
        raise ValueError("Expected 212 unique selected outlets")
    directory = Path(reference["source_inputs_dir"])
    if sha256_file(directory / "inputs.json") != reference["source_inputs_sha256"]:
        raise ValueError("Saved image inventory changed")
    images = DiskImages(directory, json.loads((directory / "inputs.json").read_text(encoding="utf-8")))
    cases = reference["cases"]
    if len(cases) != 424 or len({c["case_id"] for c in cases}) != 424:
        raise ValueError("Expected 424 unique decisions")
    expected = {(name, url, task) for name, url in selected for task in ("factuality", "bias")}
    if {(c["media_name"], c["url"], c["task"]) for c in cases} != expected:
        raise ValueError("Reference cases do not match the fixed paired subset")
    for case in cases:
        row = selected[(case["media_name"], case["url"])]
        if case["true_label"] != row[f"true_label_{case['task']}"]:
            raise ValueError("Reference target differs")
        if case["case_id"] != json_hash({k: v for k, v in case.items() if k != "case_id"}):
            raise ValueError("Reference case fingerprint differs")
        if sha256_file(case["image_path"]) != case["image_sha256"]:
            raise ValueError("Source screenshot changed")
        entry = images[case["image_path"]]
        if (entry["png_sha256"] != case["submitted_png_sha256"] or entry["image"]["truncated_px"]
                or entry["image"]["retained_height_fraction"] != 1):
            raise ValueError("Submitted tiles changed or do not cover the full page")
    return reference, cases, subset, images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-plan", default=REFERENCE)
    parser.add_argument("--subset-csv", default=SUBSET)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--min-interval", type=float, default=1)
    parser.add_argument("--effort", choices=("high",), default="high")
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--record-missing-summaries", action="store_true")
    parser.add_argument("--record-provider-refusals", action="store_true")
    parser.add_argument("--record-output-limits", action="store_true",
                        help="retain output-budget failures as terminal cases without automatic regeneration")
    parser.add_argument("--record-invalid-reports", action="store_true",
                        help="retain completed malformed assessments as terminal failures without regeneration")
    parser.add_argument("--retry-plan", type=Path,
                        help="explicit authorization for one rerun of each archived output-limit failure")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.retry_plan and not args.resume:
        parser.error("--retry-plan requires --resume")
    if (not 1 <= args.workers <= 32 or not 0 <= args.retries <= 5
            or not math.isfinite(args.min_interval) or args.min_interval < 0
            or min(args.timeout, args.max_output_tokens) <= 0):
        parser.error("Invalid concurrency, pacing, retry, timeout or token limits")
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not args.dry_run and (not key or "\n" in key or "\r" in key):
        parser.error("Export OPENROUTER_API_KEY in this shell")
    model, output = MODELS[args.model], Path(args.output_dir)
    reference_path = Path(args.reference_plan)
    reference, cases, subset, images = prepare_inputs(reference_path, Path(args.subset_csv))
    if args.max_output_tokens != reference["max_output_tokens"]:
        parser.error("Token cap must match the fixed GPT/Opus probe")
    smoke = common.select_smoke_cases(cases, subset)
    controls = request_body(model, "factuality", [], args.effort, args.max_output_tokens)
    source = Path(__file__).parent
    plan = {
        "protocol": "openrouter-explicit-prior-knowledge-probe-v1", "model": model, "model_tag": args.model,
        "cases": cases, "reference_plan_sha256": sha256_file(reference_path),
        "subset_csv_sha256": reference["subset_csv_sha256"], "source_inputs_dir": reference["source_inputs_dir"],
        "source_inputs_sha256": reference["source_inputs_sha256"], "system": common.BASE_SYSTEM,
        "prompts": reference["prompts"], "reasoning": controls["reasoning"], "provider": controls["provider"],
        "schemas": {t: assessment_schema(t) for t in ("factuality", "bias")},
        "max_output_tokens": args.max_output_tokens, "timeout_seconds": args.timeout,
        "mandatory_native_summary": not args.record_missing_summaries,
        "record_provider_refusals": args.record_provider_refusals, "smoke_case_ids": [c["case_id"] for c in smoke],
        "record_output_limits": args.record_output_limits,
        "record_invalid_reports": args.record_invalid_reports,
        "implementation_sha256": {name: sha256_file(source / name) for name in (
            Path(__file__).name, "eval_opus_summary_parallel.py", "eval_gpt56_summary_parallel.py", "resilient_api.py")},
        "manual_reference_analysis": "Pending manual coding; keep asserted/uncertain and outlet/named-family/analogy scope separate.",
        "reference_denominators": "Refusals are unobserved reports, not no-reference evidence; report valid-output coverage separately.",
    }
    if args.resume and (not (output / "plan.json").is_file()
                        or json.loads((output / "plan.json").read_text()) != plan):
        parser.error("Resume requires identical model, inputs, prompts, code and observation policy")
    print(f"{model}: 212 outlets, 424 decisions; {args.workers} concurrent requests after eight-case smoke gate.", flush=True)
    print("Same GPT/Opus prompts and full-page tile hashes verified; reasoning enabled with exclude=false.", flush=True)
    expected = {c["case_id"]: c for c in cases}
    if args.dry_run:
        if args.retry_plan:
            saved = common.load_records(output / "records", expected)
            pending = load_retry_plan(output, args.retry_plan, saved, expected)
            print(f"Authorized failed-case reruns pending: {len(pending)}; unrecorded decisions: {424-len(saved)}.")
        print("Dry run passed; no paid calls or output writes.")
        return 0
    if not args.resume:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "plan.json", plan)
    record_dir = output / "records"
    record_dir.mkdir(exist_ok=True)
    with FileLock(str(output / "run.lock"), timeout=0):
        records = common.load_records(record_dir, expected)
        if any(not acceptable(r, args.record_provider_refusals, args.record_output_limits, args.record_invalid_reports) for r in records.values()):
            parser.error("An unsuccessful output/transport check exists; inspect saved responses before resuming")
        retry_cases = load_retry_plan(output, args.retry_plan, records, expected)
        print(f"Authorized failed-case reruns pending: {len(retry_cases)}; unrecorded decisions: {424-len(records)}.", flush=True)
        stop, log_lock = threading.Event(), threading.Lock()
        gate = common.RequestGate(args.min_interval, stop)
        started = time.monotonic()
        if args.resume:
            journal = output / "predictions.jsonl"
            journaled = set()
            if journal.exists():
                for line in journal.read_bytes().splitlines(keepends=True):
                    if line.endswith(b"\n"):
                        try:
                            journaled.add(json.loads(line)["case_id"])
                        except (ValueError, KeyError):
                            pass
            for id, row in records.items():
                if id not in journaled:
                    common.append_event(journal, {**row, "recovered_from_atomic_record": True}, log_lock)

        def worker(case):
            body = request_body(model, case["task"], images[case["image_path"]]["data_urls"], args.effort, args.max_output_tokens)
            payload, http, error = {}, None, ""
            for retry in range(args.retries + 1):
                if not gate.wait():
                    return None
                called, error, retry_after = time.monotonic(), "", None
                try:
                    response = requests.post(ENDPOINT, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                                             json=body, timeout=(15, args.timeout))
                    http, retry_after = response.status_code, response.headers.get("retry-after")
                    try:
                        payload = common.redact_payload(response.json(), key)
                        if not isinstance(payload, dict):
                            payload, error = {}, "Invalid non-object provider response"
                    except ValueError:
                        payload = {"error": {"message": response.text.replace(key, "[REDACTED]")}}
                except requests.RequestException as exc:
                    http, payload, error = None, {}, type(exc).__name__
                failure = classify_failure({"response": payload, "http_status": http, "request_sent": True}, error)
                common.append_event(output / "requests.jsonl", {
                    "case_id": case["case_id"], "task": case["task"], "model": model,
                    "recorded_at": datetime.now(timezone.utc).isoformat(), "attempt": retry + 1,
                    "generation_attempt": 2 if case["case_id"] in retry_cases else 1,
                    "retry_provenance": retry_cases.get(case["case_id"]),
                    "http_status": http, "response": payload, "error": error or None,
                    "failure_classification": asdict(failure) if failure else None,
                    "elapsed_seconds": time.monotonic() - called,
                    "request_controls": {k: body[k] for k in ("model", "reasoning", "provider", "response_format", "max_tokens")},
                }, log_lock)
                if not failure or not failure.retryable or retry == args.retries:
                    break
                delay = common.retry_seconds(retry_after, retry)
                gate.defer(delay)
                print(f"[retry] {case['media_name']} {case['task']}: {failure.kind}; cooldown={delay:.1f}s", flush=True)
            parsed = decode_response(payload, case["task"], model, args.record_missing_summaries, http, error)
            row = {**case, **parsed, "correct": parsed["predicted_label"] == case["true_label"],
                   "response": payload, "http_status": http, "usage": payload.get("usage"),
                   "recorded_at": datetime.now(timezone.utc).isoformat(),
                   "generation_attempt": 2 if case["case_id"] in retry_cases else 1}
            if case["case_id"] in retry_cases:
                row["retry_provenance"] = retry_cases[case["case_id"]]
            write_json(record_dir / f"{case['case_id']}.json", row)
            if not acceptable(row, args.record_provider_refusals, args.record_output_limits, args.record_invalid_reports):
                stop.set()
            return row

        def save_summary(state):
            tasks = {}
            for task in ("factuality", "bias"):
                actual = [r for r in records.values() if r["task"] == task]
                reports = [r["structured_assessment"] for r in actual if common.acceptable(r)]
                valid = len(reports)
                tasks[task] = {
                    "expected": 212, "recorded": len(actual), "status_counts": dict(Counter(r["status"] for r in actual)),
                    "valid_structured_outputs": valid, "valid_output_coverage": valid / len(actual) if actual else None,
                    "provider_summaries": sum(bool(r["provider_summary"]) for r in actual),
                    "missing_provider_summaries": sum(not bool(r["provider_summary"]) for r in actual),
                    "correct_predictions": sum(r["correct"] for r in actual),
                    "overall_accuracy": sum(r["correct"] for r in actual) / 212 if len(actual) == 212 else None,
                    "prior_rating_influence_self_reports": dict(Counter(r["prior_rating_influence"] for r in reports)),
                    "possible_contamination_self_reports": dict(Counter(r.get("possible_contamination", "not_reported") for r in reports)),
                }
            write_json(output / "summary.json", {"status": state, "model": model, "tasks": tasks,
                "elapsed_seconds": time.monotonic() - started,
                "user_requested_reruns_recorded": sum(bool(r.get("retry_provenance")) for r in records.values()),
                "interpretation": "Explicitly elicited self-reports, not verified contamination; reference coding pending."})

        def on_record(row):
            records[row["case_id"]] = row
            common.append_event(output / "predictions.jsonl", row, log_lock)
            save_summary("running" if acceptable(row, args.record_provider_refusals, args.record_output_limits, args.record_invalid_reports) else "stopped_failed_check")
            print(f"[{len(records)}/424] {row['media_name']} {row['task']}: {row['status']}; native_reasoning_chars={len(row['provider_summary'])}; generation_attempt={row.get('generation_attempt',1)}", flush=True)

        def handle_signal(signum, frame):
            stop.set()
            print("Stopping new calls; saving in-flight responses.", flush=True)
        old = {sig: signal.signal(sig, handle_signal) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            smoke_ids = {c["case_id"] for c in smoke}
            execute([c for c in smoke if c["case_id"] not in records or c["case_id"] in retry_cases],
                    worker, min(4, args.workers), stop, on_record, args.record_provider_refusals, args.record_output_limits, args.record_invalid_reports)
            smoke_ok = (all(c["case_id"] in records and acceptable(records[c["case_id"]], args.record_provider_refusals, args.record_output_limits, args.record_invalid_reports) for c in smoke)
                        and all(any(c["task"] == t and c["case_id"] in records and common.acceptable(records[c["case_id"]])
                                    for c in smoke) for t in ("factuality", "bias")))
            if not smoke_ok:
                save_summary("stopped_smoke_failed")
                print("Smoke failed: full sweep not started; inspect saved responses.", flush=True)
                return 1
            print("SMOKE PASSED: valid structured reports observed for both tasks; provider refusals retained separately.", flush=True)
            if not args.smoke_only and not stop.is_set():
                work = ([c for c in cases if c["case_id"] in retry_cases and c["case_id"] not in smoke_ids]
                        + [c for c in cases if c["case_id"] not in records])
                execute(work, worker, args.workers, stop, on_record, args.record_provider_refusals, args.record_output_limits, args.record_invalid_reports)
        finally:
            for sig, handler in old.items():
                signal.signal(sig, handler)
        complete = len(records) == 424 and all(acceptable(r, args.record_provider_refusals, args.record_output_limits, args.record_invalid_reports) for r in records.values())
        state = ("complete_with_terminal_outcomes" if any(r["status"] in ("content_filtered", "incomplete_response", "invalid_output") for r in records.values()) else "complete") if complete else (
            "smoke_complete" if args.smoke_only else "stopped_failed_check" if any(not acceptable(r, args.record_provider_refusals, args.record_output_limits, args.record_invalid_reports) for r in records.values()) else "interrupted")
        save_summary(state)
        print(f"Finished: {state}; {len(records)}/424 decisions saved.", flush=True)
        return 0 if complete or args.smoke_only else 1


if __name__ == "__main__":
    raise SystemExit(main())
