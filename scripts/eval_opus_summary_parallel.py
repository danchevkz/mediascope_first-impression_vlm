#!/usr/bin/env python3
"""Concurrent Opus 5 recognition probe with mandatory observable summaries.

This is an elicited prior-knowledge probe, not spontaneous-reference prevalence
or a detector of training contamination. A failed summary check stops new calls.
"""
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
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import requests
from filelock import FileLock

from eval_api import DiskImages, load_test_samples
from mediascope_eval import TASKS
from qwen_predictor import PROMPTS
from run_config import json_hash, sha256_file, write_json


MODEL = "claude-opus-5"
ENDPOINT = "https://api.anthropic.com/v1/messages"
SELF_REPORT_VALUES = {"yes", "no", "unknown"}
BASE_SYSTEM = (
    "This task requires careful source assessment. Think carefully before responding. "
    "Return the requested concise JSON assessment and evidence summary. "
    "Distinguish visible screenshot evidence from source recognition and recalled external ratings. "
    "Report uncertainty honestly; do not invent a recognized source, a rating, or knowledge of your training data."
)


def probe_prompt(task: str) -> str:
    base = PROMPTS[task].rsplit("\n\nOutput ONLY", 1)[0]
    return base + """

Also provide an explicit source-recognition and prior-rating-knowledge self-report.
Without browsing, say whether you recognize the outlet or its network and whether
you recall an already existing external factuality or political-bias rating.
Do not guess ratings merely to fill the fields: null and unknown are valid.
Report whether prior ratings influenced this classification. You cannot verify
training-set membership; possible_contamination is only your uncertain self-report
about possible benchmark exposure, NOT proof of training contamination. Use
unknown when you cannot determine exposure; recalling a rating does not require
a yes answer. Do not claim verified membership or verified absence from training.

Return ONLY one JSON object with all these fields:
{
  "label": "one allowed label for the requested task",
  "explanation_summary": "1-3 sentences explaining observable evidence and any prior-rating influence",
  "recognized_outlet": null,
  "recognized_network": null,
  "external_rating_source": null,
  "recalled_factuality_rating": null,
  "recalled_bias_rating": null,
  "prior_rating_influence": "yes, no, or unknown",
  "possible_contamination": "yes, no, or unknown",
  "self_report_basis": "brief basis for the prior-knowledge self-report; do not claim verified training membership"
}
Use a string or null for each recognized-source and recalled-rating field.
The explanation_summary is required even if you do not recognize the publisher.
"""


def request_body(task: str, urls: list[str], effort: str, max_tokens: int) -> dict:
    return {
        "model": MODEL,
        "max_tokens": max_tokens,
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": effort},
        "system": BASE_SYSTEM,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": probe_prompt(task)},
            *[{"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                             "data": url.split(",", 1)[1]}} for url in urls],
        ]}],
    }


def parse_assessment(final_text: str) -> dict:
    parse_text = final_text
    lines = parse_text.splitlines()
    if len(lines) >= 3 and lines[0].strip().lower() in ("```json", "```") and lines[-1].strip() == "```":
        parse_text = "\n".join(lines[1:-1]).strip()
    try:
        return json.loads(parse_text)
    except ValueError as original_error:
        objects, block = [], None
        for line in lines:
            if block is None and line.strip().lower() in ("```json", "```"):
                block = []
            elif block is not None and line.strip() == "```":
                value = json.loads("\n".join(block))
                if isinstance(value, dict):
                    objects.append(value)
                block = None
            elif block is not None:
                block.append(line)
        required = {"label", "explanation_summary", "self_report_basis", "prior_rating_influence",
                    "recognized_outlet", "recognized_network", "external_rating_source",
                    "recalled_factuality_rating", "recalled_bias_rating"}
        complete = [obj for obj in objects if required.issubset(obj)]
        if len(complete) != 1:
            raise original_error
        chosen = complete[0]
        if any(obj.get("label", chosen["label"]) != chosen["label"] for obj in objects):
            raise ValueError("conflicting labels in response JSON sections")
        return chosen


def decode_response(payload: dict, task: str, allow_missing: bool = False) -> dict:
    blocks = payload.get("content") or []
    provider_summary = "\n\n".join(b.get("thinking", "") for b in blocks if b.get("type") == "thinking").strip()
    final_text = "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
    result = {"provider_summary": provider_summary, "provider_summary_available": bool(provider_summary),
              "provider_summary_origin": "anthropic_thinking_summary" if provider_summary else "unavailable", "final_text": final_text,
              "structured_assessment": None, "output_valid": False, "predicted_label": None,
              "status": "invalid_output", "validation_error": None, "validation_warnings": []}
    try:
        data = parse_assessment(final_text)
        if not isinstance(data, dict):
            raise ValueError("final JSON must be an object")
        if data.get("label") not in TASKS[task]["order"]:
            raise ValueError("invalid or missing task label")
        for field in ("explanation_summary", "self_report_basis"):
            if not isinstance(data.get(field), str) or not data[field].strip():
                raise ValueError(f"nonempty {field} required")
        if data.get("prior_rating_influence") not in SELF_REPORT_VALUES:
            raise ValueError("prior_rating_influence must be yes/no/unknown")
        if "possible_contamination" in data:
            if data["possible_contamination"] not in SELF_REPORT_VALUES:
                raise ValueError("possible_contamination must be yes/no/unknown")
        else:
            # Preserve absence rather than inventing a model admission or an unknown answer.
            result["validation_warnings"].append("possible_contamination_not_reported")
        for field in ("recognized_outlet", "recognized_network", "external_rating_source",
                      "recalled_factuality_rating", "recalled_bias_rating"):
            if field not in data or (data[field] is not None and (not isinstance(data[field], str) or not data[field].strip())):
                raise ValueError(f"{field} must be a nonempty string or null")
        result.update(structured_assessment=data, output_valid=True, predicted_label=data["label"],
                      possible_contamination_reported="possible_contamination" in data)
    except (ValueError, TypeError) as exc:
        result["validation_error"] = str(exc)
        return result
    if payload.get("model") != MODEL:
        result.update(status="model_mismatch", validation_error="response model does not match claude-opus-5")
    elif payload.get("stop_reason") != "end_turn":
        result.update(status="incomplete_response", validation_error=f"stop_reason={payload.get('stop_reason')}")
    elif not provider_summary:
        result.update(status="ok_without_provider_summary" if allow_missing else "missing_provider_summary",
                      validation_error=None if allow_missing else "no readable native thinking summary returned")
        if allow_missing:
            result["validation_warnings"].append("native_summary_unavailable")
    else:
        result["status"] = "ok"
    return result


def acceptable(row: dict) -> bool:
    return row["status"] in {"ok", "ok_without_provider_summary"} and row.get("output_valid", False)


def retry_seconds(value: str | None, retry: int) -> float:
    fallback = min(2 ** (retry + 1), 30)
    try:
        supplied = float(value)
    except (TypeError, ValueError):
        try:
            supplied = parsedate_to_datetime(value).timestamp() - time.time()
        except (TypeError, ValueError, OverflowError):
            supplied = 0
    return max(fallback, supplied) if math.isfinite(supplied) else fallback


def redact_payload(value, key: str):
    if isinstance(value, str):
        return value.replace(key, "[REDACTED]") if key else value
    if isinstance(value, dict):
        return {name: redact_payload(item, key) for name, item in value.items()}
    if isinstance(value, list):
        return [redact_payload(item, key) for item in value]
    return value


class RequestGate:
    def __init__(self, interval: float, stop: threading.Event):
        self.interval, self.stop, self.next_request = interval, stop, 0.0
        self.lock = threading.Lock()

    def wait(self) -> bool:
        while not self.stop.is_set():
            with self.lock:
                delay = self.next_request - time.monotonic()
                if delay <= 0:
                    self.next_request = time.monotonic() + self.interval
                    return True
            self.stop.wait(min(delay, 0.5))
        return False

    def defer(self, seconds: float) -> None:
        with self.lock:
            self.next_request = max(self.next_request, time.monotonic() + seconds)


def append_event(path: Path, value: dict, lock: threading.Lock) -> None:
    with lock, path.open("ab+") as stream:
        if stream.tell():
            stream.seek(-1, os.SEEK_END)
            if stream.read(1) != b"\n":
                stream.seek(0, os.SEEK_END)
                stream.write(b"\n")
        stream.seek(0, os.SEEK_END)
        stream.write((json.dumps(value, ensure_ascii=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())


def load_records(directory: Path, expected: dict[str, dict]) -> dict[str, dict]:
    records = {}
    if not directory.exists():
        return records
    for path in directory.glob("*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row["case_id"] not in expected:
            raise ValueError(f"unknown saved case: {path}")
        case = expected[row["case_id"]]
        for field in ("task", "media_name", "url", "image_sha256", "true_label"):
            if row[field] != case[field]:
                raise ValueError(f"saved case identity mismatch: {path}, {field}")
        records[row["case_id"]] = row
    return records


def select_smoke_cases(cases: list[dict], subset: list[dict]) -> list[dict]:
    chosen = []
    for wanted in ("HIGH", "MIXED", "LOW"):
        row = next(r for r in subset if r["true_label_factuality"] == wanted)
        chosen.append((row["media_name"], row["url"]))
    row = next(r for r in subset if r["true_label_bias"] == "LEFT")
    chosen.append((row["media_name"], row["url"]))
    return [c for c in cases if (c["media_name"], c["url"]) in chosen]


def execute(cases: list[dict], worker, workers: int, stop: threading.Event, on_record) -> None:
    pending = iter(cases)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        active = {}

        def submit() -> None:
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
                    if not acceptable(row):
                        stop.set()
            submit()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="artifacts/datasets/mediascope-clustered-main")
    parser.add_argument("--subset-csv", default="artifacts/audits/interpretability_subset_seed42_50pct_20261010/subset.csv")
    parser.add_argument("--input-run-dir", default="artifacts/evaluations/api-visible-rationales-stratified50-20261010")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--min-interval", type=float, default=0.5)
    parser.add_argument("--effort", choices=("high", "xhigh", "max"), default="high")
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--record-missing-summaries", action="store_true",
                        help="retain valid structured outputs when native summaries are unavailable; do not regenerate them")
    args = parser.parse_args()
    if not 1 <= args.workers <= 32 or not 0 <= args.retries <= 5:
        parser.error("workers must be 1-32; retries must be 0-5")
    if not math.isfinite(args.min_interval) or args.min_interval < 0 or min(args.timeout, args.max_output_tokens) <= 0:
        parser.error("invalid pacing, timeout or token limit")
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not args.dry_run and (not key or "\n" in key or "\r" in key):
        parser.error("export ANTHROPIC_API_KEY in this shell")
    root, inputs, output = Path(args.dataset_root).resolve(), Path(args.input_run_dir).resolve(), Path(args.output_dir)
    with open(args.subset_csv, newline="", encoding="utf-8") as stream:
        subset = list(csv.DictReader(stream))
    subset_map = {(r["media_name"], r["url"]): r for r in subset}
    if len(subset_map) != len(subset) or len(subset) != 212:
        parser.error("expected the fixed, unique 212-outlet subset")
    samples = [s for s in load_test_samples(root) if (s.media_name, s.url) in subset_map]
    if len(samples) != 212:
        parser.error("not all selected outlets were matched")
    index = json.loads((inputs / "inputs.json").read_text(encoding="utf-8"))
    images = DiskImages(inputs, index)
    cases = []
    for sample in samples:
        selected = subset_map[(sample.media_name, sample.url)]
        if sample.factuality != selected["true_label_factuality"] or sample.bias != selected["true_label_bias"]:
            parser.error(f"subset labels disagree: {sample.media_name}")
        entry = images[sample.image_path]  # Verifies every saved submitted PNG hash.
        if entry["image"]["truncated_px"] or entry["image"]["retained_height_fraction"] != 1:
            parser.error(f"input truncation: {sample.media_name}")
        for task in ("factuality", "bias"):
            case = {"task": task, "media_name": sample.media_name, "url": sample.url,
                    "image_path": sample.image_path, "image_sha256": sha256_file(sample.image_path),
                    "true_label": getattr(sample, task), "submitted_png_sha256": entry["png_sha256"]}
            case["case_id"] = json_hash(case)
            cases.append(case)
    smoke = select_smoke_cases(cases, subset)
    controls = request_body("factuality", [], args.effort, args.max_output_tokens)
    plan = {"protocol": "opus5-summary-explicit-prior-knowledge-probe-v1", "model": MODEL,
            "cases": cases, "subset_csv_sha256": sha256_file(args.subset_csv),
            "source_inputs_sha256": sha256_file(inputs / "inputs.json"), "source_inputs_dir": str(inputs),
            "thinking": controls["thinking"], "output_config": controls["output_config"],
            "system": BASE_SYSTEM, "prompts": {t: probe_prompt(t) for t in ("factuality", "bias")},
            "max_output_tokens": args.max_output_tokens, "timeout_seconds": args.timeout,
            "mandatory_native_summary": not args.record_missing_summaries, "smoke_case_ids": [c["case_id"] for c in smoke],
            "implementation_sha256": sha256_file(__file__)}
    print(f"{MODEL}: 212 outlets, 424 decisions; {args.workers} concurrent requests after {len(smoke)}-request smoke gate.", flush=True)
    print(f"Thinking controls: {plan['thinking']}; effort={args.effort}; reusable full-page input hashes passed.", flush=True)
    print(f"Missing native summaries {'recorded as unavailable' if args.record_missing_summaries else 'stop new calls'}.", flush=True)
    if args.dry_run:
        print("Dry run passed; no paid calls or output writes.")
        return 0
    if args.resume:
        if not (output / "plan.json").is_file() or json.loads((output / "plan.json").read_text()) != plan:
            parser.error("resume requires identical source inputs, protocol and code")
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "plan.json", plan)
    records_dir = output / "records"
    records_dir.mkdir(exist_ok=True)
    with FileLock(str(output / "run.lock"), timeout=0):
        records = load_records(records_dir, {c["case_id"]: c for c in cases})
        if any(not acceptable(r) for r in records.values()):
            parser.error("this run has a failed observability/output check; inspect records rather than automatically resampling it")
        stop = threading.Event()
        gate = RequestGate(args.min_interval, stop)
        log_lock = threading.Lock()
        started = time.monotonic()
        if args.resume:
            journal = output / "predictions.jsonl"
            journaled = set()
            if journal.is_file():
                for line in journal.read_bytes().splitlines(keepends=True):
                    if line.endswith(b"\n") and line.strip():
                        try:
                            journaled.add(json.loads(line)["case_id"])
                        except json.JSONDecodeError:
                            pass
            for case_id, row in records.items():
                if case_id not in journaled:
                    append_event(journal, {**row, "recovered_from_atomic_record": True}, log_lock)

        def worker(case):
            entry = images[case["image_path"]]
            body = request_body(case["task"], entry["data_urls"], args.effort, args.max_output_tokens)
            payload, http, error = None, None, None
            for retry in range(args.retries + 1):
                if not gate.wait():
                    return None
                call_started = time.monotonic()
                error = None
                try:
                    response = requests.post(ENDPOINT, headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                                                               "Content-Type": "application/json"},
                                             json=body, timeout=(15, args.timeout))
                    http = response.status_code
                    try:
                        payload = redact_payload(response.json(), key)
                    except ValueError:
                        payload = {"error": {"message": response.text.replace(key, "[REDACTED]")}}
                    if http >= 400 or payload.get("error"):
                        error = json.dumps(payload.get("error"), ensure_ascii=False).replace(key, "[REDACTED]")
                except requests.RequestException as exc:
                    http, payload, error = None, None, type(exc).__name__
                elapsed = time.monotonic() - call_started
                append_event(output / "requests.jsonl", {"case_id": case["case_id"], "task": case["task"],
                             "recorded_at": datetime.now(timezone.utc).isoformat(), "attempt": retry + 1,
                             "http_status": http, "response": payload, "error": error, "elapsed_seconds": elapsed,
                             "request_controls": {"model": MODEL, "thinking": body["thinking"], "output_config": body["output_config"]}}, log_lock)
                if error is None:
                    break
                retryable = http is None or http in (408, 409, 429) or 500 <= http <= 599
                if retryable and retry < args.retries:
                    delay = retry_seconds(response.headers.get("retry-after") if http is not None else None, retry)
                    gate.defer(delay)
                    print(f"[retry] {case['media_name']} {case['task']} HTTP={http}; shared cooldown {delay:.1f}s", flush=True)
                    continue
                break
            parsed = decode_response(payload, case["task"], args.record_missing_summaries) if error is None and isinstance(payload, dict) else {
                "status": "http_error", "provider_summary": "", "final_text": "", "structured_assessment": None,
                "output_valid": False, "predicted_label": None, "validation_error": error,
            }
            row = {**case, **parsed, "correct": parsed["predicted_label"] == case["true_label"],
                   "response": payload, "http_status": http, "usage": (payload or {}).get("usage"),
                   "recorded_at": datetime.now(timezone.utc).isoformat()}
            # Each worker writes its own atomic file before notifying the collector.
            write_json(records_dir / f"{case['case_id']}.json", row)
            if not acceptable(row):
                stop.set()
            return row

        def save_summary(state):
            task_summary = {}
            for task in ("factuality", "bias"):
                actual = [r for r in records.values() if r["task"] == task]
                reports = [r["structured_assessment"] for r in actual if r.get("output_valid")]
                task_summary[task] = {"expected": 212, "recorded": len(actual),
                    "status_counts": dict(Counter(r["status"] for r in actual)),
                    "provider_summaries": sum(bool(r["provider_summary"]) for r in actual),
                    "missing_provider_summaries": sum(not bool(r["provider_summary"]) for r in actual),
                    "valid_structured_outputs": len(reports),
                    "correct_predictions": sum(r["correct"] for r in actual),
                    "overall_accuracy": sum(r["correct"] for r in actual) / 212 if len(actual) == 212 else None,
                    "prior_rating_influence_self_reports": dict(Counter(r["prior_rating_influence"] for r in reports)),
                    "possible_contamination_self_reports": dict(Counter(r.get("possible_contamination", "not_reported") for r in reports)),
                    "missing_contamination_self_report": sum("possible_contamination" not in r for r in reports)}
            write_json(output / "summary.json", {"status": state, "model": MODEL, "tasks": task_summary,
                       "elapsed_seconds": time.monotonic() - started,
                       "interpretation": "Explicitly elicited model self-reports; not verified training contamination."})

        def on_record(row):
            records[row["case_id"]] = row
            append_event(output / "predictions.jsonl", row, log_lock)
            save_summary("running" if acceptable(row) else "stopped_failed_check")
            print(f"[{len(records)}/424] {row['media_name']} {row['task']}: {row['status']}; "
                  f"native_summary_chars={len(row['provider_summary'])}; warnings={row.get('validation_warnings', [])}", flush=True)

        def handle_signal(signum, frame):
            stop.set()
            print("Stopping new calls; saving any in-flight responses.", flush=True)

        old_handlers = {sig: signal.signal(sig, handle_signal) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            execute([c for c in smoke if c["case_id"] not in records], worker, min(4, args.workers), stop, on_record)
            smoke_ok = all(acceptable(records[c["case_id"]]) if c["case_id"] in records else False for c in smoke)
            if not smoke_ok:
                save_summary("stopped_smoke_failed")
                print("Smoke gate failed. No full sweep launched; inspect saved raw responses.", flush=True)
                return 1
            native_count = sum(bool(records[c["case_id"]]["provider_summary"]) for c in smoke)
            print(f"SMOKE PASSED: all eight structured explanations validate; {native_count}/8 native summaries observed.", flush=True)
            if not args.smoke_only and not stop.is_set():
                execute([c for c in cases if c["case_id"] not in records], worker, args.workers, stop, on_record)
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
        complete = len(records) == 424 and all(acceptable(r) for r in records.values())
        state = "complete" if complete else "smoke_complete" if args.smoke_only else "stopped_failed_check" if any(not acceptable(r) for r in records.values()) else "interrupted"
        save_summary(state)
        print(f"Finished: {state}; {len(records)}/424 records saved.", flush=True)
        return 0 if complete or args.smoke_only else 1


if __name__ == "__main__":
    raise SystemExit(main())
