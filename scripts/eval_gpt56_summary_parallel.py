#!/usr/bin/env python3
"""GPT-5.6 counterpart to the explicit Opus source-recognition/summary probe."""
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
from pathlib import Path

import requests
from filelock import FileLock

import eval_opus_summary_parallel as common
from eval_api import DiskImages, load_test_samples
from mediascope_eval import TASKS
from gpt_summary_stream import stream_result
from run_config import json_hash, sha256_file, write_json


MODEL = "gpt-5.6-sol"
ENDPOINT = "https://api.openai.com/v1/responses"


def assessment_schema(task: str) -> dict:
    props = {
        "label": {"type": "string", "enum": TASKS[task]["order"]},
        "explanation_summary": {"type": "string"},
        "prior_rating_influence": {"type": "string", "enum": ["yes", "no", "unknown"]},
        "possible_contamination": {"type": "string", "enum": ["yes", "no", "unknown"]},
        "self_report_basis": {"type": "string"},
    }
    for name in ("recognized_outlet", "recognized_network", "external_rating_source",
                 "recalled_factuality_rating", "recalled_bias_rating"):
        props[name] = {"type": ["string", "null"]}
    return {"type": "object", "additionalProperties": False, "properties": props, "required": list(props)}


def request_body(task: str, urls: list[str], effort: str, max_tokens: int, stream: bool = False) -> dict:
    body = {
        "model": MODEL, "store": False,
        "reasoning": {"effort": effort, "summary": "detailed"},
        "max_output_tokens": max_tokens,
        "instructions": common.BASE_SYSTEM,
        "text": {"format": {"type": "json_schema", "name": "source_rating_self_report", "strict": True,
                             "schema": assessment_schema(task)}},
        "input": [{"role": "user", "content": [
            {"type": "input_text", "text": common.probe_prompt(task)},
            *[{"type": "input_image", "image_url": url, "detail": "high"} for url in urls],
        ]}],
    }
    if stream:
        body["stream"] = True
    return body


def decode_response(payload: dict, task: str, stream_summary: str = "", allow_missing: bool = False) -> dict:
    reasoning = [r for r in payload.get("output", []) if r.get("type") == "reasoning"]
    response_summary = "\n\n".join(
        block.get("text", "") for item in reasoning for block in item.get("summary", [])
        if isinstance(block, dict) and isinstance(block.get("text"), str)
    ).strip()
    provider_summary = response_summary or stream_summary
    messages = [r for r in payload.get("output", []) if r.get("type") == "message"
                and r.get("role") == "assistant" and r.get("phase") != "commentary"]
    final_text = "\n".join(b.get("text", "") for b in messages[-1].get("content", [])
                           if b.get("type") == "output_text").strip() if messages else ""
    result = {"provider_summary": provider_summary, "provider_summary_kind": "openai_reasoning_summary",
              "provider_summary_available": bool(provider_summary),
              "provider_summary_origin": "response_reasoning_summary" if response_summary else "actual_sse_summary_events" if stream_summary else "unavailable",
              "response_summary_chars": len(response_summary), "stream_summary_chars": len(stream_summary),
              "final_text": final_text, "structured_assessment": None, "output_valid": False,
              "predicted_label": None, "status": "invalid_output", "validation_error": None}
    try:
        parse_text = final_text
        lines = parse_text.splitlines()
        if len(lines) >= 3 and lines[0].strip().lower() in ("```json", "```") and lines[-1].strip() == "```":
            parse_text = "\n".join(lines[1:-1]).strip()
        data = json.loads(parse_text)
        if not isinstance(data, dict) or data.get("label") not in TASKS[task]["order"]:
            raise ValueError("missing or invalid task label")
        for name in ("explanation_summary", "self_report_basis"):
            if not isinstance(data.get(name), str) or not data[name].strip():
                raise ValueError(f"nonempty {name} required")
        for name in ("prior_rating_influence", "possible_contamination"):
            if data.get(name) not in common.SELF_REPORT_VALUES:
                raise ValueError(f"{name} must be yes/no/unknown")
        for name in ("recognized_outlet", "recognized_network", "external_rating_source",
                     "recalled_factuality_rating", "recalled_bias_rating"):
            if name not in data or (data[name] is not None and (not isinstance(data[name], str) or not data[name].strip())):
                raise ValueError(f"{name} must be a nonempty string or null")
        result.update(structured_assessment=data, output_valid=True, predicted_label=data["label"])
    except (ValueError, TypeError) as exc:
        result["validation_error"] = str(exc)
        return result
    if payload.get("model") != MODEL:
        result.update(status="model_mismatch", validation_error="response model does not match gpt-5.6-sol")
    elif payload.get("status") != "completed":
        result.update(status="incomplete_response", validation_error=f"status={payload.get('status')}; {payload.get('incomplete_details')}")
    elif not provider_summary:
        result.update(status="ok_without_provider_summary" if allow_missing else "missing_provider_summary",
                      validation_error=None if allow_missing else "no readable native OpenAI reasoning summary returned")
    else:
        result["status"] = "ok"
    return result


def acceptable(row: dict) -> bool:
    return row["status"] in {"ok", "ok_without_provider_summary"} and row.get("output_valid", False)


def execute(cases, worker, workers, stop, on_record):
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
                    if not acceptable(row):
                        stop.set()
            submit()


def load_stream_seeds(directory: Path, cases: list[dict], max_tokens: int, effort: str) -> tuple[dict, dict]:
    diagnostic = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
    source = Path(diagnostic["source_dir"]) / "plan.json"
    if sha256_file(source) != diagnostic["source_plan_sha256"]:
        raise ValueError("diagnostic source plan changed")
    original = json.loads(source.read_text(encoding="utf-8"))
    if original["model"] != MODEL or original["system"] != common.BASE_SYSTEM:
        raise ValueError("diagnostic model/system differs")
    if original["prompts"] != {t: common.probe_prompt(t) for t in ("factuality", "bias")}:
        raise ValueError("diagnostic prompts differ")
    if original["schemas"] != {t: assessment_schema(t) for t in ("factuality", "bias")}:
        raise ValueError("diagnostic structured-output schemas differ")
    if original["max_output_tokens"] != max_tokens:
        raise ValueError("diagnostic token cap differs")
    expected = {c["case_id"]: c for c in cases}
    seed_cases = {c["case_id"]: c for c in diagnostic["cases"]}
    if len(seed_cases) != 4 or any(expected.get(id) != c for id, c in seed_cases.items()):
        raise ValueError("diagnostic case inputs/targets differ")
    files = list(directory.glob("stream_detailed-*.json"))
    if len(files) != 4:
        raise ValueError("all four streaming pilot records are required, including missing-summary cases")
    seeds, hashes = {}, {}
    for path in files:
        saved = json.loads(path.read_text(encoding="utf-8"))
        id = saved["case_id"]
        if id not in seed_cases or id in seeds or saved["profile"] != "stream_detailed":
            raise ValueError("unknown, duplicated or wrong-profile streaming seed")
        if saved["request_controls"] != {"model": MODEL, "reasoning": {"effort": effort, "summary": "detailed"}, "stream": True}:
            raise ValueError("streaming seed generation controls differ")
        streamed_response, summary = stream_result(saved["stream_events"])
        if streamed_response != saved["response"]:
            raise ValueError("stream response disagrees with saved final payload")
        parsed = decode_response(saved["response"], seed_cases[id]["task"], summary, allow_missing=True)
        if not acceptable(parsed):
            raise ValueError(f"streaming seed has invalid classification/explanation: {id}")
        seeds[id] = {**seed_cases[id], **parsed,
                     "correct": parsed["predicted_label"] == seed_cases[id]["true_label"],
                     "response": saved["response"], "stream_events": saved["stream_events"],
                     "http_status": saved["http_status"], "usage": saved.get("usage"),
                     "recorded_at": datetime.now(timezone.utc).isoformat(),
                     "seed_provenance": {"path": str(path), "sha256": sha256_file(path), "new_api_call": False}}
        hashes[str(path)] = sha256_file(path)
    if set(seeds) != set(seed_cases):
        raise ValueError("incomplete streaming pilot seed import")
    return seeds, {"profile": "stream_detailed", "files_sha256": hashes,
                   "imported": len(seeds), "missing_summaries_imported": sum(not r["provider_summary"] for r in seeds.values())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="artifacts/datasets/mediascope-clustered-main")
    parser.add_argument("--subset-csv", default="artifacts/audits/interpretability_subset_seed42_50pct_20261010/subset.csv")
    parser.add_argument("--input-run-dir", default="artifacts/evaluations/api-visible-rationales-stratified50-20261010")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--min-interval", type=float, default=0.5)
    parser.add_argument("--effort", choices=("medium", "high"), default="high")
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--record-missing-summaries", action="store_true",
                        help="keep valid structured reports when OpenAI omits native summaries; never synthesize or retry them")
    parser.add_argument("--seed-stream-diagnostic", help="import all four records from the preselected streaming pilot profile")
    args = parser.parse_args()
    if not 1 <= args.workers <= 32 or not 0 <= args.retries <= 5:
        parser.error("workers must be 1-32; retries must be 0-5")
    if not math.isfinite(args.min_interval) or args.min_interval < 0 or min(args.timeout, args.max_output_tokens) <= 0:
        parser.error("invalid pacing, timeout or token limit")
    if args.record_missing_summaries and not args.stream:
        parser.error("--record-missing-summaries requires --stream")
    if args.seed_stream_diagnostic and (not args.stream or not args.record_missing_summaries):
        parser.error("streaming seed import requires --stream --record-missing-summaries")
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not args.dry_run and (not key or "\n" in key or "\r" in key):
        parser.error("export OPENAI_API_KEY in this shell")
    root, inputs, output = Path(args.dataset_root).resolve(), Path(args.input_run_dir).resolve(), Path(args.output_dir)
    with open(args.subset_csv, newline="", encoding="utf-8") as stream:
        subset = list(csv.DictReader(stream))
    selection = {(r["media_name"], r["url"]): r for r in subset}
    if len(selection) != len(subset) or len(subset) != 212:
        parser.error("expected the fixed, unique 212-outlet subset")
    samples = [s for s in load_test_samples(root) if (s.media_name, s.url) in selection]
    if len(samples) != 212:
        parser.error("not all selected outlets were matched")
    index = json.loads((inputs / "inputs.json").read_text(encoding="utf-8"))
    images = DiskImages(inputs, index)
    cases = []
    for sample in samples:
        selected = selection[(sample.media_name, sample.url)]
        if sample.factuality != selected["true_label_factuality"] or sample.bias != selected["true_label_bias"]:
            parser.error(f"subset labels disagree: {sample.media_name}")
        entry = images[sample.image_path]
        if entry["image"]["truncated_px"] or entry["image"]["retained_height_fraction"] != 1:
            parser.error(f"input truncation: {sample.media_name}")
        digest = sha256_file(sample.image_path)
        for task in ("factuality", "bias"):
            case = {"task": task, "media_name": sample.media_name, "url": sample.url,
                    "image_path": sample.image_path, "image_sha256": digest,
                    "true_label": getattr(sample, task), "submitted_png_sha256": entry["png_sha256"]}
            case["case_id"] = json_hash(case)
            cases.append(case)
    smoke = common.select_smoke_cases(cases, subset)
    controls = request_body("factuality", [], args.effort, args.max_output_tokens, args.stream)
    seeds, seed_info = ({}, None)
    if args.seed_stream_diagnostic:
        seeds, seed_info = load_stream_seeds(Path(args.seed_stream_diagnostic), cases, args.max_output_tokens, args.effort)
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    routing = {}
    for variable, header in (("OPENAI_ORG_ID", "OpenAI-Organization"), ("OPENAI_PROJECT_ID", "OpenAI-Project")):
        if os.environ.get(variable):
            headers[header] = os.environ[variable]
            routing[header] = os.environ[variable]
    plan = {"protocol": "gpt56-summary-explicit-prior-knowledge-probe-v2", "model": MODEL, "cases": cases,
            "subset_csv_sha256": sha256_file(args.subset_csv), "source_inputs_sha256": sha256_file(inputs / "inputs.json"),
            "source_inputs_dir": str(inputs), "reasoning": controls["reasoning"], "system": common.BASE_SYSTEM,
            "prompts": {task: common.probe_prompt(task) for task in ("factuality", "bias")},
            "schemas": {task: assessment_schema(task) for task in ("factuality", "bias")},
            "max_output_tokens": args.max_output_tokens, "timeout_seconds": args.timeout, "store": False,
            "mandatory_native_summary": not args.record_missing_summaries, "stream": args.stream,
            "seed_info": seed_info, "smoke_case_ids": [c["case_id"] for c in smoke], "routing": routing,
            "implementation_sha256": sha256_file(__file__), "common_implementation_sha256": sha256_file(common.__file__),
            "stream_implementation_sha256": sha256_file(Path(__file__).parent / "gpt_summary_stream.py")}
    print(f"{MODEL}: 212 outlets, 424 decisions; {args.workers} concurrent requests after {len(smoke)}-request smoke gate.", flush=True)
    print(f"Reasoning controls: {plan['reasoning']}; same prompts and full-page tiles as Opus 5.", flush=True)
    print(f"Streaming={args.stream}; missing native summaries {'recorded as unavailable' if args.record_missing_summaries else 'stop new calls'}; imported pilot cases={len(seeds)}.", flush=True)
    if args.dry_run:
        print("Dry run passed; no paid calls or output writes.")
        return 0
    if args.resume:
        if not (output / "plan.json").is_file() or json.loads((output / "plan.json").read_text()) != plan:
            parser.error("resume requires identical sources, prompt, model and code")
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "plan.json", plan)
    record_dir = output / "records"
    record_dir.mkdir(exist_ok=True)
    with FileLock(str(output / "run.lock"), timeout=0):
        records = common.load_records(record_dir, {c["case_id"]: c for c in cases})
        if any(not acceptable(r) for r in records.values()):
            parser.error("failed output/observability check exists; inspect it rather than silently resampling")
        stop, log_lock = threading.Event(), threading.Lock()
        gate = common.RequestGate(args.min_interval, stop)
        started = time.monotonic()
        for id, row in seeds.items():
            if id not in records:
                write_json(record_dir / f"{id}.json", row)
                common.append_event(output / "predictions.jsonl", row, log_lock)
                records[id] = row
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
            for id, row in records.items():
                if id not in journaled:
                    common.append_event(journal, {**row, "recovered_from_atomic_record": True}, log_lock)

        def worker(case):
            entry = images[case["image_path"]]
            body = request_body(case["task"], entry["data_urls"], args.effort, args.max_output_tokens, args.stream)
            payload, http, error, events, streamed_summary = None, None, None, [], ""
            for retry in range(args.retries + 1):
                if not gate.wait():
                    return None
                started_call, error, events, streamed_summary = time.monotonic(), None, [], ""
                try:
                    with requests.post(ENDPOINT, headers=headers, json=body, stream=args.stream, timeout=(15, args.timeout)) as response:
                        http = response.status_code
                        if args.stream and http < 400:
                            for line in response.iter_lines():
                                if not line.startswith(b"data:"):
                                    continue
                                raw = line[5:].strip()
                                if raw == b"[DONE]":
                                    break
                                events.append(common.redact_payload(json.loads(raw), key))
                            payload, streamed_summary = stream_result(events)
                            if payload is None:
                                error = "stream ended without a terminal response event"
                        else:
                            try:
                                payload = common.redact_payload(response.json(), key)
                            except ValueError:
                                payload = {"error": {"message": response.text.replace(key, "[REDACTED]")}}
                    if http >= 400 or (payload or {}).get("error"):
                        error = json.dumps((payload or {}).get("error"), ensure_ascii=False).replace(key, "[REDACTED]")
                except (requests.RequestException, ValueError) as exc:
                    http, payload, error = None, None, type(exc).__name__
                common.append_event(output / "requests.jsonl", {"case_id": case["case_id"], "task": case["task"],
                    "recorded_at": datetime.now(timezone.utc).isoformat(), "attempt": retry + 1,
                    "http_status": http, "response": payload, "error": error,
                    "elapsed_seconds": time.monotonic() - started_call,
                    "stream_events": events,
                    "request_controls": {"model": MODEL, "reasoning": body["reasoning"], "store": False, "stream": args.stream}}, log_lock)
                if error is None:
                    break
                retryable = http is None or http in (408, 409, 429) or 500 <= http <= 599
                if retryable and retry < args.retries:
                    delay = common.retry_seconds(response.headers.get("retry-after") if http is not None else None, retry)
                    gate.defer(delay)
                    print(f"[retry] {case['media_name']} {case['task']} HTTP={http}; shared cooldown {delay:.1f}s", flush=True)
                    continue
                break
            parsed = decode_response(payload, case["task"], streamed_summary, args.record_missing_summaries) if error is None and isinstance(payload, dict) else {
                "status": "http_error", "provider_summary": "", "provider_summary_kind": "openai_reasoning_summary",
                "final_text": "", "structured_assessment": None, "output_valid": False,
                "predicted_label": None, "validation_error": error}
            row = {**case, **parsed, "correct": parsed["predicted_label"] == case["true_label"], "response": payload,
                    "stream_events": events, "http_status": http, "usage": (payload or {}).get("usage"), "recorded_at": datetime.now(timezone.utc).isoformat()}
            write_json(record_dir / f"{case['case_id']}.json", row)
            if not acceptable(row):
                stop.set()
            return row

        def save_summary(state):
            results = {}
            for task in ("factuality", "bias"):
                actual = [r for r in records.values() if r["task"] == task]
                reports = [r["structured_assessment"] for r in actual if r["output_valid"]]
                results[task] = {"expected": 212, "recorded": len(actual),
                    "status_counts": dict(Counter(r["status"] for r in actual)),
                    "provider_summaries": sum(bool(r["provider_summary"]) for r in actual),
                    "missing_provider_summaries": sum(not bool(r["provider_summary"]) for r in actual),
                    "valid_structured_outputs": len(reports), "correct_predictions": sum(r["correct"] for r in actual),
                    "overall_accuracy": sum(r["correct"] for r in actual) / 212 if len(actual) == 212 else None,
                    "prior_rating_influence_self_reports": dict(Counter(r["prior_rating_influence"] for r in reports)),
                    "possible_contamination_self_reports": dict(Counter(r["possible_contamination"] for r in reports))}
            write_json(output / "summary.json", {"status": state, "model": MODEL, "tasks": results,
                "elapsed_seconds": time.monotonic() - started,
                "interpretation": "Explicitly elicited model self-reports; not verified training contamination."})

        def on_record(row):
            records[row["case_id"]] = row
            common.append_event(output / "predictions.jsonl", row, log_lock)
            save_summary("running" if acceptable(row) else "stopped_failed_check")
            print(f"[{len(records)}/424] {row['media_name']} {row['task']}: {row['status']}; "
                  f"native_summary_chars={len(row['provider_summary'])}", flush=True)

        def handle_signal(signum, frame):
            stop.set()
            print("Stopping new calls; saving in-flight responses.", flush=True)

        old = {sig: signal.signal(sig, handle_signal) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            execute([c for c in smoke if c["case_id"] not in records], worker, min(4, args.workers), stop, on_record)
            if not all(acceptable(records[c["case_id"]]) if c["case_id"] in records else False for c in smoke):
                save_summary("stopped_smoke_failed")
                print("Smoke gate failed. No full sweep launched; inspect raw responses.", flush=True)
                return 1
            native_count = sum(bool(records[c["case_id"]]["provider_summary"]) for c in smoke)
            print(f"SMOKE PASSED: all eight structured explanations validate; {native_count}/8 native summaries observed.", flush=True)
            if not args.smoke_only and not stop.is_set():
                execute([c for c in cases if c["case_id"] not in records], worker, args.workers, stop, on_record)
        finally:
            for sig, handler in old.items():
                signal.signal(sig, handler)
        complete = len(records) == 424 and all(acceptable(r) for r in records.values())
        state = "complete" if complete else "smoke_complete" if args.smoke_only else "stopped_failed_check" if any(not acceptable(r) for r in records.values()) else "interrupted"
        save_summary(state)
        print(f"Finished: {state}; {len(records)}/424 records saved.", flush=True)
        return 0 if complete or args.smoke_only else 1


if __name__ == "__main__":
    raise SystemExit(main())
