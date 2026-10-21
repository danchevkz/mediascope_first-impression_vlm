#!/usr/bin/env python3
"""Bounded 12-call summary-observability diagnostic; never launch a full sweep."""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

import eval_gpt56_summary_parallel as gpt
import eval_opus_summary_parallel as common
from eval_api import DiskImages
from run_config import sha256_file, write_json


PROFILES = ("standard_auto", "pro_detailed", "stream_detailed")


def diagnostic_body(task: str, urls: list[str], profile: str, max_tokens: int) -> dict:
    body = gpt.request_body(task, urls, "high", max_tokens)
    if profile == "standard_auto":
        body["reasoning"]["summary"] = "auto"
    elif profile == "pro_detailed":
        body["reasoning"]["mode"] = "pro"
    elif profile == "stream_detailed":
        body["stream"] = True
    else:
        raise ValueError(f"Unknown profile: {profile}")
    return body


def stream_result(events: list[dict]) -> tuple[dict | None, str]:
    """Extract only actual summary delta/part events, not final-answer text."""
    response = None
    chunks = defaultdict(list)
    completed_parts = {}
    for event in events:
        kind = event.get("type")
        if kind in ("response.completed", "response.incomplete", "response.failed"):
            response = event.get("response")
        key = (event.get("item_id"), event.get("summary_index", 0))
        if kind == "response.reasoning_summary_text.delta" and isinstance(event.get("delta"), str):
            chunks[key].append(event["delta"])
        elif kind == "response.reasoning_summary_text.done" and isinstance(event.get("text"), str):
            completed_parts[key] = event["text"]
    keys = list(dict.fromkeys([*chunks, *completed_parts]))
    text = "\n\n".join(completed_parts.get(key, "".join(chunks[key])) for key in keys).strip()
    return response, text


def classify_observability(payload: dict, task: str, stream_text: str = "") -> dict:
    parsed = gpt.decode_response(payload, task)
    native = parsed["provider_summary"]
    observed = native or stream_text
    valid_identity = payload.get("model") == gpt.MODEL and payload.get("status") == "completed"
    return {"decoded": parsed, "final_summary_chars": len(native), "stream_summary_chars": len(stream_text),
            "observed_provider_summary": observed,
            "summary_origin": "response_reasoning_summary" if native else "actual_stream_summary_events" if stream_text else "missing",
            "observable_valid_output": bool(observed and parsed["output_valid"] and valid_identity)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default="artifacts/evaluations/gpt56-summary-probe-parallel-20261010")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("diagnostic workers must be 1-4")
    source, output = Path(args.source_dir), Path(args.output_dir)
    plan = json.loads((source / "plan.json").read_text(encoding="utf-8"))
    if plan["model"] != gpt.MODEL:
        parser.error("source must be the GPT-5.6 Sol summary probe")
    baseline = common.load_records(source / "records", {c["case_id"]: c for c in plan["cases"]})
    if len(baseline) != 4:
        parser.error("expected exactly the four recorded baseline smoke cases")
    controls = gpt.request_body("factuality", [], "high", plan["max_output_tokens"])
    if plan["system"] != controls["instructions"] or plan["prompts"] != {t: common.probe_prompt(t) for t in ("factuality", "bias")}:
        parser.error("source prompts differ; refusing unmatched diagnostic")
    inputs = Path(plan["source_inputs_dir"])
    if sha256_file(inputs / "inputs.json") != plan["source_inputs_sha256"]:
        parser.error("saved input index changed")
    images = DiskImages(inputs, json.loads((inputs / "inputs.json").read_text(encoding="utf-8")))
    cases = [c for c in plan["cases"] if c["case_id"] in baseline]
    for case in cases:
        entry = images[case["image_path"]]
        if entry["png_sha256"] != case["submitted_png_sha256"] or sha256_file(case["image_path"]) != case["image_sha256"]:
            parser.error("baseline input hash mismatch")
    profile_controls = {p: diagnostic_body("factuality", [], p, plan["max_output_tokens"])["reasoning"] for p in PROFILES}
    print(f"Baseline: four saved cases reused. Diagnostic: {len(cases)*len(PROFILES)} new calls total, {args.workers} at a time.", flush=True)
    print(f"Profiles: {profile_controls}; stream_detailed also captures real SSE summary events.", flush=True)
    if args.dry_run:
        print("Dry run passed; no writes, no model calls, no full sweep.")
        return
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key or "\n" in key or "\r" in key:
        parser.error("export OPENAI_API_KEY in this shell")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", {"source_dir": str(source.resolve()), "source_plan_sha256": sha256_file(source / "plan.json"),
               "model": gpt.MODEL, "cases": cases, "profiles": profile_controls, "maximum_new_calls": 12,
               "stream_profile": "stream_detailed", "implementation_sha256": sha256_file(__file__)})
    results = []
    for row in baseline.values():
        inspected = classify_observability(row["response"], row["task"])
        results.append({"profile": "saved_standard_detailed", "case_id": row["case_id"],
                        "media_name": row["media_name"], "task": row["task"], "new_request": False, **inspected})

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    for variable, header in (("OPENAI_ORG_ID", "OpenAI-Organization"), ("OPENAI_PROJECT_ID", "OpenAI-Project")):
        if os.environ.get(variable):
            headers[header] = os.environ[variable]

    def worker(case, profile):
        prepared = images[case["image_path"]]
        body = diagnostic_body(case["task"], prepared["data_urls"], profile, plan["max_output_tokens"])
        events, payload, streamed, http, error = [], None, "", None, None
        started = time.monotonic()
        try:
            with requests.post(gpt.ENDPOINT, headers=headers, json=body, stream=profile == "stream_detailed",
                               timeout=(15, plan["timeout_seconds"])) as response:
                http = response.status_code
                if profile == "stream_detailed" and http < 400:
                    for line in response.iter_lines():
                        if not line.startswith(b"data:"):
                            continue
                        raw = line[5:].strip()
                        if raw == b"[DONE]":
                            break
                        event = common.redact_payload(json.loads(raw), key)
                        events.append(event)
                    payload, streamed = stream_result(events)
                    if payload is None:
                        error = "stream ended without a terminal response event"
                else:
                    payload = common.redact_payload(response.json(), key)
                if http >= 400 or (payload or {}).get("error"):
                    error = json.dumps((payload or {}).get("error"), ensure_ascii=False)
        except (requests.RequestException, ValueError) as exc:
            error = f"{type(exc).__name__}: {str(exc).replace(key, '[REDACTED]')}"
        inspected = classify_observability(payload, case["task"], streamed) if payload and not error else {
            "observable_valid_output": False, "observed_provider_summary": "", "summary_origin": "unavailable",
            "final_summary_chars": 0, "stream_summary_chars": len(streamed)}
        result = {"profile": profile, "case_id": case["case_id"], "media_name": case["media_name"], "task": case["task"],
                  "new_request": True, "http_status": http, "error": error, "response": payload,
                  "stream_events": events, "elapsed_seconds": time.monotonic()-started, "usage": (payload or {}).get("usage"),
                  "request_controls": {"model": gpt.MODEL, "reasoning": body["reasoning"], "stream": body.get("stream", False)},
                  **inspected}
        write_json(output / f"{profile}-{case['case_id']}.json", result)
        return result

    def save_summary():
        report = {}
        for profile in ("saved_standard_detailed", *PROFILES):
            rows = [r for r in results if r["profile"] == profile]
            report[profile] = {"recorded": len(rows), "expected": 4,
                              "observable_valid_outputs": sum(r["observable_valid_output"] for r in rows),
                              "mean_elapsed_seconds": sum(r.get("elapsed_seconds", 0) for r in rows)/len(rows) if rows else None}
        write_json(output / "summary.json", {"profiles": report, "maximum_new_calls": 12,
                   "interpretation": "Configuration pilot, not benchmark results or selection of favorable model answers. No full sweep is launched."})

    save_summary()
    for profile in PROFILES:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(worker, case, profile) for case in cases]
            for future in as_completed(futures):
                row = future.result()
                results.append(row)
                save_summary()
                print(f"[{profile}] {row['media_name']} {row['task']}: summary_chars={len(row['observed_provider_summary'])}; "
                      f"valid={row['observable_valid_output']}; HTTP={row['http_status']}", flush=True)
        if any(r.get("http_status") in (401, 402, 403) for r in results if r["profile"] == profile):
            print("Authentication/quota failure: stopping remaining diagnostic profiles.", flush=True)
            break
    print(f"Diagnostic finished. Inspect {output/'summary.json'}. No full evaluation launched.", flush=True)


if __name__ == "__main__":
    main()
