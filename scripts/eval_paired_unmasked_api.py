#!/usr/bin/env python3
"""Case-parallel label-only API tests matching the previous masked protocol."""
import argparse
import json
import math
import os
import signal
import threading
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from filelock import FileLock

from api_predictor import PROVIDERS
from eval_api import DiskImages, EvaluationStopped, RateGate, load_test_samples
from eval_audit import EvaluationAudit
from mediascope_eval import TASKS, audit_metrics, normalize_to_label
from qwen_predictor import PROMPTS
from resilient_api import APIResponseError, ResilientAPIPredictor
from run_config import environment_info, json_hash, sha256_file, source_info, write_json


MODELS = {"gpt56": ("openai", "gpt-5.6-sol"), "opus5": ("anthropic", "claude-opus-5"),
          "kimi": ("openrouter", "moonshotai/kimi-k3"), "qwen": ("openrouter", "qwen/qwen3.8-max-prime")}
SIZING = {"tile_height": 1400, "max_tiles": 6, "max_width": 1024}


def last_request(path, sample, task, model):
    if not path.exists():
        return None
    events = [json.loads(line) for line in path.read_bytes().splitlines(keepends=True)
              if line.endswith(b"\n") and line.strip()]
    if not events:
        return None
    event = events[-1]
    if event["image_path"] != sample.image_path or event["task"] != task or event["model"] != model:
        raise ValueError("Case request journal identity differs")
    # A logged terminal response is authoritative even when filtered/truncated.
    # A scheduled but not executed transport retry remains pending.
    return event if event.get("retry_delay_seconds") is None else None


def record_from_answer(info, task, raw, details, error=None, elapsed=0):
    label = normalize_to_label(raw, TASKS[task]["order"]) if error is None else None
    return {**{k: v for k, v in info.items() if k != "labels"}, "task": task,
            "true_label": info["labels"][task], "raw_text": raw, "predicted_label": label,
            "correct": label == info["labels"][task], "status": "error" if error else "ok" if label else "unparseable",
            "error": error, "details": details, "elapsed_seconds": elapsed}


def smoke_ids(samples):
    selected = set()
    for task in TASKS:
        for label in TASKS[task]["order"]:
            match = next((s for s in samples if getattr(s, task) == label), None)
            if match is not None:
                selected.add(match.media_name)
    return selected


def execute(cases, worker, workers, stop, collector):
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
                    collector(row)
            submit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--min-interval", type=float, default=1)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 32 or not 0 <= args.retries <= 5 or not math.isfinite(args.min_interval) or args.min_interval < 0:
        parser.error("Invalid bounded concurrency/retry/pacing controls")
    provider, model = MODELS[args.model]
    key_name = PROVIDERS[provider][0]
    if args.dry_run:
        os.environ.setdefault(key_name, "dry-run-not-a-real-key")
    if not os.environ.get(key_name, "").strip():
        parser.error(f"Export {key_name} in this shell")
    samples = load_test_samples(args.dataset_root.resolve())
    if len(samples) != 133:
        parser.error("Expected all 133 paired unmasked images")
    dataset_manifest = json.loads((args.dataset_root / "dataset_manifest.json").read_text())
    if dataset_manifest["condition"] != "unmasked" or dataset_manifest["preprocessing"] != SIZING:
        parser.error("Expected the matched full-page unmasked condition")
    index_path = args.input_dir / "inputs.json"
    images = DiskImages(args.input_dir.resolve(), json.loads(index_path.read_text()))
    infos = []
    for sample in samples:
        path = str(Path(sample.image_path).resolve())
        entry = images[path]
        if entry["image"]["truncated_px"] or entry["image"]["retained_height_fraction"] != 1:
            parser.error("Input truncation detected")
        if sample.factuality not in TASKS["factuality"]["order"] or sample.bias not in TASKS["bias"]["order"]:
            parser.error("Invalid paired targets")
        infos.append({"sample_id": json_hash([sample.media_name, sample.url, path]), "media_name": sample.media_name,
                      "url": sample.url, "image_path": path, "image_sha256": sha256_file(path),
                      "labels": {t: getattr(sample, t) for t in TASKS}})
    stop = threading.Event()
    gate = RateGate(args.min_interval, stop)
    session = uuid.uuid4().hex
    probe = ResilientAPIPredictor(provider, model, images, SIZING, max_output_tokens=8192, effort="medium", timeout=300,
                                  prompts=PROMPTS, gate=gate, request_log=args.output_dir / "preflight-unused.jsonl", session_id=session,
                                  retries=args.retries)
    metadata = probe.run_metadata()
    plan = {"protocol": "paired-unmasked-label-only-v1", "condition": "unmasked", "model_tag": args.model,
            "dataset_csv_sha256": sha256_file(args.dataset_root / "test.csv"),
            "dataset_manifest_sha256": sha256_file(args.dataset_root / "dataset_manifest.json"),
            "input_inventory_sha256": sha256_file(index_path), "samples": infos, "predictor": metadata,
            "runner_sha256": sha256_file(__file__), "source": source_info(), "environment": environment_info()}
    if args.resume and (not (args.output_dir / "plan.json").exists()
                        or json.loads((args.output_dir / "plan.json").read_text()) != plan):
        parser.error("Resume requires identical data, model, prompts, settings and code")
    print(f"{model}: 133 unmasked pairs, 266 decisions; {args.workers} concurrent requests; medium reasoning, 8192-token cap.", flush=True)
    if args.dry_run:
        print("Dry run passed; exact full-page tiles verified; no paid calls or output writes.")
        return 0
    if not args.resume:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        write_json(args.output_dir / "plan.json", plan)
    requests_dir = args.output_dir / "requests"
    requests_dir.mkdir(exist_ok=True)
    identity = {"tasks": list(TASKS), "samples": infos, "parser": {"mode": "strict", "version": 1},
                "predictor": metadata, "context": plan, "source": source_info(), "environment": environment_info()}
    with FileLock(str(args.output_dir / "run.lock"), timeout=0), EvaluationAudit(args.output_dir / "predictions.csv", identity, args.resume) as audit:
        def worker(case):
            sample, task, info = case
            if stop.is_set():
                return None
            path = requests_dir / f"{info['sample_id']}-{task}.jsonl"
            previous = last_request(path, sample, task, model)
            if previous:
                details = {**previous["details"], "replayed_from_request_log": True, "request_sent": False}
                return record_from_answer(info, task, previous["raw_text"], details, previous.get("error"), previous["elapsed_seconds"])
            predictor = ResilientAPIPredictor(provider, model, images, SIZING, max_output_tokens=8192, effort="medium", timeout=300,
                                              prompts=PROMPTS, gate=gate, request_log=path, session_id=session, retries=args.retries)
            called, raw, error = time.monotonic(), None, None
            try:
                raw = predictor.predict(sample, task)
            except EvaluationStopped:
                return None
            except APIResponseError as exc:
                error = {"type": "APIResponseError", "kind": exc.failure.kind, "message": predictor.redact(exc.failure.message),
                         "retryable": exc.failure.retryable, "fatal": exc.failure.fatal}
            except Exception as exc:
                error = {"type": type(exc).__name__, "message": predictor.redact(str(exc))}
                stop.set()
            return record_from_answer(info, task, raw, predictor.last_details, error, time.monotonic()-called)

        def save(state):
            metrics = audit_metrics(audit)
            audit.save_metrics(metrics, state)
            write_json(args.output_dir / "summary.json", {"status": state, "model": model, "tasks": metrics,
                                                          "recorded": len(audit.records), "expected": 266})

        def collect(row):
            audit.append(row)
            save("running")
            print(f"[{len(audit.records)}/266] {row['media_name']} {row['task']}: {row['status']}", flush=True)

        work = [(s, t, i) for s, i in zip(samples, infos) for t in TASKS if audit.pending(i["sample_id"], t)]
        smoke_names = smoke_ids(samples)
        smoke = [c for c in work if c[0].media_name in smoke_names]
        def handle(signum, frame):
            stop.set()
            print("Stopping submissions; preserving in-flight results.", flush=True)
        old = {sig: signal.signal(sig, handle) for sig in (signal.SIGINT, signal.SIGTERM)}
        save("running")
        try:
            execute(smoke, worker, min(4, args.workers), stop, collect)
            gate_ok = all(any(r["task"] == t and r["media_name"] in smoke_names and r["status"] == "ok"
                              for r in audit.records.values()) for t in TASKS)
            if not gate_ok or stop.is_set():
                save("stopped_smoke_or_fatal")
                print("Smoke/fatal check stopped this model; saved outcomes preserved.", flush=True)
                return 1
            print("SMOKE PASSED: valid final labels for both tasks; terminal per-case failures retained.", flush=True)
            execute([c for c in work if c[0].media_name not in smoke_names], worker, args.workers, stop, collect)
        finally:
            for sig, handler in old.items():
                signal.signal(sig, handler)
            audit.export_csv()
            state = ("complete_with_failures" if any(r["status"] != "ok" for r in audit.records.values()) else "complete") if len(audit.records) == 266 else "interrupted_or_fatal"
            save(state)
        print(f"Finished: {state}; {len(audit.records)}/266 recorded.", flush=True)
        return 0 if len(audit.records) == 266 else 1


if __name__ == "__main__":
    raise SystemExit(main())
