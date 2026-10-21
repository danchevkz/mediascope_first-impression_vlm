#!/usr/bin/env python3
"""Full HF test-split API evaluation, with per-model journals and bounded concurrency."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
import os
import signal
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

from filelock import FileLock
from PIL import Image

from api_predictor import PROVIDERS, ReasoningAPIPredictor
from audit_image_sizes import geometry
from eval_finetuned_rationales import parse_label, rationale_prompt, selected_keys
from mediascope_eval import Sample, TASKS, run
from qwen_predictor import PROMPTS
from reconcile_data import filename_stem
from run_config import sha256_file, write_json
from smoke_api import MODELS, prepare_images


DEFAULT_MODELS = ["kimi", "qwen", "opus5", "gpt56"]


def load_test_samples(root):
    with (root / "test.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    samples, used = [], set()
    for row in rows:
        fact, bias = row["true_label_factuality"], row["true_label_bias"]
        if fact not in TASKS["factuality"]["order"] or bias not in TASKS["bias"]["order"] + ["UNRATED"]:
            raise ValueError(f"Unexpected target label: {row['media_name']}")
        if row.get("image_path"):
            path = (root / row["image_path"]).resolve()
        else:
            stem = filename_stem(row["media_name"])
            candidates = [root / "test" / name for name in (f"{stem}.png", f"{stem}_1.png")]
            hits = [p for p in candidates if p.is_file()]
            if len(hits) != 1:
                raise ValueError(f"Missing or ambiguous test image: {row['media_name']}")
            path = hits[0].resolve()
        if path in used:
            raise ValueError(f"Repeated test screenshot: {path}")
        used.add(path)
        samples.append(Sample(row["media_name"], row["url"], str(path), row.get("country", ""),
                              fact, None if bias == "UNRATED" else bias))
    if not samples:
        raise ValueError("Empty test split")
    return samples


class DiskImages:
    """Exact saved PNGs, encoded lazily with one cached screenshot per worker."""
    def __init__(self, directory, index):
        self.directory, self.index = directory, index
        self.local = threading.local()

    def __getitem__(self, image_path):
        if getattr(self.local, "path", None) != image_path:
            entry = self.index[image_path]
            urls = []
            for filename, expected in zip(entry["files"], entry["png_sha256"]):
                data = (self.directory / filename).read_bytes()
                if hashlib.sha256(data).hexdigest() != expected:
                    raise ValueError(f"Saved input tile changed: {filename}")
                urls.append("data:image/png;base64," + base64.b64encode(data).decode())
            self.local.value = {"image": entry["image"], "data_urls": urls, "png_sha256": entry["png_sha256"]}
            self.local.path = image_path
        return self.local.value


def save_inputs(samples, sizing, output):
    index_path = output / "inputs.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    for number, sample in enumerate(samples, 1):
        entry = index.get(sample.image_path)
        if entry:
            for filename, digest in zip(entry["files"], entry["png_sha256"]):
                if sha256_file(output / filename) != digest:
                    raise ValueError(f"Saved input tile changed: {filename}")
            continue
        prepared = prepare_images([sample], sizing)[sample.image_path]
        directory = output / "inputs" / f"sample-{number:04d}"
        directory.mkdir(parents=True, exist_ok=True)
        filenames = []
        for tile_number, (url, digest) in enumerate(zip(prepared["data_urls"], prepared["png_sha256"]), 1):
            path = directory / f"tile-{tile_number:02d}.png"
            if path.exists():
                if sha256_file(path) != digest:
                    raise ValueError(f"Saved input tile changed: {path}")
            else:
                with path.open("xb") as f:
                    f.write(base64.b64decode(url.split(",", 1)[1]))
            filenames.append(path.relative_to(output).as_posix())
        index[sample.image_path] = {"image": prepared["image"], "png_sha256": prepared["png_sha256"], "files": filenames}
        write_json(index_path, index)
        if number % 50 == 0:
            print(f"Saved full-page inputs: {number}/{len(samples)}", flush=True)
    return DiskImages(output, index)


class EvaluationStopped(BaseException):
    """Stop a worker without counting the unattempted example as a model error."""


class RateGate:
    """Share pacing and server cooldowns between models on the same provider."""
    def __init__(self, interval, stop=None):
        self.interval, self.next_request = interval, 0.0
        self.lock = threading.Lock()
        self.stop = stop or threading.Event()

    def wait(self):
        while True:
            if self.stop.is_set():
                raise EvaluationStopped()
            with self.lock:
                delay = self.next_request - time.monotonic()
                if delay <= 0:
                    self.next_request = time.monotonic() + self.interval
                    return
            self.stop.wait(min(delay, 1.0))

    def defer(self, seconds):
        with self.lock:
            self.next_request = max(self.next_request, time.monotonic() + seconds)


def retry_delay(value, retry):
    fallback = min(30 * 2 ** retry, 300)
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        try:
            seconds = parsedate_to_datetime(value).timestamp() - time.time()
        except (TypeError, ValueError, OverflowError):
            return fallback
    return max(fallback, seconds) if math.isfinite(seconds) and seconds >= 0 else fallback


class BenchmarkPredictor(ReasoningAPIPredictor):
    def __init__(self, *args, gate, request_log, session_id, retries=3, recover=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.gate, self.request_log, self.session_id, self.retries = gate, request_log, session_id, retries
        self.recovered = {}
        if recover and self.request_log.exists():
            with self.request_log.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self.recovered[(event["image_path"], event["task"])] = event

    def run_metadata(self):
        metadata = super().run_metadata()
        # Pacing/retry counts are operational controls; actual values are logged
        # per attempt/session so they can be adjusted when resuming a rate limit.
        metadata.update(automatic_retries="HTTP 429 only; every attempt journaled",
                        benchmark_implementation_sha256=sha256_file(__file__))
        return metadata

    def predict(self, sample, task_key):
        if self.gate.stop.is_set():
            raise EvaluationStopped()
        previous = self.recovered.pop((sample.image_path, task_key), None)
        if previous and previous["error"] is None:
            # Recover a response synced before a crash but not yet written to
            # the prediction journal. Do not pay to generate it a second time.
            self.last_details = {**previous["details"], "request_sent": False,
                                 "replayed_from_request_log": True, "recovery_session_id": self.session_id}
            return previous["raw_text"]
        for retry in range(self.retries + 1):
            self.gate.wait()
            started = time.perf_counter()
            error = None
            try:
                answer = super().predict(sample, task_key)
            except Exception as exc:
                error = exc
                answer = None
            self.last_details.update(session_id=self.session_id, rate_limit_attempt=retry + 1,
                                     request_event_id=uuid.uuid4().hex)
            event = {
                "recorded_at": datetime.now(timezone.utc).isoformat(), "session_id": self.session_id,
                "image_path": sample.image_path, "task": task_key, "model": self.model,
                "attempt_within_call": retry + 1, "raw_text": answer,
                "elapsed_seconds": time.perf_counter() - started,
                "error": None if error is None else {"type": type(error).__name__, "message": self.redact(str(error))},
                "details": self.last_details,
            }
            with self.request_log.open("ab+") as f:
                if f.tell():
                    f.seek(-1, os.SEEK_END)
                    last_byte = f.read(1)
                    f.seek(0, os.SEEK_END)
                    if last_byte != b"\n":
                        f.write(b"\n")  # Keep an interrupted fragment separate from the new event.
                f.write((json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n").encode())
                f.flush()
                os.fsync(f.fileno())
            if error is None:
                return answer
            if self.last_details.get("http_status") == 429 and not self.fatal_error:
                delay = retry_delay(self.last_details.get("retry_after"), retry)
                self.gate.defer(delay)
                if retry < self.retries:
                    print(f"[{self.model}] rate limited; retrying after >= {delay:.0f}s", flush=True)
                    continue
            raise error


class VisibleRationalePredictor(BenchmarkPredictor):
    """Preserve a visible rationale while returning its final label to the scorer."""

    def predict(self, sample, task_key):
        answer = super().predict(sample, task_key)
        self.last_details["visible_rationale"] = answer
        label = parse_label(answer, TASKS[task_key]["order"])
        return label if label is not None else answer


def usage_summary(path):
    """Include costs/usage from all HTTP attempts, not only the final predictions."""
    totals = Counter()
    count, errors, reported_cost, cost_count, request_seconds = 0, 0, 0.0, 0, 0.0
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue  # An interrupted write is retained verbatim in this diagnostic log.
            details = event["details"]
            count += bool(details.get("request_sent"))
            if details.get("request_sent"):
                request_seconds += event["elapsed_seconds"]
            errors += event["error"] is not None
            usage = details.get("usage") or {}
            for key in ("input_tokens", "output_tokens", "prompt_tokens", "completion_tokens", "total_tokens",
                        "cache_read_input_tokens", "cache_creation_input_tokens"):
                if isinstance(usage.get(key), (int, float)):
                    totals[key] += usage[key]
            reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
            output_details = usage.get("output_tokens_details") or {}
            if reasoning is None:
                reasoning = output_details.get("reasoning_tokens", output_details.get("thinking_tokens"))
            if reasoning is not None:
                totals["reasoning_tokens"] += reasoning
            if isinstance(usage.get("cost"), (int, float)):
                reported_cost += usage["cost"]
                cost_count += 1
    return {"http_attempts": count, "error_events": errors, "usage_totals": dict(totals),
            "http_wall_seconds": request_seconds,
            "provider_reported_cost_usd": reported_cost if cost_count else None,
            "attempts_with_reported_cost": cost_count}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", default="artifacts/datasets/mediascope-clustered-main")
    ap.add_argument("--models", nargs="+", choices=MODELS, default=DEFAULT_MODELS)
    ap.add_argument("--task", choices=("factuality", "bias", "both"), default="both")
    ap.add_argument("--limit", type=int, help="optional first-N preflight; omit for the entire test split")
    ap.add_argument("--tile-height", type=int, default=1400)
    ap.add_argument("--max-width", type=int, default=1024)
    ap.add_argument("--max-tiles", type=int, default=6)
    ap.add_argument("--max-output-tokens", type=int, default=8192)
    ap.add_argument("--effort", choices=("low", "medium", "high"), default="medium")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--workers", type=int, default=4, help="max concurrent models; one request at a time per model")
    ap.add_argument("--min-interval", type=float, default=2, help="minimum seconds between requests on each provider")
    ap.add_argument("--rate-limit-retries", type=int, default=3, help="retry HTTP 429 only; each attempt saved")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--retry-errors", action="store_true", help="retry previously recorded inference errors on resume")
    ap.add_argument("--dry-run", action="store_true", help="hash/validate inputs and print plan; no requests or writes")
    ap.add_argument("--visible-rationales", action="store_true",
                    help="request a brief visible explanation and parse its final LABEL line")
    ap.add_argument("--subset-flags",
                    help="optional record_flags.csv selecting the fixed Opus-readable task/outlet subset")
    args = ap.parse_args()
    if min(args.tile_height, args.max_width, args.max_tiles, args.max_output_tokens, args.timeout, args.workers) <= 0:
        ap.error("Sizing, token cap, timeout, and workers must be positive")
    if not math.isfinite(args.min_interval) or args.min_interval < 0 or not 0 <= args.rate_limit_retries <= 10:
        ap.error("--min-interval must be nonnegative; --rate-limit-retries must be 0-10")
    if args.limit is not None and args.limit <= 0:
        ap.error("--limit must be positive")
    if len(args.models) != len(set(args.models)):
        ap.error("Select each model only once")
    if args.retry_errors and not args.resume:
        ap.error("--retry-errors requires --resume")
    if args.subset_flags and not args.visible_rationales:
        ap.error("--subset-flags currently requires --visible-rationales")
    needed = {PROVIDERS[MODELS[tag][0]][0] for tag in args.models}
    missing = sorted(k for k in needed if not os.environ.get(k, "").strip())
    if missing and not args.dry_run:
        ap.error("Missing credentials in this shell: " + ", ".join(missing))
    tasks = ("factuality", "bias") if args.task == "both" else (args.task,)
    root = Path(args.dataset_root).resolve()
    samples = load_test_samples(root)
    if args.subset_flags:
        wanted = {
            (task, media_name, url)
            for task in tasks
            for media_name, url in selected_keys(args.subset_flags, task)
        }
        for sample in samples:
            if ("factuality", sample.media_name, sample.url) not in wanted:
                sample.factuality = None
            if ("bias", sample.media_name, sample.url) not in wanted:
                sample.bias = None
        samples = [sample for sample in samples if sample.factuality is not None or sample.bias is not None]
        matched = sum(getattr(sample, task) is not None for sample in samples for task in tasks)
        if matched != len(wanted):
            ap.error(f"matched {matched} of {len(wanted)} selected task/outlet decisions")
    if args.limit:
        samples = samples[:args.limit]
    sizing = {key: getattr(args, key) for key in ("tile_height", "max_tiles", "max_width")}
    prompts = ({task: rationale_prompt(task) for task in tasks} if args.visible_rationales
               else {task: PROMPTS[task] for task in tasks})
    inventory = []
    for sample in samples:
        with Image.open(sample.image_path) as image:
            width, height = image.size
        layout = geometry(width, height, args.max_width, args.tile_height, args.max_tiles)
        if layout["truncated_px"]:
            ap.error(f"Full-image budget exceeded: {sample.media_name}")
        inventory.append({"media_name": sample.media_name, "url": sample.url, "image_path": sample.image_path,
                          "image_sha256": sha256_file(sample.image_path), "width": width, "height": height,
                          "labels": {task: getattr(sample, task) for task in tasks}, "full_tiles": layout["full_tiles"]})
    metadata_path = root / ".cache/huggingface/download/test.csv.metadata"
    plan = {
        "dataset_root": str(root), "source_split": "test", "test_csv_sha256": sha256_file(root / "test.csv"),
        "download_revision": metadata_path.read_text().splitlines()[0] if metadata_path.exists() else None,
        "tasks": list(tasks), "samples": inventory, "prompts": prompts,
        "models": {tag: {"provider": MODELS[tag][0], "model_id": MODELS[tag][1]} for tag in args.models},
        "preprocessing": sizing, "max_output_tokens": args.max_output_tokens, "reasoning_effort": args.effort,
        "timeout_seconds": args.timeout, "script_sha256": sha256_file(__file__),
    }
    calls = len(args.models) * sum(getattr(s, task) in TASKS[task]["order"] for s in samples for task in tasks)
    print(f"Full API evaluation: {len(samples)} test rows, tasks={tasks}, {len(args.models)} models, {calls} initial calls.", flush=True)
    print("Label counts:", {task: dict(Counter(getattr(s, task) for s in samples)) for task in tasks}, flush=True)
    print("Models:", plan["models"], flush=True)
    print(f"Full-image preflight passed; max tiles={max(row['full_tiles'] for row in inventory)}", flush=True)
    if args.dry_run:
        print("Dry run complete. Missing key variables:", ", ".join(missing) or "none")
        return 0
    output = Path(args.output_dir)
    if not args.resume:
        output.mkdir(parents=True, exist_ok=False)
    elif not (output / "plan.json").is_file():
        ap.error("--resume requires this run's original plan.json")
    with FileLock(str(output / "run.lock"), timeout=0):
        if args.resume:
            if json.loads((output / "plan.json").read_text()) != plan:
                ap.error("Inputs, model settings, or evaluation code changed; choose a new output directory")
        else:
            write_json(output / "plan.json", plan)
        images = save_inputs(samples, sizing, output)
        session_id = uuid.uuid4().hex
        with (output / "sessions.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"id": session_id, "started_at": datetime.now(timezone.utc).isoformat(),
                                "workers": args.workers, "min_interval": args.min_interval,
                                "rate_limit_retries": args.rate_limit_retries, "retry_errors": args.retry_errors}) + "\n")
        stop = threading.Event()
        gates = {provider: RateGate(args.min_interval, stop) for provider in PROVIDERS}
        results = {}

        def evaluate_model(tag):
            if stop.is_set():
                raise EvaluationStopped()
            provider, model = MODELS[tag]
            print(f"Starting {tag}: {model}", flush=True)
            predictor_class = VisibleRationalePredictor if args.visible_rationales else BenchmarkPredictor
            predictor = predictor_class(provider, model, images, sizing, args.max_output_tokens, args.effort, args.timeout,
                                        prompts=prompts, gate=gates[provider],
                                        request_log=output / f"{tag}.requests.jsonl",
                                        session_id=session_id, retries=args.rate_limit_retries, recover=args.resume)
            csv_path = output / f"{tag}.csv"
            scores = run(predictor, samples, out_csv=str(csv_path), tasks=tasks,
                         resume=args.resume and csv_path.with_suffix(".audit").exists(),
                         retry_errors=args.retry_errors, audit_context=plan)
            return {"metrics": scores, "requests": usage_summary(output / f"{tag}.requests.jsonl")}

        def stop_requests(signum, frame):
            stop.set()
            print("Stopping new requests; waiting for in-flight calls to return and saving results.", flush=True)

        previous_handlers = {sig: signal.signal(sig, stop_requests) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            with ThreadPoolExecutor(max_workers=min(args.workers, len(args.models))) as pool:
                futures = {pool.submit(evaluate_model, tag): tag for tag in args.models}
                for future in as_completed(futures):
                    tag = futures[future]
                    try:
                        results[tag] = future.result()
                    except EvaluationStopped:
                        results[tag] = {"interrupted": True}
                    except Exception as exc:
                        results[tag] = {"runner_error": {"type": type(exc).__name__, "message": str(exc)}}
                        print(f"{tag} stopped: {exc}", flush=True)
                    write_json(output / "summary.json", {"source_split": "test", "models": results})
                    print(f"Finished {tag}; results saved under {output}", flush=True)
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
        failures = any("runner_error" in result or any(
            task["status_counts"].get("error", 0) or task["status_counts"].get("unparseable", 0)
            for task in result.get("metrics", {}).values()) for result in results.values())
        print(f"Finished. Inspect {output / 'summary.json'} and per-model audit journals.", flush=True)
        if stop.is_set():
            return 130
        return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
