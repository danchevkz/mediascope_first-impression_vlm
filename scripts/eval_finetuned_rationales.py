#!/usr/bin/env python3
"""Elicit auditable rationales from a task-specific fine-tuned adapter."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from mediascope_eval import TASKS, load_samples
from qwen_predictor import PROMPTS, TransformersLoraPredictor
from review_api_reasoning import PATTERNS
from run_config import load_adapter_run_config, sha256_file, write_json
from train_peft import resolve_model_id


def rationale_prompt(task: str) -> str:
    base = PROMPTS[task].rsplit("\n\nOutput ONLY", 1)[0]
    labels = ", ".join(TASKS[task]["order"])
    return (
        f"{base}\n\nBriefly explain the basis for your judgment in 1-3 sentences. "
        f"Finish with a separate final line in the form `LABEL: <label>`, where "
        f"<label> is one of: {labels}."
    )


def parse_label(text: str, labels: list[str]) -> str | None:
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if stripped.upper() in labels:
        return stripped.upper()
    alternatives = "|".join(re.escape(label) for label in sorted(labels, key=len, reverse=True))
    match = re.search(
        rf"(?im)^\s*(?:\*\*)?LABEL\s*:\s*({alternatives})\s*[.`]*(?:\*\*)?\s*$",
        stripped,
    )
    return match.group(1).upper() if match else None


def selected_keys(path: str, task: str) -> set[tuple[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        fields = set(rows.fieldnames or ())
        selected = set()
        for row in rows:
            if {"model", "task", "status", "readable"}.issubset(fields):
                include = (row["model"] == "opus" and row["task"] == task
                           and row["status"] == "ok" and row["readable"] == "True")
            elif "task" in fields:
                include = row["task"] == task
            else:
                include = True
            if include:
                selected.add((row["media_name"], row["url"]))
        return selected


def load_completed(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    completed = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                completed.add((row["media_name"], row["url"]))
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir")
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--model", help="registry key or HF model id for --base-only")
    parser.add_argument("--processor-dir", help="processor assets for --base-only")
    parser.add_argument("--revision")
    parser.add_argument("--quant", choices=("4bit", "16bit"), default="4bit")
    parser.add_argument("--task", choices=("factuality", "bias"), required=True)
    parser.add_argument("--subset-flags", required=True)
    parser.add_argument("--csv", default="artifacts/datasets/mediascope-clustered-prepared-ea1f7ec37e90/test.csv")
    parser.add_argument("--image-root", default="artifacts/datasets/mediascope-clustered-main")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    journal = output_dir / "predictions.jsonl"
    if output_dir.exists() and journal.exists() and not args.resume:
        parser.error("output already exists; pass --resume")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.base_only and args.adapter_dir:
        parser.error("--base-only and --adapter-dir are mutually exclusive")
    if not args.base_only and not args.adapter_dir:
        parser.error("--adapter-dir is required unless --base-only is selected")
    if args.base_only and (not args.model or not args.processor_dir):
        parser.error("--base-only requires --model and --processor-dir")
    config_dir = args.processor_dir if args.base_only else args.adapter_dir
    saved = load_adapter_run_config(config_dir)
    if not saved:
        parser.error("model assets have no run_config.json")
    settings = saved["settings"]
    if not args.base_only and settings["task"] != args.task:
        parser.error(f"adapter task is {settings['task']}, not {args.task}")

    wanted = selected_keys(args.subset_flags, args.task)
    samples, missing = load_samples(args.csv, args.image_root, tasks=(args.task,))
    if missing:
        parser.error(f"{len(missing)} selected dataset images are missing")
    samples = [sample for sample in samples if (sample.media_name, sample.url) in wanted]
    if len(samples) != len(wanted):
        parser.error(f"matched {len(samples)} of {len(wanted)} requested records")
    if args.limit:
        samples = samples[:args.limit]

    model_id, default_skips = resolve_model_id(args.model if args.base_only else settings["model_id"])
    prompt = rationale_prompt(args.task)
    predictor = TransformersLoraPredictor(
        model_id=model_id,
        adapter_dir=None if args.base_only else args.adapter_dir,
        processor_dir=args.processor_dir if args.base_only else None,
        quant=args.quant if args.base_only else settings.get("quant", "4bit"),
        skip_vision_modules=settings.get("skip_vision_modules", default_skips),
        **settings["preprocessing"],
        max_new_tokens=args.max_new_tokens,
        thinking="off",
        prompts={args.task: prompt},
        revision=args.revision or saved.get("model_revision"),
    )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "records": len(samples),
        "adapter": predictor.run_metadata(),
        "csv": str(Path(args.csv).resolve()),
        "csv_sha256": sha256_file(args.csv),
        "subset_flags": str(Path(args.subset_flags).resolve()),
        "subset_flags_sha256": sha256_file(args.subset_flags),
        "prompt": prompt,
        "selection": "Fixed task/outlet subset supplied by --subset-flags",
    }
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and args.resume:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("task", "csv_sha256", "subset_flags_sha256", "prompt"):
            if previous[key] != manifest[key]:
                parser.error(f"resume mismatch for {key}")
    else:
        write_json(manifest_path, manifest)

    completed = load_completed(journal) if args.resume else set()
    labels = TASKS[args.task]["order"]
    mode = "a" if args.resume else "w"
    with open(journal, mode, encoding="utf-8") as handle:
        for index, sample in enumerate(samples, 1):
            key = (sample.media_name, sample.url)
            if key in completed:
                continue
            try:
                raw = predictor.predict(sample, args.task)
                predicted = parse_label(raw, labels)
                status = "ok" if predicted else "unparseable"
                error = None
            except Exception as exc:
                raw, predicted, status, error = None, None, "error", repr(exc)
            text = raw or ""
            record = {
                "media_name": sample.media_name,
                "url": sample.url,
                "task": args.task,
                "true_label": getattr(sample, args.task),
                "predicted_label": predicted,
                "correct": predicted == getattr(sample, args.task),
                "status": status,
                "error": error,
                "raw_output": raw,
                "generated_tokens": predictor.last_details.get("generated_tokens"),
                "reached_token_budget": predictor.last_details.get("reached_token_budget"),
                "flags": {name: bool(re.search(pattern, text, re.IGNORECASE))
                          for name, pattern in PATTERNS.items()},
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            print(f"[{index}/{len(samples)}] {sample.media_name}: {status}", flush=True)


if __name__ == "__main__":
    main()
