#!/usr/bin/env python3
"""Small, auditable, reasoning-enabled API smoke test on HF training examples."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import random
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from api_predictor import PROVIDERS, ReasoningAPIPredictor
from mediascope_eval import Sample, TASKS, run
from qwen_predictor import PROMPTS, _data_url, tile_image_with_info
from reconcile_data import filename_stem
from run_config import sha256_file, write_json


MODELS = {
    "kimi": ("openrouter", "moonshotai/kimi-k3"),
    "qwen": ("openrouter", "qwen/qwen3.8-max-prime"),
    "claude": ("anthropic", "claude-sonnet-5-5"),
    "openai": ("openai", "gpt-6.1-sol"),
    "opus": ("anthropic", "claude-opus-5-5"),
    "gpt56": ("openai", "gpt-5.6-sol"),
    "opus5": ("anthropic", "claude-opus-5"),
}
DEFAULT_MODELS = ["kimi", "qwen", "opus5", "gpt56"]


def select_samples(root, task, limit, seed):
    """Seeded label cycling for plumbing coverage, not a performance estimate."""
    with (root / "train.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    groups = defaultdict(list)
    identities = set()
    for row in rows:
        label = row[f"true_label_{task}"].strip().upper()
        if label == "UNRATED" and task == "bias":
            continue
        if label not in TASKS[task]["order"]:
            raise ValueError(f"Unexpected {task} label in training CSV: {label!r}")
        key = (row["media_name"], row["url"])
        if key in identities:
            raise ValueError(f"Duplicate training outlet: {row['media_name']}")
        identities.add(key)
        groups[label].append(row)
    rng = random.Random(seed)
    for label in TASKS[task]["order"]:
        groups[label].sort(key=lambda row: (row["media_name"], row["url"]))
        rng.shuffle(groups[label])
    chosen = []
    while len(chosen) < limit:
        for label in TASKS[task]["order"]:
            if groups[label] and len(chosen) < limit:
                chosen.append(groups[label].pop())
        if not any(groups.values()) and len(chosen) < limit:
            raise ValueError("Not enough eligible training examples")
    samples = []
    for row in chosen:
        if row.get("image_path"):
            path = (root / row["image_path"]).resolve()
        else:
            stem = filename_stem(row["media_name"])
            hits = [root / "train" / name for name in (f"{stem}.png", f"{stem}_1.png")
                    if (root / "train" / name).is_file()]
            if len(hits) != 1:
                raise ValueError(f"Missing or ambiguous screenshot: {row['media_name']}")
            path = hits[0].resolve()
        bias = row["true_label_bias"].strip().upper()
        samples.append(Sample(row["media_name"], row["url"], str(path), row.get("country", ""),
                              row["true_label_factuality"].strip().upper(), None if bias == "UNRATED" else bias))
    if len({s.image_path for s in samples}) != len(samples):
        raise ValueError("Repeated screenshot in selected examples")
    return samples


def prepare_images(samples, sizing):
    prepared = {}
    for sample in samples:
        tiles, info = tile_image_with_info(sample.image_path, **sizing)
        try:
            if info["truncated_px"]:
                raise ValueError(f"{sample.media_name}: {info['truncated_px']} pixels would be omitted; increase --max-tiles")
            urls = [_data_url(tile) for tile in tiles]
            hashes = [hashlib.sha256(base64.b64decode(url.split(",", 1)[1])).hexdigest() for url in urls]
            prepared[sample.image_path] = {"image": info, "data_urls": urls, "png_sha256": hashes}
        finally:
            for tile in tiles:
                tile.close()
    return prepared


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", default="artifacts/datasets/mediascope-clustered-main")
    ap.add_argument("--models", nargs="+", choices=MODELS, default=DEFAULT_MODELS)
    ap.add_argument("--task", choices=TASKS, default="factuality")
    ap.add_argument("--limit", type=int, default=5, help="1-20 training examples; this entry point is smoke-only")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tile-height", type=int, default=1400)
    ap.add_argument("--max-width", type=int, default=1024)
    ap.add_argument("--max-tiles", type=int, default=6)
    ap.add_argument("--max-output-tokens", type=int, default=8192, help="per-request cap including reasoning")
    ap.add_argument("--effort", choices=("low", "medium", "high"), default="medium")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--output-dir", help="new directory; default unique directory under artifacts/evaluations/")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--retry-errors", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="validate and print the plan without requests or writing outputs")
    args = ap.parse_args()
    if not 1 <= args.limit <= 20:
        ap.error("--limit must be between 1 and 20 for a smoke test")
    if min(args.tile_height, args.max_width, args.max_tiles, args.max_output_tokens, args.timeout) <= 0:
        ap.error("Sizing, token cap, and timeout must be positive")
    if len(set(args.models)) != len(args.models):
        ap.error("Select each model only once")
    if args.resume and not args.output_dir:
        ap.error("--resume requires --output-dir")
    if args.retry_errors and not args.resume:
        ap.error("--retry-errors requires --resume")
    needed = {PROVIDERS[MODELS[tag][0]][0] for tag in args.models}
    missing = sorted(key for key in needed if not os.environ.get(key, "").strip())
    if missing and not args.dry_run:
        ap.error("Missing credentials in this shell: " + ", ".join(missing))
    root = Path(args.dataset_root).resolve()
    samples = select_samples(root, args.task, args.limit, args.seed)
    sizing = {key: getattr(args, key) for key in ("tile_height", "max_tiles", "max_width")}
    prepared = prepare_images(samples, sizing)
    selected = [{"media_name": s.media_name, "url": s.url, "image_path": s.image_path,
                 "image_sha256": sha256_file(s.image_path), "label": getattr(s, args.task),
                 "image": prepared[s.image_path]["image"], "submitted_png_sha256": prepared[s.image_path]["png_sha256"]}
                for s in samples]
    metadata_path = root / ".cache/huggingface/download/train.csv.metadata"
    plan = {
        "smoke_only": True, "dataset_root": str(root), "source_split": "train",
        "train_csv_sha256": sha256_file(root / "train.csv"),
        "download_revision": metadata_path.read_text().splitlines()[0] if metadata_path.exists() else None,
        "selection": "seeded round-robin over task labels", "seed": args.seed, "task": args.task,
        "models": {tag: {"provider": MODELS[tag][0], "model_id": MODELS[tag][1]} for tag in args.models},
        "max_output_tokens": args.max_output_tokens, "reasoning_effort": args.effort,
        "timeout_seconds": args.timeout, "preprocessing": sizing, "prompt": PROMPTS[args.task],
        "samples": selected, "script_sha256": sha256_file(__file__),
    }
    print(f"Smoke: {len(samples)} training screenshots x {len(args.models)} models = at most {len(samples) * len(args.models)} API calls; no automatic retries.")
    for sample, info in zip(samples, selected):
        print(f"  {sample.media_name}: {getattr(sample, args.task)}; {info['image']['tile_count']} tiles; 100% stored-image coverage")
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        print("Missing key variables (no requests made):", ", ".join(missing) or "none")
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.output_dir or f"artifacts/evaluations/api-smoke-{stamp}-{uuid.uuid4().hex[:8]}")
    if args.resume:
        saved = json.loads((out / "plan.json").read_text())
        if saved != plan:
            ap.error("Smoke plan differs from saved run; use a new output directory")
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / "plan.json", plan)
    # Keep the exact PNG tiles sent to all providers, for visual inspection.
    for index, sample in enumerate(samples):
        directory = out / "inputs" / f"sample-{index + 1:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        image = prepared[sample.image_path]
        for number, (url, expected_hash) in enumerate(zip(image["data_urls"], image["png_sha256"]), 1):
            path = directory / f"tile-{number:02d}.png"
            if path.exists():
                if sha256_file(path) != expected_hash:
                    raise ValueError(f"Saved input tile changed: {path}")
            else:
                with path.open("xb") as f:
                    f.write(base64.b64decode(url.split(",", 1)[1]))
    results = {}
    for tag in args.models:
        provider, model = MODELS[tag]
        print(f"\n=== {tag}: {provider} / {model} ===", flush=True)
        predictor = ReasoningAPIPredictor(provider, model, prepared, sizing,
                                          args.max_output_tokens, args.effort, args.timeout)
        csv_path = out / f"{tag}.csv"
        audit_exists = csv_path.with_suffix(".audit").exists()
        metrics = run(predictor, samples, out_csv=str(csv_path), tasks=(args.task,),
                      resume=args.resume and audit_exists, retry_errors=args.retry_errors,
                      audit_context=plan)
        results[tag] = metrics[args.task]
        write_json(out / "summary.json", {"smoke_only": True, "models": results})
    print(f"\nSaved smoke outputs and exact input tiles: {out.resolve()}")
    print("These five-example scores verify plumbing, not held-out model quality.")
    failed = any(result["status_counts"].get("error", 0) or result["status_counts"].get("unparseable", 0)
                 for result in results.values())
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
