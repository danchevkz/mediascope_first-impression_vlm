#!/usr/bin/env python3
"""Portable entry points using the actual recorded paper-run settings."""
from __future__ import annotations

import argparse
import csv
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from run_config import sha256_file


ROOT = Path(__file__).resolve().parents[1]
SIZING = {"tile_height": 1400, "max_tiles": 6, "max_width": 1024}


def load(path):
    return json.loads(path.read_text())


def manifest_for(dataset, split="test"):
    return ROOT / "data" / (f"main/{split}.csv" if dataset == "main" else f"paired/{dataset}.csv")


def command(args):
    """Construct a command without loading models, making requests, or writing files."""
    common = [sys.executable]
    if args.action == "train":
        key = f"{args.model}_{args.task}"
        if key == "qwen35_factuality":
            key += "_main"
        config = load(ROOT / "configs/training" / f"{key}.json")
        saved = config["settings"]["arguments"]
        result = common + [str(ROOT / "scripts/train_peft.py"), "--model", args.model, "--task", args.task,
                           "--csv", str(manifest_for("main", "train")), "--image-root", str(args.images.resolve()),
                           "--output-dir", str(args.output.resolve()), "--revision", config["model_revision"]]
        for key in ("quant", "thinking", "target_modules", "lora_r", "lr", "warmup_steps", "num_epochs",
                    "batch_size", "grad_accum", "checkpoint_steps", "save_total_limit", "tile_height",
                    "max_tiles", "max_width", "seed", "num_workers"):
            if key in saved and saved[key] is not None:
                result.extend(["--" + key.replace("_", "-"), str(saved[key])])
        if saved.get("require_full_image"):
            result.append("--require-full-image")
        if saved.get("no_grad_checkpointing"):
            result.append("--no-grad-checkpointing")
        return result
    if args.action == "local":
        settings = load(ROOT / "configs/local_evaluation.json")[args.dataset][f"{args.model}_{args.variant}"][args.task]
        key = settings["adapter_key"]
        result = common + [str(ROOT / "scripts/mediascope_eval.py"), "--predictor", "local", "--model", args.model,
                           "--task", args.task, "--csv", str(manifest_for(args.dataset)),
                           "--image-root", str(args.images.resolve()), "--quant", settings["quant"],
                           "--thinking", settings["thinking"],
                           "--max-new-tokens", str(settings["generation"]["max_new_tokens"]),
                           "--parser", "strict", "--out", str(args.output.resolve()),
                           "--revision", settings["model_revision"]]
        for field, value in settings["preprocessing"].items():
            result.extend(["--" + field.replace("_", "-"), str(value)])
        if args.variant == "base":
            # Match the saved base-run processor without attaching LoRA weights.
            processor_key = f"{args.model}_factuality"
            if args.dataset == "main" and args.model == "qwen35":
                processor_key += "_main"
            result += ["--base-only", "--processor-dir", str(args.adapters.resolve() / processor_key)]
        else:
            result += ["--adapter-dir", str(args.adapters.resolve() / key)]
        if args.limit:
            result += ["--limit", str(args.limit)]
        if args.resume:
            result += ["--resume"]
        return result
    if args.action == "api":
        if args.dataset in ("main", "masked"):
            result = common + [str(ROOT / "scripts/eval_api.py"), "--models", args.model,
                               "--dataset-root", str(args.inputs.resolve()), "--output-dir", str(args.output.resolve()),
                               "--max-tiles", "6", "--max-width", "1024", "--tile-height", "1400",
                               "--effort", "medium", "--max-output-tokens", "8192"]
        else:
            result = common + [str(ROOT / "scripts/eval_paired_unmasked_api.py"), "--model", args.model,
                               "--dataset-root", str(args.inputs.resolve()),
                               "--input-dir", str(args.output.resolve().parent / "prepared_inputs_unmasked"),
                               "--output-dir", str(args.output.resolve())]
        if args.resume:
            result.append("--resume")
        return result
    raise ValueError(args.action)


def prepare(args):
    """Bind immutable release manifests to a local screenshot directory and verify hashes."""
    if args.output.exists():
        raise FileExistsError("Choose a fresh prepared-input directory")
    images = args.images.resolve()
    inventory = load(ROOT / "data/main/image_inventory.json")
    parts = {}
    for split in (("train", "test") if args.dataset == "main" else ("test",)):
        with manifest_for(args.dataset, split).open(newline="") as handle:
            reader = csv.DictReader(handle)
            data, fields = list(reader), reader.fieldnames
        hashes = {r["path"]: r["sha256"] for r in inventory["splits"][split]["images"]} if args.dataset == "main" else {}
        for row in data:
            path = images / row["image_path"]
            digest = hashes[row["image_path"]] if args.dataset == "main" else row["image_sha256"]
            if sha256_file(path) != digest:
                raise ValueError(f"Screenshot hash mismatch: {path.name}")
            row["image_path"] = str(path)
        parts[split] = (fields, data)
    args.output.mkdir(parents=True)
    for split, (fields, data) in parts.items():
        with (args.output / f"{split}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(data)
    if args.dataset == "main":
        shutil.copyfile(ROOT / "data/main/clusters_13.csv", args.output / "clusters_13.csv")
    (args.output / "release_input.json").write_text(json.dumps({"dataset": args.dataset, "images": str(images)}, indent=2) + "\n")
    (args.output / "dataset_manifest.json").write_text(json.dumps(
        {"condition": args.dataset, "preprocessing": SIZING, "n": len(parts["test"][1])}, indent=2) + "\n")
    print(f"Verified and prepared {args.dataset}: {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--dataset", choices=("main", "masked", "unmasked"), required=True)
    prepare_parser.add_argument("--images", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    for action in ("train", "local", "api"):
        p = sub.add_parser(action)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--dry-run", action="store_true", help="print the exact command; no files, models or requests")
        if action != "api":
            p.add_argument("--model", choices=("qwen35", "ministral"), required=True)
            p.add_argument("--task", choices=("factuality", "bias"), required=True)
            p.add_argument("--images", type=Path, required=True)
        else:
            p.add_argument("--model", choices=("gpt56", "opus5", "kimi", "qwen"), required=True)
            p.add_argument("--inputs", type=Path, required=True)
        if action != "train":
            p.add_argument("--dataset", choices=("main", "masked", "unmasked"), required=True)
            p.add_argument("--resume", action="store_true")
        if action == "local":
            p.add_argument("--variant", choices=("base", "finetuned"), required=True)
            p.add_argument("--adapters", type=Path, default=ROOT / "assets/adapters")
            p.add_argument("--limit", type=int)
    p = sub.add_parser("prepare-api-inputs")
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args)
        return
    if args.action == "prepare-api-inputs":
        from eval_api import load_test_samples, save_inputs
        if args.output.exists():
            parser.error("Choose a fresh tile directory")
        args.output.mkdir(parents=True)
        save_inputs(load_test_samples(args.inputs), SIZING, args.output)
        return
    if args.action == "api" and not args.dry_run:
        prepared = load(args.inputs / "release_input.json")
        if prepared["dataset"] != args.dataset:
            parser.error("Prepared images belong to a different condition")
        if args.dataset == "unmasked":
            from eval_api import load_test_samples, save_inputs
            tiles = args.output.resolve().parent / "prepared_inputs_unmasked"
            tiles.mkdir(parents=True, exist_ok=True)
            save_inputs(load_test_samples(args.inputs), SIZING, tiles)
    run = command(args)
    print(shlex.join(run), flush=True)
    if not args.dry_run:
        subprocess.run(run, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
