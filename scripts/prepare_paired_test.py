#!/usr/bin/env python3
"""Validate uploaded pairs and derive immutable evaluator CSVs without copying images."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from audit_image_sizes import geometry
from mediascope_eval import TASKS
from run_config import sha256_file, write_json
from run_masked_local_suite import MODELS as LOCAL_MODELS


def inspect_pairs(source):
    source = source.resolve()
    with (source / "labels.csv").open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        required = {"masked_image", "unmasked_image", "true_label_factuality", "true_label_bias"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Missing paired label columns")
    if not rows:
        raise ValueError("Empty paired dataset")
    for field in ("masked_image", "unmasked_image"):
        names = [r[field] for r in rows]
        if len(names) != len(set(names)) or any(Path(n).name != n for n in names):
            raise ValueError(f"Duplicate/unsafe filename: {field}")
        folder = "masked" if field == "masked_image" else "unmasked"
        actual = {p.name for p in (source / folder).glob("*.png")}
        if actual != set(names):
            raise ValueError(f"Missing/extra {folder} images")
    pairs, seen = [], {"masked": set(), "unmasked": set()}
    for row in rows:
        pair = {"pair_id": Path(row["masked_image"]).stem,
                "labels": {t: row[f"true_label_{t}"] for t in TASKS}}
        if any(pair["labels"][t] not in TASKS[t]["order"] for t in TASKS):
            raise ValueError(f"Invalid label: {pair['pair_id']}")
        for condition in ("masked", "unmasked"):
            name = row[f"{condition}_image"]
            path = source / condition / name
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
                if image.format != "PNG":
                    raise ValueError(f"Expected PNG: {path}")
            digest = sha256_file(path)
            if digest in seen[condition]:
                raise ValueError(f"Duplicate {condition} image content: {name}")
            seen[condition].add(digest)
            layout = geometry(width, height, 1024, 1400, 6)
            if layout["truncated_px"]:
                raise ValueError(f"Full-page input exceeds the matched six-tile budget: {name}")
            pair[condition] = {"filename": name, "image_path": str(path), "image_sha256": digest,
                               "width": width, "height": height, "full_tiles": layout["full_tiles"]}
        if (pair["masked"]["width"], pair["masked"]["height"]) != (pair["unmasked"]["width"], pair["unmasked"]["height"]):
            raise ValueError(f"Paired image dimensions differ: {pair['pair_id']}")
        pairs.append(pair)
    return pairs


def previous_masked_match(pairs, root):
    if root is None:
        return None
    with (root / "labels.csv").open(newline="", encoding="utf-8") as handle:
        labels = {r["image"]: r for r in csv.DictReader(handle)}
    matched_images, matched_labels = [], []
    for pair in pairs:
        name = pair["masked"]["filename"]
        image = root / "images" / name
        if image.is_file() and sha256_file(image) == pair["masked"]["image_sha256"]:
            matched_images.append(pair["pair_id"])
        if name in labels and all(labels[name][f"true_label_{t}"] == pair["labels"][t] for t in TASKS):
            matched_labels.append(pair["pair_id"])
    return {"root": str(root.resolve()), "image_hash_matches": len(matched_images), "label_matches": len(matched_labels),
            "all_images_and_labels_match": len(matched_images) == len(matched_labels) == len(pairs)}


def training_overlap(pairs):
    result, pixels = {}, {}
    for model, config in LOCAL_MODELS.items():
        for task in TASKS:
            adapter = Path(config[f"{task}_adapter"])
            path = adapter / "run_config.json"
            if not path.is_file():
                path = adapter.parent / "run_config.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            hashes = {r["image_sha256"] for r in data["training_samples"]}
            names = {Path(r["image_path"]).name: r for r in data["training_samples"]}
            pixel_matches, missing_references, changed_labels = [], [], []
            for pair in pairs:
                image = pair["unmasked"]
                if image["filename"] not in names:
                    continue
                trained = names[image["filename"]]
                reference = Path(trained["image_path"])
                if not reference.is_file():
                    reference = Path("artifacts/datasets/mediascope-clustered-main/train") / image["filename"]
                if not reference.is_file():
                    missing_references.append(pair["pair_id"])
                    continue
                if sha256_file(reference) != trained["image_sha256"]:
                    raise ValueError(f"Training reference changed: {reference}")
                key = (image["image_path"], trained["image_sha256"])
                if key not in pixels:
                    with Image.open(image["image_path"]) as a, Image.open(reference) as b:
                        pixels[key] = (a.size == b.size and hashlib.sha256(a.convert("RGB").tobytes()).digest()
                                       == hashlib.sha256(b.convert("RGB").tobytes()).digest())
                if pixels[key]:
                    pixel_matches.append(pair["pair_id"])
                if pair["labels"][task] != trained["label"]:
                    changed_labels.append(pair["pair_id"])
            result[f"{model}_{task}"] = {
                "training_config_sha256": sha256_file(path),
                "exact_image_hash_overlap": [p["pair_id"] for p in pairs if p["unmasked"]["image_sha256"] in hashes],
                "filename_overlap": [p["pair_id"] for p in pairs if p["unmasked"]["filename"] in names],
                "pixel_identical_training_images": pixel_matches,
                "missing_training_references": missing_references,
                "training_label_disagreements": changed_labels,
            }
    return result


def prepare(source, output, previous=None, audit_training=False, dry_run=False):
    pairs = inspect_pairs(source)
    counts = {t: dict(Counter(p["labels"][t] for p in pairs)) for t in TASKS}
    manifest = {"protocol": "paired-masked-unmasked-test-v1", "source_root": str(source.resolve()),
                "labels_sha256": sha256_file(source / "labels.csv"), "n": len(pairs), "pairs": pairs,
                "label_counts": counts, "preprocessing": {"tile_height": 1400, "max_tiles": 6, "max_width": 1024},
                "textual_identity_policy": "Generic paired site IDs are metadata only; model input contains only task prompts and images.",
                "previous_masked_match": previous_masked_match(pairs, previous)}
    if audit_training:
        manifest["adapter_training_overlap"] = training_overlap(pairs)
    if not dry_run:
        output.mkdir(parents=True, exist_ok=False)
        fields = ["media_name", "url", "country", "image_path", "true_label_bias", "true_label_factuality",
                  "image_sha256", "width", "height", "masked_image", "unmasked_image"]
        for condition in ("masked", "unmasked"):
            directory = output / condition
            directory.mkdir()
            with (directory / "test.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for pair in pairs:
                    image = pair[condition]
                    writer.writerow({"media_name": pair["pair_id"], "url": "", "country": "", "image_path": image["image_path"],
                                     **{f"true_label_{t}": pair["labels"][t] for t in TASKS},
                                     **{k: image[k] for k in ("image_sha256", "width", "height")},
                                     "masked_image": pair["masked"]["filename"], "unmasked_image": pair["unmasked"]["filename"]})
            write_json(directory / "dataset_manifest.json", {"condition": condition, **manifest})
        write_json(output / "manifest.json", manifest)
    print(json.dumps({"paired_images": len(pairs), "label_counts": counts,
                      "previous_masked_match": manifest["previous_masked_match"],
                      "training_overlap": manifest.get("adapter_training_overlap"), "dry_run": dry_run,
                      "output": str(output)}, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--previous-masked-root", type=Path)
    parser.add_argument("--audit-training-overlap", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    prepare(args.source_root, args.output_root, args.previous_masked_root, args.audit_training_overlap, args.dry_run)
