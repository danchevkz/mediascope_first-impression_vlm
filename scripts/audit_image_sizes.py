#!/usr/bin/env python3
"""Inspect stored screenshot coverage without loading models or altering data.

Supports reconciled CSV image_path columns and HF train/ and test/ image folders.
The report measures the stored image, not whether the original web capture was
complete. Tile geometry follows qwen_predictor's resizing/sliver rule.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from reconcile_data import filename_stem
from run_config import IMAGE_DEFAULTS, sha256_file, write_json

Image.MAX_IMAGE_PIXELS = None


def geometry(width, height, max_width, tile_height, cap):
    if min(width, height, max_width, tile_height, cap) <= 0:
        raise ValueError("Image dimensions and budget must be positive")
    w = min(width, max_width)
    h = max(1, int(height * (max_width / width))) if width > max_width else height
    intervals = []
    y = 0
    while y < h:
        end = min(y + tile_height, h)
        if 0 < h - end < max(50, w // 100):
            end = h
        intervals.append([y, end])
        y = end
    covered = intervals[min(cap, len(intervals)) - 1][1]
    return {"resized_width": w, "resized_height": h,
            "full_tiles": len(intervals), "full_pixels": w * h,
            "capped_pixels": w * covered, "truncated_px": h - covered,
            "retained_fraction": covered / h}


def describe(values):
    values = sorted(values)
    return {"min": values[0], "p50": values[math.ceil(0.50 * len(values)) - 1],
            "p90": values[math.ceil(0.90 * len(values)) - 1],
            "p95": values[math.ceil(0.95 * len(values)) - 1],
            "p99": values[math.ceil(0.99 * len(values)) - 1], "max": values[-1],
            "mean": sum(values) / len(values)}


def summarize(rows, widths):
    summary = {
        "n": len(rows), "source_width": describe([r["width"] for r in rows]),
        "source_height": describe([r["height"] for r in rows]),
        "source_megapixels": describe([r["width"] * r["height"] / 1e6 for r in rows]),
        "aspect_height_over_width": describe([r["height"] / r["width"] for r in rows]),
        "height_frequency": dict(Counter(r["height"] for r in rows).most_common(10)),
        "sizing": {},
    }
    for width in widths:
        prefix = f"w{width}_"
        summary["sizing"][str(width)] = {
            "resized_height": describe([r[prefix + "resized_height"] for r in rows]),
            "full_tiles": describe([r[prefix + "full_tiles"] for r in rows]),
            "tile_histogram": dict(sorted(Counter(r[prefix + "full_tiles"] for r in rows).items())),
            "truncated_by_current_cap": sum(r[prefix + "truncated_px"] > 0 for r in rows),
            "retained_fraction": describe([r[prefix + "retained_fraction"] for r in rows]),
            "full_to_capped_total_pixel_ratio": sum(r[prefix + "full_pixels"] for r in rows)
                                                / sum(r[prefix + "capped_pixels"] for r in rows),
        }
    summary["largest_15"] = [
        {k: row[k] for k in ("split", "media_name", "image_path", "width", "height",
                             f"w{widths[0]}_resized_height", f"w{widths[0]}_full_tiles")}
        for row in sorted(rows, key=lambda r: r[f"w{widths[0]}_resized_height"], reverse=True)[:15]
    ]
    return summary


def inspect(root, widths, tile_height, cap):
    records, inputs, matched, available = [], {}, set(), set()
    for split in ("train", "test"):
        csv_path = root / f"{split}.csv"
        inputs[split] = {"path": str(csv_path), "sha256": sha256_file(csv_path)}
        hf_metadata = root / ".cache/huggingface/download" / f"{split}.csv.metadata"
        if hf_metadata.exists():
            inputs[split]["downloaded_revision"] = hf_metadata.read_text().splitlines()[0]
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            has_paths = "image_path" in (reader.fieldnames or [])
            rows = list(reader)
        if not rows:
            raise ValueError(f"Empty split: {csv_path}")
        image_tree = root / "mediascopeAll" if has_paths else root / split
        available.update(p.resolve() for p in image_tree.rglob("*.png"))
        for row in rows:
            if has_paths:
                path = (root / row["image_path"]).resolve()
            else:
                stem = filename_stem(row["media_name"])
                candidates = [root / split / name for name in (f"{stem}.png", f"{stem}_1.png")]
                hits = [p for p in candidates if p.is_file()]
                if len(hits) != 1:
                    raise ValueError(f"{split}: ambiguous or missing image for {row['media_name']!r}: {hits}")
                path = hits[0].resolve()
            if path in matched:
                raise ValueError(f"Screenshot repeated in dataset: {path}")
            matched.add(path)
            with Image.open(path) as image:
                width, height = image.size
            record = {"split": split, "media_name": row["media_name"], "url": row["url"],
                      "image_path": str(path), "image_sha256": sha256_file(path),
                      "file_bytes": path.stat().st_size, "width": width, "height": height}
            for target_width in widths:
                record.update({f"w{target_width}_{key}": value for key, value in
                               geometry(width, height, target_width, tile_height, cap).items()})
            records.append(record)
    return records, {"csv_inputs": inputs, "unreferenced_images": [str(p) for p in sorted(available - matched)]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", default="data")
    ap.add_argument("--widths", type=int, nargs="+", default=[1024, 768])
    ap.add_argument("--tile-height", type=int, default=IMAGE_DEFAULTS["tile_height"])
    ap.add_argument("--current-cap", type=int, default=IMAGE_DEFAULTS["max_tiles"])
    ap.add_argument("--output-dir", required=True, help="new audit directory; never overwritten")
    args = ap.parse_args()
    if min(*args.widths, args.tile_height, args.current_cap) <= 0:
        ap.error("Widths, tile height, and cap must be positive")
    out = Path(args.output_dir)
    if out.exists():
        ap.error(f"Output directory already exists: {out}")
    root = Path(args.dataset_root).resolve()
    records, inputs = inspect(root, args.widths, args.tile_height, args.current_cap)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(), "dataset_root": str(root),
        "method": "PNG header dimensions; nearest-rank quantiles; no GPU/model forward; stored-image coverage only",
        "script_sha256": sha256_file(__file__),
        "settings": {"widths": args.widths, "tile_height": args.tile_height, "current_cap": args.current_cap},
        **inputs,
        "partitions": {part: summarize([r for r in records if part == "all" or r["split"] == part], args.widths)
                       for part in ("train", "test", "all")},
    }
    out.mkdir(parents=True, exist_ok=False)
    with (out / "image_sizes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    write_json(out / "summary.json", report)
    for partition, result in report["partitions"].items():
        print(partition, "n=", result["n"], "height=", json.dumps(result["source_height"]))
        for width, sizing in result["sizing"].items():
            print(f"  width={width}: full tiles={sizing['tile_histogram']}; "
                  f"truncated at {args.current_cap}={sizing['truncated_by_current_cap']}; "
                  f"full/capped pixels={sizing['full_to_capped_total_pixel_ratio']:.3f}")
    print(f"Saved {out / 'image_sizes.csv'} and {out / 'summary.json'}")


if __name__ == "__main__":
    main()
