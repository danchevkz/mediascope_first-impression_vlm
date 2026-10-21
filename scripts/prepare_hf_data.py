#!/usr/bin/env python3
"""Validate an HF split and write separate CSV manifests with screenshot paths."""

import argparse
import csv
from collections import Counter
from pathlib import Path

from PIL import Image

from audit_image_sizes import geometry
from mediascope_eval import FACTUALITY_ORDER, BIAS_ORDER
from reconcile_data import filename_stem
from run_config import sha256_file, write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--output-dir", required=True, help="new manifest directory; source files are preserved")
    args = ap.parse_args()
    root, output = Path(args.dataset_root).resolve(), Path(args.output_dir)
    if output.exists():
        ap.error(f"Output already exists: {output}")
    parts, identities, files = {}, {}, {}
    report = {"dataset_root": str(root), "image_root_for_training": str(root), "splits": {}}
    for split in ("train", "test"):
        source = root / f"{split}.csv"
        with source.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows, fields = list(reader), list(reader.fieldnames or [])
        if not rows:
            raise ValueError(f"Empty split: {split}")
        inputs, max_tiles = [], 0
        for row in rows:
            name, url = row["media_name"], row["url"]
            if (name, url) in identities or name in identities or url in identities:
                raise ValueError(f"Repeated outlet across rows/splits: {name}")
            identities[(name, url)] = identities[name] = identities[url] = split
            if row["true_label_factuality"] not in FACTUALITY_ORDER:
                raise ValueError(f"Invalid factuality target: {name}")
            if row["true_label_bias"] not in BIAS_ORDER + ["UNRATED"]:
                raise ValueError(f"Invalid bias target: {name}")
            stem = filename_stem(name)
            matches = [root / split / fn for fn in (f"{stem}.png", f"{stem}_1.png")
                       if (root / split / fn).is_file()]
            if len(matches) != 1:
                raise ValueError(f"Missing/ambiguous image for {name}: {matches}")
            path = matches[0]
            if path.name in files:
                raise ValueError(f"Repeated screenshot name across splits: {path.name}")
            with Image.open(path) as image:
                width, height = image.size
                image.verify()
            layout = geometry(width, height, 1024, 1400, 6)
            if layout["truncated_px"]:
                raise ValueError(f"{name} exceeds the six-tile full-coverage budget")
            max_tiles = max(max_tiles, layout["full_tiles"])
            row["image_path"] = path.relative_to(root).as_posix()
            inputs.append({"path": row["image_path"], "sha256": sha256_file(path),
                           "width": width, "height": height})
            files[path.name] = {"split": split, "row": row}
        available = {p.name for p in (root / split).glob("*.png")}
        expected = {Path(row["image_path"]).name for row in rows}
        if available != expected:
            raise ValueError(f"Unmatched screenshot inventory for {split}")
        metadata = root / ".cache/huggingface/download" / f"{split}.csv.metadata"
        report["splits"][split] = {
            "rows": len(rows), "source_csv_sha256": sha256_file(source),
            "revision": metadata.read_text().splitlines()[0] if metadata.exists() else None,
            "factuality": dict(Counter(row["true_label_factuality"] for row in rows)),
            "bias": dict(Counter(row["true_label_bias"] for row in rows)),
            "max_full_page_tiles_at_width_1024": max_tiles, "images": inputs,
        }
        parts[split] = rows, fields + (["image_path"] if "image_path" not in fields else [])
    clusters = root / "clusters_13.csv"
    if clusters.exists():
        with clusters.open(newline="", encoding="utf-8") as f:
            for record in csv.DictReader(f):
                matched = files.get(record["image"])
                if not matched or matched["split"] != record["split"]:
                    raise ValueError(f"Cluster/split membership mismatch: {record['image']}")
                for column, target in (("factuality", "true_label_factuality"), ("bias", "true_label_bias")):
                    if record[column] != matched["row"][target]:
                        raise ValueError(f"Cluster/CSV label mismatch: {record['image']} {column}")
        report["clusters_csv_sha256"] = sha256_file(clusters)
    output.mkdir(parents=True, exist_ok=False)
    for split, (rows, fields) in parts.items():
        path = output / f"{split}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        report["splits"][split]["prepared_csv_sha256"] = sha256_file(path)
        print(split, len(rows), report["splits"][split]["factuality"], "max tiles", report["splits"][split]["max_full_page_tiles_at_width_1024"])
    write_json(output / "validation.json", report)
    print(f"Prepared manifests: {output}; use --image-root {root}")


if __name__ == "__main__":
    main()
