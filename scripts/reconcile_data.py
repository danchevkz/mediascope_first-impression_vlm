#!/usr/bin/env python3
"""Validate the current split against groups_json and attach screenshot paths."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import unicodedata
from collections import Counter


def filename_stem(media_name: str) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", media_name).strip("_")
    return unicodedata.normalize("NFKD", safe_name).encode("ascii", "ignore").decode()


def load_group_entries(groups_dir: str, split: str) -> dict[str, dict]:
    entries = {}
    for path in sorted(glob.glob(os.path.join(groups_dir, f"*_{split}.json"))):
        with open(path, encoding="utf-8") as f:
            for entry in json.load(f):
                image = entry["image"]
                if image in entries:
                    raise ValueError(f"duplicate groups_json image: {image}")
                entries[image] = entry
    return entries


def image_paths(image_root: str) -> dict[str, str]:
    paths = {}
    for path in glob.glob(os.path.join(image_root, "**", "*.png"), recursive=True):
        name = os.path.basename(path)
        if name in paths:
            raise ValueError(f"duplicate screenshot basename: {name}")
        paths[name] = path
    return paths


def reconcile(csv_path: str, split: str, groups_dir: str, image_root: str, write: bool):
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = list(reader.fieldnames or [])

    groups = load_group_entries(groups_dir, split)
    screenshots = image_paths(image_root)
    if len(rows) != len(groups):
        raise ValueError(f"{split}: {len(rows)} CSV rows != {len(groups)} groups_json rows")

    used = set()
    for row in rows:
        stem = filename_stem(row["media_name"])
        candidates = [f"{stem}.png", f"{stem}_1.png"]
        matches = [name for name in candidates if name in groups and name not in used]
        if len(matches) != 1:
            raise ValueError(
                f"{split}: {row['media_name']!r} matched {matches}, expected exactly one image"
            )
        image = matches[0]
        entry = groups[image]
        for label in ("true_label_factuality", "true_label_bias"):
            if row[label].strip().upper() != entry[label].strip().upper():
                raise ValueError(
                    f"{split}: {row['media_name']!r} {label} is {row[label]!r} in CSV "
                    f"but {entry[label]!r} in groups_json"
                )
        if image not in screenshots:
            raise ValueError(f"{split}: missing screenshot {image}")
        row["image_path"] = os.path.relpath(screenshots[image], os.path.dirname(image_root))
        used.add(image)

    unused = set(groups) - used
    if unused:
        raise ValueError(f"{split}: {len(unused)} unmatched groups_json images: {sorted(unused)[:5]}")

    if "image_path" not in fields:
        fields.append("image_path")
    if write:
        tmp_path = f"{csv_path}.tmp"
        with open(tmp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_path, csv_path)

    print(f"{split}: {len(rows)} rows reconciled; labels={dict(Counter(r['true_label_factuality'] for r in rows))}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--write", action="store_true", help="add/update image_path in train.csv and test.csv")
    args = ap.parse_args()

    for split in ("train", "test"):
        reconcile(
            os.path.join(args.data_dir, f"{split}.csv"),
            split,
            os.path.join(args.data_dir, "groups_json"),
            os.path.join(args.data_dir, "mediascopeAll"),
            args.write,
        )


if __name__ == "__main__":
    main()
