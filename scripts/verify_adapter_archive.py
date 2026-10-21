#!/usr/bin/env python3
"""Verify the companion adapter archive without extracting large weight files."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path

from run_config import sha256_file


def verify(root, archive_path):
    expected = json.loads((root / "release_manifest.json").read_text())["adapter_archive"]
    if sha256_file(archive_path) != expected["sha256"]:
        raise ValueError("Adapter archive SHA-256 mismatch")
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)) or any(not m.isfile() for m in members):
            raise ValueError("Duplicate or non-regular archive member")
        for name in names:
            path = Path(name)
            if path.is_absolute() or ".." in path.parts or not name.startswith("assets/adapters/"):
                raise ValueError("Unexpected archive path")
        inventory = json.load(archive.extractfile("assets/adapters/inventory.json"))
        if set(names) != {r["path"] for r in inventory} | {"assets/adapters/inventory.json"}:
            raise ValueError("Archive inventory differs from members")
        licensing = json.loads((root / "licenses/metadata.json").read_text())
        adapters = json.loads((root / "configs/adapters.json").read_text())
        for key, entry in adapters.items():
            base = f"assets/adapters/{key}/"
            for name in ("LICENSE", "BASE_MODEL_LICENSE", "NOTICE", "README.md"):
                if base + name not in names:
                    raise ValueError(f"Missing adapter licensing file: {base + name}")
            upstream = licensing["upstream_models"][entry["model_id"]]
            if archive.extractfile(base + "LICENSE").read() != (root / licensing["adapters"]["file"]).read_bytes():
                raise ValueError(f"Model-license text differs: {key}")
            if archive.extractfile(base + "BASE_MODEL_LICENSE").read() != (root / upstream["license_file"]).read_bytes():
                raise ValueError(f"Upstream-license text differs: {key}")
            notice = archive.extractfile(base + "NOTICE").read().decode()
            if any(value not in notice for value in (entry["model_id"], entry["model_revision"], upstream["notice"])):
                raise ValueError(f"Incomplete adapter attribution: {key}")
        for row in inventory:
            digest = hashlib.sha256()
            length = 0
            with archive.extractfile(row["path"]) as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
                    length += len(chunk)
            if digest.hexdigest() != row["sha256"] or length != row["bytes"]:
                raise ValueError(f"Adapter asset differs: {row['path']}")
    print(f"Verified {len(inventory)} adapter assets, per-adapter licenses/notices, and the archive checksum.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    verify(Path(__file__).resolve().parents[1], args.archive)
