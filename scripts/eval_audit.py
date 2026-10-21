"""Durable evaluation journal. CSV and metrics are derived from this journal."""

from __future__ import annotations

import csv
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from filelock import FileLock

from run_config import json_hash, write_json


def timestamp():
    return datetime.now(timezone.utc).isoformat()


class EvaluationAudit:
    def __init__(self, out_csv, identity, resume=False):
        self.out_csv = Path(out_csv)
        self.directory = self.out_csv.with_suffix(".audit")
        self.identity = json.loads(json.dumps(identity))
        self.resume = resume
        self.records = {}
        self.journal = None
        self.lock = None

    def __enter__(self):
        if self.out_csv.exists() and not self.resume:
            raise FileExistsError(f"Refusing to overwrite {self.out_csv}; use a new --out or --resume")
        if self.resume:
            if not (self.directory / "manifest.json").is_file():
                raise ValueError("--resume requires the original .audit/manifest.json and predictions.jsonl")
        else:
            self.directory.mkdir(parents=True, exist_ok=False)
        self.lock = FileLock(str(self.directory / "writer.lock"))
        self.lock.acquire(timeout=0)
        try:
            manifest_path = self.directory / "manifest.json"
            if self.resume:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest["identity"] != self.identity:
                    raise ValueError("Cannot resume: data, model, prompt, settings, parser, code, or environment changed")
                self._read_journal()
            else:
                write_json(manifest_path, {"schema_version": 1, "created_at": timestamp(),
                                          "identity_sha256": json_hash(self.identity), "identity": self.identity})
            self.journal = (self.directory / "predictions.jsonl").open("ab")
            return self
        except BaseException:
            self.lock.release()
            raise

    def _read_journal(self):
        path = self.directory / "predictions.jsonl"
        if not path.exists():
            raise ValueError(f"Missing evaluation journal: {path}")
        expected = {(s["sample_id"], task) for s in self.identity["samples"] for task in self.identity["tasks"]}
        with path.open("rb") as f:
            good_end = 0
            while line := f.readline():
                if not line.endswith(b"\n"):
                    # Only a torn final write can be repaired automatically.
                    # Preserve the fragment for inspection before truncating it.
                    fragment = self.directory / f"interrupted-write-{uuid.uuid4().hex}.bin"
                    fragment.write_bytes(line)
                    with path.open("r+b") as repair:
                        repair.truncate(good_end)
                    break
                record = json.loads(line)
                key = (record["sample_id"], record["task"])
                if key not in expected:
                    raise ValueError(f"Unexpected sample/task in evaluation journal: {key}")
                self.records[key] = record
                good_end = f.tell()

    def pending(self, sample_id, task, retry_errors=False):
        record = self.records.get((sample_id, task))
        return record is None or (retry_errors and record["status"] == "error")

    def append(self, record):
        key = (record["sample_id"], record["task"])
        record = {**record, "recorded_at": timestamp(),
                  "attempt": self.records.get(key, {}).get("attempt", 0) + 1}
        self.journal.write((json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n").encode())
        self.journal.flush()
        os.fsync(self.journal.fileno())
        self.records[key] = record

    def export_csv(self):
        tasks = self.identity["tasks"]
        fields = ["sample_id", "media_name", "url", "image_path", "image_sha256"]
        for task in tasks:
            fields += [f"{task}_{suffix}" for suffix in
                       ("true", "pred", "raw", "status", "error", "elapsed_seconds", "attempt")]
        tmp = self.out_csv.with_name(self.out_csv.name + ".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for sample in self.identity["samples"]:
                row = {k: sample[k] for k in fields[:5]}
                for task in tasks:
                    record = self.records.get((sample["sample_id"], task))
                    row[f"{task}_true"] = sample["labels"][task]
                    row[f"{task}_status"] = "pending" if record is None else record["status"]
                    if record is not None:
                        for suffix, key in (("pred", "predicted_label"), ("raw", "raw_text"),
                                            ("elapsed_seconds", "elapsed_seconds"), ("attempt", "attempt")):
                            row[f"{task}_{suffix}"] = record.get(key)
                        row[f"{task}_error"] = json.dumps(record["error"], ensure_ascii=False) if record["error"] else ""
                writer.writerow(row)
        tmp.replace(self.out_csv)

    def save_metrics(self, metrics, status):
        expected = len(self.identity["samples"]) * len(self.identity["tasks"])
        write_json(self.directory / "metrics.json", {
            "schema_version": 1, "updated_at": timestamp(), "status": status,
            "expected_records": expected, "completed_records": len(self.records), "tasks": metrics,
        })

    def __exit__(self, *exc):
        if self.journal is not None:
            self.journal.close()
        if self.lock is not None:
            self.lock.release()
