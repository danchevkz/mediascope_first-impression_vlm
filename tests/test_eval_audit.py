"""Behavioral tests for the durable evaluation contract; no network or GPU."""

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from mediascope_eval import Sample, normalize_to_label, run
from qwen_predictor import OpenRouterQwenPredictor, tile_image_with_info
from run_config import load_adapter_run_config, resolve_image_settings


class FakePredictor:
    def __init__(self, answers, settings=None):
        self.answers = answers
        self.calls = []
        self.settings = settings or {"thinking": "off"}

    def run_metadata(self):
        return {"backend": "test", **self.settings}

    def predict(self, sample, task):
        self.calls.append((sample.media_name, task))
        self.last_details = {"image": {"truncated_px": 12}, "generated_token_ids": [1, 2]}
        answer = self.answers[sample.media_name]
        if isinstance(answer, BaseException):
            raise answer
        return answer


class EvaluationAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.out = self.root / "preds.csv"
        self.audit = self.root / "preds.audit"
        self.samples = []
        for name, fact, bias in (("one", "LOW", "LEFT"), ("two", "MIXED", None), ("three", "HIGH", "RIGHT")):
            path = self.root / f"{name}.png"
            Image.new("RGB", (100, 200), "white").save(path)
            self.samples.append(Sample(name, f"https://{name}.example", str(path), "", fact, bias))

    def evaluate(self, predictor, **kwargs):
        return run(predictor, self.samples, out_csv=str(self.out), tasks=("factuality",), **kwargs)

    def journal(self):
        return [json.loads(line) for line in (self.audit / "predictions.jsonl").read_text().splitlines()]

    def test_raw_errors_and_invalid_answers_remain_inspectable(self):
        predictor = FakePredictor({"one": " low \n", "two": "not LOW", "three": RuntimeError("fixture failure")})
        result = self.evaluate(predictor)["factuality"]
        records = self.journal()
        self.assertEqual([r["status"] for r in records], ["ok", "unparseable", "error"])
        self.assertEqual(records[0]["raw_text"], " low \n")
        self.assertEqual(records[1]["raw_text"], "not LOW")
        self.assertIn("RuntimeError", records[2]["error"]["traceback"])
        self.assertEqual(records[0]["details"]["generated_token_ids"], [1, 2])
        self.assertAlmostEqual(result["accuracy"], 1 / 3)
        self.assertEqual(result["confusion_matrix"]["counts"], [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1]])
        self.assertEqual(result["status_counts"], {"ok": 1, "unparseable": 1, "error": 1})
        with self.out.open(newline="") as f:
            exported = list(csv.DictReader(f))
        self.assertEqual(exported[0]["factuality_raw"], " low \n")
        self.assertEqual(json.loads((self.audit / "metrics.json").read_text())["status"], "complete_with_failures")

    def test_interruption_torn_write_and_resume_without_duplicate_predictions(self):
        with self.assertRaises(KeyboardInterrupt):
            self.evaluate(FakePredictor({"one": "LOW", "two": KeyboardInterrupt(), "three": "HIGH"}))
        self.assertEqual(len(self.journal()), 1)
        self.assertEqual(json.loads((self.audit / "metrics.json").read_text())["status"], "interrupted")
        with (self.audit / "predictions.jsonl").open("ab") as f:
            f.write(b'{"unfinished":')
        resumed = FakePredictor({"one": AssertionError("must not repeat"), "two": "MIXED", "three": "HIGH"})
        result = self.evaluate(resumed, resume=True)
        self.assertEqual(resumed.calls, [("two", "factuality"), ("three", "factuality")])
        self.assertEqual(len(self.journal()), 3)
        self.assertEqual(result["factuality"]["accuracy"], 1.0)
        self.assertEqual(len(list(self.audit.glob("interrupted-write-*.bin"))), 1)

    def test_retry_preserves_error_history_and_counts_latest_attempt_once(self):
        self.evaluate(FakePredictor({"one": "LOW", "two": RuntimeError("temporary"), "three": "HIGH"}))
        retry = FakePredictor({"two": "MIXED"})
        result = self.evaluate(retry, resume=True, retry_errors=True)
        self.assertEqual(retry.calls, [("two", "factuality")])
        records = self.journal()
        self.assertEqual(len(records), 4)
        self.assertEqual(records[1]["status"], "error")
        self.assertEqual(records[-1]["attempt"], 2)
        self.assertEqual(result["factuality"]["n"], 3)
        self.assertEqual(result["factuality"]["accuracy"], 1)

    def test_overwrite_and_incompatible_resume_are_rejected(self):
        answers = {"one": "LOW", "two": "MIXED", "three": "HIGH"}
        self.evaluate(FakePredictor(answers))
        original = self.out.read_bytes()
        with self.assertRaises(FileExistsError):
            self.evaluate(FakePredictor(answers))
        with self.assertRaisesRegex(ValueError, "Cannot resume"):
            self.evaluate(FakePredictor(answers, {"thinking": "template-default"}), resume=True)
        Image.new("RGB", (100, 200), "black").save(self.samples[0].image_path)
        with self.assertRaisesRegex(ValueError, "Cannot resume"):
            self.evaluate(FakePredictor(answers), resume=True)
        self.assertEqual(self.out.read_bytes(), original)

    def test_unrated_bias_does_not_reduce_factuality_coverage(self):
        result = run(FakePredictor({"one": "LOW", "two": "MIXED", "three": "HIGH"}),
                     self.samples, out_csv=str(self.out))
        self.assertEqual(result["factuality"]["n"], 3)
        self.assertEqual(result["bias"]["n"], 2)
        self.assertEqual(result["bias"]["status_counts"]["not_applicable"], 1)

    def test_report_only_rebuilds_outputs_without_model(self):
        self.evaluate(FakePredictor({"one": "LOW", "two": "MIXED", "three": "HIGH"}))
        self.out.unlink()
        completed = subprocess.run([sys.executable, str(ROOT / "scripts/mediascope_eval.py"),
                                    "--report-only", "--out", str(self.out)], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(self.out.is_file())
        self.assertEqual(json.loads((self.audit / "metrics.json").read_text())["tasks"]["factuality"]["accuracy"], 1)

    def test_api_audit_preserves_reasoning_but_does_not_parse_it_as_answer(self):
        predictor = OpenRouterQwenPredictor(model="qwen-fixture", api_key="SECRET_NOT_FOR_AUDIT",
                                          base_url="http://localhost:9999/v1", max_tiles=1, max_width=64)
        response = Mock(status_code=200)
        response.json.return_value = {"id": "fixture-id", "model": "fixture-model", "usage": {"total_tokens": 10},
                                      "choices": [{"message": {"content": None, "reasoning": "HIGH"}, "finish_reason": "length"}]}
        with patch("qwen_predictor.requests.post", return_value=response) as request:
            run(predictor, self.samples[:1], out_csv=str(self.out), tasks=("factuality",))
        record = self.journal()[0]
        self.assertEqual(record["status"], "unparseable")
        self.assertEqual(record["details"]["choice"]["message"]["reasoning"], "HIGH")
        self.assertEqual(record["details"]["image"]["submitted_image_sizes"], [[64, 128]])
        self.assertEqual(request.call_args.kwargs["json"]["chat_template_kwargs"], {"enable_thinking": False})
        self.assertNotIn("SECRET_NOT_FOR_AUDIT", (self.audit / "manifest.json").read_text())


class InputContractTests(unittest.TestCase):
    def test_strict_parser_rejects_negation_and_multiple_labels(self):
        labels = ["LOW", "MIXED", "HIGH"]
        self.assertEqual(normalize_to_label(" high\n", labels), "HIGH")
        for raw in ("not LOW", "LOW or HIGH", "LOW.", None, "", {"label": "LOW"}):
            self.assertIsNone(normalize_to_label(raw, labels))
        self.assertEqual(normalize_to_label("not LOW", labels, mode="legacy"), "LOW")

    def test_sizing_metadata_and_inheritance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "image.png"
            Image.new("RGB", (512, 1000)).save(path)
            tiles, info = tile_image_with_info(path, tile_height=100, max_tiles=2, max_width=128)
            self.assertEqual(info["original_size"], [512, 1000])
            self.assertEqual(info["resized_size"], [128, 250])
            self.assertEqual(info["truncated_px"], 50)
            self.assertEqual(info["retained_height_fraction"], 0.8)
            self.assertEqual([t.size for t in tiles], [(128, 100), (128, 100)])
            checkpoint = root / "checkpoint-25"
            checkpoint.mkdir()
            config = {"settings": {"preprocessing": {"max_tiles": 2, "max_width": 768, "tile_height": 1000}}}
            (root / "run_config.json").write_text(json.dumps(config))
            saved = load_adapter_run_config(checkpoint)
            self.assertEqual(resolve_image_settings({}, saved["settings"]["preprocessing"]), config["settings"]["preprocessing"])
            self.assertEqual(resolve_image_settings({"max_width": 512}, saved["settings"]["preprocessing"])["max_width"], 512)
            with self.assertRaises(ValueError):
                resolve_image_settings({"max_tiles": 0})


if __name__ == "__main__":
    unittest.main()
