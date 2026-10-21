import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from api_predictor import ReasoningAPIPredictor
from eval_paired_unmasked_api import MODELS, last_request, record_from_answer
from qwen_predictor import PROMPTS


class PairedAPITests(unittest.TestCase):
    def test_uses_the_four_requested_versions_and_original_label_only_controls(self):
        self.assertEqual(MODELS["gpt56"][1], "gpt-5.6-sol")
        self.assertEqual(MODELS["opus5"][1], "claude-opus-5")
        self.assertEqual(MODELS["qwen"][1], "qwen/qwen3.8-max-prime")
        keys = {"OPENAI_API_KEY": "test", "ANTHROPIC_API_KEY": "test", "OPENROUTER_API_KEY": "test"}
        with patch.dict("os.environ", keys):
            for provider, model in MODELS.values():
                predictor = ReasoningAPIPredictor(provider, model, {}, {}, 8192, "medium", 300)
                body = predictor.request_body(PROMPTS["bias"], ["data:image/png;base64,test"])
                self.assertEqual(body["model"], model)
                self.assertIn("Output ONLY the label text", PROMPTS["bias"])
                self.assertNotIn("source-recognition", PROMPTS["bias"])
                if provider == "anthropic":
                    self.assertEqual(body["output_config"]["effort"], "medium")
                else:
                    self.assertEqual(body["reasoning"]["effort"], "medium")

    def test_strict_label_parsing_and_failures_are_scored_not_substituted(self):
        info = {"sample_id": "s", "media_name": "site_001", "url": "", "image_path": "image.png",
                "image_sha256": "hash", "labels": {"factuality": "HIGH"}}
        valid = record_from_answer(info, "factuality", "MIXED", {})
        self.assertEqual(valid["status"], "ok")
        self.assertFalse(valid["correct"])
        self.assertEqual(record_from_answer(info, "factuality", "It is HIGH", {})["status"], "unparseable")
        failed = record_from_answer(info, "factuality", "HIGH", {"native": "HIGH"}, {"kind": "provider_content_filter"})
        self.assertEqual(failed["status"], "error")
        self.assertIsNone(failed["predicted_label"])
        self.assertFalse(failed["correct"])

    def test_logged_wrong_labels_and_terminal_failures_are_reused_without_resampling(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "requests.jsonl"
            sample = SimpleNamespace(image_path="image.png")
            for error in (None, {"kind": "provider_content_filter"}):
                event = {"image_path": "image.png", "task": "bias", "model": "qwen/qwen3.8-max-prime",
                         "raw_text": "LEFT", "details": {}, "error": error, "retry_delay_seconds": None}
                path.write_text(json.dumps(event) + "\n" + "torn fragment")
                self.assertEqual(last_request(path, sample, "bias", event["model"]), event)
                with self.assertRaises(ValueError):
                    last_request(path, sample, "factuality", event["model"])
            event["retry_delay_seconds"] = 5
            path.write_text(json.dumps(event) + "\n")
            self.assertIsNone(last_request(path, sample, "bias", event["model"]))


if __name__ == "__main__":
    unittest.main()
