"""Full-run orchestration and transport recovery tests; requests are mocked."""

import contextlib
import csv
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from api_predictor import PROVIDERS
from eval_api import BenchmarkPredictor, EvaluationStopped, RateGate, main, retry_delay, usage_summary
from mediascope_eval import Sample


class FullAPITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.keys = patch.dict(os.environ, {value[0]: "fixture-key" for value in PROVIDERS.values()})
        self.keys.start()
        self.addCleanup(self.keys.stop)

    @staticmethod
    def response(status, payload, headers=None):
        response = Mock(status_code=status, headers=headers or {})
        response.json.return_value = payload
        return response

    def predictor(self, recover=False, retries=3):
        gate = RateGate(0)
        gate.wait = Mock()
        gate.defer = Mock()
        images = {"fixture.png": {"image": {"truncated_px": 0}, "data_urls": [], "png_sha256": []}}
        return BenchmarkPredictor("openrouter", "fixture-model", images, {}, gate=gate,
                                  request_log=self.root / "requests.jsonl", session_id="test",
                                  retries=retries, recover=recover)

    def test_rate_limit_attempts_saved_and_success_recovered_without_new_charge(self):
        sample = Sample("fixture", "https://fixture.example", "fixture.png", "", "LOW", "LEFT")
        predictor = self.predictor()
        limited = self.response(429, {"error": {"message": "rate limit"}}, {"retry-after": "65"})
        success = self.response(200, {"choices": [{"message": {"content": "LOW"}, "finish_reason": "stop"}],
                                     "usage": {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.001}})
        with patch("api_predictor.requests.post", side_effect=[limited, success]) as post:
            self.assertEqual(predictor.predict(sample, "factuality"), "LOW")
            self.assertEqual(post.call_count, 2)
        predictor.gate.defer.assert_called_once_with(65.0)
        events = [json.loads(line) for line in predictor.request_log.read_text().splitlines()]
        self.assertEqual([e["details"]["http_status"] for e in events], [429, 200])
        recovered = self.predictor(recover=True)
        with patch("api_predictor.requests.post") as post:
            self.assertEqual(recovered.predict(sample, "factuality"), "LOW")
            post.assert_not_called()
        self.assertTrue(recovered.last_details["replayed_from_request_log"])
        summary = usage_summary(predictor.request_log)
        self.assertEqual(summary["http_attempts"], 2)
        self.assertEqual(summary["provider_reported_cost_usd"], 0.001)

    def test_quota_exhaustion_is_not_retried(self):
        sample = Sample("fixture", "", "fixture.png", "", "LOW", "LEFT")
        predictor = self.predictor()
        error = self.response(429, {"error": {"code": "insufficient_quota", "message": "quota"}})
        with patch("api_predictor.requests.post", return_value=error) as post:
            with self.assertRaises(RuntimeError):
                predictor.predict(sample, "factuality")
            self.assertEqual(post.call_count, 1)
        predictor.gate.defer.assert_not_called()

    def test_stop_interrupts_wait_without_model_error_or_request(self):
        stop = threading.Event()
        stop.set()
        gate = RateGate(5, stop)
        with self.assertRaises(EvaluationStopped):
            gate.wait()
        self.assertEqual(retry_delay(None, 0), 30)
        self.assertEqual(retry_delay("120", 1), 120)
        self.assertEqual(retry_delay("nan", 0), 30)

    def test_full_both_task_run_and_resume(self):
        dataset = self.root / "dataset"
        (dataset / "test").mkdir(parents=True)
        fields = ["media_name", "url", "country", "true_label_factuality", "true_label_bias"]
        with (dataset / "test.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for index in range(3):
                name = f"Outlet_{index}"
                Image.new("RGB", (64, 128), "white").save(dataset / "test" / f"{name}.png")
                writer.writerow(dict(zip(fields, [name, f"https://{name}.example", "", "LOW", "LEFT"])))
        output = self.root / "evaluation"
        argv = ["eval_api.py", "--dataset-root", str(dataset), "--output-dir", str(output),
                "--task", "both", "--workers", "4", "--min-interval", "0"]

        def reply(url, **kwargs):
            body = kwargs["json"]
            prompt = str(body.get("messages", body.get("input")))
            label = "LOW" if "FACTUAL REPORTING" in prompt else "LEFT"
            if url.endswith("/responses"):
                payload = {"status": "completed", "output": [{"type": "message", "role": "assistant",
                            "content": [{"type": "output_text", "text": label}]}]}
            elif url.endswith("/messages"):
                payload = {"stop_reason": "end_turn", "content": [{"type": "text", "text": label}]}
            else:
                payload = {"choices": [{"message": {"content": label}, "finish_reason": "stop"}]}
            return self.response(200, {**payload, "id": "fixture", "model": body["model"],
                                       "usage": {"input_tokens": 10, "output_tokens": 1}})

        with patch.object(sys, "argv", argv), patch("api_predictor.requests.post", side_effect=reply) as post:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 0)
            self.assertEqual(post.call_count, 24)  # 3 rows x 2 tasks x 4 models.
            self.assertEqual({call.kwargs["json"]["model"] for call in post.call_args_list},
                             {"moonshotai/kimi-k3", "qwen/qwen3.8-max-prime", "claude-opus-5", "gpt-5.6-sol"})
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(set(summary["models"]), {"kimi", "qwen", "opus5", "gpt56"})
        for model in summary["models"].values():
            self.assertEqual(model["requests"]["http_attempts"], 6)
            for task in model["metrics"].values():
                self.assertEqual(task["n"], 3)
                self.assertEqual(task["accuracy"], 1)
        with patch.object(sys, "argv", argv + ["--resume"]), patch("api_predictor.requests.post") as post:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 0)
            post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
