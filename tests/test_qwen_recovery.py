"""Exercise real failure envelopes and derived-run recovery without network calls."""

import contextlib
import csv
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import eval_api
from eval_api import EvaluationStopped, RateGate
from mediascope_eval import Sample
from recover_qwen_api import MODEL, SOURCE_FILES, main
from resilient_api import APIResponseError, ResilientAPIPredictor
from run_config import sha256_file


def response(status=200, label="LOW", finish="stop", error=None, top_error=None):
    result = Mock(status_code=status, headers={})
    payload = {"id": "fixture", "model": MODEL, "usage": {"prompt_tokens": 10, "completion_tokens": 2},
               "choices": [{"finish_reason": finish,
                            "message": {"content": label, "reasoning": "A different label HIGH was considered."}}]}
    if error:
        payload["choices"][0]["error"] = error
    if top_error:
        payload["error"] = top_error
    result.json.return_value = payload
    return result


class ResilientAPITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.keys = patch.dict(os.environ, {"OPENROUTER_API_KEY": "SECRET_FIXTURE_KEY"})
        self.keys.start()
        self.addCleanup(self.keys.stop)
        self.sample = Sample("fixture", "https://example.org", "fixture.png", "", "LOW", "LEFT")

    def predictor(self, retries=3, recover=False):
        gate = RateGate(0)
        images = {"fixture.png": {"image": {"truncated_px": 0}, "data_urls": [], "png_sha256": []}}
        return ResilientAPIPredictor("openrouter", MODEL, images, {}, gate=gate,
                                     request_log=self.root / "requests.jsonl", session_id="fixture",
                                     retries=retries, backoff_base=0, recover=recover)

    def events(self):
        return [json.loads(line) for line in (self.root / "requests.jsonl").read_text().splitlines()]

    def test_image_timeout_400_retries_and_does_not_disable_following_requests(self):
        predictor = self.predictor()
        bad = response(400, top_error={"code": 400, "message": "Provider returned error", "metadata": {
            "raw": 'data: {"error":{"message":"Download multimodal file timed out"}}'}})
        with patch("api_predictor.requests.post", side_effect=[bad, response(), response()]) as post:
            self.assertEqual(predictor.predict(self.sample, "factuality"), "LOW")
            self.assertEqual(predictor.predict(self.sample, "factuality"), "LOW")
            self.assertEqual(post.call_count, 3)
        self.assertIsNone(predictor.fatal_error)
        self.assertEqual(self.events()[0]["error"]["kind"], "image_download_timeout")
        self.assertEqual(self.events()[0]["details"]["http_status"], 400)

    def test_http_200_embedded_transient_error_is_not_scored_as_partial_label(self):
        predictor = self.predictor()
        bad = response(label="HIGH", finish="error", error={"code": 502, "message": "Upstream temporarily unavailable"})
        with patch("api_predictor.requests.post", side_effect=[bad, response()]) as post:
            self.assertEqual(predictor.predict(self.sample, "factuality"), "LOW")
            self.assertEqual(post.call_count, 2)
        self.assertEqual(self.events()[0]["raw_text"], "HIGH")
        self.assertEqual(self.events()[0]["error"]["kind"], "transient_upstream")

    def test_provider_filter_disguised_as_502_is_terminal_for_that_sample_only(self):
        predictor = self.predictor()
        blocked = response(label=None, finish="error", error={
            "code": 502, "message": "Upstream error from Alibaba: Output data may contain inappropriate content.",
            "metadata": {"error_type": "provider_unavailable"}})
        with patch("api_predictor.requests.post", side_effect=[blocked, response()]) as post:
            with self.assertRaises(APIResponseError) as error:
                predictor.predict(self.sample, "factuality")
            self.assertEqual(error.exception.failure.kind, "provider_content_filter")
            self.assertEqual(post.call_count, 1)
            self.assertFalse(predictor.gate.stop.is_set())
            self.assertEqual(predictor.predict(self.sample, "factuality"), "LOW")
            self.assertEqual(post.call_count, 2)

    def test_network_retries_are_bounded_and_budget_exhaustion_is_not_retried(self):
        predictor = self.predictor()
        with patch("api_predictor.requests.post", side_effect=requests.exceptions.ReadTimeout()) as post:
            with self.assertRaises(APIResponseError):
                predictor.predict(self.sample, "factuality")
            self.assertEqual(post.call_count, 4)
        self.assertEqual([e["attempt_within_call"] for e in self.events()], [1, 2, 3, 4])
        with patch("api_predictor.requests.post", return_value=response(label=None, finish="length")) as post:
            with self.assertRaises(APIResponseError) as error:
                predictor.predict(self.sample, "factuality")
            self.assertEqual(error.exception.failure.kind, "output_limit")
            self.assertEqual(post.call_count, 1)
            self.assertEqual(post.call_args.kwargs["json"]["max_tokens"], 8192)

    def test_authentication_and_quota_stop_without_poisoning_the_remaining_audit(self):
        for status, body in ((401, {"message": "bad credentials"}),
                             (429, {"code": "insufficient_quota", "message": "quota exhausted"})):
            predictor = self.predictor()
            with patch("api_predictor.requests.post", return_value=response(status, top_error=body)) as post:
                with self.assertRaises(APIResponseError):
                    predictor.predict(self.sample, "factuality")
                with self.assertRaises(EvaluationStopped):
                    predictor.predict(self.sample, "bias")
                self.assertEqual(post.call_count, 1)
            self.assertTrue(predictor.gate.stop.is_set())

    def test_committed_transport_success_is_replayed_without_another_charge(self):
        predictor = self.predictor()
        with patch("api_predictor.requests.post", return_value=response()):
            predictor.predict(self.sample, "factuality")
        recovered = self.predictor(recover=True)
        with patch("api_predictor.requests.post") as post:
            self.assertEqual(recovered.predict(self.sample, "factuality"), "LOW")
            post.assert_not_called()
        self.assertTrue(recovered.last_details["replayed_from_request_log"])
        self.assertEqual(len(self.events()), 1)

    def test_final_rate_limit_still_defers_the_next_samples_request(self):
        predictor = self.predictor(retries=0)
        predictor.gate.defer = Mock()
        limited = response(429, top_error={"code": 429, "message": "Rate limit"})
        limited.headers = {"retry-after": "65"}
        with patch("api_predictor.requests.post", return_value=limited) as post:
            with self.assertRaises(APIResponseError):
                predictor.predict(self.sample, "factuality")
            self.assertEqual(post.call_count, 1)
        predictor.gate.defer.assert_called_once_with(65)
        self.assertIsNone(self.events()[0]["retry_delay_seconds"])
        self.assertEqual(self.events()[0]["provider_cooldown_seconds"], 65)

    def test_request_error_does_not_poison_model_and_error_text_is_redacted(self):
        predictor = self.predictor()
        bad = response(400, top_error={"code": 400, "message": "Invalid image: SECRET_FIXTURE_KEY"})
        with patch("api_predictor.requests.post", side_effect=[bad, response()]) as post:
            with self.assertRaises(APIResponseError):
                predictor.predict(self.sample, "factuality")
            self.assertEqual(predictor.predict(self.sample, "factuality"), "LOW")
            self.assertEqual(post.call_count, 2)
        self.assertNotIn("SECRET_FIXTURE_KEY", (self.root / "requests.jsonl").read_text())

    def test_reasoning_words_do_not_trigger_filters_and_model_mismatch_stops(self):
        predictor = self.predictor()
        healthy = response()
        healthy.json.return_value["choices"][0]["message"]["reasoning"] = "The article discusses inappropriate content."
        wrong_model = response()
        wrong_model.json.return_value["model"] = "different-model"
        with patch("api_predictor.requests.post", side_effect=[healthy, wrong_model]) as post:
            self.assertEqual(predictor.predict(self.sample, "factuality"), "LOW")
            with self.assertRaises(APIResponseError) as error:
                predictor.predict(self.sample, "factuality")
            self.assertEqual(error.exception.failure.kind, "model_mismatch")
            self.assertTrue(predictor.gate.stop.is_set())
            self.assertEqual(post.call_count, 2)


class DerivedRecoveryTests(unittest.TestCase):
    def test_recovery_retains_wrong_successes_preserves_source_and_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            (dataset / "test").mkdir(parents=True)
            with (dataset / "test.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["media_name", "url", "true_label_factuality", "true_label_bias"])
                for i in range(5):
                    Image.new("RGB", (64, 128), "white").save(dataset / f"test/Outlet_{i}.png")
                    writer.writerow([f"Outlet_{i}", f"https://outlet-{i}.example", "LOW", "LEFT"])
            source, recovered = root / "original", root / "recovered"
            original_args = ["eval_api.py", "--dataset-root", str(dataset), "--models", "qwen",
                             "--output-dir", str(source), "--min-interval", "0", "--rate-limit-retries", "0"]
            originals = [
                response(label="HIGH"), response(label="LEFT"),  # First is a valid but WRONG answer.
                response(label=None, finish="error", error={"code": 502, "message": "Output data may contain inappropriate content."}),
                response(label=None, finish="length"),
                response(label=None, finish="error", error={"code": 502, "message": "Temporary upstream error"}),
                requests.exceptions.ReadTimeout(),
                response(400, top_error={"code": 400, "message": "Download multimodal file timed out"}),
            ]
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "fixture-key"}), contextlib.redirect_stdout(io.StringIO()):
                with patch.object(sys, "argv", original_args), patch("api_predictor.requests.post", side_effect=originals) as post:
                    self.assertEqual(eval_api.main(), 1)
                    self.assertEqual(post.call_count, 7)  # Old fatal bug skips the last three requests.
                hashes = {name: sha256_file(source / name) for name in SOURCE_FILES}
                argv = ["recover_qwen_api.py", "--source-dir", str(source), "--output-dir", str(recovered),
                        "--min-interval", "0", "--backoff-base", "0"]

                def success(url, **kwargs):
                    prompt = kwargs["json"]["messages"][0]["content"][0]["text"]
                    return response(label="LOW" if "FACTUAL REPORTING" in prompt else "LEFT")

                with patch.object(sys, "argv", argv), patch("api_predictor.requests.post", return_value=response(401)) as post:
                    self.assertEqual(main(), 1)
                    self.assertEqual(post.call_count, 1)
                blocked = json.loads((recovered / "summary.json").read_text())["models"]["qwen"]
                self.assertEqual(blocked["status"], "blocked")
                self.assertEqual(blocked["processed_this_session"], 1)
                self.assertEqual(blocked["remaining_initial_recovery"], 5)
                self.assertEqual(len((recovered / "qwen.audit/predictions.jsonl").read_text().splitlines()), 11)
                with patch.object(sys, "argv", argv + ["--resume", "--retry-errors"]), patch("api_predictor.requests.post", side_effect=success) as post:
                    self.assertEqual(main(), 1)  # Genuine filter and budget outcomes remain visible.
                    self.assertEqual(post.call_count, 6)
                self.assertEqual({name: sha256_file(source / name) for name in SOURCE_FILES}, hashes)
                self.assertEqual({name: sha256_file(recovered / "source_snapshot" / name) for name in SOURCE_FILES}, hashes)
                records = [json.loads(line) for line in (recovered / "qwen.audit/predictions.jsonl").read_text().splitlines()]
                self.assertEqual(len(records), 17)  # Ten imports, one auth failure, six new predictions.
                self.assertEqual(records[0]["predicted_label"], "HIGH")
                self.assertFalse(records[0]["correct"])
                self.assertEqual(records[2]["recovery_provenance"]["source_status"], "unparseable")
                self.assertEqual(records[2]["error"]["kind"], "provider_content_filter")
                result = json.loads((recovered / "summary.json").read_text())["models"]["qwen"]
                self.assertEqual(result["outcome_counts"], {"ok": 8, "provider_content_filter": 1, "output_limit": 1})
                self.assertEqual(result["metrics"]["factuality"]["accuracy"], .6)
                self.assertEqual(result["metrics"]["bias"]["accuracy"], .8)
                self.assertEqual(result["remaining_initial_recovery"], 0)
                with patch.object(sys, "argv", argv + ["--resume"]), patch("api_predictor.requests.post") as post:
                    self.assertEqual(main(), 1)
                    post.assert_not_called()
                self.assertEqual(len((recovered / "qwen.audit/predictions.jsonl").read_text().splitlines()), 17)
                with patch.object(sys, "argv", argv + ["--resume", "--retries", "1"]), patch("api_predictor.requests.post") as post:
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        main()
                    post.assert_not_called()
                Image.new("RGB", (64, 128), "black").save(dataset / "test/Outlet_0.png")
                with patch.object(sys, "argv", argv + ["--resume"]), patch("api_predictor.requests.post") as post:
                    with self.assertRaisesRegex(ValueError, "screenshot changed"):
                        main()
                    post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
