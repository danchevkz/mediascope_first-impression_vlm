"""Native provider request/response checks without network calls."""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from api_predictor import PROVIDERS, ReasoningAPIPredictor
from mediascope_eval import Sample


class APISmokeTests(unittest.TestCase):
    def setUp(self):
        self.sample = Sample("OUTLET_SENTINEL", "https://sentinel.invalid", "fixture.png", "", "LOW", "RIGHT")
        self.images = {"fixture.png": {"image": {"tile_count": 1, "truncated_px": 0},
                                       "data_urls": ["data:image/png;base64,aW1hZ2U="], "png_sha256": ["fixture"]}}
        self.keys = {spec[0]: "fixture-secret" for spec in PROVIDERS.values()}
        self.env = patch.dict(os.environ, self.keys)
        self.env.start()
        self.addCleanup(self.env.stop)

    def predictor(self, provider):
        return ReasoningAPIPredictor(provider, "fixture-model", self.images,
                                      {"tile_height": 1400, "max_tiles": 6, "max_width": 1024})

    def test_native_request_controls_and_final_answer_extraction(self):
        payloads = {
            "openrouter": {"choices": [{"message": {"content": "LOW", "reasoning": "HIGH"}, "finish_reason": "stop"}]},
            "anthropic": {"content": [{"type": "thinking", "thinking": "HIGH"}, {"type": "text", "text": "LOW"}], "stop_reason": "end_turn"},
            "openai": {"status": "completed", "output": [
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "HIGH"}]},
                {"type": "message", "role": "assistant", "phase": "commentary", "content": [{"type": "output_text", "text": "HIGH"}]},
                {"type": "message", "role": "assistant", "phase": "final_answer", "content": [{"type": "output_text", "text": "LOW"}]},
            ]},
        }
        for provider, payload in payloads.items():
            with self.subTest(provider=provider):
                predictor = self.predictor(provider)
                payload = {**payload, "id": "request-123", "model": "fixture-model", "usage": {"output_tokens": 10}}
                response = Mock(status_code=200, headers={"request-id": "header-request-id"})
                response.json.return_value = payload
                with patch("api_predictor.requests.post", return_value=response) as post:
                    self.assertEqual(predictor.predict(self.sample, "factuality"), "LOW")
                args = post.call_args.kwargs
                body = args["json"]
                self.assertNotIn("OUTLET_SENTINEL", str(body))
                self.assertNotIn("sentinel.invalid", str(body))
                self.assertNotIn("temperature", body)
                self.assertEqual(predictor.last_details["response"], payload)
                self.assertEqual(predictor.last_details["usage"], {"output_tokens": 10})
                self.assertNotIn("fixture-secret", str(predictor.run_metadata()))
                if provider == "openai":
                    self.assertEqual(body["reasoning"]["effort"], "medium")
                    self.assertEqual(body["max_output_tokens"], 8192)
                    self.assertFalse(body["store"])
                    self.assertEqual(body["input"][0]["content"][1]["type"], "input_image")
                elif provider == "anthropic":
                    self.assertEqual(body["thinking"], {"type": "adaptive"})
                    self.assertEqual(body["output_config"], {"effort": "medium"})
                    self.assertEqual(args["headers"]["anthropic-version"], "2023-06-01")
                    self.assertEqual(body["messages"][0]["content"][1]["source"]["media_type"], "image/png")
                else:
                    self.assertTrue(body["reasoning"]["enabled"])
                    self.assertEqual(body["reasoning"]["effort"], "medium")

    def test_incomplete_reasoning_is_not_used_as_final_answer(self):
        predictor = self.predictor("openai")
        payload = {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"},
                   "output": [{"type": "reasoning", "summary": [{"text": "LOW"}]}]}
        self.assertIsNone(predictor.final_text(payload))
        self.assertTrue(predictor.last_details["reached_token_budget"])

    def test_auth_failure_is_not_repeated_or_leaked(self):
        predictor = self.predictor("openrouter")
        response = Mock(status_code=401, headers={}, text="fixture-secret")
        with patch("api_predictor.requests.post", return_value=response) as post:
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, "HTTP 401") as caught:
                    predictor.predict(self.sample, "factuality")
                self.assertNotIn("fixture-secret", str(caught.exception))
                self.assertNotIn("fixture-secret", str(predictor.last_details))
        self.assertEqual(post.call_count, 1)

    def test_network_exception_does_not_echo_credentials(self):
        predictor = self.predictor("openai")
        with patch("api_predictor.requests.post", side_effect=requests.exceptions.InvalidHeader("fixture-secret")):
            with self.assertRaises(RuntimeError) as caught:
                predictor.predict(self.sample, "factuality")
        self.assertNotIn("fixture-secret", str(caught.exception))

    def test_truncated_image_is_rejected_before_request(self):
        predictor = self.predictor("anthropic")
        self.images["fixture.png"]["image"]["truncated_px"] = 1
        with patch("api_predictor.requests.post") as post:
            with self.assertRaisesRegex(ValueError, "truncated"):
                predictor.predict(self.sample, "factuality")
            post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
