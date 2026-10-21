"""Reasoning-enabled screenshot inference through three native HTTP APIs."""

from __future__ import annotations

import os

import requests

from qwen_predictor import PROMPTS
from run_config import sha256_file


PROVIDERS = {
    "openrouter": ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1/chat/completions"),
    "anthropic": ("ANTHROPIC_API_KEY", "https://api.anthropic.com/v1/messages"),
    "openai": ("OPENAI_API_KEY", "https://api.openai.com/v1/responses"),
}


class ReasoningAPIPredictor:
    def __init__(self, provider, model, prepared_images, preprocessing,
                 max_output_tokens=8192, effort="medium", timeout=300, prompts=None):
        self.provider, self.model = provider, model
        self.key_variable, self.endpoint = PROVIDERS[provider]
        self.api_key = os.environ.get(self.key_variable, "").strip()
        if not self.api_key or "\n" in self.api_key or "\r" in self.api_key:
            raise ValueError(f"Export a valid {self.key_variable} in this terminal")
        self.images, self.preprocessing = prepared_images, preprocessing
        self.max_output_tokens, self.effort, self.timeout = max_output_tokens, effort, timeout
        self.prompts = dict(PROMPTS if prompts is None else prompts)
        self.last_details = {}
        self.fatal_error = None

    def request_body(self, prompt, urls):
        if self.provider == "openai":
            return {
                "model": self.model, "store": False,
                "reasoning": {"effort": self.effort, "summary": "auto"},
                "max_output_tokens": self.max_output_tokens,
                "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]
                           + [{"type": "input_image", "image_url": url, "detail": "high"} for url in urls]}],
            }
        if self.provider == "anthropic":
            return {
                "model": self.model, "max_tokens": self.max_output_tokens,
                "thinking": {"type": "adaptive"}, "output_config": {"effort": self.effort},
                "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]
                              + [{"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                               "data": url.split(",", 1)[1]}} for url in urls]}],
            }
        return {
            "model": self.model, "reasoning": {"enabled": True, "effort": self.effort},
            "max_tokens": self.max_output_tokens, "provider": {"require_parameters": True},
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]
                          + [{"type": "image_url", "image_url": {"url": url, "detail": "high"}} for url in urls]}],
        }

    def run_metadata(self):
        body = self.request_body("", [])
        controls = {k: v for k, v in body.items() if k not in ("input", "messages")}
        return {
            "backend": self.provider, "model_id": self.model, "endpoint": self.endpoint,
            "prompts": self.prompts, "preprocessing": self.preprocessing, "full_coverage_required": True,
            "request_controls": controls, "timeout_seconds": self.timeout,
            "image_detail": "provider-native" if self.provider == "anthropic" else "high",
            "implementation_sha256": sha256_file(__file__), "automatic_retries": 0,
        }

    def redact(self, value):
        if isinstance(value, str):
            return value.replace(self.api_key, "[REDACTED]")
        if isinstance(value, dict):
            return {k: self.redact(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.redact(v) for v in value]
        return value

    def final_text(self, response):
        if self.provider == "openrouter":
            choice = response["choices"][0]
            self.last_details["finish_reason"] = choice.get("finish_reason")
            self.last_details["reached_token_budget"] = choice.get("finish_reason") == "length"
            content = choice["message"].get("content")
            if isinstance(content, list):
                return "\n".join(block["text"] for block in content if block.get("type") == "text") or None
            return content  # Never substitute the reasoning field for the final answer.
        if self.provider == "anthropic":
            self.last_details["finish_reason"] = response.get("stop_reason")
            self.last_details["reached_token_budget"] = response.get("stop_reason") == "max_tokens"
            return "\n".join(block["text"] for block in response.get("content", [])
                             if block.get("type") == "text") or None
        self.last_details["finish_reason"] = response.get("status")
        self.last_details["incomplete_details"] = response.get("incomplete_details")
        self.last_details["reached_token_budget"] = (response.get("incomplete_details") or {}).get("reason") == "max_output_tokens"
        messages = [item for item in response.get("output", [])
                    if item.get("type") == "message" and item.get("role") == "assistant"
                    and item.get("phase") != "commentary"]
        if not messages:
            return None
        return "\n".join(block["text"] for block in messages[-1].get("content", [])
                         if block.get("type") == "output_text") or None

    def predict(self, sample, task_key):
        self.last_details = {}
        if self.fatal_error:
            self.last_details["request_sent"] = False
            raise RuntimeError(self.fatal_error)
        prepared = self.images[sample.image_path]
        if prepared["image"]["truncated_px"]:
            raise ValueError("Full screenshot does not fit; refusing a truncated API request")
        body = self.request_body(self.prompts[task_key], prepared["data_urls"])
        self.last_details = {
            "image": prepared["image"], "prompt": self.prompts[task_key], "request_sent": False,
            "submitted_png_sha256": prepared["png_sha256"],
        }
        if self.provider == "anthropic":
            headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}
        else:
            headers = {"Authorization": f"Bearer {self.api_key}"}
        if self.provider == "openai":
            for variable, header in (("OPENAI_ORG_ID", "OpenAI-Organization"), ("OPENAI_PROJECT_ID", "OpenAI-Project")):
                if os.environ.get(variable):
                    headers[header] = os.environ[variable]
        try:
            self.last_details["request_sent"] = True
            response = requests.post(self.endpoint, headers=headers, json=body, timeout=(15, self.timeout))
        except requests.RequestException as exc:
            # Exception strings may include request-header values. Record the
            # network error type without leaking authentication material.
            raise RuntimeError(f"{self.provider} request failed: {type(exc).__name__}") from None
        self.last_details["http_status"] = response.status_code
        self.last_details["retry_after"] = response.headers.get("retry-after")
        self.last_details["request_id"] = response.headers.get("request-id") or response.headers.get("x-request-id")
        if response.status_code in (401, 403):
            self.fatal_error = f"{self.provider}: HTTP {response.status_code}; check key, model access, and billing"
            raise RuntimeError(self.fatal_error)
        try:
            payload = response.json()
        except ValueError:
            self.last_details["response_body"] = self.redact(response.text)
            raise RuntimeError(f"{self.provider}: HTTP {response.status_code}, non-JSON response") from None
        self.last_details.update({"response": self.redact(payload), "usage": payload.get("usage"),
                                  "response_id": payload.get("id"), "response_model": payload.get("model")})
        if response.status_code >= 400 or payload.get("error"):
            message = self.redact(str(payload.get("error", "API error")))
            quota_exhausted = isinstance(payload.get("error"), dict) and payload["error"].get("code") == "insufficient_quota"
            if response.status_code in (400, 402, 404, 422) or quota_exhausted:
                self.fatal_error = f"{self.provider}: HTTP {response.status_code}; {message}"
            raise RuntimeError(f"{self.provider}: HTTP {response.status_code}; {message}")
        return self.final_text(payload)
