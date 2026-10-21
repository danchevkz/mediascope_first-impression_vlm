"""Opt-in transport recovery with explicit per-response failure classification.

The historical API runner stays byte-for-byte reproducible. New/recovery runs
use this adapter and record its implementation and retry policy separately.
"""

import json
import math
import os
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from api_predictor import ReasoningAPIPredictor
from eval_api import EvaluationStopped
from run_config import sha256_file


@dataclass(frozen=True)
class Failure:
    kind: str
    message: str
    retryable: bool = False
    fatal: bool = False


class APIResponseError(RuntimeError):
    def __init__(self, failure):
        self.failure = failure
        super().__init__(f"{failure.kind}: {failure.message}")


def classify_failure(details, error_message=""):
    """Classify actual error fields; never search reasoning text for errors."""
    response = details.get("response") or {}
    choices = response.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    errors = [e for e in (response.get("error"), choice.get("error")) if e]
    text = " ".join([error_message, *[json.dumps(e, ensure_ascii=False) for e in errors]])
    lowered = text.lower()
    http = details.get("http_status")
    finish = choice.get("finish_reason", details.get("finish_reason"))
    codes = {str(e.get("code", e.get("type", ""))).lower() for e in errors if isinstance(e, dict)}
    numeric_codes = {int(code) for code in codes if code.isdigit()}

    if http == 401 or 401 in numeric_codes or "invalid_api_key" in codes:
        return Failure("authentication", text or "HTTP 401", fatal=True)
    if http == 402 or 402 in numeric_codes or codes & {"insufficient_quota", "quota_exceeded", "billing_hard_limit_reached"}:
        return Failure("quota", text or "Insufficient quota", fatal=True)
    filtered = any(term in lowered for term in (
        "inappropriate content", "content_filter", "content filter", "content policy",
        "content moderation", "safety policy", "safety_system", "sensitive content"))
    if filtered or message.get("refusal") or finish in ("content_filter", "safety") or response.get("stop_reason") == "refusal":
        return Failure("provider_content_filter", text or str(message.get("refusal") or finish or "Refusal"))
    if http == 403 or 403 in numeric_codes:
        return Failure("authentication", text or "HTTP 403", fatal=True)
    if http == 404 or 404 in numeric_codes or codes & {"model_not_found", "invalid_model"}:
        return Failure("model_unavailable", text or "Model/endpoint unavailable", fatal=True)
    if finish == "length" or choice.get("native_finish_reason") == "length" or details.get("reached_token_budget"):
        return Failure("output_limit", "Output budget exhausted; the generation settings are preserved")
    if any(term in lowered for term in (
        "download multimodal file timed out", "image download timed out", "image fetch timed out")):
        return Failure("image_download_timeout", text, retryable=True)
    if http == 429 or 429 in numeric_codes:
        return Failure("rate_limit", text or "Rate limited", retryable=True)
    if http in (408, 409, 425) or (isinstance(http, int) and 500 <= http <= 599) or any(500 <= code <= 599 for code in numeric_codes):
        return Failure("transient_upstream", text or f"HTTP {http}", retryable=True)
    if any(name.lower() in lowered for name in ("Timeout", "ConnectionError", "ChunkedEncodingError", "RemoteDisconnected")):
        return Failure("network_error", text, retryable=True)
    if finish == "error":
        return Failure("upstream_response_error", text or "Provider returned finish_reason=error", retryable=True)
    if errors or (isinstance(http, int) and http >= 400):
        return Failure("request_error", text or f"HTTP {http}")
    if response and "choices" in response and not choices:
        return Failure("invalid_response", "Provider returned an empty choices array", retryable=True)
    if error_message:
        return Failure("local_or_unclassified_error", text, fatal=not bool(details.get("request_sent")))
    return None


def classify_record(record):
    error = record.get("error") or {}
    if record["status"] == "error" and record.get("details", {}).get("request_sent") is False:
        return Failure("previously_skipped", str(error.get("message", "Original runner suppressed this request")), retryable=True)
    failure = classify_failure(record.get("details") or {}, str(error.get("message", "")))
    if failure:
        return failure
    if record["status"] == "error":
        return Failure("unclassified_recorded_error", str(error.get("message", "Unknown original error")))
    if record["status"] == "unparseable":
        return Failure("invalid_final_label", "Completed model output was not a whole allowed label")
    return None


def append_event(path, event):
    """Keep a torn final fragment separate; never overwrite the request history."""
    with path.open("ab+") as stream:
        if stream.tell():
            stream.seek(-1, os.SEEK_END)
            end = stream.read(1)
            stream.seek(0, os.SEEK_END)
            if end != b"\n":
                stream.write(b"\n")
        stream.write((json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())


class ResilientAPIPredictor(ReasoningAPIPredictor):
    def __init__(self, *args, gate, request_log, session_id, retries=3,
                 backoff_base=5, backoff_max=120, recover=False, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0 <= retries <= 10 or not (math.isfinite(backoff_base) and math.isfinite(backoff_max)
                                         and 0 <= backoff_base <= backoff_max and backoff_max > 0):
            raise ValueError("Invalid bounded retry policy")
        self.gate, self.request_log, self.session_id = gate, request_log, session_id
        self.retries, self.backoff_base, self.backoff_max = retries, backoff_base, backoff_max
        self.recovered = {}
        if recover and request_log.exists():
            for line in request_log.read_text(encoding="utf-8").splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self.recovered[(event["image_path"], event["task"])] = event

    def run_metadata(self):
        return {**super().run_metadata(), "resilient_implementation_sha256": sha256_file(__file__),
                "automatic_retries": "Bounded classified transient failures; every attempt journaled",
                "retry_policy": {"retries": self.retries, "backoff_base": self.backoff_base,
                                 "backoff_max": self.backoff_max,
                                 "retryable": "network, 408/409/425/429/5xx, image-download timeout, non-filter upstream response errors",
                                 "terminal_per_sample": "content filters, output budget, invalid final labels, other request errors",
                                 "stop_run": "authentication, quota, missing model/endpoint, local input failure"}}

    def delay(self, retry):
        delay = min(self.backoff_base * 2 ** retry, self.backoff_max)
        value = self.last_details.get("retry_after")
        try:
            supplied = float(value)
        except (TypeError, ValueError):
            try:
                supplied = parsedate_to_datetime(value).timestamp() - time.time()
            except (TypeError, ValueError, OverflowError):
                supplied = 0
        if math.isfinite(supplied) and supplied > 0:
            delay = max(delay, min(supplied, self.backoff_max))
        return delay

    def predict(self, sample, task_key):
        if self.gate.stop.is_set():
            raise EvaluationStopped()
        previous = self.recovered.pop((sample.image_path, task_key), None)
        if previous and previous.get("error") is None:
            self.last_details = {**previous["details"], "request_sent": False,
                                 "replayed_from_request_log": True, "recovery_session_id": self.session_id}
            return previous["raw_text"]
        for retry in range(self.retries + 1):
            self.gate.wait()
            started = time.perf_counter()
            caught, answer = "", None
            try:
                answer = super().predict(sample, task_key)
            except Exception as exc:
                caught = self.redact(str(exc))
            failure = classify_failure(self.last_details, caught)
            actual_model = self.last_details.get("response_model")
            if failure is None and actual_model and actual_model != self.model:
                failure = Failure("model_mismatch", f"Requested {self.model}, received {actual_model}", fatal=True)
            # The old one-shot client marks all HTTP 400s as global failures.
            # Retain that state only for a genuinely run-fatal classification.
            self.fatal_error = failure.message if failure and failure.fatal else None
            self.last_details.update(session_id=self.session_id, transport_attempt=retry + 1,
                                     request_event_id=uuid.uuid4().hex,
                                     failure_classification=asdict(failure) if failure else None)
            delay = self.delay(retry) if failure and failure.retryable and retry < self.retries else None
            cooldown = self.delay(retry) if failure and failure.retryable and (
                delay is not None or failure.kind == "rate_limit" or self.last_details.get("retry_after")) else None
            event = {"recorded_at": datetime.now(timezone.utc).isoformat(), "session_id": self.session_id,
                     "image_path": sample.image_path, "task": task_key, "model": self.model,
                     "attempt_within_call": retry + 1, "raw_text": answer,
                     "elapsed_seconds": time.perf_counter() - started,
                     "error": {"type": "APIResponseError", **asdict(failure)} if failure else None,
                     "retry_delay_seconds": delay, "provider_cooldown_seconds": cooldown, "details": self.last_details}
            append_event(self.request_log, self.redact(event))
            if failure is None:
                return answer
            if failure.fatal:
                self.gate.stop.set()
            if cooldown is not None:
                self.gate.defer(cooldown)
            if delay is not None:
                print(f"[{self.model}] {failure.kind}; retry {retry + 1}/{self.retries} after {delay:.0f}s", flush=True)
                continue
            raise APIResponseError(failure)
