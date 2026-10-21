import json
import tempfile
import threading
import unittest
from pathlib import Path

from eval_opus_summary_parallel import acceptable, decode_response, execute, load_records, probe_prompt, redact_payload, request_body, retry_seconds
from repair_opus_probe_parsing import reparse, validate_unchanged_request


def valid_assessment():
    return {"label": "HIGH", "explanation_summary": "The screenshot cites named wire services.",
            "recognized_outlet": None, "recognized_network": None, "external_rating_source": None,
            "recalled_factuality_rating": None, "recalled_bias_rating": None,
            "prior_rating_influence": "unknown", "possible_contamination": "unknown",
            "self_report_basis": "I cannot verify training membership or recall a source rating."}


def response(summary="A brief source assessment.", assessment=None):
    return {"model": "claude-opus-5", "stop_reason": "end_turn", "content": [
        {"type": "thinking", "thinking": summary, "signature": "test signature"},
        {"type": "text", "text": json.dumps(assessment if assessment is not None else valid_assessment())},
    ]}


class OpusSummaryTests(unittest.TestCase):
    def test_request_uses_correct_model_and_explicit_summary_display(self):
        body = request_body("factuality", ["data:image/png;base64,abc"], "high", 8192)
        self.assertEqual(body["model"], "claude-opus-5")
        self.assertEqual(body["thinking"], {"type": "adaptive", "display": "summarized"})
        self.assertEqual(body["output_config"], {"effort": "high"})
        self.assertEqual(body["messages"][0]["content"][1]["source"]["data"], "abc")

    def test_prompt_does_not_demand_admission_or_verified_training_membership(self):
        text = probe_prompt("bias")
        self.assertIn("null and unknown are valid", text)
        self.assertIn("NOT proof of training contamination", text)
        self.assertNotIn("Output ONLY the label", text)

    def test_valid_summary_and_structured_output(self):
        row = decode_response(response(), "factuality")
        self.assertEqual(row["status"], "ok")
        self.assertTrue(row["output_valid"])
        self.assertEqual(row["predicted_label"], "HIGH")

    def test_valid_json_in_code_fence_preserves_original_response_text(self):
        payload = response()
        raw = "```json\n" + payload["content"][1]["text"] + "\n```"
        payload["content"][1]["text"] = raw
        row = decode_response(payload, "factuality")
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["final_text"], raw)
        self.assertEqual(row["predicted_label"], "HIGH")

    def test_incomplete_fence_or_extra_prose_is_not_silently_parsed(self):
        payload = response()
        payload["content"][1]["text"] = "```json\n" + payload["content"][1]["text"]
        self.assertEqual(decode_response(payload, "factuality")["status"], "invalid_output")
        payload = response()
        payload["content"][1]["text"] = "Extra prose\n" + payload["content"][1]["text"]
        self.assertEqual(decode_response(payload, "factuality")["status"], "invalid_output")

    def test_unique_complete_report_after_partial_report_is_parsed_without_resampling(self):
        payload = response()
        partial = json.dumps({"label": "HIGH", "explanation_summary": "Visible sourcing."})
        full = payload["content"][1]["text"]
        raw = f"```json\n{partial}\n```\n\nSource-recognition report:\n\n```json\n{full}\n```"
        payload["content"][1]["text"] = raw
        parsed = decode_response(payload, "factuality")
        self.assertEqual(parsed["status"], "ok")
        self.assertEqual(parsed["final_text"], raw)
        self.assertEqual(parsed["structured_assessment"], valid_assessment())

    def test_conflicting_or_multiple_complete_reports_are_not_selected_for_correctness(self):
        payload = response()
        full = payload["content"][1]["text"]
        payload["content"][1]["text"] = f"```json\n{{\"label\": \"LOW\"}}\n```\n```json\n{full}\n```"
        self.assertEqual(decode_response(payload, "factuality")["status"], "invalid_output")
        payload["content"][1]["text"] = f"```json\n{full}\n```\n```json\n{full}\n```"
        self.assertEqual(decode_response(payload, "factuality")["status"], "invalid_output")

    def test_empty_signed_thinking_block_is_not_success(self):
        row = decode_response(response(summary=""), "factuality")
        self.assertEqual(row["status"], "missing_provider_summary")
        self.assertTrue(row["output_valid"])
        self.assertEqual(row["provider_summary"], "")

    def test_absent_thinking_is_not_filled_from_final_explanation(self):
        payload = response()
        payload["content"] = payload["content"][1:]
        self.assertEqual(decode_response(payload, "factuality")["status"], "missing_provider_summary")

    def test_missing_explanation_is_invalid(self):
        assessment = valid_assessment()
        assessment["explanation_summary"] = ""
        row = decode_response(response(assessment=assessment), "factuality")
        self.assertEqual(row["status"], "invalid_output")
        self.assertFalse(row["output_valid"])

    def test_unknown_contamination_allowed_even_when_prior_rating_influenced_answer(self):
        assessment = valid_assessment()
        assessment["prior_rating_influence"] = "yes"
        self.assertEqual(decode_response(response(assessment=assessment), "factuality")["status"], "ok")

    def test_self_report_requires_an_explicit_allowed_value(self):
        assessment = valid_assessment()
        assessment["possible_contamination"] = True
        self.assertEqual(decode_response(response(assessment=assessment), "factuality")["status"], "invalid_output")

    def test_missing_secondary_contamination_field_is_not_invented_or_run_fatal(self):
        assessment = valid_assessment()
        assessment.pop("possible_contamination")
        row = decode_response(response(assessment=assessment), "factuality")
        self.assertEqual(row["status"], "ok")
        self.assertTrue(row["output_valid"])
        self.assertFalse(row["possible_contamination_reported"])
        self.assertNotIn("possible_contamination", row["structured_assessment"])
        self.assertIn("possible_contamination_not_reported", row["validation_warnings"])

    def test_missing_secondary_field_does_not_relax_native_summary_requirement(self):
        assessment = valid_assessment()
        assessment.pop("possible_contamination")
        self.assertEqual(decode_response(response(summary="", assessment=assessment), "factuality")["status"], "missing_provider_summary")

    def test_rejects_wrong_model_or_truncated_response(self):
        payload = response()
        payload["model"] = "claude-opus-5-5"
        self.assertEqual(decode_response(payload, "factuality")["status"], "model_mismatch")
        payload = response()
        payload["stop_reason"] = "max_tokens"
        self.assertEqual(decode_response(payload, "factuality")["status"], "incomplete_response")

    def test_rate_limit_cooldown_honors_retry_after(self):
        self.assertEqual(retry_seconds("120", 0), 120)
        self.assertEqual(retry_seconds("invalid", 0), 2)

    def test_redacts_credential_if_error_payload_echoes_it(self):
        payload = {"error": {"message": "echo secret-key"}, "details": ["secret-key"]}
        saved = redact_payload(payload, "secret-key")
        self.assertNotIn("secret-key", json.dumps(saved))

    def test_failed_observability_stops_new_work_but_keeps_inflight_records(self):
        stop = threading.Event()
        barrier = threading.Barrier(4)
        seen, collected = [], []
        lock = threading.Lock()

        def worker(case):
            with lock:
                seen.append(case)
            barrier.wait(timeout=5)
            if case == 0:
                stop.set()
            else:
                self.assertTrue(stop.wait(timeout=5))
            return {"status": "missing_provider_summary" if case == 0 else "ok", "output_valid": True, "case": case}

        execute(list(range(20)), worker, 4, stop, collected.append)
        self.assertEqual(len(seen), 4)
        self.assertEqual(len(collected), 4)
        self.assertTrue(stop.is_set())

    def test_resume_rejects_changed_target(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            case = {"case_id": "case1", "task": "factuality", "media_name": "source",
                    "url": "url", "image_sha256": "hash", "true_label": "HIGH"}
            (root / "case1.json").write_text(json.dumps({**case, "true_label": "LOW"}))
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                load_records(root, {"case1": case})

    def test_offline_repair_reuses_response_and_records_previous_parser_failure(self):
        payload = response()
        payload["content"][1]["text"] = "```json\n" + payload["content"][1]["text"] + "\n```"
        original_response = json.dumps(payload)
        row = {"case_id": "case1", "task": "factuality", "true_label": "HIGH", "status": "invalid_output",
               "validation_error": "Expecting value", "correct": False, "response": payload}
        repaired = reparse(row)
        self.assertEqual(repaired["status"], "ok")
        self.assertTrue(repaired["correct"])
        self.assertEqual(json.dumps(repaired["response"]), original_response)
        self.assertEqual(repaired["parser_recovery_provenance"]["previous_status"], "invalid_output")
        self.assertTrue(repaired["parser_recovery_provenance"]["response_reused"])

    def test_parser_only_upgrade_refuses_changed_request_prompt(self):
        body = request_body("factuality", [], "high", 8192)
        plan = {"model": body["model"], "thinking": body["thinking"], "output_config": body["output_config"],
                "system": body["system"], "max_output_tokens": 8192,
                "prompts": {task: probe_prompt(task) for task in ("factuality", "bias")}}
        validate_unchanged_request(plan)
        plan["prompts"]["factuality"] = "Changed prompt"
        with self.assertRaisesRegex(ValueError, "Request protocol changed"):
            validate_unchanged_request(plan)

    def test_explicit_missing_summary_policy_preserves_empty_summary_and_valid_label(self):
        row = decode_response(response(summary=""), "factuality", allow_missing=True)
        self.assertEqual(row["status"], "ok_without_provider_summary")
        self.assertFalse(row["provider_summary_available"])
        self.assertEqual(row["provider_summary_origin"], "unavailable")
        self.assertEqual(row["provider_summary"], "")
        self.assertTrue(acceptable(row))
        self.assertFalse(acceptable(decode_response(response(summary=""), "factuality")))

    def test_missing_summary_policy_does_not_accept_invalid_fields(self):
        data = valid_assessment()
        data["label"] = "WRONG"
        self.assertFalse(acceptable(decode_response(response(summary="", assessment=data), "factuality", allow_missing=True)))

    def test_scheduler_continues_on_explicit_unavailable_summary(self):
        stop, collected = threading.Event(), []
        def worker(case):
            return {"status": "ok_without_provider_summary", "output_valid": True, "case": case}
        execute(list(range(8)), worker, 4, stop, collected.append)
        self.assertEqual(len(collected), 8)
        self.assertFalse(stop.is_set())

    def test_observation_policy_migration_reuses_the_original_empty_summary(self):
        payload = response(summary="")
        row = {"case_id": "case1", "task": "factuality", "true_label": "HIGH", "status": "missing_provider_summary",
               "validation_error": "no summary", "correct": True, "response": payload}
        repaired = reparse(row, allow_missing=True)
        self.assertEqual(repaired["status"], "ok_without_provider_summary")
        self.assertEqual(repaired["provider_summary"], "")
        self.assertEqual(repaired["response"], payload)
        self.assertTrue(repaired["parser_recovery_provenance"]["response_reused"])


if __name__ == "__main__":
    unittest.main()
