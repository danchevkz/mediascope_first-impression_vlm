import json
import tempfile
import threading
import unittest
from pathlib import Path

import eval_opus_summary_parallel as common
from eval_openrouter_summary_parallel import MODELS, acceptable, decode_response, execute, load_retry_plan, reasoning_text, request_body
from run_config import sha256_file


def assessment():
    return {"label": "HIGH", "explanation_summary": "Visible sources are attributed.",
            "recognized_outlet": None, "recognized_network": None, "external_rating_source": None,
            "recalled_factuality_rating": None, "recalled_bias_rating": None,
            "prior_rating_influence": "unknown", "possible_contamination": "unknown",
            "self_report_basis": "I do not recall a published rating and cannot determine benchmark exposure."}


def response(model=MODELS["kimi"], reasoning="Native provider reasoning."):
    return {"model": model, "provider": "test-provider", "choices": [{"finish_reason": "stop",
        "message": {"role": "assistant", "content": json.dumps(assessment()), "reasoning": reasoning}}]}


class OpenRouterRecognitionTests(unittest.TestCase):
    def test_same_prompts_and_original_benchmark_models_with_required_controls(self):
        self.assertEqual(MODELS["qwen"], "qwen/qwen3.8-max-prime")
        for model in MODELS.values():
            body = request_body(model, "bias", ["data:image/png;base64,test"], "high", 8192)
            self.assertEqual(body["model"], model)
            self.assertEqual(body["reasoning"], {"enabled": True, "effort": "high", "exclude": False})
            self.assertTrue(body["provider"]["require_parameters"])
            self.assertEqual(body["messages"][0]["content"], common.BASE_SYSTEM)
            self.assertEqual(body["messages"][1]["content"][0]["text"], common.probe_prompt("bias"))
            self.assertEqual(body["messages"][1]["content"][1]["image_url"]["detail"], "high")
            self.assertTrue(body["response_format"]["json_schema"]["strict"])

    def test_valid_report_preserves_final_text_and_provider_identity(self):
        payload = response()
        row = decode_response(payload, "factuality", MODELS["kimi"])
        self.assertTrue(acceptable(row))
        self.assertEqual(row["structured_assessment"], assessment())
        self.assertEqual(row["final_text"], payload["choices"][0]["message"]["content"])
        self.assertEqual(row["response_provider"], "test-provider")
        self.assertEqual(row["provider_summary"], "Native provider reasoning.")

    def test_reasoning_content_alias_and_summary_details_are_captured(self):
        self.assertEqual(reasoning_text({"reasoning_content": "Actual native text."})[0], "Actual native text.")
        text, kind = reasoning_text({"reasoning_details": [
            {"type": "reasoning.summary", "summary": "Native summary."},
            {"type": "reasoning.text", "text": "Native text."},
            {"type": "reasoning.encrypted", "data": "not readable", "text": "not a visible trace"},
        ]})
        self.assertEqual(text, "Native summary.\n\nNative text.")
        self.assertEqual(kind, "reasoning.summary+reasoning.text")

    def test_plain_reasoning_is_not_duplicated_by_equivalent_detail_fields(self):
        self.assertEqual(reasoning_text({"reasoning": "Native text.", "reasoning_details": [
            {"type": "reasoning.text", "text": "Native text."}]})[0], "Native text.")

    def test_missing_reasoning_is_never_filled_from_final_explanation(self):
        payload = response(reasoning="")
        row = decode_response(payload, "factuality", MODELS["kimi"], allow_missing=True)
        self.assertTrue(acceptable(row))
        self.assertEqual(row["status"], "ok_without_provider_summary")
        self.assertEqual(row["provider_summary"], "")
        self.assertEqual(row["provider_summary_origin"], "unavailable")
        self.assertFalse(acceptable(decode_response(payload, "factuality", MODELS["kimi"])))

    def test_encrypted_reasoning_does_not_count_as_observable_text(self):
        payload = response(reasoning="")
        payload["choices"][0]["message"]["reasoning_details"] = [{"type": "reasoning.encrypted", "data": "signature"}]
        row = decode_response(payload, "factuality", MODELS["kimi"], allow_missing=True)
        self.assertFalse(row["provider_summary_available"])

    def test_wrong_model_is_rejected_even_for_a_filtered_response(self):
        payload = response(model="qwen/qwen3.8-max-0902")
        self.assertEqual(decode_response(payload, "factuality", MODELS["qwen"])["status"], "model_mismatch")
        payload["choices"][0]["finish_reason"] = "content_filter"
        self.assertEqual(decode_response(payload, "factuality", MODELS["qwen"])["status"], "model_mismatch")

    def test_truncated_report_is_not_accepted_even_with_parseable_json(self):
        payload = response()
        payload["choices"][0]["finish_reason"] = "length"
        row = decode_response(payload, "factuality", MODELS["kimi"], allow_missing=True)
        self.assertEqual(row["status"], "incomplete_response")
        self.assertFalse(row["output_valid"])
        self.assertFalse(row["failure_classification"]["retryable"])

    def test_missing_summary_policy_does_not_relax_core_validation(self):
        payload = response(reasoning="")
        data = assessment()
        data["prior_rating_influence"] = "maybe"
        payload["choices"][0]["message"]["content"] = json.dumps(data)
        row = decode_response(payload, "factuality", MODELS["kimi"], allow_missing=True)
        self.assertEqual(row["status"], "invalid_output")
        self.assertFalse(acceptable(row, record_refusals=True))

    def test_secondary_contamination_omission_is_preserved(self):
        payload = response()
        data = assessment()
        data.pop("possible_contamination")
        payload["choices"][0]["message"]["content"] = json.dumps(data)
        row = decode_response(payload, "factuality", MODELS["kimi"])
        self.assertTrue(acceptable(row))
        self.assertNotIn("possible_contamination", row["structured_assessment"])
        self.assertIn("possible_contamination_not_reported", row["validation_warnings"])

    def test_provider_filter_is_explicit_unobserved_report_not_a_negative_reference(self):
        payload = {"error": {"code": 400, "message": "Provider rejected inappropriate content"}}
        row = decode_response(payload, "bias", MODELS["qwen"], allow_missing=True, http_status=400)
        self.assertEqual(row["status"], "content_filtered")
        self.assertIsNone(row["structured_assessment"])
        self.assertIsNone(row["predicted_label"])
        self.assertFalse(row["failure_classification"]["retryable"])
        self.assertTrue(acceptable(row, record_refusals=True))
        self.assertFalse(acceptable(row))

    def test_refusal_field_is_preserved_as_terminal_provider_outcome(self):
        payload = response()
        payload["choices"][0]["message"]["refusal"] = "Cannot process this image."
        row = decode_response(payload, "bias", MODELS["kimi"])
        self.assertEqual(row["status"], "content_filtered")
        self.assertFalse(row["output_valid"])

    def test_reasoning_text_about_content_filters_is_not_classified_as_an_api_error(self):
        row = decode_response(response(reasoning="This headline mentions a content filter."), "factuality", MODELS["kimi"])
        self.assertTrue(acceptable(row))

    def test_authentication_and_quota_errors_stop_rather_than_becoming_allowed_refusals(self):
        for http in (401, 402):
            with self.subTest(http=http):
                row = decode_response({"error": {"code": http, "message": "Access denied"}}, "bias", MODELS["kimi"], http_status=http)
                self.assertFalse(acceptable(row, record_refusals=True))
                self.assertTrue(row["failure_classification"]["fatal"])

    def test_scheduler_continues_after_recorded_refusals_but_stops_on_invalid_output(self):
        stop, collected = threading.Event(), []
        execute(list(range(12)), lambda c: {"status": "content_filtered", "output_valid": False},
                3, stop, collected.append, record_refusals=True)
        self.assertEqual(len(collected), 12)
        self.assertFalse(stop.is_set())
        execute(list(range(12)), lambda c: {"status": "invalid_output", "output_valid": False},
                1, stop, collected.append, record_refusals=True)
        self.assertTrue(stop.is_set())
        self.assertEqual(len(collected), 13)

    def test_recorded_output_limits_continue_without_relaxing_other_failures(self):
        row = {"status": "incomplete_response", "output_valid": False,
               "failure_classification": {"kind": "output_limit"}}
        self.assertFalse(acceptable(row, True))
        self.assertTrue(acceptable(row, True, True))
        stop, collected = threading.Event(), []
        execute(list(range(5)), lambda c: dict(row), 2, stop, collected.append, True, True)
        self.assertEqual(len(collected), 5)
        self.assertFalse(stop.is_set())
        row["failure_classification"]["kind"] = "invalid_response"
        self.assertFalse(acceptable(row, True, True))

    def test_authorized_retry_is_single_generation_even_if_it_fails_again(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "records").mkdir()
            (root / "retry_backups").mkdir()
            case = {"case_id": "failed", "task": "factuality", "media_name": "Example",
                    "url": "https://example.test", "image_sha256": "image", "true_label": "HIGH"}
            original = {**case, "status": "incomplete_response", "output_valid": False,
                        "failure_classification": {"kind": "output_limit"}}
            current = root / "records/failed.json"
            backup = root / "retry_backups/failed.json"
            current.write_text(json.dumps(original))
            backup.write_bytes(current.read_bytes())
            digest = sha256_file(current)
            plan_path = root / "output_limit_reruns.json"
            plan_path.write_text(json.dumps({"generation_attempts_authorized": 1, "cases": [{
                "case_id": "failed", "original_record_sha256": digest, "backup": "retry_backups/failed.json"}]}))
            pending = load_retry_plan(root, plan_path, {"failed": original}, {"failed": case})
            self.assertEqual(set(pending), {"failed"})
            rerun = {**original, "generation_attempt": 2, "retry_provenance": pending["failed"]}
            current.write_text(json.dumps(rerun))
            self.assertEqual(load_retry_plan(root, plan_path, {"failed": rerun}, {"failed": case}), {})
            backup.write_text("{}")
            with self.assertRaisesRegex(ValueError, "backup changed"):
                load_retry_plan(root, plan_path, {"failed": rerun}, {"failed": case})

    def test_retry_authorization_cannot_include_successful_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "records").mkdir()
            case = {"case_id": "valid", "task": "factuality", "media_name": "Example",
                    "url": "https://example.test", "image_sha256": "image", "true_label": "HIGH"}
            row = {**case, "status": "ok", "output_valid": True}
            backup = root / "valid.json"
            backup.write_text(json.dumps(row))
            (root / "records/valid.json").write_bytes(backup.read_bytes())
            plan = root / "retry.json"
            plan.write_text(json.dumps({"generation_attempts_authorized": 1, "cases": [{
                "case_id": "valid", "original_record_sha256": sha256_file(backup), "backup": "valid.json"}]}))
            with self.assertRaisesRegex(ValueError, "Only the authorized output-limit"):
                load_retry_plan(root, plan, {"valid": row}, {"valid": case})

    def test_completed_schema_echo_is_terminal_without_relaxing_transport_or_model_checks(self):
        payload = response(model=MODELS["qwen"])
        payload["choices"][0]["message"]["content"] = json.dumps([{"type": "object", "properties": {"label": {"type": "string"}}}])
        row = {**decode_response(payload, "bias", MODELS["qwen"], allow_missing=True),
               "http_status": 200, "response": payload}
        self.assertEqual(row["status"], "invalid_output")
        self.assertFalse(row["output_valid"])
        self.assertFalse(acceptable(row, True, True))
        self.assertTrue(acceptable(row, True, True, True))
        for mutation in ({"status": "model_mismatch"}, {"status": "http_error"},
                         {"http_status": 401}, {"finish_reason": "tool_calls"}):
            self.assertFalse(acceptable({**row, **mutation}, True, True, True))
        stop, collected = threading.Event(), []
        execute(list(range(4)), lambda c: dict(row), 2, stop, collected.append, True, True, True)
        self.assertEqual(len(collected), 4)
        self.assertFalse(stop.is_set())


if __name__ == "__main__":
    unittest.main()
