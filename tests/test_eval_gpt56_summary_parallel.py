import json
import tempfile
import threading
import unittest
from pathlib import Path

import eval_opus_summary_parallel as opus
from eval_gpt56_summary_parallel import acceptable, assessment_schema, decode_response, execute, load_stream_seeds, request_body
from run_config import sha256_file


def assessment():
    return {"label": "HIGH", "explanation_summary": "The page uses named wire sources.",
            "recognized_outlet": None, "recognized_network": None, "external_rating_source": None,
            "recalled_factuality_rating": None, "recalled_bias_rating": None,
            "prior_rating_influence": "unknown", "possible_contamination": "unknown",
            "self_report_basis": "No recalled external rating; training membership is unknown."}


def response():
    return {"model": "gpt-5.6-sol", "status": "completed", "output": [
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "Brief provider summary."}],
         "encrypted_content": "not-readable-summary"},
        {"type": "message", "role": "assistant", "phase": "final_answer",
         "content": [{"type": "output_text", "text": json.dumps(assessment())}]},
    ]}


class GPTSummaryProbeTests(unittest.TestCase):
    def test_correct_model_and_explicit_detailed_summary(self):
        body = request_body("factuality", ["data:image/png;base64,abc"], "high", 8192)
        self.assertEqual(body["model"], "gpt-5.6-sol")
        self.assertEqual(body["reasoning"], {"effort": "high", "summary": "detailed"})
        self.assertFalse(body["store"])
        self.assertTrue(body["text"]["format"]["strict"])

    def test_prompt_and_system_are_identical_to_opus_probe(self):
        for task in ("factuality", "bias"):
            g = request_body(task, [], "high", 8192)
            a = opus.request_body(task, [], "high", 8192)
            self.assertEqual(g["instructions"], a["system"])
            self.assertEqual(g["input"][0]["content"][0]["text"], a["messages"][0]["content"][0]["text"])

    def test_schema_requires_self_reports_and_correct_task_labels(self):
        schema = assessment_schema("bias")
        self.assertEqual(set(schema["required"]), set(assessment()))
        self.assertIn("RIGHT-CENTER", schema["properties"]["label"]["enum"])
        self.assertNotIn("HIGH", schema["properties"]["label"]["enum"])
        self.assertIn("unknown", schema["properties"]["possible_contamination"]["enum"])

    def test_extracts_real_summary_and_final_assessment(self):
        row = decode_response(response(), "factuality")
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["provider_summary"], "Brief provider summary.")
        self.assertEqual(row["predicted_label"], "HIGH")

    def test_encrypted_reasoning_is_not_counted_as_readable_summary(self):
        payload = response()
        payload["output"][0]["summary"] = []
        row = decode_response(payload, "factuality")
        self.assertEqual(row["status"], "missing_provider_summary")
        self.assertEqual(row["provider_summary"], "")

    def test_final_explanation_does_not_substitute_for_provider_summary(self):
        payload = response()
        payload["output"] = payload["output"][1:]
        self.assertEqual(decode_response(payload, "factuality")["status"], "missing_provider_summary")

    def test_commentary_message_is_not_parsed_as_final_answer(self):
        payload = response()
        payload["output"].append({"type": "message", "role": "assistant", "phase": "commentary",
                                  "content": [{"type": "output_text", "text": "Still analyzing."}]})
        self.assertEqual(decode_response(payload, "factuality")["status"], "ok")

    def test_rejects_incomplete_or_wrong_model_response(self):
        payload = response()
        payload["status"] = "incomplete"
        payload["incomplete_details"] = {"reason": "max_output_tokens"}
        self.assertEqual(decode_response(payload, "factuality")["status"], "incomplete_response")
        payload = response()
        payload["model"] = "gpt-6.1-sol"
        self.assertEqual(decode_response(payload, "factuality")["status"], "model_mismatch")

    def test_json_fences_are_handled_without_changing_saved_text(self):
        payload = response()
        raw = "```json\n" + payload["output"][1]["content"][0]["text"] + "\n```"
        payload["output"][1]["content"][0]["text"] = raw
        row = decode_response(payload, "factuality")
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["final_text"], raw)

    def test_unknown_contamination_is_valid_with_reported_prior_rating_influence(self):
        payload = response()
        data = assessment()
        data["prior_rating_influence"] = "yes"
        payload["output"][1]["content"][0]["text"] = json.dumps(data)
        self.assertEqual(decode_response(payload, "factuality")["status"], "ok")

    def test_missing_summary_is_explicitly_retained_only_when_requested(self):
        payload = response()
        payload["output"][0]["summary"] = []
        row = decode_response(payload, "factuality", allow_missing=True)
        self.assertEqual(row["status"], "ok_without_provider_summary")
        self.assertFalse(row["provider_summary_available"])
        self.assertEqual(row["provider_summary_origin"], "unavailable")
        self.assertEqual(row["provider_summary"], "")
        self.assertTrue(acceptable(row))
        self.assertFalse(acceptable(decode_response(payload, "factuality")))

    def test_real_stream_text_preserved_as_a_separate_summary_origin(self):
        payload = response()
        payload["output"][0]["summary"] = []
        row = decode_response(payload, "factuality", stream_summary="Actual SSE summary.")
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["provider_summary_origin"], "actual_sse_summary_events")
        self.assertEqual(row["response_summary_chars"], 0)
        self.assertEqual(row["stream_summary_chars"], len("Actual SSE summary."))

    def test_missing_summary_policy_does_not_accept_invalid_json_or_wrong_model(self):
        payload = response()
        payload["output"][1]["content"][0]["text"] = "not JSON"
        self.assertFalse(acceptable(decode_response(payload, "factuality", allow_missing=True)))
        payload = response()
        payload["model"] = "gpt-6.1-sol"
        self.assertFalse(acceptable(decode_response(payload, "factuality", allow_missing=True)))

    def test_scheduler_continues_after_flagged_missing_summaries(self):
        stop = threading.Event()
        collected = []
        def worker(case):
            return {"status": "ok_without_provider_summary", "output_valid": True, "case": case}
        execute(list(range(8)), worker, 4, stop, collected.append)
        self.assertEqual(len(collected), 8)
        self.assertFalse(stop.is_set())

    def test_seed_import_retains_all_four_cases_including_missing_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, diagnostic = root / "source", root / "diagnostic"
            source.mkdir(); diagnostic.mkdir()
            cases = [{"case_id": f"case{i}", "task": "factuality", "media_name": f"Source{i}",
                      "true_label": "HIGH"} for i in range(4)]
            plan = {"model": "gpt-5.6-sol", "system": opus.BASE_SYSTEM, "max_output_tokens": 8192,
                    "prompts": {t: opus.probe_prompt(t) for t in ("factuality", "bias")},
                    "schemas": {t: assessment_schema(t) for t in ("factuality", "bias")}}
            (source / "plan.json").write_text(json.dumps(plan))
            (diagnostic / "plan.json").write_text(json.dumps({"source_dir": str(source),
                "source_plan_sha256": sha256_file(source / "plan.json"), "cases": cases}))
            for i, case in enumerate(cases):
                payload = response()
                if i == 3:
                    payload["output"][0]["summary"] = []
                saved = {"case_id": case["case_id"], "profile": "stream_detailed", "http_status": 200,
                         "request_controls": {"model": "gpt-5.6-sol", "reasoning": {"effort": "high", "summary": "detailed"}, "stream": True},
                         "response": payload, "stream_events": [{"type": "response.completed", "response": payload}]}
                (diagnostic / f"stream_detailed-{case['case_id']}.json").write_text(json.dumps(saved))
            seeds, metadata = load_stream_seeds(diagnostic, cases, 8192, "high")
            self.assertEqual(len(seeds), 4)
            self.assertEqual(metadata["missing_summaries_imported"], 1)
            self.assertEqual(seeds["case3"]["status"], "ok_without_provider_summary")
            self.assertFalse(seeds["case3"]["seed_provenance"]["new_api_call"])
            (diagnostic / "stream_detailed-case3.json").unlink()
            with self.assertRaisesRegex(ValueError, "all four"):
                load_stream_seeds(diagnostic, cases, 8192, "high")


if __name__ == "__main__":
    unittest.main()
