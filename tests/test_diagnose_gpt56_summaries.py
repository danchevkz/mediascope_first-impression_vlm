import json
import unittest

from diagnose_gpt56_summaries import classify_observability, diagnostic_body, stream_result


class GPTSummaryDiagnosticTests(unittest.TestCase):
    def test_profiles_change_only_requested_configuration(self):
        standard = diagnostic_body("bias", [], "standard_auto", 8192)
        pro = diagnostic_body("bias", [], "pro_detailed", 8192)
        streaming = diagnostic_body("bias", [], "stream_detailed", 8192)
        self.assertEqual(standard["reasoning"], {"effort": "high", "summary": "auto"})
        self.assertEqual(pro["reasoning"]["mode"], "pro")
        self.assertTrue(streaming["stream"])
        self.assertEqual(standard["input"], pro["input"])
        self.assertEqual(standard["input"], streaming["input"])

    def test_extracts_actual_summary_deltas_even_when_final_array_empty(self):
        response = {"model": "gpt-5.6-sol", "status": "completed", "output": []}
        events = [
            {"type": "response.reasoning_summary_text.delta", "item_id": "rs1", "summary_index": 0, "delta": "Source "},
            {"type": "response.reasoning_summary_text.delta", "item_id": "rs1", "summary_index": 0, "delta": "assessment."},
            {"type": "response.completed", "response": response},
        ]
        payload, text = stream_result(events)
        self.assertEqual(payload, response)
        self.assertEqual(text, "Source assessment.")

    def test_final_output_text_and_encryption_are_not_summary_fallbacks(self):
        events = [
            {"type": "response.output_text.delta", "delta": "Final explanation."},
            {"type": "response.output_item.done", "item": {"encrypted_content": "opaque"}},
            {"type": "response.completed", "response": {"status": "completed", "output": []}},
        ]
        _, text = stream_result(events)
        self.assertEqual(text, "")

    def test_done_event_does_not_duplicate_streamed_text(self):
        events = [
            {"type": "response.reasoning_summary_text.delta", "item_id": "rs1", "summary_index": 0, "delta": "Summary."},
            {"type": "response.reasoning_summary_text.done", "item_id": "rs1", "summary_index": 0, "text": "Summary."},
        ]
        _, text = stream_result(events)
        self.assertEqual(text, "Summary.")

    def test_missing_terminal_event_is_detected(self):
        payload, _ = stream_result([{ "type": "response.reasoning_summary_text.delta", "delta": "Summary."}])
        self.assertIsNone(payload)

    def test_stream_observation_requires_valid_completed_model_response(self):
        assessment = {"label": "HIGH", "explanation_summary": "Visible source evidence.",
                      "recognized_outlet": None, "recognized_network": None, "external_rating_source": None,
                      "recalled_factuality_rating": None, "recalled_bias_rating": None,
                      "prior_rating_influence": "unknown", "possible_contamination": "unknown",
                      "self_report_basis": "Unknown training membership."}
        payload = {"model": "gpt-5.6-sol", "status": "completed", "output": [
            {"type": "reasoning", "summary": []},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": json.dumps(assessment)}]},
        ]}
        observed = classify_observability(payload, "factuality", "Actual stream summary text.")
        self.assertTrue(observed["observable_valid_output"])
        self.assertEqual(observed["summary_origin"], "actual_stream_summary_events")
        self.assertEqual(observed["decoded"]["provider_summary"], "")
        payload["status"] = "incomplete"
        self.assertFalse(classify_observability(payload, "factuality", "Stream summary.")["observable_valid_output"])


if __name__ == "__main__":
    unittest.main()
