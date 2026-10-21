import unittest

from audit_interpretability_outputs import summary


class InterpretabilityAuditTests(unittest.TestCase):
    def test_same_reference_in_both_channels_counts_once(self):
        text = "MBFC rates this outlet HIGH."
        row = {"status": "ok", "correct": True, "details": {
            "visible_rationale": text,
            "response": {"choices": [{"message": {"reasoning": text}}]},
        }}
        result = summary([row], expected=1)
        self.assertEqual(result["strict_lexical_candidates_visible"]["matched_records"], 1)
        self.assertEqual(result["strict_lexical_candidates_provider"]["matched_records"], 1)
        self.assertEqual(result["strict_lexical_candidates_either_deduplicated"]["matched_records"], 1)

    def test_provider_only_reference_not_lost(self):
        row = {"status": "ok", "correct": False, "details": {
            "visible_rationale": "The headlines look neutral.\nLABEL: HIGH",
            "response": {"output": [{"type": "reasoning", "summary": [
                {"text": "MBFC rates this outlet MIXED."}
            ]}]},
        }}
        result = summary([row], expected=1)
        self.assertEqual(result["strict_lexical_candidates_visible"]["matched_records"], 0)
        self.assertEqual(result["strict_lexical_candidates_provider"]["matched_records"], 1)
        self.assertEqual(result["strict_lexical_candidates_provider"]["correct_predictions"], 0)

    def test_partial_run_does_not_claim_full_accuracy(self):
        result = summary([{"status": "ok", "correct": True, "details": {}}], expected=212)
        self.assertFalse(result["complete"])
        self.assertIsNone(result["overall_subset_accuracy"])
        self.assertEqual(result["accuracy_on_recorded"], 1)


if __name__ == "__main__":
    unittest.main()
