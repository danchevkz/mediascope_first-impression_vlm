import unittest

from summarize_paired_unmasked import score_rows


class PairedScoreSubsetTests(unittest.TestCase):
    def test_errors_count_as_incorrect_and_splits_use_fixed_pair_ids(self):
        labels = {"site_001": {"factuality": "HIGH"}, "site_003": {"factuality": "LOW"}}
        rows = {
            ("site_001", "factuality"): {"media_name": "site_001", "true_label": "HIGH", "predicted_label": "HIGH", "correct": True, "status": "ok"},
            ("site_003", "factuality"): {"media_name": "site_003", "true_label": "LOW", "predicted_label": None, "correct": False, "status": "error"},
        }
        full = score_rows(rows, set(labels), labels, "factuality")
        self.assertTrue(full["complete"])
        self.assertEqual(full["accuracy"], 0.5)
        self.assertEqual(full["mae"], 1)
        self.assertEqual(full["mse"], 2)
        self.assertEqual(full["valid_output_rate"], 0.5)
        seen = score_rows(rows, {"site_001"}, labels, "factuality")
        self.assertEqual(seen["accuracy"], 1)
        pending = score_rows({("site_001", "factuality"): rows[("site_001", "factuality")]}, set(labels), labels, "factuality")
        self.assertFalse(pending["complete"])

    def test_mismatched_target_or_correctness_is_rejected(self):
        labels = {"site_001": {"bias": "RIGHT"}}
        row = {"media_name": "site_001", "true_label": "RIGHT", "predicted_label": "RIGHT", "correct": False, "status": "ok"}
        with self.assertRaises(ValueError):
            score_rows({("site_001", "bias"): row}, set(labels), labels, "bias")
        row.update(true_label="LEFT", correct=True)
        with self.assertRaises(ValueError):
            score_rows({("site_001", "bias"): row}, set(labels), labels, "bias")


if __name__ == "__main__":
    unittest.main()
