import unittest

from plot_reference_analysis import plot_values


class ReferencePlotTests(unittest.TestCase):
    def qwen(self):
        return {"model": "qwen", "task": "factuality", "n": "212", "valid_reports": "181",
                "references": "30", "correct": "79", "correct_references": "23"}

    def test_failed_reports_remain_in_correct_prediction_bar_denominator(self):
        data = plot_values(self.qwen())
        self.assertAlmostEqual(data["reference_rate_valid_reports_percent"], 100 * 30 / 181)
        self.assertAlmostEqual(data["correct_references_fraction_all_attempts_percent"], 100 * 23 / 212)
        self.assertAlmostEqual(data["reference_share_of_correct_predictions_percent"], 100 * 23 / 79)
        self.assertAlmostEqual(data["overall_accuracy_percent"], 100 * 79 / 212)

    def test_old_valid_only_correct_prediction_bar_is_rejected(self):
        row = {**self.qwen(), "correct_references_fraction_all_attempts": 23 / 181}
        with self.assertRaisesRegex(ValueError, "count-based denominator"):
            plot_values(row)

    def test_impossible_correct_counts_are_rejected(self):
        row = {**self.qwen(), "correct_references": "31"}
        with self.assertRaisesRegex(ValueError, "Inconsistent"):
            plot_values(row)


if __name__ == "__main__":
    unittest.main()
