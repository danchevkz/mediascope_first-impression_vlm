"""Sanity checks for paired resampling and unequal-family weighting."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from quantify_api_risk import MODELS, model_blocks, paired_bootstrap, score_matrices


class RiskQuantificationTests(unittest.TestCase):
    def test_metrics_include_invalid_predictions_in_denominator(self):
        matrix = np.array([[2, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
        result = score_matrices(matrix)
        self.assertAlmostEqual(result["accuracy"], 4 / 7)
        self.assertAlmostEqual(result["macro_f1"], .6)

    def test_identical_models_have_exactly_zero_paired_interval(self):
        blocks = np.array([[[1, 0, 0], [0, 0, 0]],
                           [[0, 0, 0], [1, 0, 0]],
                           [[0, 0, 0], [0, 1, 0]]], dtype=float)
        result = paired_bootstrap({model: blocks.copy() for model in MODELS}, 1000, np.random.default_rng(7))
        for metric in ("accuracy", "macro_f1"):
            self.assertEqual(result["opus_minus_openai"][metric]["point"], 0)
            self.assertEqual(result["opus_minus_openai"][metric]["percentile_95_interval"], [0, 0])

    def test_equal_family_weight_is_not_equal_outlet_weight(self):
        keys = [(f"Outlet {i}", f"https://{i}.example") for i in range(5)]
        groups = {key: "large-family" if i < 4 else "singleton" for i, key in enumerate(keys)}
        rows = {(*key, "factuality"): {"true_label": "MIXED" if i < 4 else "HIGH",
                                       "predicted_label": "MIXED" if i < 4 else "LOW"}
                for i, key in enumerate(keys)}
        records = {model: rows for model in MODELS}
        unweighted = model_blocks(keys, "factuality", records, groups)["opus"]
        balanced = model_blocks(keys, "factuality", records, groups, True)["opus"]
        self.assertEqual(score_matrices(unweighted.sum(axis=0))["accuracy"], .8)
        self.assertEqual(score_matrices(balanced.sum(axis=0))["accuracy"], .5)
        np.testing.assert_array_equal(balanced.sum(axis=(1, 2)), [1, 1])


if __name__ == "__main__":
    unittest.main()
