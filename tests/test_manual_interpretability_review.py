import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from manual_interpretability_review import aggregate


class ManualReviewAggregationTests(unittest.TestCase):
    def make_review(self, root, batch):
        rows = [
            {"id": "gpt56:factuality:001", "model": "gpt56", "task": "factuality", "visible": "Neutral headlines.", "provider": "According to MBFC, X is HIGH.", "correct": False, "token_limit": False},
            {"id": "gpt56:factuality:002", "model": "gpt56", "task": "factuality", "visible": "Typical of outlets MBFC rates poorly.", "provider": "", "correct": True, "token_limit": False},
        ]
        (root / "snapshot.json").write_text(json.dumps({"records": rows, "expected_per_task": 2}))
        (root / "annotations").mkdir()
        (root / "annotations/batch.json").write_text(json.dumps(batch))

    def run_review(self, root):
        with contextlib.redirect_stdout(io.StringIO()):
            return aggregate(root, ("gpt56",))

    def test_uses_manual_categories_and_keeps_prediction_correctness_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_review(root, {"model": "gpt56", "task": "factuality", "reviewed_ranges": [[1, 2]], "decisions": {
                "gpt56:factuality:001": {"category": "asserted", "channel": "provider", "evidence": "According to MBFC, X is HIGH.", "note": "Existing external rating."},
                "gpt56:factuality:002": {"category": "generic", "channel": "visible", "evidence": "Typical of outlets MBFC rates poorly.", "note": "Generic analogy."},
            }})
            m = self.run_review(root)["models"]["gpt56"]["factuality"]
            self.assertEqual(m["reference_rate"], 0.5)
            self.assertEqual(m["accuracy_among_references"], 0)
            self.assertEqual(m["overall_accuracy"], 0.5)
            self.assertEqual(m["categories"]["generic"], 1)

    def test_unreviewed_cases_do_not_default_to_negative(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_review(root, {"model": "gpt56", "task": "factuality", "reviewed_ranges": [[1, 1]], "decisions": {}})
            m = self.run_review(root)["models"]["gpt56"]["factuality"]
            self.assertFalse(m["complete_review"])
            self.assertIsNone(m["reference_rate"])

    def test_rejects_invented_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_review(root, {"model": "gpt56", "task": "factuality", "reviewed_ranges": [[1, 2]], "decisions": {
                "gpt56:factuality:001": {"category": "asserted", "channel": "provider", "evidence": "Not in saved text.", "note": "Wrong quote."},
            }})
            with self.assertRaisesRegex(ValueError, "not a saved excerpt"):
                self.run_review(root)


if __name__ == "__main__":
    unittest.main()
