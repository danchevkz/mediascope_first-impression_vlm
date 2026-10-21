import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import review_probe_selfreports as review
from review_probe_selfreports import aggregate
from run_config import sha256_file


class ProbeSelfReportReviewTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.subset = Path(temporary.name) / "subset.csv"
        self.subset.write_text("media_name,url\nOutlet 1,https://example.test\n", encoding="utf-8")
        patcher = patch.object(review, "SUBSET", self.subset)
        patcher.start()
        self.addCleanup(patcher.stop)

    def make_review(self, root, opus_count=2):
        (root / "sources").mkdir()
        (root / "annotations").mkdir()
        (root / "JUDGE_PROMPT.md").write_text("Frozen test rubric.", encoding="utf-8")
        rows = []
        for model, count in (("opus5", opus_count), ("gpt56", 3)):
            for task in ("factuality", "bias"):
                decisions = {"1": "asserted", "2": "uncertain_recall"} if task == "factuality" else {}
                if model == "gpt56" and task == "factuality":
                    decisions = {"1": "asserted", "2": "generic", "3": "uncertain_recall"}
                if model == "gpt56" and task == "bias":
                    decisions = {"1": "asserted", "2": "ambiguous"}
                batch = {"model": model, "task": task, "reviewed_ranges": [[1, count]], "decisions": decisions}
                (root / "annotations" / f"{model}_{task}.json").write_text(json.dumps(batch), encoding="utf-8")
                for i in range(1, count + 1):
                    id = f"{model}:{task}:{i:03d}"
                    true_label = "HIGH" if task == "factuality" else "RIGHT"
                    label = "LOW" if task == "factuality" and i == 1 else true_label
                    summary = "Native-only external-rating claim." if model == "opus5" or i == 2 else ""
                    original = {"task": task, "media_name": f"Outlet {i}", "true_label": true_label,
                                "correct": label == true_label, "provider_summary": summary,
                                "structured_assessment": {"label": label}, "status": "ok", "output_valid": True}
                    source = root / "sources" / f"{model}_{task}_{i}.json"
                    source.write_text(json.dumps(original), encoding="utf-8")
                    rows.append({"id": id, "model": model, "task": task, "media_name": original["media_name"],
                                 "structured": original["structured_assessment"], "provider_summary": summary,
                                 "true_label": true_label, "predicted_label": label, "correct": original["correct"],
                                 "source": str(source), "source_sha256": sha256_file(source)})
        saved = {"expected_per_task": 3, "subset_sha256": sha256_file(self.subset), "records": rows}
        (root / "snapshot.json").write_text(json.dumps(saved), encoding="utf-8")
        return saved

    def run_review(self, root):
        with contextlib.redirect_stdout(io.StringIO()):
            return aggregate(root)

    def test_matched_denominators_and_manual_codes_not_native_only_claims(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_review(root)
            result = self.run_review(root)
            self.assertEqual(result["reviewed_decisions"], 10)
            factuality = result["models"]["gpt56"]["factuality"]
            self.assertEqual(factuality["explicit_references"], 2)
            self.assertEqual(factuality["correct_reference_predictions"], 1)
            self.assertEqual(factuality["accuracy_among_references"], 0.5)
            self.assertEqual(factuality["overall_accuracy"], 2 / 3)
            self.assertEqual(factuality["provider_summaries"], 1)
            self.assertEqual(result["matched"]["outlet_indices"], [1, 2])
            matched = result["matched"]["models"]["gpt56"]["factuality"]
            self.assertEqual(matched["n"], 2)
            self.assertEqual(matched["explicit_references"], 1)
            self.assertEqual(matched["accuracy_among_references"], 0)
            self.assertEqual(matched["overall_accuracy"], 0.5)
            self.assertFalse(result["models"]["opus5"]["factuality"]["complete"])
            bias = result["models"]["opus5"]["bias"]
            self.assertEqual(bias["categories"]["none"], 2)
            self.assertIsNone(bias["accuracy_among_references"])
            self.assertTrue((root / "REPORT.md").exists())
            self.assertEqual(len((root / "coded_records.csv").read_text().splitlines()), 11)

    def test_unreviewed_cases_cannot_default_to_none(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_review(root)
            path = root / "annotations/gpt56_bias.json"
            batch = json.loads(path.read_text())
            batch["reviewed_ranges"] = [[1, 2]]
            path.write_text(json.dumps(batch))
            with self.assertRaisesRegex(ValueError, "Incomplete review: 1"):
                self.run_review(root)
            self.assertFalse((root / "summary.json").exists())

    def test_completed_comparison_does_not_claim_partial_prefix_coverage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_review(root, opus_count=3)
            result = self.run_review(root)
            self.assertEqual(result["matched"]["outlet_count"], 3)
            text = (root / "REPORT.md").read_text()
            self.assertIn("Both models completed the same fixed stratified subset", text)
            self.assertNotIn("Opus stopped early", text)
            self.assertNotIn("shared prefix is still not", text)

    def test_changed_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            saved = self.make_review(root)
            Path(saved["records"][0]["source"]).write_text("{}")
            with self.assertRaisesRegex(ValueError, "Source hash mismatch"):
                self.run_review(root)

    def test_snapshot_prediction_cannot_disagree_with_frozen_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            saved = self.make_review(root)
            saved["records"][0]["predicted_label"] = "HIGH"
            (root / "snapshot.json").write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError, "Correctness mismatch"):
                self.run_review(root)

    def test_duplicate_source_decisions_and_annotations_are_rejected(self):
        for duplicate in ("record", "annotation"):
            with self.subTest(duplicate=duplicate), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                saved = self.make_review(root)
                if duplicate == "record":
                    saved["records"].append(saved["records"][0])
                    (root / "snapshot.json").write_text(json.dumps(saved))
                else:
                    (root / "annotations/duplicate.json").write_text((root / "annotations/opus5_bias.json").read_text())
                with self.assertRaisesRegex(ValueError, "Duplicate source|duplicate annotation"):
                    self.run_review(root)

    def test_invalid_category_and_out_of_range_decision_are_rejected(self):
        for decisions, message in (({"1": "made_up"}, "Invalid category"),
                                   ({"1": "asserted", "4": "asserted"}, "outside reviewed range")):
            with self.subTest(decisions=decisions), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                self.make_review(root)
                path = root / "annotations/gpt56_bias.json"
                batch = json.loads(path.read_text())
                batch["decisions"] = decisions
                path.write_text(json.dumps(batch))
                with self.assertRaisesRegex(ValueError, message):
                    self.run_review(root)

    def test_matched_indices_must_identify_the_same_outlet(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            saved = self.make_review(root)
            row = next(r for r in saved["records"] if r["id"] == "gpt56:factuality:002")
            row["media_name"] = "Different outlet"
            source = Path(row["source"])
            original = json.loads(source.read_text())
            original["media_name"] = row["media_name"]
            source.write_text(json.dumps(original))
            row["source_sha256"] = sha256_file(source)
            (root / "snapshot.json").write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError, "Matched outlet identity mismatch"):
                self.run_review(root)


if __name__ == "__main__":
    unittest.main()
