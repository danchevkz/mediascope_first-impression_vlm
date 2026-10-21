import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from review_recognition_probe_v2 import aggregate
from run_config import sha256_file


class ConservativeReferenceAggregationTests(unittest.TestCase):
    def make_review(self, root):
        (root / "annotations").mkdir()
        (root / "sources").mkdir()
        (root / "JUDGE_PROMPT.md").write_text("Test conservative rubric.")
        texts = ["I recall MBFC rating Outlet 1 High.",
                 "I vaguely recall MBFC rating Named Network Mixed.",
                 "Similar local networks are often rated right."]
        rows = []
        for task in ("factuality", "bias"):
            for index in range(1, 4):
                id = f"{task}:{index:03d}"
                a = {"recognized_outlet": f"Outlet {index}", "recognized_network": None,
                     "external_rating_source": None, "recalled_factuality_rating": None,
                     "recalled_bias_rating": None, "self_report_basis": texts[index-1] if task == "factuality" else "No recalled rating.",
                     "explanation_summary": "Visible evidence only.", "prior_rating_influence": "yes" if task == "factuality" and index in (1, 3) else "no",
                     "possible_contamination": "yes" if task == "bias" and index == 1 else "unknown"}
                source = root / "sources" / f"{task}_{index}.json"
                source.write_text(json.dumps(a))
                rows.append({"id": id, "task": task, "media_name": f"Outlet {index}", "observed": True,
                             "structured": a, "provider_summary": "Native-only recalled external rating.",
                             "correct": True, "status": "ok", "source": str(source), "source_sha256": sha256_file(source)})
        (root / "snapshot.json").write_text(json.dumps({"model": "test", "expected_per_task": 3, "records": rows}))
        decisions = {
            "1": {"category": "asserted", "rating_scope": "outlet", "family_link": "not_applicable", "evidence": [texts[0]]},
            "2": {"category": "uncertain_recall", "rating_scope": "named_family", "family_link": "tentative_or_inferred", "evidence": [texts[1]]},
            "3": {"category": "generic", "rating_scope": "generic_type", "family_link": "not_applicable", "evidence": [texts[2]]},
        }
        for task in ("factuality", "bias"):
            (root / "annotations" / f"{task}.json").write_text(json.dumps({"task": task, "reviewed_ranges": [[1, 3]],
                                                                        "decisions": decisions if task == "factuality" else {}}))
        return rows

    def run_review(self, root):
        with contextlib.redirect_stdout(io.StringIO()):
            return aggregate(root)

    def test_keeps_reference_confidence_scope_and_model_admissions_distinct(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_review(root)
            report = self.run_review(root)
            fact = report["tasks"]["factuality"]
            self.assertEqual(fact["references"], 2)
            self.assertEqual(fact["reference_rate"], 2 / 3)
            self.assertEqual(fact["asserted_only_rate"], 1 / 3)
            self.assertEqual(fact["outlet_references"], 1)
            self.assertEqual(fact["family_references"], 1)
            self.assertEqual(fact["tentative_family_references"], 1)
            self.assertEqual(fact["categories"]["generic"], 1)
            self.assertEqual(fact["correct_reference_predictions"], 2)
            self.assertEqual(fact["accuracy_among_references"], 1)
            self.assertEqual(fact["prior_rating_influence"]["yes"], 2)
            bias = report["tasks"]["bias"]
            self.assertEqual(bias["references"], 0)
            self.assertEqual(bias["possible_contamination"]["yes"], 1)
            self.assertEqual(bias["provider_reasoning_available"], 3)

    def test_unreviewed_cases_are_not_assumed_negative(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_review(root)
            path = root / "annotations/bias.json"
            batch = json.loads(path.read_text())
            batch["reviewed_ranges"] = [[1, 2]]
            path.write_text(json.dumps(batch))
            with self.assertRaisesRegex(ValueError, "Incomplete manual review"):
                self.run_review(root)

    def test_invented_evidence_and_generic_primary_attribution_are_rejected(self):
        for mutation, message in (({"evidence": ["Not in the saved response"]}, "exact saved excerpt"),
                                  ({"rating_scope": "generic_type"}, "no anchored attribution")):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                self.make_review(root)
                path = root / "annotations/factuality.json"
                batch = json.loads(path.read_text())
                batch["decisions"]["1"].update(mutation)
                path.write_text(json.dumps(batch))
                with self.assertRaisesRegex(ValueError, message):
                    self.run_review(root)

    def test_source_hash_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = self.make_review(root)
            Path(rows[0]["source"]).write_text("Changed response")
            with self.assertRaisesRegex(ValueError, "Saved response changed"):
                self.run_review(root)

    def test_unobserved_reports_are_excluded_from_reference_denominator(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = self.make_review(root)
            rows[-1].update(observed=False, structured=None, status="content_filtered")
            (root / "snapshot.json").write_text(json.dumps({"model": "test", "expected_per_task": 3, "records": rows}))
            report = self.run_review(root)
            self.assertEqual(report["tasks"]["bias"]["recorded"], 3)
            self.assertEqual(report["tasks"]["bias"]["valid_reports"], 2)
            self.assertEqual(report["tasks"]["bias"]["reference_rate"], 0)
            self.assertEqual(report["tasks"]["bias"]["overall_accuracy"], 2 / 3)
            self.assertIsNone(report["tasks"]["bias"]["accuracy_among_references"])
            self.assertNotIn("unobserved", report["tasks"]["bias"]["categories"])


if __name__ == "__main__":
    unittest.main()
