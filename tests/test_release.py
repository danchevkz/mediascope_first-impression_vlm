"""Release checks exercise portability and corrupted provenance, without inference."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import paper
import reproduce_release as release


class ReleaseTests(unittest.TestCase):
    def test_commands_bind_recorded_training_settings_after_relocation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            configs = root / "configs/training"
            configs.mkdir(parents=True)
            (configs / "qwen35_factuality_main.json").write_text(json.dumps({
                "model_revision": "immutable-revision", "settings": {"arguments": {
                    "lr": 5e-5, "max_tiles": 6, "thinking": "off", "require_full_image": True,
                    "warmup_steps": 16, "grad_accum": 16,
                }}}))
            args = argparse.Namespace(action="train", model="qwen35", task="factuality",
                                      images=root / "external images", output=root / "new training")
            with patch.object(paper, "ROOT", root):
                command = paper.command(args)
            self.assertEqual(command[command.index("--lr") + 1], "5e-05")
            self.assertEqual(command[command.index("--max-tiles") + 1], "6")
            self.assertIn("--require-full-image", command)
            self.assertIn(str(root / "external images"), command)
            self.assertIn(str(root / "data/main/train.csv"), command)

    def test_main_inference_selects_its_recorded_adapter_and_token_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "configs").mkdir()
            (root / "configs/local_evaluation.json").write_text(json.dumps({"main": {"qwen35_finetuned": {
                "factuality": {"adapter_key": "qwen35_factuality_main", "quant": "4bit", "thinking": "off",
                               "model_revision": "pinned", "generation": {"max_new_tokens": 64},
                               "preprocessing": {"max_tiles": 6, "max_width": 1024, "tile_height": 1400}}}}}))
            args = argparse.Namespace(action="local", model="qwen35", variant="finetuned", task="factuality",
                                      dataset="main", images=root / "images", output=root / "predictions.csv",
                                      adapters=root / "adapters", limit=None, resume=False)
            with patch.object(paper, "ROOT", root):
                command = paper.command(args)
            self.assertEqual(command[command.index("--max-new-tokens") + 1], "64")
            self.assertEqual(command[command.index("--adapter-dir") + 1], str(root / "adapters/qwen35_factuality_main"))

    def test_reference_evidence_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit = root / "results/reference_audits/kimi_v2"
            (audit / "annotations").mkdir(parents=True)
            (audit / "snapshot.json").write_text(json.dumps({"records": [{
                "id": "bias:001", "observed": True, "structured": {"self_report_basis": "No recalled rating"},
            }]}))
            (audit / "annotations/batch.json").write_text(json.dumps({"task": "bias", "reviewed_ranges": [[1, 1]],
                "decisions": {"1": {"category": "asserted", "evidence": ["Invented rating"]}}}))
            with self.assertRaisesRegex(ValueError, "Evidence not found"):
                release.reference_annotations(root)

    def test_unreviewed_valid_report_is_not_treated_as_none(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit = root / "results/reference_audits/kimi_v2"
            (audit / "annotations").mkdir(parents=True)
            (audit / "snapshot.json").write_text(json.dumps({"records": [{
                "id": "bias:001", "observed": True, "structured": {},
            }]}))
            with self.assertRaisesRegex(ValueError, "Unreviewed valid report"):
                release.reference_annotations(root)

    def test_release_checksum_detects_changed_prediction_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "prediction.json").write_text("changed")
            (root / "release_manifest.json").write_text(json.dumps({"files": {
                "prediction.json": {"sha256": "0" * 64}}}))
            with self.assertRaisesRegex(ValueError, "Changed release file"):
                release.verify_manifest(root)

    def test_masked_api_uses_masked_capable_runner_and_paper_model(self):
        args = argparse.Namespace(action="api", model="opus5", dataset="masked",
                                  inputs=Path("prepared inputs"), output=Path("run output"), resume=False)
        command = paper.command(args)
        self.assertTrue(command[1].endswith("eval_api.py"))
        self.assertEqual(command[command.index("--models") + 1], "opus5")
        self.assertIn(str(Path("prepared inputs").resolve()), command)


if __name__ == "__main__":
    unittest.main()
