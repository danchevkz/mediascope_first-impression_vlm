import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from prepare_paired_test import inspect_pairs, prepare
from mediascope_eval import load_samples


class PairedDatasetPreparationTests(unittest.TestCase):
    def source(self, root):
        source = root / "uploaded"
        (source / "masked").mkdir(parents=True)
        (source / "unmasked").mkdir()
        rows = [{"masked_image": "site_001.png", "unmasked_image": "Known_Outlet.png", "true_label_factuality": "HIGH", "true_label_bias": "LEFT"},
                {"masked_image": "site_003.png", "unmasked_image": "Another_Outlet.png", "true_label_factuality": "LOW", "true_label_bias": "RIGHT"}]
        for i, row in enumerate(rows):
            Image.new("RGB", (20, 30), (i * 50, 10, 20)).save(source / "masked" / row["masked_image"])
            Image.new("RGB", (20, 30), (i * 50, 40, 60)).save(source / "unmasked" / row["unmasked_image"])
        self.labels(source, rows)
        return source, rows

    def labels(self, source, rows):
        with (source / "labels.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_csvs_keep_pair_ids_and_absolute_source_paths_without_copying_images(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, _ = self.source(root)
            output = root / "derived"
            with contextlib.redirect_stdout(io.StringIO()):
                manifest = prepare(source, output)
            self.assertEqual(manifest["n"], 2)
            for condition in ("masked", "unmasked"):
                samples, missing = load_samples(str(output / condition / "test.csv"), str(output / condition))
                self.assertFalse(missing)
                self.assertEqual([s.media_name for s in samples], ["site_001", "site_003"])
                self.assertTrue(all(Path(s.image_path).is_absolute() for s in samples))
                self.assertFalse((output / condition / "images").exists())
            with self.assertRaises(FileExistsError):
                prepare(source, output)

    def test_rejects_missing_images_duplicate_maps_and_invalid_labels(self):
        for kind in ("missing", "duplicate", "label"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                source, rows = self.source(Path(temp))
                if kind == "missing":
                    (source / "unmasked/Known_Outlet.png").unlink()
                elif kind == "duplicate":
                    rows[1]["unmasked_image"] = rows[0]["unmasked_image"]
                    self.labels(source, rows)
                else:
                    rows[1]["true_label_bias"] = "UNRATED"
                    self.labels(source, rows)
                with self.assertRaises(ValueError):
                    inspect_pairs(source)

    def test_dry_run_makes_no_derived_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, _ = self.source(root)
            output = root / "derived"
            with contextlib.redirect_stdout(io.StringIO()):
                prepare(source, output, dry_run=True)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
