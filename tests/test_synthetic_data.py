"""Synthetic label integrity, changed-file rejection, and train-only mixing."""
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
from datasets import Dataset, Image as ImageFeature
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prepare_synthetic import read_pool
from preprocessing import eval_transform, get_dataset, get_loader
from synthetic_data import load_synthetic_manifest
from train_classifier import parse_args


class SyntheticDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.data = Path(self.temp.name)
        self.pool = self.data / "synthetic" / "pool"
        self.names = ["001.Species_A", "002.Species_B"]
        records = []
        for label, name in enumerate(self.names):
            path = self.pool / name / "sample.jpg"
            path.parent.mkdir(parents=True)
            Image.new("RGB", (256, 256), (label * 255, 80, 20)).save(path)
            records.append({"image_id": path.relative_to(self.data).with_suffix("").as_posix(),
                            "label": label, "class_name": name,
                            "md5": hashlib.md5(path.read_bytes()).hexdigest()})
        self.metadata = pd.DataFrame(records)
        self.metadata.to_csv(self.pool / "metadata.csv", index=False)

    def prepare(self):
        with patch("prepare_synthetic.DATA_DIR", self.data):
            rows = read_pool(self.pool, self.names, expected_per_class=1)
        # Contact sheets and unrelated images must not enter the manifest.
        self.assertEqual(len(rows), 2)
        rows["path"] = [str(Path(p).relative_to(self.pool)) for p in rows.path]
        path = self.pool / "manifest.csv"
        rows.to_csv(path, index=False)
        return path

    def test_metadata_excludes_contact_sheets_and_hashes_protect_images(self):
        sheets = self.pool / "sheets"
        sheets.mkdir()
        Image.new("RGB", (8, 8)).save(sheets / "contact.jpg")
        path = self.prepare()
        rows = load_synthetic_manifest(path, self.names)
        self.assertEqual(rows.label.tolist(), [0, 1])
        Path(rows.iloc[0].path).write_bytes(b"changed image")
        with self.assertRaisesRegex(ValueError, "changed since preparation"):
            load_synthetic_manifest(path, self.names)
        with patch("prepare_synthetic.DATA_DIR", self.data):
            with self.assertRaisesRegex(ValueError, "MD5 mismatch"):
                read_pool(self.pool, self.names, expected_per_class=1)

    def test_wrong_labels_missing_species_and_duplicate_ids_are_rejected(self):
        path = self.prepare()
        manifest = pd.read_csv(path)
        manifest.loc[0, "label"] = 1
        manifest.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "label/species mismatch"):
            load_synthetic_manifest(path, self.names)
        self.metadata.iloc[:1].to_csv(self.pool / "metadata.csv", index=False)
        with patch("prepare_synthetic.DATA_DIR", self.data):
            with self.assertRaisesRegex(ValueError, "Expected 1"):
                read_pool(self.pool, self.names, expected_per_class=1)
        repeated = pd.concat([self.metadata, self.metadata.iloc[:1]])
        repeated.to_csv(self.pool / "metadata.csv", index=False)
        with patch("prepare_synthetic.DATA_DIR", self.data):
            with self.assertRaisesRegex(ValueError, "repeated image IDs"):
                read_pool(self.pool, self.names, expected_per_class=1)

    def test_real_and_synthetic_batches_mix_with_canonical_labels_and_seed(self):
        path = self.prepare()
        real_manifest = pd.DataFrame({"image_id": ["real/a", "real/b"], "label": [0, 1],
                                      "class_name": self.names})
        real = Dataset.from_dict({"image_id": real_manifest.image_id.tolist(), "label": [0, 1],
                                  "image": [Image.new("L", (256, 256), 120),
                                            Image.new("RGB", (256, 256), "blue")]})
        real = real.cast_column("image", ImageFeature())
        with patch("preprocessing.load_manifest", return_value=real_manifest), patch(
                "preprocessing.load_from_disk", return_value={"train": real}):
            def batches():
                torch.manual_seed(0)
                return list(get_loader("train", seed=0, batch_size=3,
                                       num_workers=0, synthetic_manifest=path))
            first, second = batches(), batches()
            self.assertEqual(sum(len(batch["label"]) for batch in first), 4)
            self.assertEqual(sorted(torch.cat([b["label"] for b in first]).tolist()), [0, 0, 1, 1])
            for a, b in zip(first, second):
                self.assertEqual(tuple(a["pixel_values"].shape[1:]), (3, 224, 224))
                self.assertTrue(torch.equal(a["pixel_values"], b["pixel_values"]))
                self.assertTrue(torch.equal(a["label"], b["label"]))
        for split in ("val", "test"):
            with self.assertRaisesRegex(ValueError, "train split only"):
                get_dataset(split, synthetic_manifest=path)

    def test_mixed_training_center_crops_both_sources_deterministically_without_flipping(self):
        path = self.prepare()
        # Center stripes expose both the crop (blue borders disappear) and mirroring.
        image = Image.new("RGB", (448, 224), "blue")
        image.paste("red", (120, 0, 224, 224))
        image.paste("green", (224, 0, 328, 224))
        synthetic_path = self.pool / self.names[0] / "sample.jpg"
        image.save(synthetic_path)
        self.metadata.loc[0, "md5"] = hashlib.md5(synthetic_path.read_bytes()).hexdigest()
        self.metadata.to_csv(self.pool / "metadata.csv", index=False)
        path = self.prepare()
        real_manifest = pd.DataFrame({"image_id": ["real/a", "real/b"], "label": [0, 1],
                                      "class_name": self.names})
        real = Dataset.from_dict({"image_id": real_manifest.image_id.tolist(), "label": [0, 1],
                                  "image": [image, image]}).cast_column("image", ImageFeature())
        with patch("preprocessing.load_manifest", return_value=real_manifest), patch(
                "preprocessing.load_from_disk", return_value={"train": real}):
            dataset = get_dataset("train", synthetic_manifest=path)
            baseline = get_dataset("train")
            self.assertTrue(torch.equal(dataset[0]["pixel_values"], baseline[0]["pixel_values"]))
            with Image.open(synthetic_path) as generated:
                expected = eval_transform()(generated.convert("RGB"))
            self.assertTrue(torch.equal(dataset[2]["pixel_values"], expected))
            for index in (0, 2):  # Real image and corresponding synthetic image.
                torch.manual_seed(0)
                first = dataset[index]["pixel_values"]
                torch.manual_seed(123)
                second = dataset[index]["pixel_values"]
                self.assertTrue(torch.equal(first, second))
                self.assertGreater(first[0, 112, 0].item(), first[2, 112, 0].item())
                self.assertGreater(first[1, 112, -1].item(), first[0, 112, -1].item())

    def test_combining_standard_augmentation_and_synthetic_data_is_rejected(self):
        with self.assertRaises(SystemExit):
            parse_args(["--augment", "--synthetic-manifest", "manifest.csv"])
        with self.assertRaisesRegex(ValueError, "no random crops/flips"):
            get_dataset("train", augment=True, synthetic_manifest="manifest.csv")


if __name__ == "__main__":
    unittest.main()
