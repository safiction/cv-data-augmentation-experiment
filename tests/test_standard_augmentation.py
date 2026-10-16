"""Original plus augmented pools retain one deterministic entry of each type per image."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
from datasets import Dataset, Image as ImageFeature
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from preprocessing import get_dataset, get_loader


class StandardAugmentationTests(unittest.TestCase):
    def setUp(self):
        image = Image.new("RGB", (448, 224), "blue")
        image.paste("red", (0, 0, 100, 224))
        image.paste("green", (348, 0, 448, 224))
        self.manifest = pd.DataFrame({"image_id": ["real/a", "real/b"], "label": [0, 1],
                                      "class_name": ["001.A", "002.B"]})
        self.real = Dataset.from_dict({"image_id": self.manifest.image_id.tolist(),
                                       "label": [0, 1], "image": [image, image]})
        self.real = self.real.cast_column("image", ImageFeature())
        self.patches = [patch("preprocessing.load_manifest", return_value=self.manifest),
                        patch("preprocessing.load_from_disk", return_value={"train": self.real})]
        for mock in self.patches:
            mock.start()
            self.addCleanup(mock.stop)

    def test_pool_contains_original_and_augmented_entry_for_every_real_image(self):
        dataset = get_dataset("train", augment=True, seed=0)
        self.assertEqual(len(dataset), 2 * len(self.real))
        self.assertEqual([dataset[i]["label"] for i in range(4)], [0, 1, 0, 1])
        baseline = get_dataset("train")
        for i in range(2):
            original = dataset[i]["pixel_values"]
            augmented = dataset[i + 2]["pixel_values"]
            self.assertEqual(tuple(original.shape), (3, 224, 224))
            self.assertEqual(tuple(augmented.shape), (3, 224, 224))
            self.assertTrue(torch.equal(original, baseline[i]["pixel_values"]))
            # The old center crop removes the red/green borders of this wide image.
            self.assertGreater(original[2, 112, 0].item(), original[0, 112, 0].item())
            self.assertGreater(original[2, 112, -1].item(), original[1, 112, -1].item())
            self.assertFalse(torch.equal(original, augmented))
        rows = list(get_loader("train", augment=True, seed=0, batch_size=3, num_workers=0))
        self.assertEqual(sorted(torch.cat([b["label"] for b in rows]).tolist()), [0, 0, 1, 1])

    def test_augmented_variants_are_fixed_and_preserve_the_callers_rng(self):
        dataset = get_dataset("train", augment=True, seed=0)
        torch.manual_seed(17)
        rng_before = torch.get_rng_state().clone()
        first = dataset[2]["pixel_values"]
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
        torch.manual_seed(999)
        again = dataset[2]["pixel_values"]
        self.assertTrue(torch.equal(first, again))
        reordered = self.real.select([1, 0])
        with patch("preprocessing.load_from_disk", return_value={"train": reordered}):
            rebuilt = get_dataset("train", augment=True, seed=0)
        self.assertTrue(torch.equal(first, rebuilt[2]["pixel_values"]))

    def test_full_image_control_retains_only_originals_and_preserves_edges(self):
        control = get_dataset("train", full_image=True)
        comparison = get_dataset("train", augment=True)
        self.assertEqual(len(control), len(self.real))
        self.assertEqual([control[i]["label"] for i in range(2)], [0, 1])
        for i in range(2):
            pixels = control[i]["pixel_values"]
            self.assertFalse(torch.equal(pixels, comparison[i]["pixel_values"]))
            self.assertGreater(pixels[0, 112, 0].item(), pixels[2, 112, 0].item())
            self.assertGreater(pixels[1, 112, -1].item(), pixels[0, 112, -1].item())
        for split in ("val", "test"):
            with self.assertRaisesRegex(ValueError, "train split only"):
                get_dataset(split, full_image=True)
        with self.assertRaisesRegex(ValueError, "real-only control"):
            get_dataset("train", full_image=True, augment=True)


if __name__ == "__main__":
    unittest.main()
