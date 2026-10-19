"""Diffusion augmentation arm (crops/flips + synthetic) and review-based synthetic selection."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import torch
from datasets import Dataset, Image as ImageFeature
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prepare_synthetic import apply_review, main as prepare_synthetic, read_pool, take_per_class
from preprocessing import eval_transform, get_dataset, get_loader
from train_classifier import parse_args

NAMES = ["001.Species_A", "002.Species_B"]


def striped_image(color):
    """Off-center stripes, so crops and flips change the tensor."""
    image = Image.new("RGB", (448, 224), "blue")
    image.paste(color, (0, 0, 100, 224))
    image.paste("green", (348, 0, 448, 224))
    return image


class DiffusionAugmentationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(temp.cleanup)
        pool = Path(temp.name)
        rows = []
        for label, name in enumerate(NAMES):
            path = pool / name / "sample.jpg"
            path.parent.mkdir(parents=True)
            striped_image("yellow").save(path)
            rows.append({"image_id": f"synthetic/pool/{name}/sample", "path": path.relative_to(pool).as_posix(),
                         "label": label, "class_name": name,
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        self.synthetic_path = pool / NAMES[0] / "sample.jpg"
        self.manifest_path = pool / "manifest.csv"
        pd.DataFrame(rows).to_csv(self.manifest_path, index=False)
        real_manifest = pd.DataFrame({"image_id": ["real/a", "real/b"], "label": [0, 1], "class_name": NAMES})
        real = Dataset.from_dict({"image_id": real_manifest.image_id.tolist(), "label": [0, 1],
                                  "image": [striped_image("red"), striped_image("red")]})
        real = real.cast_column("image", ImageFeature())
        for mock in (patch("preprocessing.load_manifest", return_value=real_manifest),
                     patch("preprocessing.load_from_disk", return_value={"train": real})):
            mock.start()
            self.addCleanup(mock.stop)

    def test_every_real_and_synthetic_image_gets_an_original_and_a_fixed_augmented_entry(self):
        dataset = get_dataset("train", augment=True, seed=0, synthetic_manifest=self.manifest_path)
        self.assertEqual(len(dataset), 8)
        ids = [dataset[i]["image_id"] for i in range(8)]
        self.assertEqual(ids[:4], ids[4:])
        self.assertEqual(ids[:2], ["real/a", "real/b"])
        self.assertTrue(all(i.startswith("synthetic/") for i in ids[2:4]))
        self.assertEqual([dataset[i]["label"] for i in range(8)], [0, 1, 0, 1] * 2)
        # Originals match the real + synthetic arm; augmented synthetic views differ from them.
        plain = get_dataset("train", synthetic_manifest=self.manifest_path)
        for i in range(4):
            self.assertTrue(torch.equal(dataset[i]["pixel_values"], plain[i]["pixel_values"]))
        with Image.open(self.synthetic_path) as image:
            center = eval_transform()(image.convert("RGB"))
        self.assertTrue(torch.equal(dataset[2]["pixel_values"], center))
        self.assertFalse(torch.equal(dataset[6]["pixel_values"], center))

    def test_augmented_synthetic_views_are_fixed_and_leave_global_rng_alone(self):
        dataset = get_dataset("train", augment=True, seed=0, synthetic_manifest=self.manifest_path)
        torch.manual_seed(0)
        state = torch.get_rng_state()
        first = dataset[6]["pixel_values"]
        self.assertTrue(torch.equal(torch.get_rng_state(), state))
        torch.manual_seed(123)
        self.assertTrue(torch.equal(dataset[6]["pixel_values"], first))
        other_seed = get_dataset("train", augment=True, seed=1, synthetic_manifest=self.manifest_path)
        views = {tuple(other_seed[i]["pixel_values"].flatten()[:50].tolist()) for i in (6, 7)}
        views |= {tuple(dataset[i]["pixel_values"].flatten()[:50].tolist()) for i in (6, 7)}
        self.assertGreater(len(views), 1)

    def test_loader_batches_are_reproducible(self):
        def batches():
            return list(get_loader("train", seed=0, augment=True, batch_size=3, num_workers=0,
                                   synthetic_manifest=self.manifest_path))
        first, second = batches(), batches()
        self.assertEqual(sum(len(b["label"]) for b in first), 8)
        for a, b in zip(first, second):
            self.assertEqual(a["image_id"], b["image_id"])
            self.assertTrue(torch.equal(a["pixel_values"], b["pixel_values"]))

    def test_cli_accepts_the_diffusion_augmentation_arm(self):
        args = parse_args(["--augment", "--synthetic-manifest", str(self.manifest_path)])
        self.assertTrue(args.augment)
        self.assertEqual(args.synthetic_manifest, self.manifest_path)


class ReviewSelectionTests(unittest.TestCase):
    def pool(self, per_class=3):
        return pd.DataFrame([{"image_id": f"synthetic/pool/{name}/{name[4:]}_{k:04d}", "label": label,
                              "class_name": name}
                             for label, name in enumerate(NAMES) for k in range(per_class)])

    def review(self, pool, decisions):
        return pd.DataFrame({"image_id": pool.image_id, "decision": decisions})

    def test_only_ok_and_artifact_images_are_kept(self):
        pool = self.pool()
        decisions = ["ok", "wrong_species", "artifact", "not_a_bird", "leak", "ok"]
        kept = apply_review(pool, self.review(pool, decisions))
        self.assertEqual(kept.image_id.tolist(), pool.image_id[[0, 2, 5]].tolist())

    def test_unreviewed_or_unknown_decisions_are_rejected(self):
        pool = self.pool()
        with self.assertRaisesRegex(ValueError, "1 pool images have no review decision"):
            apply_review(pool, self.review(pool, ["ok"] * 5 + [""]))
        with self.assertRaisesRegex(ValueError, "no review decision"):
            apply_review(pool, self.review(pool.iloc[:4], ["ok"] * 4))
        with self.assertRaisesRegex(ValueError, "Unknown review decisions"):
            apply_review(pool, self.review(pool, ["ok"] * 5 + ["maybe"]))
        repeated = self.review(pool, ["ok"] * 6)
        with self.assertRaisesRegex(ValueError, "repeats image IDs"):
            apply_review(pool, pd.concat([repeated, repeated.iloc[:1]]))

    def test_first_n_per_species_and_short_species_keep_all_remaining(self):
        pool = self.pool(per_class=4)
        kept = apply_review(pool, self.review(pool, ["wrong_species", "ok", "ok", "artifact",
                                                     "ok", "wrong_species", "not_a_bird", "leak"]))
        selected, short = take_per_class(kept.sample(frac=1, random_state=0), 2, NAMES)
        # Species A: the rejected first seed is replaced by the next accepted ones.
        self.assertEqual(selected.image_id.tolist(), pool.image_id[[1, 2, 4]].tolist())
        self.assertEqual(short, [NAMES[1]])

    def test_species_with_no_remaining_images_are_reported_as_short(self):
        pool = self.pool(per_class=2)
        for remaining in (pool.iloc[:2], pool.iloc[:0]):
            with self.subTest(images=len(remaining)):
                selected, short = take_per_class(remaining, 2, NAMES)
                self.assertEqual(selected.image_id.tolist(), remaining.image_id.tolist())
                self.assertEqual(short, NAMES[1:] if len(remaining) else NAMES)

    def test_preparation_reports_actual_counts_and_shortages_after_leakage(self):
        temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(temp.cleanup)
        pool_dir = Path(temp.name)
        pool = self.pool(per_class=4)
        pool["path"] = [str(pool_dir / f"{i}.jpg") for i in range(len(pool))]
        pool["md5"] = [f"hash-{i}" for i in range(len(pool))]
        pool["sha256"] = [f"sha-{i}" for i in range(len(pool))]
        pool.to_csv(pool_dir / "metadata.csv", index=False)
        review_path = pool_dir / "review.csv"
        self.review(pool, ["wrong_species", "ok", "ok", "artifact",
                           "ok", "ok", "not_a_bird", "leak"]).to_csv(review_path, index=False)
        names = NAMES + ["003.Species_C"]
        holdout = pd.DataFrame({"image_id": ["real/heldout"]})
        raw = Mock()
        raw.cast_column.return_value = {"image_id": ["real/heldout"],
                                        "image": [{"bytes": b"heldout", "path": None}]}
        output = pool_dir / "prepared"
        with patch("prepare_synthetic.class_names_from_manifest", return_value=names), patch(
                "prepare_synthetic.load_manifest", return_value=holdout), patch(
                "prepare_synthetic.read_pool", return_value=pool), patch(
                "prepare_synthetic.resnet18"), patch("prepare_synthetic.features"), patch(
                "prepare_synthetic.load_from_disk", return_value={"val": raw, "test": raw}), patch(
                "prepare_synthetic.get_loader"), patch(
                "prepare_synthetic.duplicate_pairs", side_effect=[[(4, 0, 1.0)], []]):
            prepare_synthetic(["--pool", str(pool_dir), "--review", str(review_path),
                               "--per-class", "2", "--device", "cpu"])
        selected = pd.read_csv(output / "manifest.csv")
        self.assertEqual(selected.image_id.tolist(), pool.image_id[[1, 2, 4]].tolist())
        report = json.loads((output / "leakage_report.json").read_text())
        self.assertEqual(report["accepted_images"], 3)
        self.assertEqual(report["excluded_images"], 1)
        self.assertEqual(report["accepted_per_class"], {"0": 2, "1": 1, "2": 0})
        self.assertEqual(report["quality_review"]["species_below_target"], names[1:])
        self.assertEqual(report["quality_review"]["species_without_synthetic"], names[2:])

    def test_reviewed_pools_may_hold_extra_seeds(self):
        temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(temp.cleanup)
        data = Path(temp.name)
        records = []
        for label, name in enumerate(NAMES):
            for k in range(1 + label):  # species B has an extra seed
                path = data / "synthetic" / "pool" / name / f"img_{k}.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (32, 32), (label * 100, k * 100, 0)).save(path)
                records.append({"image_id": path.relative_to(data).with_suffix("").as_posix(),
                                "label": label, "class_name": name,
                                "md5": hashlib.md5(path.read_bytes()).hexdigest()})
        pool = data / "synthetic" / "pool"
        pd.DataFrame(records).to_csv(pool / "metadata.csv", index=False)
        with patch("prepare_synthetic.DATA_DIR", data):
            with self.assertRaisesRegex(ValueError, "Expected 1"):
                read_pool(pool, NAMES, expected_per_class=1)
            self.assertEqual(len(read_pool(pool, NAMES, expected_per_class=None)), 3)


if __name__ == "__main__":
    unittest.main()
