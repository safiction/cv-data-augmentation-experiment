"""Smoke orchestration must keep paired settings and report real, correctly linked mistakes."""
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from datasets import Dataset, Image as ImageFeature
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smoke_experiment import SETUPS, main, parse_args, run_script, write_report


class SmokeExperimentTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        dependencies = patch("smoke_experiment.find_spec", return_value=object())
        dependencies.start()
        self.addCleanup(dependencies.stop)

    def test_paired_training_finishes_before_test_and_only_intended_data_is_added(self):
        calls = []
        output = self.root / "run"
        review = self.root / "review.csv"
        review.touch()
        with patch("smoke_experiment.PROCESSED_DATA_DIR", self.root), \
                patch("torch.cuda.is_available", return_value=False), \
                patch("smoke_experiment.run_script", side_effect=lambda s, a: calls.append((s, a))), \
                patch("smoke_experiment.write_report") as report:
            main(["--output", str(output), "--device", "cpu", "--allow-cpu-generation",
                  "--seed", "2", "--k-shot", "5", "--per-class", "3", "--max-steps", "7",
                  "--eval-every", "3", "--pool", "test_pool", "--review", str(review)])
        self.assertEqual([script for script, _ in calls], [
            "generate.py", "prepare_synthetic.py", *(["train_classifier.py"] * 3),
            *(["evaluate_classifier.py"] * 3)])
        self.assertIn("--allow-cpu", calls[0][1])
        self.assertEqual(calls[0][1][calls[0][1].index("--per-class") + 1], 3)
        self.assertEqual(calls[1][1][calls[1][1].index("--review") + 1], review.resolve())
        for setup, (_, arguments) in zip(SETUPS, calls[2:5]):
            for flag, expected in (("--seed", 2), ("--k-shot", 5), ("--max-steps", 7),
                                   ("--eval-every", 3)):
                self.assertEqual(arguments[arguments.index(flag) + 1], expected)
            self.assertEqual("--augment" in arguments, setup == "real_standard_augmented")
            if setup == "real_standard_augmented":
                self.assertEqual(arguments[arguments.index("--augment-per-class") + 1], 1)
            else:
                self.assertNotIn("--augment-per-class", arguments)
            self.assertEqual("--synthetic-manifest" in arguments, setup == "real_generated")
            self.assertNotIn("--split", arguments)
        for setup, (_, arguments) in zip(SETUPS, calls[5:]):
            self.assertEqual(arguments[0], output / setup / "train" / "best.pt")
            self.assertEqual(arguments[arguments.index("--split") + 1], "test")
        settings = json.loads((output / "smoke_config.json").read_text(encoding="utf-8"))
        self.assertEqual(settings["pool"], "test_pool")
        self.assertEqual(settings["setups"], list(SETUPS))
        report.assert_called_once_with(output)

    def test_stage_failure_propagates_without_evaluating_test(self):
        with patch("smoke_experiment.PROCESSED_DATA_DIR", self.root), \
                patch("torch.cuda.is_available", return_value=False), \
                patch("smoke_experiment.run_script", side_effect=subprocess.CalledProcessError(1, "generate")) as run, \
                patch("smoke_experiment.write_report") as report:
            with self.assertRaises(subprocess.CalledProcessError):
                main(["--output", str(self.root / "failed"), "--device", "cpu", "--allow-cpu-generation"])
        self.assertEqual(run.call_count, 1)
        report.assert_not_called()

    def test_missing_generation_dependencies_fail_before_creating_output(self):
        output = self.root / "missing_dependencies"
        with patch("smoke_experiment.PROCESSED_DATA_DIR", self.root), \
                patch("smoke_experiment.find_spec", return_value=None), \
                patch("smoke_experiment.run_script") as run:
            with self.assertRaisesRegex(ModuleNotFoundError, "diffusers, transformers, accelerate"):
                main(["--output", str(output)])
        self.assertFalse(output.exists())
        run.assert_not_called()

    def test_subprocess_uses_current_python_project_root_and_fails_fast(self):
        with patch("smoke_experiment.subprocess.run") as run, contextlib.redirect_stdout(io.StringIO()):
            run_script("train_classifier.py", ["--max-steps", 2])
        args, kwargs = run.call_args
        self.assertEqual(args[0], [sys.executable, str(ROOT / "src" / "train_classifier.py"),
                                  "--max-steps", "2"])
        self.assertEqual(kwargs, {"cwd": ROOT, "check": True})

    def test_existing_output_and_invalid_generation_inputs_are_rejected(self):
        with patch("smoke_experiment.run_script") as run:
            with self.assertRaises(FileExistsError):
                main(["--output", str(self.root)])
        run.assert_not_called()
        for arguments in (["--per-class", "0"], ["--generation-steps", "-1"], ["--pool", "../pool"]):
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                parse_args(arguments)

    def test_report_exports_up_to_five_actual_wrong_images_and_handles_perfect_predictions(self):
        output = self.root / "report with spaces"
        ids = [f"validation/species/photo_{i}" for i in range(8)]
        images = [Image.new("RGB", (12, 10), (i * 20, 40, 90)) for i in range(8)]
        # Dataset order deliberately differs from prediction order: joining by position is wrong.
        test = Dataset.from_dict({"image_id": ids[::-1], "label": [0] * 8,
                                  "image": images[::-1]}).cast_column("image", ImageFeature())
        for setup, mistakes in zip(SETUPS, (6, 2, 0)):
            train_dir, test_dir = output / setup / "train", output / setup / "test"
            train_dir.mkdir(parents=True)
            test_dir.mkdir()
            config = {"train_images": 12, "real_train_images": 8, "synthetic_train_images": 4,
                      "augmented_train_images": 0}
            (train_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
            metrics = {"loss": 1.25, "accuracy": (8 - mistakes) / 8, "macro_f1": 0.1, "images": 8}
            (test_dir / "test_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
            pd.DataFrame({"image_id": ids, "label": [0] * 8, "class_name": ["species_0"] * 8,
                          "prediction": [1] * mistakes + [0] * (8 - mistakes),
                          "predicted_class_name": ["species_1"] * mistakes + ["species_0"] * (8 - mistakes)
                          }).to_csv(test_dir / "test_predictions.csv", index=False)

        with patch("smoke_experiment.load_from_disk", return_value={"test": test}), \
                contextlib.redirect_stdout(io.StringIO()) as console:
            write_report(output)
        summary = pd.read_csv(output / "metrics_summary.csv")
        self.assertEqual(summary.setup.tolist(), list(SETUPS))
        self.assertEqual(summary.accuracy.tolist(), [0.25, 0.75, 1.0])
        markdown = (output / "summary.md").read_text(encoding="utf-8")
        for setup, count in zip(SETUPS, (5, 2, 0)):
            wrong = pd.read_csv(output / setup / "test" / "wrong_predictions.csv")
            self.assertEqual(len(wrong), count)
            self.assertTrue((wrong.label != wrong.prediction).all())
            for rank, row in enumerate(wrong.itertuples(index=False)):
                path = output / row.image_path
                self.assertTrue(path.is_file())
                self.assertEqual(row.image_url, path.resolve().as_uri())
                self.assertIn(f"]({row.image_path})", markdown)
                self.assertIn(row.image_url, console.getvalue())
                with Image.open(path) as image:
                    self.assertEqual(image.getpixel((0, 0)), images[rank].getpixel((0, 0)))
        self.assertIn("Showing 0 of 0 wrong predictions", markdown)


if __name__ == "__main__":
    unittest.main()
