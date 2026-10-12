"""Regression checks for freezing, split metrics and best-checkpoint selection."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from classifier import BirdClassifier, evaluate, load_checkpoint, save_evaluation
from train_classifier import parse_args, train_loop


class ClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_training_changes_only_head_and_keeps_batchnorm_fixed(self):
        model = BirdClassifier(pretrained=False)
        before = {key: value.clone() for key, value in model.backbone.state_dict().items()}
        head_before = model.head.weight.detach().clone()
        model.train()
        self.assertTrue(model.head.training)
        self.assertFalse(model.backbone.training)
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 102600)
        optimizer = torch.optim.AdamW(model.head.parameters(), lr=0.01)
        loss = nn.CrossEntropyLoss()(model(torch.randn(2, 3, 64, 64)), torch.tensor([0, 1]))
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.equal(head_before, model.head.weight))
        for key, value in model.backbone.state_dict().items():
            self.assertTrue(torch.equal(before[key], value), key)
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))

    def test_metrics_cover_all_classes_and_weight_partial_batch_correctly(self):
        logits = torch.zeros(4, 200)
        logits[range(4), [0, 1, 1, 1]] = torch.tensor([4.0, 3.0, 2.0, 1.0])
        truth = torch.tensor([0, 1, 0, 1])

        class FixedPredictions(nn.Module):
            def forward(self, indices):
                return logits[indices.long()]

        rows = [{"pixel_values": torch.tensor(i), "label": truth[i]} for i in range(4)]
        result, actual, predictions = evaluate(
            FixedPredictions(), DataLoader(rows, batch_size=3), torch.device("cpu"))
        self.assertEqual(actual, truth.tolist())
        self.assertEqual(predictions, [0, 1, 1, 1])
        self.assertAlmostEqual(result["accuracy"], 0.75)
        self.assertAlmostEqual(result["macro_f1"], (2 / 3 + 4 / 5) / 200)
        self.assertAlmostEqual(result["loss"], nn.CrossEntropyLoss()(logits, truth).item(), places=6)
        with self.assertRaises(ValueError):
            evaluate(FixedPredictions(), [], torch.device("cpu"))

    def test_best_checkpoint_is_not_the_last_step_and_reloads_without_download(self):
        model = BirdClassifier(pretrained=False)
        rows = [{"pixel_values": torch.randn(3, 64, 64), "label": torch.tensor(i)}
                for i in range(2)]
        loader = DataLoader(rows, batch_size=2)
        config = {"lr": 0.01, "weight_decay": 0.0, "amp": False,
                  "max_steps": 2, "eval_every": 1, "log_every": 1,
                  "class_names": [f"species_{i}" for i in range(200)]}
        metrics = [{"loss": 2.0, "accuracy": 0.5, "macro_f1": score} for score in (0.6, 0.4)]
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output = Path(directory)
            with patch("train_classifier.evaluate", side_effect=[(m, [], []) for m in metrics]):
                runtime = train_loop(model, loader, loader, config, output, torch.device("cpu"))
            self.assertEqual(runtime["best_step"], 1)
            # Any attempt to download pretrained weights during reload fails this test.
            from torchvision.models import resnet18 as original_resnet18
            def offline_resnet18(*, weights):
                self.assertIsNone(weights)
                return original_resnet18(weights=weights)
            with patch("classifier.resnet18", side_effect=offline_resnet18):
                restored, checkpoint = load_checkpoint(output / "best.pt", torch.device("cpu"))
            self.assertEqual(checkpoint["step"], 1)
            self.assertFalse(torch.equal(restored.head.weight, model.head.weight))
            restored.eval()
            with torch.no_grad():
                expected = restored(rows[0]["pixel_values"].unsqueeze(0))
                again, _ = load_checkpoint(output / "best.pt", torch.device("cpu"))
                self.assertTrue(torch.equal(expected, again(rows[0]["pixel_values"].unsqueeze(0))))
            history = pd.read_csv(output / "history.csv")
            self.assertEqual(history["step"].tolist(), [1, 2])

    def test_export_preserves_ids_and_rejects_misaligned_rows(self):
        manifest = pd.DataFrame({"image_id": ["a", "b"], "label": [0, 1],
                                 "class_name": ["species_0", "species_1"]})
        names = [f"species_{i}" for i in range(200)]
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output = Path(directory)
            save_evaluation(output, "val", manifest, names, ({}, [0, 1], [1, 1]))
            table = pd.read_csv(output / "val_predictions.csv")
            self.assertEqual(table["image_id"].tolist(), ["a", "b"])
            self.assertEqual(table["predicted_class_name"].tolist(), ["species_1", "species_1"])
            with self.assertRaises(ValueError):
                save_evaluation(output, "val", manifest, names, ({}, [1, 0], [1, 1]))

    def test_nonfinite_hyperparameters_are_rejected(self):
        for arguments in (["--lr", "nan"], ["--weight-decay", "inf"], ["--max-steps", "0"]):
            with self.subTest(arguments=arguments), self.assertRaises(SystemExit):
                parse_args(arguments)


if __name__ == "__main__":
    unittest.main()
