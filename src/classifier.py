"""Frozen ImageNet ResNet-18 and shared classification/evaluation helpers."""
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

NUM_CLASSES = 200


class BirdClassifier(nn.Module):
    """Train a linear head while keeping backbone weights and BatchNorm fixed."""

    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)
        features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.backbone.requires_grad_(False)
        self.head = nn.Linear(features, NUM_CLASSES)
        self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        # requires_grad=False does not freeze BatchNorm running statistics.
        self.backbone.eval()
        return self

    def forward(self, images):
        # Only the head needs an autograd graph in this fixed-feature experiment.
        with torch.no_grad():
            features = self.backbone(images)
        return self.head(features)


def seed_everything(seed):
    """Seed model initialization and augmentation; loaders seed their own shuffle."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Supported devices: auto, cpu, cuda, cuda:<index>")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; use --device cpu or install CUDA PyTorch")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {device.index} does not exist")
    return device


def class_names_from_manifest(manifest):
    """Validate the canonical numeric label-to-species mapping."""
    mapping = manifest[["label", "class_name"]].drop_duplicates().sort_values("label")
    if mapping["label"].tolist() != list(range(NUM_CLASSES)):
        raise ValueError("Manifest must map each of labels 0..199 to exactly one species")
    if mapping["class_name"].nunique() != NUM_CLASSES:
        raise ValueError("Species names must be unique")
    if manifest["image_id"].duplicated().any():
        raise ValueError("Manifest contains repeated image IDs")
    return mapping["class_name"].tolist()


@torch.no_grad()
def evaluate(model, loader, device):
    """Compute metrics once over the entire split, including the last partial batch."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss_sum = 0.0
    truth, predictions = [], []
    for batch in loader:
        images = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        logits = model(images)
        loss_sum += criterion(logits, labels).item()
        truth.extend(labels.cpu().tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
    if not truth:
        raise ValueError("Cannot evaluate an empty dataset")
    metrics = {
        "loss": loss_sum / len(truth),
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(
            truth, predictions, labels=list(range(NUM_CLASSES)),
            average="macro", zero_division=0)),
        "images": len(truth),
    }
    return metrics, truth, predictions


def save_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_evaluation(output, split, manifest, names, result):
    """Attach predictions to manifest IDs in the unshuffled evaluation order."""
    metrics, truth, predictions = result
    if manifest["label"].tolist() != truth or len(predictions) != len(manifest):
        raise ValueError("Evaluation rows do not match the manifest order")
    output = Path(output)
    table = manifest.copy()
    table["prediction"] = predictions
    table["predicted_class_name"] = [names[label] for label in predictions]
    table.to_csv(output / f"{split}_predictions.csv", index=False)
    save_json(output / f"{split}_metrics.json", metrics)
    save_json(output / f"{split}_per_class.json", classification_report(
        truth, predictions, labels=list(range(NUM_CLASSES)), target_names=names,
        output_dict=True, zero_division=0))


def load_checkpoint(path, device):
    """Restore the entire network without downloading ImageNet weights again."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported classifier checkpoint format")
    names = checkpoint["config"]["class_names"]
    if len(names) != NUM_CLASSES or len(set(names)) != NUM_CLASSES:
        raise ValueError("Checkpoint contains an invalid class mapping")
    model = BirdClassifier(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint
