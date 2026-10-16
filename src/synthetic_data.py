"""Validated synthetic manifests and lazy image loading for mixed training."""
import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


def load_synthetic_manifest(path, class_names):
    """Read a prepared manifest; paths are relative to the manifest's directory.

    Validate labels and content hashes before training, so files cannot silently change
    after the leakage check. Contact sheets are never discovered as training images.
    """
    path = Path(path).resolve()
    manifest = pd.read_csv(path)
    required = {"image_id", "path", "label", "class_name", "sha256"}
    if not required.issubset(manifest.columns) or manifest.empty:
        raise ValueError(f"Synthetic manifest must contain nonempty columns {sorted(required)}")
    if manifest[list(required)].isna().any().any():
        raise ValueError("Synthetic manifest contains missing values")
    if manifest["image_id"].duplicated().any():
        raise ValueError("Synthetic image IDs must be unique")
    resolved = []
    for row in manifest.itertuples(index=False):
        if (row.label not in range(len(class_names))
                or class_names[int(row.label)] != row.class_name):
            raise ValueError(f"Synthetic label/species mismatch: {row.image_id}")
        image_path = (path.parent / row.path).resolve()
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != row.sha256:
            raise ValueError(f"Synthetic content changed since preparation: {row.image_id}")
        resolved.append(str(image_path))
    if len(set(resolved)) != len(resolved):
        raise ValueError("Synthetic manifest repeats image paths")
    manifest["path"] = resolved
    manifest["label"] = manifest["label"].astype(int)
    return manifest


class SyntheticImages(Dataset):
    """Return the same pixel_values/label dictionaries as the real-image loader."""

    def __init__(self, manifest, transform):
        self.rows = manifest.to_dict("records")
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["path"]) as image:
            pixels = self.transform(image.convert("RGB"))
        return {"pixel_values": pixels, "label": int(row["label"])}
