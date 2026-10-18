"""Validated synthetic manifests and lazy image loading for mixed training."""
import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

# Review decisions (docs/generation.md). Accepted images have the right species label; defects
# alone do not drop an image.
DECISIONS = ("ok", "wrong_species", "artifact", "not_a_bird", "leak")
ACCEPTED = ("ok", "artifact")


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

    @property
    def image_ids(self):
        return [row["image_id"] for row in self.rows]

    def load(self, index):
        """Untransformed RGB image, label and ID."""
        row = self.rows[index]
        with Image.open(row["path"]) as image:
            return image.convert("RGB"), int(row["label"]), row["image_id"]

    def __getitem__(self, index):
        image, label, image_id = self.load(index)
        return {"pixel_values": self.transform(image), "label": label, "image_id": image_id}
