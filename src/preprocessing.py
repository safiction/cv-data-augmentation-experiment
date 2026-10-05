"""Online preprocessing and data loaders shared by every experiment.

Images are converted to RGB (the dataset has 8 grayscale "L" images) and normalized with fixed
ImageNet statistics, because the classifier is an ImageNet-pretrained ResNet-18. The statistics
are not estimated from our data, so normalization cannot leak val/test information.

Which images a run uses comes from the manifests in splits/manifests (see make_manifests.py).
"""
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from torchvision import transforms as T

from load_data import PROCESSED_DATA_DIR, SUBSET_SEEDS, load_manifest

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
RESIZE_SIZE = 256
CROP_SIZE = 224
# Smallest crop area for augmentation. torchvision's default (0.08) often cuts the bird out of the
# frame, which turns a fine-grained sample into a mislabeled background patch.
MIN_CROP_SCALE = 0.5


def eval_transform():
    """Deterministic: shorter side to 256, center crop 224. Used for val/test and no-aug training."""
    return T.Compose([
        T.Resize(RESIZE_SIZE),
        T.CenterCrop(CROP_SIZE),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def train_aug_transform():
    """Standard augmentation arm: random resized crop and horizontal flip."""
    return T.Compose([
        T.RandomResizedCrop(CROP_SIZE, scale=(MIN_CROP_SCALE, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def to_rgb(img):
    """Grayscale, palette and RGBA images become 3-channel RGB; RGB images pass through."""
    return img if img.mode == "RGB" else img.convert("RGB")


class ApplyTransform:
    """Batch transform for `Dataset.with_transform`; a class so DataLoader workers can pickle it."""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, batch):
        return {
            "pixel_values": [self.transform(to_rgb(img)) for img in batch["image"]],
            "label": batch["label"],
        }


def get_dataset(split, k_shot=None, seed=SUBSET_SEEDS[0], augment=False):
    """Return the images of a manifest, transformed, in manifest order.

    split: "train", "val" or "test".
    k_shot: for train only, the subset with k images per class drawn with `seed`
        (None keeps the whole train pool). Subsets are nested: 5-shot is inside 10-shot.
    augment: random crop + flip instead of the deterministic eval transform (train only).
    """
    if split != "train" and (k_shot is not None or augment):
        raise ValueError("k_shot and augment apply to the train split only")

    manifest = load_manifest(split, k_shot, seed if k_shot is not None else None)
    ds = load_from_disk(PROCESSED_DATA_DIR)[split]
    row_of = {image_id: i for i, image_id in enumerate(ds["image_id"])}
    ds = ds.select([row_of[image_id] for image_id in manifest["image_id"]])
    assert ds["label"] == manifest["label"].tolist(), "manifest labels differ from the dataset"

    transform = train_aug_transform() if augment else eval_transform()
    return ds.with_transform(ApplyTransform(transform))


def get_loader(split, k_shot=None, seed=SUBSET_SEEDS[0], augment=False, batch_size=64, num_workers=2):
    """DataLoader over `get_dataset(...)`; batches are {"pixel_values": (B, 3, 224, 224), "label": (B,)}.

    Train loaders shuffle with a generator seeded by `seed`, so the batch order is reproducible and
    paired across arms. DataLoader also derives each worker's torch seed from that generator, which
    makes the random crops/flips reproducible when num_workers > 0. With num_workers=0 the
    augmentation uses the global torch RNG: call torch.manual_seed(seed) before training.
    """
    shuffle = split == "train"
    return DataLoader(
        get_dataset(split, k_shot, seed, augment),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )
