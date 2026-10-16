"""Online preprocessing and data loaders shared by every experiment.

Images are converted to RGB (the dataset has 8 grayscale "L" images) and normalized with fixed
ImageNet statistics, because the classifier is an ImageNet-pretrained ResNet-18. The statistics
are not estimated from our data, so normalization cannot leak val/test information.

Which images a run uses comes from the manifests in splits/manifests (see make_manifests.py).
"""
import hashlib

import torch
from datasets import load_from_disk
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms as T

from load_data import PROCESSED_DATA_DIR, SUBSET_SEEDS, load_manifest
from synthetic_data import SyntheticImages, load_synthetic_manifest

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


def full_image_transform():
    """Keep the whole image: direct resize to 224 square, tensor and normalization."""
    return T.Compose([
        T.Resize((CROP_SIZE, CROP_SIZE)),
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
            "image_id": batch["image_id"],  # to save predictions per image
        }


class FixedAugmentedImages(Dataset):
    """One fixed crop/flip version per real image, independent of loading order/workers."""

    def __init__(self, dataset, seed):
        self.dataset = dataset
        self.transform = train_aug_transform()
        self.seeds = [int.from_bytes(hashlib.sha256(f"{seed}:{image_id}".encode()).digest()[:8],
                                    "little") % (2 ** 63)
                      for image_id in dataset["image_id"]]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.dataset[index]
        # Random torchvision transforms use the CPU RNG. Restore it after this sample
        # so fixed image variants do not change model initialization or other randomness.
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator().manual_seed(self.seeds[index]).get_state())
            pixels = self.transform(to_rgb(row["image"]))
        return {"pixel_values": pixels, "label": int(row["label"]), "image_id": row["image_id"]}


def get_dataset(split, k_shot=None, seed=SUBSET_SEEDS[0], augment=False,
                synthetic_manifest=None, full_image=False):
    """Return the images of a manifest, transformed, in manifest order.

    split: "train", "val" or "test".
    k_shot: for train only, the subset with k images per class drawn with `seed`
        (None keeps the whole train pool). Subsets are nested: 5-shot is inside 10-shot.
    augment: concatenate original center-cropped views with one fixed crop/flip view of each
        real image, doubling the number of training entries (train only).
    synthetic_manifest: prepared CSV to append to real training images (train only).
        Both sources use the baseline resize/center-crop transform, with no random crops/flips.
        augment must be False.
    full_image: real-only control using full-image resizing instead of center cropping (train only).
    """
    if split != "train" and (k_shot is not None or augment or synthetic_manifest is not None or full_image):
        raise ValueError("k_shot, augment, synthetic_manifest and full_image apply to the train split only")
    if synthetic_manifest is not None and augment:
        raise ValueError("Real + synthetic training uses no random crops/flips; augment must be False")
    if full_image and (augment or synthetic_manifest is not None):
        raise ValueError("full_image selects the real-only control; do not combine it with added images")

    manifest = load_manifest(split, k_shot, seed if k_shot is not None else None)
    ds = load_from_disk(PROCESSED_DATA_DIR)[split]
    row_of = {image_id: i for i, image_id in enumerate(ds["image_id"])}
    ds = ds.select([row_of[image_id] for image_id in manifest["image_id"]])
    assert ds["label"] == manifest["label"].tolist(), "manifest labels differ from the dataset"

    transform = full_image_transform() if full_image else eval_transform()
    real = ds.with_transform(ApplyTransform(transform))
    if augment:
        return ConcatDataset([real, FixedAugmentedImages(ds, seed)])
    if synthetic_manifest is None:
        return real
    names = manifest[["label", "class_name"]].drop_duplicates().sort_values("label")
    synthetic = load_synthetic_manifest(synthetic_manifest, names["class_name"].tolist())
    if set(manifest["image_id"]) & set(synthetic["image_id"]):
        raise ValueError("Real and synthetic image IDs overlap")
    return ConcatDataset([real, SyntheticImages(synthetic, transform)])


def get_loader(split, k_shot=None, seed=SUBSET_SEEDS[0], augment=False, batch_size=64, num_workers=2,
               synthetic_manifest=None, full_image=False):
    """DataLoader over `get_dataset(...)`; batches are {"pixel_values": (B, 3, 224, 224), "label": (B,),
    "image_id": list of B str}.

    Train loaders shuffle with a generator seeded by `seed`, so the batch order is reproducible.
    Combined training shuffles original and additional entries together. Augmented entries
    are fixed per image and seed, so revisits retain the same view.
    Fixed augmented views use an image-ID/seed-derived CPU RNG state and restore the
    caller's RNG afterwards, independently of worker count and access order.
    """
    shuffle = split == "train"
    return DataLoader(
        get_dataset(split, k_shot, seed, augment, synthetic_manifest, full_image),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )
