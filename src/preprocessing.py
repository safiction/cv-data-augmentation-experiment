"""Online preprocessing shared by every experiment.

Images are converted to RGB (the dataset has 8 grayscale "L" images) and normalized with fixed
ImageNet statistics, because the classifier is an ImageNet-pretrained ResNet-18. The statistics
are not estimated from our data, so normalization cannot leak val/test information.
"""
from datasets import load_from_disk
from torchvision import transforms as T

from load_data import PROCESSED_DATA_DIR, SUBSET_SEEDS, load_splits

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


class ApplyTransform:
    """Batch transform for `Dataset.with_transform`; a class so DataLoader workers can pickle it."""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, batch):
        return {
            "pixel_values": [self.transform(img.convert("RGB")) for img in batch["image"]],
            "label": batch["label"],
        }


def get_dataset(split, k_shot=None, seed=SUBSET_SEEDS[0], augment=False):
    """Return a split ready for `torch.utils.data.DataLoader`.

    split: "train", "val" or "test".
    k_shot: for train only, keep k images per class from the subset drawn with `seed`
        (None keeps the whole train pool). Subsets are nested: 5-shot is inside 10-shot.
    augment: random crop + flip instead of the deterministic eval transform (train only).
    """
    if split != "train" and (k_shot is not None or augment):
        raise ValueError("k_shot and augment apply to the train split only")

    ds = load_from_disk(PROCESSED_DATA_DIR)[split]
    if k_shot is not None:
        splits = load_splits()
        keep = set(splits.loc[splits[f"rank_seed{seed}"] < k_shot, "image_id"])
        ds = ds.filter(lambda ids: [i in keep for i in ids], input_columns="image_id", batched=True)
        assert len(ds) == k_shot * ds.features["label"].num_classes, "subset is not class-balanced"

    transform = train_aug_transform() if augment else eval_transform()
    return ds.with_transform(ApplyTransform(transform))
