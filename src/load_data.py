"""Download CUB-200-2011 and save it with the project's train/val/test split.

The split is read from splits/cub_splits.csv (tracked in git, created by src/make_splits.py), so
every machine gets identical splits without recomputing the duplicate search.
"""
from pathlib import Path

import pandas as pd
from datasets import ClassLabel, DatasetDict, load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed" / "cub_200_2011"
SPLITS_DIR = PROJECT_ROOT / "splits"
SPLITS_CSV = SPLITS_DIR / "cub_splits.csv"
DROPPED_CSV = SPLITS_DIR / "dropped_duplicates.csv"
MANIFESTS_DIR = SPLITS_DIR / "manifests"

HF_DATASET = "birder-project/CUB_200_2011"
SUBSET_SEEDS = (0, 1, 2)


def parse_key(key):
    split, class_name, _ = key.split("/")
    return split, class_name


def load_raw():
    """All 11,788 images with labels and the official split ("train"/"test") restored."""
    raw = load_dataset(HF_DATASET, split="train")

    parsed = [parse_key(k) for k in raw["__key__"]]
    class_names = sorted({c for _, c in parsed})
    # the HF copy calls the official test split "validation"
    official_split = ["train" if s == "training" else "test" for s, _ in parsed]

    return (
        raw.remove_columns("__url__")  # absolute path to local HF cache, useless
        .rename_columns({"jpg": "image", "__key__": "image_id"})
        .add_column("label", [c for _, c in parsed])
        .cast_column("label", ClassLabel(names=class_names))
        .add_column("official_split", official_split)
    )


def load_splits():
    """Split table: image_id, label, class_name, split, rank_seed{s} (train only, else -1)."""
    return pd.read_csv(SPLITS_CSV)


def manifest_path(split, k_shot=None, seed=None):
    """splits/manifests/{split}.csv, or train_{k}shot_seed{s}.csv for a k-shot subset."""
    if k_shot is None:
        return MANIFESTS_DIR / f"{split}.csv"
    return MANIFESTS_DIR / f"{split}_{k_shot}shot_seed{seed}.csv"


def load_manifest(split, k_shot=None, seed=None):
    """Images of one split or k-shot subset: image_id, label, class_name (see make_manifests.py)."""
    path = manifest_path(split, k_shot, seed)
    if not path.exists():
        available = sorted(p.name for p in MANIFESTS_DIR.glob("*.csv"))
        raise FileNotFoundError(f"{path.name} not found; available: {available}")
    return pd.read_csv(path)


def download_and_save():
    """Download CUB-200-2011 from Hugging Face, apply the saved split, save locally."""
    ds = load_raw()
    split_of = dict(zip(*load_splits()[["image_id", "split"]].to_numpy().T))
    row_split = [split_of.get(image_id) for image_id in ds["image_id"]]  # None = dropped duplicate

    dataset = DatasetDict({
        name: ds.select([i for i, s in enumerate(row_split) if s == name]).remove_columns("official_split")
        for name in ("train", "val", "test")
    })
    assert sum(len(d) for d in dataset.values()) == len(split_of), "splits CSV does not match the download"

    PROCESSED_DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(PROCESSED_DATA_DIR)
    return dataset


if __name__ == "__main__":
    ds = download_and_save()
    print(ds)
