"""Create splits/cub_splits.csv and splits/dropped_duplicates.csv. Run once; the CSVs are in git.

1. Remove duplicates (see src/leakage.py). Every duplicate group keeps its official test images, so
   the benchmark test split is never modified, and drops its train copies; a group without test
   images keeps its first image_id. This removes train-to-test leakage and makes sure no photo can
   end up in both train and val.
2. test = official CUB test split.
3. val = VAL_PER_CLASS random images per class from the rest of the official train split.
4. train = the remaining official train images, the pool for k-shot subsets.
5. For every seed in SUBSET_SEEDS each class's train images get a random order `rank_seed{s}`.
   The k-shot subset of seed s is `rank_seed{s} < k`: subsets are nested (5-shot inside 10-shot)
   and the experiment arms that share a seed train on identical real images.

The reasoning behind the numbers is in docs/data_split.md.
"""
import numpy as np
import pandas as pd

from leakage import duplicate_pairs, image_features
from load_data import DROPPED_CSV, SPLITS_CSV, SPLITS_DIR, SUBSET_SEEDS, load_raw

SPLIT_SEED = 42
VAL_PER_CLASS = 5
MAX_SHOT = 20  # largest k-shot setting planned; every class must keep this many train images


def find_duplicates(md5s, embs, labels, image_ids, official_split):
    """{dropped image_id: kept image_id} following the rule in the module docstring."""
    parent = list(range(len(image_ids)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j, _ in duplicate_pairs(md5s, embs, labels):
        parent[root(i)] = root(j)
    groups = {}
    for i in range(len(image_ids)):
        groups.setdefault(root(i), []).append(i)

    dropped = {}
    for members in groups.values():
        test = [i for i in members if official_split[i] == "test"]
        keep = test or [min(members, key=lambda i: image_ids[i])]
        for i in set(members) - set(keep):
            dropped[image_ids[i]] = image_ids[keep[0]]
    return dropped


def make_splits():
    ds = load_raw()
    image_ids, labels, official = ds["image_id"], ds["label"], ds["official_split"]
    md5s, embs = image_features(ds)
    dropped = find_duplicates(md5s, embs, labels, image_ids, official)

    df = pd.DataFrame({"image_id": image_ids, "label": labels, "split": official})
    df = df[~df["image_id"].isin(dropped)].sort_values("image_id", ignore_index=True)
    class_names = ds.features["label"].names
    df.insert(2, "class_name", [class_names[label] for label in df["label"]])

    rng = np.random.default_rng(SPLIT_SEED)
    for _, idx in df[df["split"] == "train"].groupby("label").groups.items():
        df.loc[rng.choice(idx, size=VAL_PER_CLASS, replace=False), "split"] = "val"

    train = df[df["split"] == "train"]
    assert train["label"].value_counts().min() >= MAX_SHOT, "train pool too small for MAX_SHOT"
    for seed in SUBSET_SEEDS:
        rng = np.random.default_rng(seed)
        df[f"rank_seed{seed}"] = -1
        for _, idx in train.groupby("label").groups.items():
            df.loc[idx, f"rank_seed{seed}"] = rng.permutation(len(idx))

    SPLITS_DIR.mkdir(exist_ok=True)
    df.to_csv(SPLITS_CSV, index=False)
    pd.DataFrame(sorted(dropped.items()), columns=["image_id", "duplicate_of"]).to_csv(DROPPED_CSV, index=False)
    return df, dropped


if __name__ == "__main__":
    df, dropped = make_splits()
    print("Dropped duplicates:", len(dropped))
    print(df["split"].value_counts())
    print("Images per class:")
    print(df.groupby("split")["label"].value_counts().groupby("split").agg(["min", "max"]))
