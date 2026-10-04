"""How noisy is val macro-F1 for a given number of val images per class?

Fits the planned classifier (linear head on frozen ImageNet ResNet-18 features) on a 10-shot
subset of the deduplicated official train split, then bootstraps val sets of v images per class
from the remaining official train images. Prints the std of val macro-F1 per v and, for scale,
the bootstrap std on the full official test split. Numbers are quoted in docs/data_split.md.

Usage: python src/val_size_analysis.py  (reuses data/features.npz from make_splits.py)
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from leakage import image_features
from load_data import SUBSET_SEEDS, load_raw, load_splits

SHOT = 10
VAL_SIZES = (3, 5, 8, 10)
N_BOOT = 300


def main():
    ds = load_raw()
    kept = set(load_splits()["image_id"])
    ds = ds.select([i for i, image_id in enumerate(ds["image_id"]) if image_id in kept])
    _, feats = image_features(ds)
    labels = np.array(ds["label"])
    official = np.array(ds["official_split"])
    pool, test = np.flatnonzero(official == "train"), np.flatnonzero(official == "test")
    classes = np.unique(labels)

    val_std = {v: [] for v in VAL_SIZES}
    test_std = []
    for seed in SUBSET_SEEDS:
        rng = np.random.default_rng(seed)
        shot, rest = [], {}
        for c in classes:
            idx = rng.permutation(pool[labels[pool] == c])
            shot += list(idx[:SHOT])
            rest[c] = idx[SHOT:]
        clf = LogisticRegression(max_iter=2000).fit(feats[shot], labels[shot])
        pred = clf.predict(feats)

        for v in VAL_SIZES:
            scores = []
            for _ in range(N_BOOT):
                idx = np.concatenate([rng.choice(rest[c], v) for c in classes])
                scores.append(f1_score(labels[idx], pred[idx], average="macro"))
            val_std[v].append(np.std(scores))
        scores = []
        for _ in range(N_BOOT):
            idx = rng.choice(test, len(test))
            scores.append(f1_score(labels[idx], pred[idx], average="macro"))
        test_std.append(np.std(scores))

    for v in VAL_SIZES:
        print(f"val {v}/class ({v * len(classes)} images): std {100 * np.mean(val_std[v]):.1f} pp")
    print(f"test ({len(test)} images): std {100 * np.mean(test_std):.1f} pp")


if __name__ == "__main__":
    main()
