"""Write one manifest CSV per training subset and per eval split, from splits/cub_splits.csv.

A manifest lists exactly the images one run uses (image_id, label, class_name), so a run is fully
described by its manifest name, e.g. splits/manifests/train_10shot_seed0.csv. The generation team
reads the same files to know which species and real images each seed covers.

Usage: python src/make_manifests.py  (seconds; no download or embeddings needed)
"""
from load_data import MANIFESTS_DIR, SUBSET_SEEDS, load_splits, manifest_path

SHOTS = (5, 10, 20)  # core experiment uses 10; 5 and 20 are the planned extensions


def write_manifests():
    splits = load_splits()
    n_classes = splits["label"].nunique()
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    columns = ["image_id", "label", "class_name"]

    written = {}
    for split in ("train", "val", "test"):
        written[manifest_path(split)] = splits.loc[splits["split"] == split, columns]
    for seed in SUBSET_SEEDS:
        ranks = splits[f"rank_seed{seed}"]
        for k in SHOTS:
            subset = splits.loc[(splits["split"] == "train") & (ranks < k), columns]
            assert (subset["label"].value_counts() == k).all() and subset["label"].nunique() == n_classes
            written[manifest_path("train", k, seed)] = subset

    for path, df in written.items():
        df.sort_values("image_id").to_csv(path, index=False)
    return written


if __name__ == "__main__":
    for path, df in write_manifests().items():
        print(f"{path.name}: {len(df)} images")
