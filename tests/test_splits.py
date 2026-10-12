"""Subset IDs: splits/cub_splits.csv and the manifests agree and keep the split rules.

Needs only the CSVs in git, no download.
"""
import hashlib

import pandas as pd
import pytest

from load_data import (
    DROPPED_CSV, NUM_CLASSES, SUBSET_SEEDS, canonical_sha256, class_names, load_manifest,
    load_splits, sha256_matches,
)
from make_manifests import SHOTS, build_manifests


@pytest.fixture(scope="module")
def splits():
    return load_splits()


def test_manifests_in_git_match_splits_csv():
    for path, expected in build_manifests().items():
        pd.testing.assert_frame_equal(pd.read_csv(path), expected, obj=path.name)


def test_text_hashes_ignore_line_endings(tmp_path):
    lf = tmp_path / "lf.csv"
    crlf = tmp_path / "crlf.csv"
    lf.write_bytes(b"a,b\n1,2\n")
    crlf.write_bytes(b"a,b\r\n1,2\r\n")
    assert canonical_sha256(lf) == canonical_sha256(crlf)
    assert sha256_matches(lf, canonical_sha256(crlf))
    assert sha256_matches(lf, hashlib.sha256(crlf.read_bytes()).hexdigest())


def test_split_sizes(splits):
    assert splits["split"].value_counts().to_dict() == {"test": 5794, "train": 4952, "val": 1000}


def test_image_ids_unique_and_splits_disjoint(splits):
    assert splits["image_id"].is_unique
    ids = {s: set(load_manifest(s)["image_id"]) for s in ("train", "val", "test")}
    assert not ids["train"] & ids["val"] and not ids["train"] & ids["test"] and not ids["val"] & ids["test"]


def test_dropped_duplicates_not_used(splits):
    dropped = pd.read_csv(DROPPED_CSV)
    assert len(dropped) == 42
    assert not set(dropped["image_id"]) & set(splits["image_id"])
    assert dropped["image_id"].str.startswith("training/").all(), "official test images must never be dropped"


def test_test_split_is_official_test(splits):
    test = splits[splits["split"] == "test"]
    # the HF copy files the official test split under "validation/"
    assert test["image_id"].str.startswith("validation/").all()
    assert splits.loc[splits["split"] != "test", "image_id"].str.startswith("training/").all()


def test_val_has_five_per_class(splits):
    counts = splits.loc[splits["split"] == "val", "label"].value_counts()
    assert len(counts) == NUM_CLASSES and (counts == 5).all()


def test_labels_match_class_names(splits):
    names = class_names()
    assert len(names) == NUM_CLASSES
    assert (splits["class_name"] == [names[label] for label in splits["label"]]).all()
    assert (splits["image_id"].str.split("/").str[1] == splits["class_name"]).all()


@pytest.mark.parametrize("seed", SUBSET_SEEDS)
def test_ranks_are_per_class_permutations(splits, seed):
    train = splits[splits["split"] == "train"]
    for _, ranks in train.groupby("label")[f"rank_seed{seed}"]:
        assert sorted(ranks) == list(range(len(ranks)))
    assert (splits.loc[splits["split"] != "train", f"rank_seed{seed}"] == -1).all()


@pytest.mark.parametrize("seed", SUBSET_SEEDS)
@pytest.mark.parametrize("k", SHOTS)
def test_kshot_subset_balanced_and_inside_train(seed, k):
    subset = load_manifest("train", k, seed)
    counts = subset["label"].value_counts()
    assert len(counts) == NUM_CLASSES and (counts == k).all()
    assert set(subset["image_id"]) <= set(load_manifest("train")["image_id"])


@pytest.mark.parametrize("seed", SUBSET_SEEDS)
def test_kshot_subsets_are_nested(seed):
    ids = [set(load_manifest("train", k, seed)["image_id"]) for k in sorted(SHOTS)]
    assert all(small < large for small, large in zip(ids, ids[1:]))


def test_seeds_give_different_subsets():
    ids = [set(load_manifest("train", 10, seed)["image_id"]) for seed in SUBSET_SEEDS]
    assert ids[0] != ids[1] != ids[2] != ids[0]
