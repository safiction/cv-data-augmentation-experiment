from pathlib import Path

from datasets import ClassLabel, DatasetDict, load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed" / "cub_200_2011"

HF_DATASET = "birder-project/CUB_200_2011"
SEED = 42
VAL_SIZE = 0.15  # fraction of the official train split held out for validation


def parse_key(key):
    split, class_name, _ = key.split("/")
    return split, class_name

def download_and_save():
    """Download CUB-200-2011 from Hugging Face, restore labels and splits, save locally."""
    raw = load_dataset(HF_DATASET, split="train")

    keys = raw["__key__"]
    parsed = [parse_key(k) for k in keys]
    official_splits = [s for s, _ in parsed]
    class_names = sorted({c for _, c in parsed})

    ds = (
        raw.remove_columns("__url__")  # absolute path to local HF cache, useless
        .rename_columns({"jpg": "image", "__key__": "image_id"})
        .add_column("label", [c for _, c in parsed])
        .cast_column("label", ClassLabel(names=class_names))
    )

    train_idx = [i for i, s in enumerate(official_splits) if s == "training"]
    test_idx = [i for i, s in enumerate(official_splits) if s == "validation"]

    train_val = ds.select(train_idx).train_test_split(
        test_size=VAL_SIZE, stratify_by_column="label", seed=SEED
    )
    dataset = DatasetDict(
        train=train_val["train"],
        val=train_val["test"],
        test=ds.select(test_idx),
    )

    PROCESSED_DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(PROCESSED_DATA_DIR)
    return dataset

if __name__ == "__main__":
    ds = download_and_save()
    print(ds)
