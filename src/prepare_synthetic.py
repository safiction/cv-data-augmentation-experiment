"""Validate an SD pool and exclude val/test duplicates before mixed-data training.

Only metadata-listed images are included, never contact sheets. This checks file
integrity and leakage.

With --review, species fidelity comes from the manual review table (docs/generation.md): only
ACCEPTED images are kept, leakage matches are dropped, then the first --per-class images of each
species by image ID are selected, so dropped images are replaced by the next seeds. A species with
fewer accepted images keeps all remaining images and is listed in the report.

Usage: python src/prepare_synthetic.py --pool data/synthetic/sd15_wildlife \
           --review reports/generation/review_sd15_wildlife.csv --per-class 10
"""
import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Image as ImageFeature, load_from_disk
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

from classifier import class_names_from_manifest, resolve_device, save_json
from leakage import DUP_MIN_SIM, duplicate_pairs
from load_data import DATA_DIR, PROCESSED_DATA_DIR, load_manifest
from preprocessing import eval_transform, get_loader
from synthetic_data import ACCEPTED, DECISIONS, SyntheticImages


def read_pool(pool, class_names, expected_per_class=10):
    """Resolve metadata IDs, verify labels, hashes and image decoding.

    expected_per_class: exact image count per species; None skips the check (reviewed pools may
    have extra seeds for some species).
    """
    pool = Path(pool).resolve()
    metadata = pd.read_csv(pool / "metadata.csv")
    required = {"image_id", "label", "class_name", "md5"}
    if not required.issubset(metadata.columns) or metadata.empty:
        raise ValueError(f"Metadata must contain {sorted(required)}")
    if metadata[list(required)].isna().any().any() or metadata.image_id.duplicated().any():
        raise ValueError("Metadata contains missing values or repeated image IDs")
    records = []
    for row in metadata.itertuples(index=False):
        if row.label not in range(len(class_names)) or class_names[int(row.label)] != row.class_name:
            raise ValueError(f"Metadata label/species mismatch: {row.image_id}")
        # Generator IDs are data-relative paths without the .jpg suffix.
        path = (DATA_DIR / (row.image_id + ".jpg")).resolve()
        if not path.is_relative_to(pool) or path.parent.name != row.class_name:
            raise ValueError(f"Image path is outside its pool/class: {row.image_id}")
        content = path.read_bytes()
        md5 = hashlib.md5(content).hexdigest()
        if md5 != row.md5:
            raise ValueError(f"Generation metadata MD5 mismatch: {row.image_id}")
        with Image.open(path) as image:
            image.convert("RGB").load()
        records.append({"image_id": row.image_id, "path": str(path), "label": int(row.label),
                        "class_name": row.class_name, "md5": md5,
                        "sha256": hashlib.sha256(content).hexdigest()})
    manifest = pd.DataFrame(records)
    counts = manifest.groupby("label").size().reindex(range(len(class_names)), fill_value=0)
    if expected_per_class is not None and not (counts == expected_per_class).all():
        raise ValueError(f"Expected {expected_per_class} synthetic images per species; got {counts.to_dict()}")
    if manifest.md5.duplicated().any():
        raise ValueError("Synthetic pool contains byte-identical images")
    return manifest


def apply_review(manifest, review):
    """Keep pool images whose review decision is ACCEPTED; every pool image must be reviewed."""
    review = review.assign(decision=review["decision"].fillna("").astype(str).str.strip())
    if review["image_id"].duplicated().any():
        raise ValueError("Review table repeats image IDs")
    decisions = review.set_index("image_id")["decision"].reindex(manifest["image_id"])
    missing = decisions.isna() | (decisions == "")
    if missing.any():
        raise ValueError(f"{int(missing.sum())} pool images have no review decision, "
                         f"e.g. {decisions[missing].index[0]}")
    unknown = sorted(set(decisions) - set(DECISIONS))
    if unknown:
        raise ValueError(f"Unknown review decisions {unknown}; allowed: {DECISIONS}")
    return manifest[decisions.isin(ACCEPTED).to_numpy()].reset_index(drop=True)


def take_per_class(manifest, per_class, class_names):
    """Keep up to per_class images per species by image ID; report species below the target."""
    ordered = manifest.sort_values("image_id")
    counts = ordered.groupby("label").size().reindex(range(len(class_names)), fill_value=0)
    short = [class_names[label] for label in counts.index[counts < per_class]]
    selected = ordered.groupby("label").head(per_class)
    return selected.sort_values(["label", "image_id"]).reset_index(drop=True), short


@torch.no_grad()
def features(model, loader, device):
    rows = []
    for batch in loader:
        pixels = batch["pixel_values"].to(device)
        rows.append(torch.nn.functional.normalize(model(pixels), dim=1).cpu().numpy())
    return np.concatenate(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=DATA_DIR / "synthetic" / "sd15_wildlife")
    parser.add_argument("--output", type=Path, help="New directory; defaults to <pool>/prepared")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--expected-per-class", type=int, default=10,
                        help="Exact images per species in an unreviewed pool")
    parser.add_argument("--review", type=Path,
                        help="Review table (reports/generation/review_{pool}.csv); keep accepted images only")
    parser.add_argument("--per-class", type=int, default=10,
                        help="With --review: maximum synthetic images per species after review and leakage filtering")
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.expected_per_class <= 0 or args.per_class <= 0:
        parser.error("batch size and per-class counts must be positive")
    output = (args.output or args.pool / "prepared").resolve()
    if output.exists():
        raise FileExistsError(f"Preparation output already exists: {output}")
    device = resolve_device(args.device)
    names = class_names_from_manifest(load_manifest("train"))
    manifest = read_pool(args.pool, names, None if args.review else args.expected_per_class)
    pool_images = len(manifest)
    if args.review:
        manifest = apply_review(manifest, pd.read_csv(args.review, keep_default_na=False))
        print(f"Review accepted {len(manifest)} of {pool_images} images", flush=True)
    print(f"Validated {len(manifest)} images across {manifest.label.nunique()} species; "
          f"extracting features on {device}", flush=True)
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
    model.fc = torch.nn.Identity()
    model.eval()
    loader = DataLoader(SyntheticImages(manifest, eval_transform()), batch_size=args.batch_size)
    synthetic_features = features(model, loader, device)
    dataset = load_from_disk(PROCESSED_DATA_DIR)
    excluded, matches = set(), {}
    for split in ("val", "test"):
        print(f"Checking synthetic images against real {split} (labels ignored)...", flush=True)
        holdout = load_manifest(split)
        raw = dataset[split].cast_column("image", ImageFeature(decode=False))
        raw_by_id = dict(zip(raw["image_id"], raw["image"]))
        hashes = []
        for image_id in holdout.image_id:
            image = raw_by_id[image_id]
            content = image["bytes"] if image["bytes"] is not None else Path(image["path"]).read_bytes()
            hashes.append(hashlib.md5(content).hexdigest())
        heldout_features = features(model, get_loader(split, batch_size=args.batch_size, num_workers=0), device)
        pairs = duplicate_pairs(manifest.md5.tolist(), synthetic_features, manifest.label.tolist(),
                                hashes, heldout_features, None)
        matches[split] = [{"synthetic_id": manifest.iloc[i].image_id,
                           "real_id": holdout.iloc[j].image_id, "similarity": similarity}
                          for i, j, similarity in pairs]
        excluded.update(i for i, _, _ in pairs)
        print(f"{split}: {len(pairs)} duplicate pairs", flush=True)
    accepted = manifest.drop(index=sorted(excluded)).copy()
    short = []
    if args.review:
        accepted, short = take_per_class(accepted, args.per_class, names)
        print(f"Selected {len(accepted)} images; {len(short)} species have fewer than "
              f"{args.per_class} accepted images and keep all remaining images", flush=True)
    if accepted.empty:
        raise ValueError("No synthetic images remain after leakage filtering")
    accepted["path"] = [Path(os.path.relpath(p, output)).as_posix() for p in accepted.path]
    counts = accepted.groupby("label").size().reindex(range(len(names)), fill_value=0)
    report = {
        "pool_images": pool_images, "input_images": len(manifest), "accepted_images": len(accepted),
        "excluded_images": len(excluded), "min_similarity": DUP_MIN_SIM,
        "detector": "MD5 equality or cosine similarity of normalized ImageNet ResNet-18 features",
        "labels_ignored": True, "device": str(device), "matches": matches,
        "accepted_per_class": {str(k): int(v) for k, v in counts.items()},
        "metadata_sha256": hashlib.sha256((args.pool / "metadata.csv").read_bytes()).hexdigest(),
        "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ("prepare_synthetic.py", "leakage.py", "preprocessing.py", "synthetic_data.py")},
        "quality_review": ({"review_table": str(args.review), "accepted_decisions": list(ACCEPTED),
                            "review_sha256": hashlib.sha256(args.review.read_bytes()).hexdigest(),
                            "per_class": args.per_class, "species_below_target": short,
                            "species_without_synthetic": [names[label] for label in counts.index[counts == 0]]}
                           if args.review else "No manual species-fidelity review recorded by this script"),
        "evaluation": "Val/test used only for duplicate detection; no classifier metrics or quality selection",
    }
    output.mkdir(parents=True, exist_ok=False)
    accepted.to_csv(output / "manifest.csv", index=False)
    save_json(output / "leakage_report.json", report)
    print(f"Prepared {len(accepted)} synthetic images: {output / 'manifest.csv'}", flush=True)


if __name__ == "__main__":
    main()
