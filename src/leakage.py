"""Duplicate and data-leakage detection.

Two detectors:
- md5 of the encoded file finds byte-identical copies;
- cosine similarity of ImageNet ResNet-18 embeddings (the classifier's own backbone) finds the
  same photo re-encoded, re-framed or cropped, and burst shots of the same bird.

A pair is a duplicate when the md5 matches, or when the similarity is at least DUP_MIN_SIM and both
images have the same label. The threshold comes from reviewing every pair above 0.90 by eye: above
0.937 all pairs were the same photo or burst shots; between 0.93 and 0.937 a few true duplicates
remain (e.g. a framed copy at 0.933) next to different photos; below 0.93 no duplicates were seen.
A false positive only costs one surplus training image, so the threshold errs on the low side.
Cross-label pairs at this similarity were different photos of look-alike species, except one photo
filed under two species inside the official test split (label noise, not leakage; left as is).
pHash was tried and rejected: at Hamming distance <= 6 it matched unrelated photos with plain sky or
water backgrounds.

The same functions will check the diffusion pool against val/test before training: Stable Diffusion
was trained on web images and can reproduce CUB photos.

Usage: python src/leakage.py  (checks the saved splits, writes reports/leakage_report.json)
"""
import hashlib
import io
import json
from itertools import combinations_with_replacement

import numpy as np
import torch
from datasets import Image as ImageFeature
from datasets import load_from_disk
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18

from load_data import DATA_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT
from preprocessing import eval_transform

DUP_MIN_SIM = 0.93
FEATURES_CACHE = DATA_DIR / "features.npz"


@torch.no_grad()
def embeddings(images, batch_size=64):
    """L2-normalized ResNet-18 penultimate features of PIL images."""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model.eval()
    transform = eval_transform()
    feats = []
    for start in range(0, len(images), batch_size):
        batch = torch.stack([transform(img.convert("RGB")) for img in images[start:start + batch_size]])
        feats.append(torch.nn.functional.normalize(model(batch), dim=1))
    return torch.cat(feats).numpy()


def image_features(ds):
    """(md5 list, embedding matrix) aligned with the rows of `ds`; cached by image_id."""
    cache = {}
    if FEATURES_CACHE.exists():
        stored = np.load(FEATURES_CACHE)
        cache = {i: (m, e) for i, m, e in zip(stored["image_id"], stored["md5"], stored["emb"])}

    missing = [k for k, image_id in enumerate(ds["image_id"]) if image_id not in cache]
    if missing:
        sub = ds.select(missing)
        raw = sub.cast_column("image", ImageFeature(decode=False))["image"]
        md5s = [hashlib.md5(x["bytes"]).hexdigest() for x in raw]
        embs = embeddings([Image.open(io.BytesIO(x["bytes"])) for x in raw])
        cache.update({i: (m, e) for i, m, e in zip(sub["image_id"], md5s, embs)})
        FEATURES_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(FEATURES_CACHE, image_id=list(cache), md5=[v[0] for v in cache.values()],
                 emb=np.stack([v[1] for v in cache.values()]))

    rows = [cache[image_id] for image_id in ds["image_id"]]
    return [m for m, _ in rows], np.stack([e for _, e in rows])


def duplicate_pairs(md5_a, emb_a, labels_a, md5_b=None, emb_b=None, labels_b=None,
                    min_sim=DUP_MIN_SIM, chunk=1024):
    """(i, j, similarity) duplicate pairs between sets a and b, or inside a when b is omitted.

    Pass labels_b=None with a b set to ignore labels, e.g. for synthetic images vs. test.
    """
    within = md5_b is None
    if within:
        md5_b, emb_b, labels_b = md5_a, emb_a, labels_a
    pairs = {}
    for start in range(0, len(emb_a), chunk):
        sim = emb_a[start:start + chunk] @ emb_b.T
        for i, j in zip(*np.nonzero(sim >= min_sim)):
            i_abs = int(i) + start
            if (not within or i_abs < j) and (labels_b is None or labels_a[i_abs] == labels_b[j]):
                pairs[i_abs, int(j)] = float(sim[i, j])
    by_md5 = {}
    for j, m in enumerate(md5_b):
        by_md5.setdefault(m, []).append(j)
    for i, m in enumerate(md5_a):
        for j in by_md5.get(m, []):
            if not within or i < j:
                pairs[i, j] = 1.0
    return [(i, j, s) for (i, j), s in sorted(pairs.items())]


def check_splits(dataset):
    """Duplicate pairs inside and between all splits of a DatasetDict."""
    feats = {s: (*image_features(dataset[s]), dataset[s]["label"], dataset[s]["image_id"]) for s in dataset}
    report = {"sizes": {s: len(dataset[s]) for s in dataset}, "pairs": {}}
    for a, b in combinations_with_replacement(dataset, 2):
        md5_a, emb_a, lab_a, ids_a = feats[a]
        md5_b, emb_b, lab_b, ids_b = feats[b]
        pairs = (duplicate_pairs(md5_a, emb_a, lab_a) if a == b
                 else duplicate_pairs(md5_a, emb_a, lab_a, md5_b, emb_b, lab_b))
        report["pairs"][f"{a}/{b}"] = [{"a": ids_a[i], "b": ids_b[j], "similarity": round(s, 4)}
                                       for i, j, s in pairs]
    ids = {s: set(f[3]) for s, f in feats.items()}
    report["shared_image_ids"] = {f"{a}/{b}": len(ids[a] & ids[b])
                                  for a, b in combinations_with_replacement(dataset, 2) if a != b}
    return report


if __name__ == "__main__":
    report = check_splits(load_from_disk(PROCESSED_DATA_DIR))
    out = PROJECT_ROOT / "reports" / "leakage_report.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print("Sizes:", report["sizes"])
    print("Duplicate pairs:", {k: len(v) for k, v in report["pairs"].items()})
    print("Shared image_ids:", report["shared_image_ids"])
    print("Report:", out.relative_to(PROJECT_ROOT))
