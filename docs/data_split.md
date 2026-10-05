# Data split, leakage check and preprocessing

## Final split

| Split | Images | Per class | Source |
| --- | --- | --- | --- |
| train (pool) | 4,952 | 20–25 | official CUB train, minus val and duplicates |
| val | 1,000 | exactly 5 | official CUB train, stratified random, seed 42 |
| test | 5,794 | 11–30 | official CUB test, untouched |

42 training images were removed as duplicates (list: `splits/dropped_duplicates.csv`).
Every kept image and its split are in `splits/cub_splits.csv`, which is tracked in git.

Experiments train on k-shot subsets of the pool, not on the whole pool: 10 images per class
(2,000) in the core runs, 5 or 20 in the extensions. A subset is defined by a seed: `rank_seed{s} < k`
in `cub_splits.csv`.

Each run reads its images from a **manifest** in `splits/manifests/` (`image_id, label, class_name`):

| Manifest | Images |
| --- | --- |
| `train_{5,10,20}shot_seed{0,1,2}.csv` | 1,000 / 2,000 / 4,000 |
| `train.csv` (whole pool), `val.csv`, `test.csv` | 4,952 / 1,000 / 5,794 |

A run is fully described by its manifest name, and the generation pipeline uses the same files
to know which real images each seed covers. `python src/make_manifests.py` rebuilds them from
`cub_splits.csv` in seconds.

| Seed | Used for |
| --- | --- |
| 42 | the val split, drawn once and never changed |
| 0, 1, 2 | the three paired k-shot subsets; all arms with the same seed see identical real images |

Subsets are nested (the 5-shot subset is inside the 10-shot subset of the same seed), so comparing
5/10/20 shots changes only the amount of data, not which photos are used.

## Why this split

**Test is the official CUB test split.** Results stay comparable with published CUB numbers, and
none of our decisions touch the data we report on. It is also large: the bootstrap standard
deviation of test macro-F1 is 0.6 pp, so a difference of a few pp between arms is measurable.
Re-splitting all 11,788 images (e.g. 70/15/15) would not help: training uses only 10 images per
class, so moving test images into the pool adds nothing to training and only shrinks the test set.

**Val comes from the official train split, not from test**, so test is used once, at the end.

**Val size is 5 images per class.** Two constraints set it:

1. *The pool must keep 20 images per class* for the 20-shot extension. After duplicate removal
   the smallest class has 25 official training images, so 5 is the largest val that keeps every
   class at 20 or more.
2. *Val must be precise enough for checkpoint selection.* Bootstrap standard deviation of val
   macro-F1 for a 10-shot linear probe on frozen ResNet-18 features (the planned classifier):

   | Val per class | Val images | Std of val macro-F1 |
   | --- | --- | --- |
   | 3 | 600 | 1.7 pp |
   | **5** | **1,000** | **1.4 pp** |
   | 8 | 1,600 | 1.0 pp |
   | 10 | 2,000 | 1.0 pp |

   (`python src/val_size_analysis.py`.) Noise falls with the square root of val size, so going
   beyond 5 per class gains at most 0.4 pp and would break constraint 1. Val only selects checkpoints and settings; the comparison between
   arms is made on test with three paired seeds.

**A fixed count per class, not a fraction.** The previous 15% split gave 4 or 5 images per class.
An equal count makes every class weigh the same in val macro-F1, as in the primary metric.

**Random within each class (stratified), with a fixed seed.** Stratification keeps all 200
classes in every split; randomness avoids picking images by file order or photographer id.

**The pool is large enough for distinct seeds.** Two 10-shot subsets share on average 4 of 10
images per class, so seeds vary the training data and not only the initialization. (20-shot
subsets share 16 of 20; seed variation there mostly reflects initialization.)

## Leakage check

`src/leakage.py` compares every pair of images inside and between splits with two detectors:

- **md5 of the file** — byte-identical copies;
- **cosine similarity of ResNet-18 embeddings** — the same photo re-encoded, re-framed, cropped,
  or burst shots of the same bird.

Rule: duplicate if md5 matches, or similarity ≥ 0.93 with the same label. The threshold was set by
looking at all pairs above 0.90: above 0.937 every pair was the same photo or a burst; between 0.93
and 0.937 true duplicates were mixed with different photos (e.g. a framed copy of one photo at
0.933); below 0.93 none were duplicates. A false positive costs one surplus training image, a
missed duplicate leaks test data, so the threshold errs low.
A perceptual hash (pHash) was also tried and rejected: at Hamming distance ≤ 6 it matched unrelated
photos with plain sky or water backgrounds.

**Findings on the official CUB split:**

| Pairs | Count | Action |
| --- | --- | --- |
| official train ↔ official test | 30 (27 distinct test images, 0.5% of test) | drop the train copy |
| official train ↔ official train | 15 | keep one per group, so no photo can be in both train and val |
| official test ↔ official test | 19 | keep: test is the official benchmark, and repeated test photos do not leak training data |

The pair reported in the EDA (byte-identical Pigeon Guillemot 0018/0081, both in official train) is
one of the 15. One photo in the official test split is filed under two species (Chuck-will's-widow
0042 and Whip-poor-will 0002): label noise, kept as is.

After removal, `python src/leakage.py` finds no duplicate pairs between train, val and test and no
shared image ids (`reports/leakage_report.json`).

**Leakage outside our control:**

- *ImageNet pretraining.* The CUB authors warn that CUB images overlap with ImageNet. The ResNet-18
  backbone may have seen some test photos, inflating absolute scores. All arms share the backbone,
  so differences between arms remain valid; absolute numbers should be read with this caveat.
- *Stable Diffusion training data.* SD 1.5 was trained on web images and can reproduce CUB photos.
  Before training, check the synthetic pool against val and test with
  `leakage.duplicate_pairs(syn_md5, syn_emb, syn_labels, md5_test, emb_test, None)` (labels ignored)
  and drop matches.
- *Process.* Val and test images are never used for prompts, generation or synthetic-image
  selection. Normalization uses fixed ImageNet statistics, not statistics of our data.

## Preprocessing

Offline (`src/make_splits.py`, `src/make_manifests.py`, `src/load_data.py`): restore labels and the
official split from the HF keys, remove duplicates, assign splits, write manifests.

Online (`src/preprocessing.py`):

| Step | Eval / no-aug training | Augmented training |
| --- | --- | --- |
| Color | convert to RGB (8 images are grayscale: 4 in train, 4 in test) | same |
| Geometry | resize shorter side to 256, center crop 224 | random resized crop 224, area 50–100% |
| Flip | — | horizontal, p = 0.5 |
| Normalize | ImageNet mean/std | same |

The crop area starts at 50% instead of torchvision's 8%: small crops often cut the bird out,
which turns a fine-grained sample into a mislabeled background patch.

```python
from preprocessing import get_loader  # run from src/

train = get_loader("train", k_shot=10, seed=0, augment=True, batch_size=64)
val = get_loader("val")
test = get_loader("test")
batch = next(iter(train))  # {"pixel_values": (64, 3, 224, 224), "label": (64,)}
```

Loaders:

- **train** shuffles with a generator seeded by `seed`. Arms with the same seed get the same images
  in the same batch order (with or without augmentation), so arm comparisons are paired.
  With `num_workers > 0` the random crops and flips are reproducible too; with `num_workers=0`,
  call `torch.manual_seed(seed)` before training.
- **val/test** are not shuffled and use the deterministic transform.
- `get_dataset(...)` returns the same data without the loader, e.g. for a custom sampler.

## Reproducing

```bash
python src/load_data.py    # download, apply splits/cub_splits.csv, save to data/processed
python src/leakage.py      # verify: no duplicates across splits
python src/make_splits.py  # only to regenerate the CSVs (~7 min on CPU, computes embeddings)
python src/make_manifests.py  # rebuild splits/manifests from cub_splits.csv
python src/val_size_analysis.py  # val-size table above
```

`make_splits.py` is deterministic given the embeddings, but borderline similarities can differ
slightly across hardware; that is why the CSVs, not the script, define the split.
