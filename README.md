# Synthetic Data Augmentation with Diffusion Models
## Project Question
Can diffusion-based synthetic data augmentation improve image classification performance when training data is limited, and under what conditions do synthetic samples help versus hurt?
## ML Task
Supervised image classification on a deliberately limited labeled dataset.
We compare:
• Baseline: classifier trained only on real images
• Augmented: the same classifier trained on real + diffusion-generated synthetic images for selected classes
Validation and test sets remain real-only, so performance is measured on real unseen data.
## Main Hypothesis
Adding high-quality diffusion-generated samples for underrepresented or data-scarce classes will improve classifier generalization on real unseen data compared with training on real data alone.
## Main Comparison
Real-only training vs. Real + synthetic training
Keep the following fixed:
• classifier architecture
• training procedure
• number of real training samples
• validation/test data
• evaluation metrics
## Metrics
• Macro F1
• Per-class precision
• Per-class recall
• Per-class F1
• Accuracy
Macro F1 is especially useful because each class contributes equally to the final score.
## Additional Research Question 1
How does the ratio of synthetic to real training samples affect classifier performance?
Possible ratios:
• 0% synthetic
• 25%
• 50%
• 100%
• 200%
This checks whether there is an optimal amount of synthetic augmentation and whether too much synthetic data starts to reduce performance.
## Additional Research Question 2
How do the quality and diversity of diffusion-generated images affect classifier performance?
Investigate whether artifacts, low diversity, or unrealistic synthetic features cause the classifier to learn synthetic-specific shortcuts instead of useful class features.


## Data description summary
Dataset: CUB-200-2011, 200 bird species, 11,788 images.

| Split | Images | Per class |
| --- | --- | --- |
| train (pool for k-shot subsets) | 4,952 | 20–25 |
| validation | 1,000 | 5 |
| test (official CUB test) | 5,794 | 11–30 |

42 duplicate training images were removed, 30 of them copies of test images.
Split reasoning, leakage check and preprocessing: [docs/data_split.md](docs/data_split.md).

## Download the data
Run script:
```bash
python src/load_data.py   # downloads and applies splits/cub_splits.csv
python src/leakage.py     # optional: verify there are no duplicates across splits
```

Data loaders (from `src/`); images per run are listed in `splits/manifests/`:
```python
from preprocessing import get_loader
train = get_loader("train", k_shot=10, seed=0, augment=True)  # 4,000 entries: 2,000 originals + 2,000 augmented
val, test = get_loader("val"), get_loader("test")
```

EDA is available in __notebooks/eda.ipynb__ (run on the earlier 15% validation split)

## Train the classifier

The classifier is ImageNet-pretrained ResNet-18 with a frozen backbone and frozen BatchNorm
statistics. Only a new 512-to-200 linear head is trained. The default real-only baseline uses
`train_10shot_seed0.csv` (2,000 real images), deterministic preprocessing, and all 1,000 validation
images. Training selects the best checkpoint by validation macro F1 and never evaluates test.

From the project root, using the environment containing `requirements.txt` dependencies:

```bash
python src/train_classifier.py --seed 0
python src/train_classifier.py --seed 0 --device cuda --amp --output outputs/baseline_gpu_seed0
```

To isolate the training-preprocessing change, run a real-only full-image control with the same
2,000 real images and training settings:

```bash
python src/train_classifier.py --seed 0 --full-image --device cuda --amp
```

This saves `outputs/baseline_full_image_10shot_seed0`. It uses direct whole-image resizing to
224 x 224 for training, documenting the earlier full-image preprocessing variant. There are no
additional augmented/synthetic entries. Validation/test retain the shared resize/center-crop
preprocessing. `--full-image` is a separate real-only control and cannot be combined with
`--augment` or `--synthetic-manifest`.

The default device is CUDA when available, otherwise CPU. `--device cuda` requires a CUDA-enabled
PyTorch installation. AMP is optional and requires CUDA. The first training run downloads ImageNet
weights if they are not cached. Data must already be saved with `python src/load_data.py`.

Defaults are a starting configuration for validation pilots: AdamW, learning rate `0.001`, weight
decay `0.0001`, batch size `64`, `1,000` optimizer steps, validation every `100` steps. These values
are configurable; freeze the chosen settings and step budget across experimental arms.
Use `--max-steps 2 --eval-every 2 --output outputs/pilot` for a short pipeline check (validation
still uses the complete split). `--num-workers 0` is the portable default for Windows and notebooks.

For standard augmentation, start a new run with `--augment`; this concatenates the real originals
with one fixed augmented view per real image, doubling the dataset to 4,000 entries at 10 shots.
The original half uses the baseline's resize-shorter-side-to-256 and center-crop-to-224 transform.
Use the same seed and other settings.
For the three paired subsets, repeat with `--seed 0`, `--seed 1`, and `--seed 2`. The optional
`--k-shot 5` and `--k-shot 20` settings use the corresponding existing manifests.

Each run creates a new output directory containing `config.json` (parameters, software/hardware,
class mapping, source and manifest hashes), `history.csv`, `best.pt`, `runtime.json`, validation
metrics, per-class precision/recall/F1, and predictions associated with image IDs. Existing output
directories are rejected to preserve previous results. The runtime measures the training loop,
periodic validation and checkpoint writes, excluding setup and final evaluation. Checkpoints
support inference; exact interrupted-training resume is not implemented.

Evaluate a saved checkpoint without downloading weights again:

```bash
python src/evaluate_classifier.py outputs/baseline_10shot_seed0/best.pt --output outputs/val_review
# Run test only after all experimental settings and checkpoint choices are fixed.
python src/evaluate_classifier.py outputs/baseline_10shot_seed0/best.pt --split test --output outputs/final_test_seed0
```

## Repeat the augmentation comparison

Keep the same real subset and training settings as the baseline. The saved seed-0 baseline used
10 shots per class, 1,000 optimizer steps, batch size 64, AdamW with learning rate 0.001 and
weight decay 0.0001, validation every 100 steps, CUDA and AMP. The standard arm uses 2,000 real
originals plus 2,000 fixed augmented views, one crop/flip version of each real image. Originals
use resize-shorter-side-to-256 and center-crop-to-224 preprocessing, matching the original baseline.
The real + synthetic arm appends 10 generated images per class to
the same real subset. Both real and generated images use the baseline's resize-shorter-side-to-256
and center-crop-to-224 preprocessing, with no random crops/flips.
Both comparison arms therefore contain 4,000 entries: 2,000 originals plus 2,000 additional images.
Augmented views are recreated lazily using a seed derived from the image ID and run seed. The
same augmented entry stays identical across accesses and workers; no extra JPG files are required.

Extract the generated pool to `data/synthetic/sd15_wildlife/`, retaining `metadata.csv` and the
200 CUB species directories. The metadata's image IDs are relative to `data/` without `.jpg`.
The `sheets/` directory contains contact sheets and is excluded from training.

```bash
# Standard augmentation: 2,000 originals + 2,000 fixed crop/flip versions = 4,000 entries.
python src/train_classifier.py --seed 0 --augment --device cuda --amp

# Validate labels, image decoding and generation MD5s; check val/test duplicates.
python src/prepare_synthetic.py --device cuda

# Real + synthetic: 2,000 real plus 2,000 generated images, both resize + center crop.
python src/train_classifier.py --seed 0 --device cuda --amp --synthetic-manifest data/synthetic/sd15_wildlife/prepared/manifest.csv
```

Preparation writes a manifest with image SHA-256 hashes and `leakage_report.json` to a new
`prepared/` directory. It checks MD5 equality and ResNet-18 embedding cosine similarity at the
project's 0.93 threshold against real validation and test images, ignoring class labels. Matches
are excluded and accepted counts are reported; this uses no classifier test metrics. Manual
species-fidelity/quality review is separate and is not performed by the script.

Both comparison loaders shuffle a combined dataset of 4,000 entries. Approximately half are
original real views and half are augmented or synthetic views. Keep the optimizer-step budget fixed across
arms, rather than the number of epochs. Both sources in real + synthetic training receive RGB
conversion, aspect-ratio-preserving resize of the shorter side to 256, center crop to 224 x 224,
tensor conversion and ImageNet normalization. The baseline, both arms' original halves,
synthetic images and validation/test now share this deterministic preprocessing. The standard
arm's additional 2,000 views retain their fixed random crops/flips. Compare both arms against
the center-crop real-only baseline. The full-image control is a historical diagnostic reference.
`--augment` cannot
be combined with `--synthetic-manifest`. Validation/test remain real-only. Images are read lazily,
and their hashes are verified before training. The run config records the synthetic manifest hash,
total/original/augmented/synthetic image counts, augmentation policy and synthetic per-class counts.

Default output directories are `outputs/standard_aug_10shot_seed0` and
`outputs/real_synthetic_10shot_seed0`. Existing results are never overwritten; choose a new
`--output` to rerun. Repeat with seeds 1 and 2 when extending the paired comparison, including a
matching real-only baseline for each seed. The synthetic pool is reused across seeds.

Regression checks run offline with `python -m unittest discover -s tests -v`; they do not require
ImageNet downloads.

Main observations:
- There is almost no class disbalance in the dataset, no missing images.
- Some pictures are in a grayscale, so need to convert them to RGB during training
- Image size is different, so need to resize or crop
- 2 duplicate images found by md5; an embedding search found 64 duplicate pairs in total (see docs/data_split.md)
