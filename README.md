# Synthetic Data Augmentation with Diffusion Models

Does diffusion-generated training data improve bird classification when real labeled data is limited?
We compare real-only training, standard augmentation and reviewed diffusion images on all
200 CUB-200-2011 species. The hypothesis is that correctly labeled synthetic images improve
generalization; realistic images of the wrong species may hurt it.

## Experiment and current status

All primary runs use 10 real images per species, an ImageNet-pretrained ResNet-18 with
frozen backbone/BatchNorm and a trainable 200-class head. Seeds 0–2 pair real subsets and
classifier initialization across approaches. AdamW, batch size 64 and 1,000 optimizer steps
are fixed; validation macro F1 selects checkpoints every 100 steps.

| Primary approach | Real images | Added entries | Total | Available seeds |
| --- | ---: | ---: | ---: | --- |
| Real-only | 2,000 | 0 | 2,000 | 0, 1, 2 |
| Standard augmentation | 2,000 | 2,000 fixed crop/flip views | 4,000 | 0, 1, 2 |
| Reviewed synthetic | 2,000 | 1,440 generated images | 3,440 | 0, 1, 2 |

As checked against `main` on **9 November 2026**, all nine primary runs and six
20-shot references have validation and test results. The 20-shot groups compare 4,000 real
images against 4,000 real + 1,440 synthetic images. Standard seed 0 uses Karina's complete
validation/test/runtime bundle (`archive/standard_aug_10shot_seed0_original`); seeds 1–2
use Artem's T4 runs, as recorded in `src/evaluation_selection.py`. The T4 seed-0 run remains
an alternative covered by the earlier replay, not the final selected bundle.

Final 10-shot test macro F1 (mean ± sample SD): real-only **54.82% ± 0.59 pp**,
standard augmentation **55.25% ± 0.34 pp**, reviewed synthetic **52.25% ± 0.96 pp**.
See the [evaluation report](docs/WEEKLY_REPORT_2026-11-03.md) and
[evaluation files](reports/evaluation_2026-11-03/README.md) for paired comparisons,
20-shot results, failure examples and provenance limits. The 20-shot reference uses twice
as many real images and cannot isolate augmentation effects.

Macro F1 is primary; also report accuracy, per-class precision/recall/F1, paired changes
and mean ± sample SD across seeds. Record generation and training costs separately with
hardware. Controlled synthetic ratios, quality/diversity comparisons, 5-shot and fine-tuning
remain optional research questions; the reviewed-pool comparison changes quality and quantity together.

## Setup

Run all commands from the repository root. Use Python 3.13 for the validated CPU replay:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-replay.txt
```

On Windows, use `py -3.13 -m venv .venv` and activate with
`.venv\Scripts\Activate.ps1`. The replay dependencies cover data loading, classifier training,
evaluation and tests. Install `requirements.txt` for generation, figures and notebooks too.
For GPU runs, install a matching CUDA-enabled torch/torchvision pair instead of the CPU
wheels before installing dependencies. See [environment setup and troubleshooting](docs/environment.md).
Pillow **12.2.0** is required for the validated image preprocessing; other builds previously
changed validation predictions. These files pin direct replay dependencies, not a complete
lock of every original training environment.

## Data

CUB-200-2011 contains 11,788 images of 200 bird species. The saved split removes 42 training
duplicates, including 30 train copies of test photos, and preserves the official test set.

| Split | Images | Per species |
| --- | ---: | ---: |
| Training pool | 4,952 | 20–25 |
| Validation | 1,000 | 5 |
| Official test | 5,794 | 11–30 |

```bash
python src/load_data.py
python src/check_reproducibility.py --output outputs/reproducibility.json
```

The first command downloads `birder-project/CUB_200_2011` and saves the processed dataset
under `data/processed/cub_200_2011/`. It applies tracked split IDs; do not regenerate splits
for ordinary runs. The second command checks saved manifests and result artifacts offline.
`--require-complete` requires all nine primary runs and now passes; the audit covers 15
validation runs. It retains the original T4 seed-0 source-audit scope, while final reporting
uses the selected bundle described above.

`train_{5,10,20}shot_seed{0,1,2}.csv` manifests select balanced, nested real subsets;
`train.csv`, `val.csv` and `test.csv` describe full splits. Labels are zero-based 0–199.
The HF `validation/` image-ID prefix denotes the **official test set**, not our validation split.

All original and synthetic images are converted to RGB, resized to a 256-pixel shorter side,
center-cropped to 224 × 224 and normalized with fixed ImageNet statistics. The standard
arm adds one fixed random resized crop (50–100% area) and possible horizontal flip per real
image. These views are derived from image ID and seed and stay fixed across accesses/workers.
Validation/test always use deterministic preprocessing and real images only.

The reviewed synthetic pool has 1,440 images across 185 species, including 15 species with
none. Review keeps `ok` and correct-species `artifact` images; quantities are therefore uneven.
Held-out images are used only for evaluation and duplicate exclusion, never prompts or
species-quality review. Duplicate filtering compares synthetic images against validation/test
without labels; it does not use classifier test scores.

Details: [report-ready data section](docs/DATA_SECTION_2026-11-10.md),
[split and leakage protocol](docs/data_split.md), [generation protocol](docs/generation.md).
The older [EDA notebook](notebooks/eda.ipynb) uses the earlier 15% validation split;
its split counts do not describe the final manifests.

## Verify and inspect data

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q -W error
python src/preview_transforms.py
```

Tests need no network downloads. Seven processed-image checks skip if data is absent;
run `load_data.py` first to include them. The preview needs matplotlib and writes
`reports/figures/transforms_preview.jpg`. An optional image-level leakage recheck is
`python src/leakage.py`; it needs processed images and ImageNet weights and rewrites the
leakage report. The ordinary offline audit does not repeat embedding-based leakage detection.

Loader example (with `src/` on Python's import path):

```python
from preprocessing import get_loader

train = get_loader("train", k_shot=10, seed=0, augment=True, num_workers=0)
val = get_loader("val", num_workers=0)
# batch: pixel_values (B, 3, 224, 224), label (B,), image_id (list of strings)
```

## Train and evaluate

```bash
# Each command creates a new directory; existing outputs are rejected.
python src/train_classifier.py --seed 0 --device cuda --amp
python src/train_classifier.py --seed 0 --augment --device cuda --amp
python src/train_classifier.py --seed 0 --device cuda --amp --synthetic-manifest data/synthetic/sd15_wildlife/prepared_accepted/manifest.csv
```

Repeat with seeds 1 and 2 for paired runs. Synthetic training requires the actual accepted
image files under `data/synthetic/sd15_wildlife/`; the archived CSV alone is insufficient.
Its manifest and leakage report are archived in `reports/generation/prepared_sd15_wildlife_accepted/`.
Their image paths are relative to the original prepared directory; after restoring images,
copy the two archived files back before training:

```bash
mkdir -p data/synthetic/sd15_wildlife/prepared_accepted
cp reports/generation/prepared_sd15_wildlife_accepted/manifest.csv data/synthetic/sd15_wildlife/prepared_accepted/
cp reports/generation/prepared_sd15_wildlife_accepted/leakage_report.json data/synthetic/sd15_wildlife/prepared_accepted/
```

Do this restoration before running the synthetic command above. On Windows use equivalent
PowerShell `New-Item -ItemType Directory -Force` and `Copy-Item` commands.
Default outputs are `outputs/baseline_10shot_seed0`, `outputs/standard_aug_10shot_seed0`
and `outputs/real_synthetic_10shot_seed0`.

Without a GPU, use `--device cpu` and omit `--amp`. A pipeline smoke check can use
`--max-steps 2 --eval-every 2 --output outputs/pilot`; it still evaluates all validation images
and is not a comparison run. The first training run downloads ImageNet weights unless cached.

Runs save configuration (including source/input hashes and hardware/software), history,
`best.pt`, runtime, validation metrics, per-class results and image-level predictions.
Training never computes test scores. Checkpoints support inference, not exact interrupted-training resume.

```bash
python src/evaluate_classifier.py outputs/baseline_10shot_seed0/best.pt --split val --device cpu --output outputs/val_review
```

The supplied test predictions are already stored as `reports/test/<run>/test_predictions.csv.gz`.
Recompute their metrics with `python src/report_test_results.py` (requires matplotlib).
No repeat export is needed for this deliverable. For a separately authorized inference replay,
use `--split test` and a fresh output directory with the corresponding selected checkpoint.
The original inference environment records are unavailable; matching CSVs and checkpoint
metadata do not independently prove inference provenance.
The existing [CPU replay evidence](docs/reproducibility.md) confirms all 3,000 standard-arm
validation predictions match the saved T4 results; full retraining was not verified.

Supplementary controls: `--k-shot 20` for more real data, `--all-train` for the full real pool,
and `--full-image` for direct 224 × 224 training resizing. The full-image control cannot be
combined with added images. Combining `--augment --synthetic-manifest ...` adds fixed views
of both real and synthetic images (6,880 entries for the reviewed pool); it is an optional
combined arm, not the primary reviewed-synthetic experiment.

## Generate and review synthetic images

Install `requirements.txt` and CUDA-enabled PyTorch. Restore existing image archives instead
of regenerating them when reproducing the recorded classifier runs.

```bash
python src/generate.py --pilot
python src/review_synthetic.py --pool pilot_photo
python src/generate.py --template wildlife --per-class 10
python src/check_synthetic.py --pool sd15_wildlife
python src/prepare_synthetic.py --device cuda --review reports/generation/review_sd15_wildlife.csv --per-class 10
```

Generation uses Stable Diffusion 1.5, species-name text prompts, 512 × 512 images and FP16
on CUDA. Review each pool image before preparation. Preparation validates image hashes,
keeps `ok`/`artifact` decisions, removes held-out duplicates and selects up to 10 accepted
images per species, retaining partial species. It creates a new prepared directory and a
leakage report; use the emitted manifest path for a new training run. Check
[the generation instructions](docs/generation.md) for pool layout, review rules and seeds.

## Repository and handoff

| Path | Contents |
| --- | --- |
| `src/` | Data, training, generation, evaluation and audit scripts |
| `splits/` | Fixed split IDs, removed duplicates and per-run manifests |
| `tests/` | Split, loader, augmentation, classifier and audit checks |
| `reports/classifier/` | Small saved validation artifacts; see its README for run identities |
| `reports/generation/` | Reviews, prepared manifest and leakage evidence |
| `docs/` | Protocols, report sections, environment and reproducibility evidence |
| `data/`, `outputs/` | Ignored local images, checkpoints and replay outputs |

Checkpoints and image archives are stored outside Git; obtain them from the team for replay.
A fresh clone supports the offline artifact audit without those archives. Original/repeated
runs remain distinct; final standard selection is Karina’s complete seed-0 bundle and T4 seeds 1–2.
The earlier T4 seed-0 validation replay remains separate.
See [PROJECT_PLAN.md](PROJECT_PLAN.md) for responsibilities and
[the 10 November handoff](docs/ARTEM_HANDOFF_2026-11-10.md) for completed work and open dependencies.
