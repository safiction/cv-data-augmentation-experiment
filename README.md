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
train = get_loader("train", k_shot=10, seed=0, augment=True)  # 2,000 images, train_10shot_seed0.csv
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

The default device is CUDA when available, otherwise CPU. `--device cuda` requires a CUDA-enabled
PyTorch installation. AMP is optional and requires CUDA. The first training run downloads ImageNet
weights if they are not cached. Data must already be saved with `python src/load_data.py`.

Defaults are a starting configuration for validation pilots: AdamW, learning rate `0.001`, weight
decay `0.0001`, batch size `64`, `1,000` optimizer steps, validation every `100` steps. These values
are configurable; freeze the chosen settings and step budget across experimental arms.
Use `--max-steps 2 --eval-every 2 --output outputs/pilot` for a short pipeline check (validation
still uses the complete split). `--num-workers 0` is the portable default for Windows and notebooks.

For standard augmentation, start a new run with `--augment`; use the same seed and other settings.
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

The diffusion arm needs a mixed real/synthetic loader and is a later extension. Regression checks
run offline with `python -m unittest discover -s tests -v`; they do not require ImageNet downloads.

Main observations:
- There is almost no class disbalance in the dataset, no missing images.
- Some pictures are in a grayscale, so need to convert them to RGB during training
- Image size is different, so need to resize or crop
- 2 duplicate images found by md5; an embedding search found 64 duplicate pairs in total (see docs/data_split.md)
