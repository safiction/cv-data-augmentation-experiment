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

## Generate synthetic data
Stable Diffusion 1.5, text-to-image from species names; needs a CUDA GPU (8 GB is enough):
```bash
python src/generate.py --pilot                   # 10 species x 5 images, data/synthetic/pilot_photo
python src/review_synthetic.py --pool pilot_photo   # contact sheets, review table, val/test leakage check
python src/generate.py --per-class 10            # full pool: 2,000 images, data/synthetic/sd15_photo
```
Each pool gets a manifest `splits/manifests/synthetic_{pool}.csv` (`image_id, label, class_name` + prompt, seed, time).
Settings, seeds, review and acceptance rules: [docs/generation.md](docs/generation.md).

EDA is available in __notebooks/eda.ipynb__ (run on the earlier 15% validation split)

Main observations:
- There is almost no class disbalance in the dataset, no missing images.
- Some pictures are in a grayscale, so need to convert them to RGB during training
- Image size is different, so need to resize or crop
- 2 duplicate images found by md5; an embedding search found 64 duplicate pairs in total (see docs/data_split.md)
