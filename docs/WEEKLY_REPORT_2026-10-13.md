# First Working Model Report — 13 October 2026

## Data loading and preprocessing

We use CUB-200-2011, containing 11,788 images from 200 bird species. The implemented split contains 4,952 training-pool images, 1,000 validation images (5 per class), and the unchanged official test set of 5,794 images. We removed 42 duplicate training images to prevent leakage. The core baseline uses the seed-0 manifest with 10 real training images per class, or 2,000 images in total.

`src/load_data.py` downloads and prepares the dataset. `src/preprocessing.py` builds reproducible PyTorch datasets and loaders from the CSV manifests in `splits/manifests/`. Each sample contains the normalized image tensor, numeric label, and image ID.

All images are converted to RGB. Baseline training and evaluation resize the shorter side to 256 pixels, center-crop to 224 × 224, and apply ImageNet normalization. The validation and test loaders are deterministic and do not shuffle. The standard-augmentation pipeline, prepared for the next experiment, uses a random resized crop covering 50–100% of the image and a horizontal flip.

## Baseline model and training

The first working baseline is an ImageNet-pretrained ResNet-18 (`IMAGENET1K_V1`). Its convolutional backbone and BatchNorm statistics are frozen; only a new 512-to-200 linear classification head is trained with cross-entropy loss. The implementation is in `src/classifier.py` and `src/train_classifier.py`.

| Setting | Value |
| --- | --- |
| Training data | 2,000 real images, 10 per class, seed 0 |
| Validation data | 1,000 real images, 5 per class |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Batch size | 64 |
| Training budget | 1,000 optimizer steps |
| Validation interval | 100 steps |
| Checkpoint selection | Highest validation macro F1 |
| Hardware | NVIDIA GeForce RTX 4060 Laptop GPU, mixed precision |

The run saves its configuration, training history, runtime, best checkpoint, aggregate and per-class metrics, and image-level predictions. The reported checkpoint is stored as `models/week1.pt`; its evaluation artifacts are in `reports/classifier/baseline_seed0/`.

## Initial results

| Validation measure | Result |
| --- | ---: |
| Macro F1 | **0.5315** |
| Top-1 accuracy | **0.5330** |
| Cross-entropy loss | 1.8820 |
| Correct predictions | 533 / 1,000 |
| Best training step | 700 |
| Training time, including periodic validation | 444.6 s (7.4 min) |

These are validation results from seed 0. The test set has not been used. Final conclusions require the same experiment for seeds 1 and 2 and comparison with standard and diffusion augmentation.

`src/summarize_results.py` validates each saved run against the validation manifest, recomputes its metrics from image-level predictions, and collects the configuration, best step, metrics, and runtime into one comparison CSV. The current table contains the seed-0 baseline and will use the same format for the remaining runs.

## Additional progress: synthetic data

Safina implemented the Stable Diffusion 1.5 generation and review pipeline. A 50-image pilot compared the `photo` and `wildlife` prompt templates using the same generation seeds. The `wildlife` template produced fewer wrong-species images (3 versus 6), so it was selected for the full pool. The resulting pool contains 2,000 images, 10 per class, and took approximately 97 minutes to generate on an RTX 3070. No matches with validation or test images were detected. Manual review of the full pool is still pending, and these images were not used in the baseline reported above.

## Problems and observations

- Eight source images are grayscale, so explicit RGB conversion is required.
- Image dimensions vary, requiring consistent resizing and cropping.
- Duplicate detection found leakage between the original train and test partitions; removing the 42 affected training images resolved it without changing the official test set.
- Validation macro F1 peaks at step 700 and falls to 0.5162 by step 1,000 while training loss continues to decrease, indicating overfitting after the selected checkpoint.
- Frequent errors occur between visually similar species, including Yellow-bellied versus Least Flycatcher and Ring-billed versus Herring Gull.
- During evaluation, we found that manifest hashes differed between Windows and Linux because of line endings. This is fixed: hashes are now platform-independent, all manifests are verified, and the existing baseline checkpoint remains supported. The fix affects only integrity checks, so baseline retraining is not required.
- The baseline was trained before `image_id` was added to loader batches. The pixel transforms are identical in the current version, so this metadata-only change does not affect model inputs or results.

## Repository

[GitHub repository](https://github.com/safiction/cv-data-augmentation-experiment)

Implementation details are available in [`data_split.md`](data_split.md). The seed-0 evaluation artifacts are in [`reports/classifier/baseline_seed0/`](../reports/classifier/baseline_seed0/), and the checkpoint is [`models/week1.pt`](../models/week1.pt).
