# Real-only baseline — seed 0

## Setup

- Model: ImageNet-pretrained ResNet-18 with a frozen backbone and a trainable 200-class head
- Training data: 2,000 real images (10 per class), seed 0
- Validation data: 1,000 real images (5 per class)
- Training: 1,000 optimizer steps, batch size 64, AdamW, learning rate 0.001
- Hardware: NVIDIA GeForce RTX 4060 Laptop GPU, mixed precision
- Checkpoint selection: highest validation macro F1

## Validation results

| Measure | Result |
| --- | ---: |
| Macro F1 | **0.5315** |
| Top-1 accuracy | **0.5330** |
| Loss | 1.8820 |
| Correct predictions | 533 / 1,000 |
| Best step | 700 |
| Training time, including periodic validation | 444.6 s (7.4 min) |

Macro F1 is close to accuracy because validation contains the same number of images for every class. The best checkpoint is step 700. By step 1,000, training loss falls from 0.290 to 0.145 while validation macro F1 falls from 0.5315 to 0.5162, which indicates overfitting after the selected checkpoint.

## Frequent confusions

| True class | Predicted class | Count |
| --- | --- | ---: |
| Yellow-bellied Flycatcher | Least Flycatcher | 3 |
| Blue Grosbeak | Indigo Bunting | 3 |
| Ring-billed Gull | Herring Gull | 3 |
| Song Sparrow | Savannah Sparrow | 3 |
| Elegant Tern | Caspian Tern | 3 |

These errors mostly involve visually similar species, which is expected in fine-grained bird classification. Eleven classes have validation F1 equal to zero, but each class has only five validation images, so per-class results are noisy and should be treated as diagnostic examples rather than stable estimates.

This is a validation result for one seed. Do not draw a final conclusion about augmentation until all three approaches have been run with seeds 0, 1, and 2. The test set remains unused until the experimental choices are fixed.

## Validation checks

- All 1,000 prediction rows match `splits/manifests/val.csv` in image ID, order, label, and class name.
- The validation set contains exactly five images for each of the 200 classes and no repeated image IDs.
- Accuracy and macro F1 recomputed from the predictions match `val_metrics.json`; per-class precision, recall, F1, and support match `val_per_class.json`.
- The best history row matches `runtime.json` and the saved aggregate metrics.
- The checkpoint file is present. Independent checkpoint inference was not rerun during this artifact-level review.
