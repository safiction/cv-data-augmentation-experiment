# Project Progress Report — 27 October 2026

## Completed experiments

The project investigates whether diffusion-generated training images help classify 200 CUB-200-2011 bird species when real labeled data is limited. Here, 10-shot and 20-shot mean 10 and 20 real training images per species; augmented and synthetic images are counted separately. We retain the original question and add 20-shot real-only training as a reference for increasing the amount of real data. The matched 10-shot real-only baseline is available for seed 0; seeds 1 and 2 are scheduled for next week.

The current comparison contains nine completed runs, three seeds per approach:

| Approach | Real images | Additional training entries | Total entries per run | Seeds |
| --- | ---: | ---: | ---: | --- |
| Real-only, 20-shot | 4,000 | 0 | 4,000 | 0, 1, 2 |
| Standard augmentation, 10-shot | 2,000 | 2,000 fixed crop/flip views | 4,000 | 0, 1, 2 |
| Accepted synthetic, 10-shot | 2,000 | 1,440 generated images | 3,440 | 0, 1, 2 |

Standard augmentation adds one fixed random resized crop (50–100% image area) and possible horizontal flip (probability 0.5) per real image. The synthetic arm adds manually reviewed Stable Diffusion 1.5 images without crop/flip variants. Standard and synthetic runs share the corresponding seed's real subset and head-initialization seed. Across seeds, the real subset, head initialization and shuffled training order change; the synthetic pool remains fixed.

All runs use ImageNet-pretrained ResNet-18 with frozen backbone and BatchNorm statistics, a trainable 200-class linear head, cross-entropy and AdamW. Settings are learning rate 0.001, weight decay 0.0001, batch size 64 and 1,000 optimizer steps. Real originals and synthetic images use RGB conversion, aspect-ratio-preserving resize to a 256-pixel shorter side, 224 × 224 center crop and ImageNet normalization. Checkpoints are selected by macro F1 on the same 1,000 real validation images, evaluated every 100 steps.

Earlier experiments include the 10-shot real-only baseline, the original unfiltered synthetic pool, a full-image-resize diagnostic and a full-real-training-pool run, each at seed 0. The 10-shot baseline anchors the comparison at fixed real-data availability; the other runs remain supplementary experiments. The current standard-augmentation summary uses the three T4 runs supplied together, rather than substituting the earlier standard seed-0 run.

## Preliminary comparison

Best-checkpoint validation results, mean ± sample standard deviation across seeds 0–2:

| Approach | Macro F1 | Accuracy |
| --- | ---: | ---: |
| Real-only, 20-shot | 59.69% ± 0.71 pp | 59.77% ± 0.65 pp |
| Standard augmentation, 10-shot | 53.51% ± 0.54 pp | 54.03% ± 0.42 pp |
| Accepted synthetic, 10-shot | 51.31% ± 2.11 pp | 52.10% ± 1.85 pp |

Here, pp means percentage points; SD is computed with `ddof=1`. Real-only 20-shot achieves the highest score, but uses twice as many real images as either augmentation arm. Its advantage therefore cannot establish that augmentation itself is harmful.

For the paired standard-versus-synthetic comparison, synthetic macro F1 changes by −4.57, +0.10 and −2.14 pp for seeds 0, 1 and 2, respectively: −2.20 pp on average. Synthetic does not outperform standard augmentation consistently, and its scores vary more across seeds.

At seed 0, the unchanged 10-shot real-only baseline scores 53.15% macro F1. Standard augmentation scores 53.53% (+0.38 pp), and accepted synthetic scores 48.96% (−4.19 pp). The reviewed-pool run scores 0.80 pp above the original synthetic-pool run (48.16%). This difference does not isolate the effect of quality: filtering also changes the number of images and their distribution across species.

## Observations

- **Review changes the training distribution.** Of 2,000 generated images, 1,120 were marked `ok`, 320 `artifact` (correct species with visible defects), 536 `wrong_species` and 24 `not_a_bird`. The selection code now retains all available accepted images up to 10 per species. All 1,440 accepted images pass the recorded leakage checks. They cover 185 species; 118 species have fewer than 10 synthetic images, including 15 with none. Additional generation is postponed.
- **Several factors may explain the synthetic results.** Unequal class counts, remaining visual defects and differences between generated and real images are plausible contributors, not established causes. At a fixed step budget, synthetic entries also replace some real-image presentations. The available comparisons do not separate these effects.
- **Additional research questions remain partly open.** The original and reviewed pools provide observations about synthetic quantity and quality, but these factors changed together. The prompt pilot informed generation settings; it did not establish the effect of synthetic diversity on classifier performance. Controlled ratio, quality and diversity comparisons remain optional extensions.
- **Results are preliminary.** Three seeds provide descriptive variability, not proof of a general advantage. The same validation set selects checkpoints and supplies reported scores. No classifier test metrics have been computed. Training used different GPUs (RTX 4060 Laptop, T4 and RTX 3070), so runtimes should be reported with hardware rather than ranked as method efficiency.
- **Saved outputs are consistent.** For all nine runs, validation IDs, labels, aggregate/per-class metrics and best-checkpoint history were checked against the supplied prediction files. The standard-augmentation summary in `rez.csv` agrees with its archive.

## Remaining work

1. By 3 November, complete the two remaining 10-shot real-only baseline runs, seeds 1 and 2, using the same real manifests and training settings as the corresponding augmentation runs. Reuse the existing seed-0 baseline. Compute paired macro-F1 changes for standard and synthetic augmentation versus the baseline across seeds 0–2.
2. Update the project plan to retain the matched 10-shot comparison as the primary experiment and describe 20-shot real-only as an additional reference. Record the smaller, uneven reviewed synthetic pool and the revised run budget. Defer optional ratio, quality and diversity experiments until the primary comparison is complete.
3. Consolidate run configurations, predictions, summaries and manifest versions in the repository, keeping checkpoints in shared storage. Preserve separate identities for original and reviewed synthetic pools and repeated seed-0 runs. Produce the final comparison figure and concise per-species analysis.
4. Freeze experimental choices and selected checkpoints, then evaluate on the unchanged 5,794-image real test set. Report macro F1, accuracy, per-class metrics and seed variability, with training and generation costs listed separately. Analyze at least three test failures and prepare the evaluation section of the final report and presentation.
