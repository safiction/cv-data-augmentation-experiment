# Project Progress Report — 27 October 2026

## Completed experiments

The project investigates whether diffusion-generated training images help classify 200 CUB-200-2011 bird species when real labeled data is limited. Here, 10-shot and 20-shot mean 10 and 20 real training images per species; augmented and synthetic images are counted separately. We retain the original question and add 20-shot real-only training as a reference for increasing the amount of real data. Karina's newly supplied 10-shot real-only seeds 1 and 2 complete the matched baseline.

All nine primary 10-shot runs are complete, plus three 20-shot reference runs:

| Approach | Real images | Additional training entries | Total entries per run | Seeds |
| --- | ---: | ---: | ---: | --- |
| Real-only baseline, 10-shot | 2,000 | 0 | 2,000 | 0, 1, 2 |
| Standard augmentation, 10-shot | 2,000 | 2,000 fixed crop/flip views | 4,000 | 0, 1, 2 |
| Accepted synthetic, 10-shot | 2,000 | 1,440 generated images | 3,440 | 0, 1, 2 |
| Real-only reference, 20-shot | 4,000 | 0 | 4,000 | 0, 1, 2 |

Standard augmentation adds one fixed random resized crop (50–100% image area) and possible horizontal flip (probability 0.5) per real image. The synthetic arm adds manually reviewed Stable Diffusion 1.5 images without crop/flip variants. All three 10-shot approaches share the corresponding seed's real subset and head-initialization seed. Across seeds, the real subset, head initialization and shuffled training order change; the synthetic pool remains fixed.

All runs use ImageNet-pretrained ResNet-18 with frozen backbone and BatchNorm statistics, a trainable 200-class linear head, cross-entropy and AdamW. Settings are learning rate 0.001, weight decay 0.0001, batch size 64 and 1,000 optimizer steps. Real originals and synthetic images use RGB conversion, aspect-ratio-preserving resize to a 256-pixel shorter side, 224 × 224 center crop and ImageNet normalization. Checkpoints are selected by macro F1 on the same 1,000 real validation images, evaluated every 100 steps.

Earlier supplementary experiments include the original unfiltered synthetic pool, a full-image-resize diagnostic and a full-real-training-pool run, each at seed 0. The original 10-shot baseline seed 0 is reused in the primary comparison. The standard-augmentation summary uses the three T4 runs supplied together; the earlier standard seed-0 run remains archived separately.

## Preliminary comparison

Best-checkpoint validation results, mean ± sample standard deviation across seeds 0–2:

| Approach | Macro F1 | Accuracy |
| --- | ---: | ---: |
| Real-only baseline, 10-shot | 53.56% ± 0.43 pp | 53.87% ± 0.49 pp |
| Standard augmentation, 10-shot | 53.51% ± 0.54 pp | 54.03% ± 0.42 pp |
| Accepted synthetic, 10-shot | 51.31% ± 2.11 pp | 52.10% ± 1.85 pp |
| Real-only reference, 20-shot | 59.69% ± 0.71 pp | 59.77% ± 0.65 pp |

Here, pp means percentage points; SD is computed with `ddof=1`. Real-only 20-shot achieves the highest score, but uses twice as many real images as either augmentation arm. Its advantage therefore cannot establish that augmentation itself is harmful.

Paired macro-F1 changes use the matching real subset for each seed. Positive values favor the first approach:

| Comparison | Seed 0 | Seed 1 | Seed 2 | Mean ± sample SD |
| --- | ---: | ---: | ---: | ---: |
| Standard − real-only 10-shot | +0.38 pp | −0.57 pp | +0.03 pp | −0.05 ± 0.48 pp |
| Accepted synthetic − real-only 10-shot | −4.19 pp | −0.47 pp | −2.11 pp | −2.25 ± 1.86 pp |
| Accepted synthetic − standard | −4.57 pp | +0.10 pp | −2.14 pp | −2.20 ± 2.34 pp |

Standard augmentation gives nearly the same mean macro F1 as the matched real-only baseline, with changes in both directions across seeds. Accepted synthetic scores lower than real-only in all three seeds and varies more across seeds. Under this frozen-backbone setup and fixed training-step budget, the reviewed synthetic pool does not improve validation macro F1.

At seed 0, the reviewed-pool run scores 0.80 pp above the original synthetic-pool run (48.96% vs. 48.16%). This difference does not isolate the effect of quality: filtering also changes the number of images and their distribution across species.

## Observations

- **Review changes the training distribution.** Of 2,000 generated images, 1,120 were marked `ok`, 320 `artifact` (correct species with visible defects), 536 `wrong_species` and 24 `not_a_bird`. The selection code now retains all available accepted images up to 10 per species. All 1,440 accepted images pass the recorded leakage checks. They cover 185 species; 118 species have fewer than 10 synthetic images, including 15 with none. Additional generation is postponed.
- **Several factors may explain the synthetic results.** Unequal class counts, remaining visual defects and differences between generated and real images are plausible contributors, not established causes. At a fixed step budget, synthetic entries also replace some real-image presentations. The available comparisons do not separate these effects.
- **Additional research questions remain partly open.** The original and reviewed pools provide observations about synthetic quantity and quality, but these factors changed together. The prompt pilot informed generation settings; it did not establish the effect of synthetic diversity on classifier performance. Controlled ratio, quality and diversity comparisons remain optional extensions.
- **Results are preliminary.** Three seeds provide descriptive variability, not proof of a general advantage. The same validation set selects checkpoints and supplies reported scores. No classifier test metrics have been computed. Training used different GPUs (RTX 4060 Laptop, T4 and RTX 3070) and library versions, limiting reproducibility across environments; runtimes should be reported with hardware rather than ranked as method efficiency.
- **Saved outputs are consistent.** For all 12 runs, validation IDs, labels, aggregate/per-class metrics and best-checkpoint history were checked against the supplied prediction files. The new baseline runs match the corresponding augmentation runs' real manifests and shared training settings; their recorded code and split hashes match the repository. The [run table](../reports/classifier/comparison_2026-10-27.csv) and [paired changes](../reports/classifier/paired_comparison_2026-10-27.csv) preserve the unrounded results.

## Remaining work

1. By 3 November, freeze experimental choices and validation-selected checkpoints for the completed comparison, confirm shared checkpoint availability and evaluate on the unchanged 5,794-image real test set. No additional baseline training is needed for the planned seeds.
2. Report test macro F1, accuracy, per-class metrics, seed variability and paired changes. Produce the final comparison figure and concise per-species analysis; keep generation and training costs separate and record hardware/software versions.
3. Analyze at least three real test failures and prepare the evaluation section of the final report and presentation. Controlled ratio, quality and diversity experiments remain optional extensions; they are not required to complete the primary comparison.
