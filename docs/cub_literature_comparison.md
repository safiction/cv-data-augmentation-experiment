# CUB-200-2011: comparison with published work

Updated 3 November 2026 from the supplied 1 November draft. This version distinguishes test from validation, includes all three real-only 10-shot seeds, and retains numerical literature comparisons checked in primary papers.

## Our current results

ImageNet-pretrained ResNet-18, frozen backbone/BatchNorm, 200-class linear head, 224-pixel crop, 1,000 optimizer steps. Checkpoints maximize macro F1 on 1,000 validation images. The table below uses the separate **5,794-image official test split**. Mean ± sample SD is across the seeds shown.

| Approach | Real images/species | Seeds | Test macro F1 | Test accuracy |
| --- | ---: | --- | ---: | ---: |
| Real-only | 10 | 0, 1, 2 | 54.82% ± 0.59 pp | 54.50% ± 0.49 pp |
| Standard augmentation — incomplete | 10 | 1, 2 only | 55.14% ± 0.40 pp | 54.96% ± 0.31 pp |
| Reviewed synthetic | 10 | 0, 1, 2 | 52.25% ± 0.96 pp | 52.30% ± 0.86 pp |
| Real-only | 20 | 0, 1, 2 | 61.31% ± 0.19 pp | 61.30% ± 0.21 pp |
| Reviewed synthetic | 20 | 0, 1, 2 | 59.40% ± 0.66 pp | 59.50% ± 0.70 pp |

The selected standard-augmentation T4 seed-0 export is missing. Its supplied replacement is an older checkpoint and is excluded. Metric reconstruction and provenance limitations are documented in the [weekly report](WEEKLY_REPORT_2026-11-03.md).

## Verified published reference points

Compare **accuracy with accuracy**, not published accuracy with our macro F1. These protocols differ, so the values are contextual references rather than a leaderboard for our method.

| Source and exact location | Training setting | Reported CUB accuracy |
| --- | --- | ---: |
| [Visual Engagement, Table 2](https://arxiv.org/html/2104.07767v2) | ImageNet-supervised ResNet-50, frozen features + linear classifier | 62.70% |
| [Visual Engagement, Table 2](https://arxiv.org/html/2104.07767v2) | Same pretrained backbone, end-to-end fine-tuning | 72.15% |
| [Self-Tuning, Table 1, fine-tuning baseline](https://arxiv.org/pdf/2102.12903) | Pretrained ResNet-50; 15%, 30%, 50%, 100% labels | 45.25%, 59.68%, 70.12%, 78.01% |
| [Diff-Mix, Table 1, unaugmented baseline](https://arxiv.org/html/2403.19600v1) | Full-data ResNet-50, 448-pixel input | 86.64% |

Visual Engagement calls its 5,794-image CUB evaluation split “validation” (Table 8), while this project calls the official held-out split “test.” It uses 5,994 training images and a 300-epoch schedule (Table 13), unlike our 2,000/4,000-image, 1,000-step runs. Its frozen-feature result is a relevant architectural reference, but cannot establish the expected score of our smaller backbone. [Paper](https://arxiv.org/html/2104.07767v2).

Self-Tuning's label fractions are not exact balanced 10-/20-shot subsets. Its fine-tuning baseline changes backbone weights; our classifier does not. The paper's Self-Tuning method additionally uses unlabeled data and is not included as a directly comparable baseline. Do not interpolate its learning curve to claim a measured penalty from freezing our backbone. [Paper, Table 1](https://arxiv.org/pdf/2102.12903).

## What the diffusion literature supports

Diff-Mix explicitly reports that unadapted text-to-image generation (Real-Gen) falls below baseline at 10 shots. Its few-shot experiments use ResNet-50 at 224 pixels and synthetic replacement probability 0.2 for 10-shot, versus 0.1 for full-data training. Its proposed method adapts the generator and translates reference images between classes, unlike our unchanged SD 1.5 species-name generation. These observations motivate checking species fidelity and sampling ratios; they do not prove the cause of our result. [Diff-Mix, Section 4.1](https://arxiv.org/html/2403.19600v1).

Our reviewed pool occupies 1,440/3,440 = **41.86%** of 10-shot training entries and 1,440/5,440 = **26.47%** of 20-shot entries. Within each shot count, paired test macro F1 falls by **2.57 pp** and **1.91 pp**, respectively. More real data and a lower synthetic ratio change together, so their effects are not isolated.

## Fixed-step budget: limitation, not a demonstrated explanation

All approaches share 1,000 optimizer steps. Adding synthetic entries reduces the expected fraction of presentations containing real originals. This is a valid comparison under a fixed step budget, but it does not control real-image exposure. Uneven class counts and remaining visual defects are additional possible factors.

The draft's interpolated baseline-at-step-735 comparison is **not an equal-exposure experiment**. It mixes approximate exposure, interpolated validation curves and checkpoint-selected scores; it cannot establish that the synthetic deficit disappears. The loader also retains short final batches, so 1,000 steps need not mean exactly 64,000 image presentations. A controlled follow-up would explicitly log real/synthetic draws and predefine the exposure budget before evaluation, using validation for choices. No such experiment has been completed, and the current test results must not be used for further tuning.

## Changes from the supplied draft

- Replaced outdated “baseline seeds 1–2 missing” and “test not evaluated” statements with the verified current status.
- Separated contextual literature results from matched within-project comparisons.
- Removed the interpolated “~13 pp cost of freezing” and “deficit disappears” claims; neither is a controlled result.
- Did not carry unverified search-excerpt values or inferred training recipes into the numerical reference table. Additional papers can be restored after checking their exact protocol and tables.
