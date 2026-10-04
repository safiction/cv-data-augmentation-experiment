# Project Plan

## 1. Project scope

**Question:** Does diffusion-generated training data improve bird classification when real labeled data is limited?

**Task:** single-label image classification across all 200 CUB-200-2011 bird species. Compare real-only training, standard augmentation, and diffusion augmentation using the same classifier. We hypothesize that correctly labeled synthetic images improve generalization; inaccurate species labels may hurt it.

## 2. Dataset preparation

- **Dataset:** CUB-200-2011; input: bird image; target: species label. Safina completed the dataset review and EDA.
- **Split:** 4,952 train pool / 1,000 validation / 5,794 test images. Preserve the official test set; validation is 5 random images per class from the official training data (stratified, seed 42). A fixed count keeps validation balanced and leaves at least 20 training images per class for the 20-shot extension. Reasoning and noise estimates: `docs/data_split.md`.
- **Limited-data subset:** sample 10 real training images per class, giving 2,000 images per run. This deliberately creates data scarcity while keeping all species. Subset IDs for seeds 0, 1, 2 are saved in `splits/cub_splits.csv`; subsets are nested across 5/10/20 shots. Unused training images stay outside the core experiments.
- **EDA findings:** no missing images; training classes are nearly balanced.
- **Duplicates and leakage:** an md5 + ResNet-18 embedding search found 30 official train images that duplicate 27 test images, and 15 duplicate pairs inside official training data (incl. the exact pair from the EDA). The train copies (42 images) were removed; the official test set is unchanged. No duplicates remain across train, validation and test.
- **Preprocessing:** convert grayscale images to RGB, resize the shorter side to 256, center-crop to 224×224, and use ImageNet normalization to match the classifier's pretrained inputs. Augmented training uses random resized crops (50–100% of the area) and horizontal flips instead; evaluation preprocessing stays deterministic.

Validation and test contain only real images, so evaluation measures performance on real data. Never use their images for generation or synthetic-image selection.

## 3. Experimental plan

**Classifier:** ImageNet-pretrained [ResNet-18](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html), a compact baseline, with a new 200-class head. Train only the head using cross-entropy; freeze the backbone and BatchNorm statistics. This limits trainable parameters and computation with only 10 examples per class. Results apply to this fixed-feature setting.

**Generator:** pretrained [Stable Diffusion 1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5), using species-name prompts and 512×512 images. Use FP16 and one image at a time to reduce GPU memory demand. Reusing the generator avoids diffusion-model training. Check species fidelity in the pilot: a realistic bird may still be the wrong species.

| Approach | Training data | Purpose |
| --- | --- | --- |
| Real-only baseline | 2,000 real images | Reference performance |
| Standard augmentation | Same real images + random crops/flips | Tests whether ordinary augmentation already helps |
| Diffusion augmentation | Standard augmentation + 2,000 synthetic images, 10 per class | Tests added value at a simple 1:1 synthetic-to-real ratio |

**Budget:** three approaches × three paired seeds = **9 classifier runs**, plus setup/pilots. Pair real subsets and classifier initialization across approaches; keep optimizer, batch size, and training-step budget fixed to avoid giving one approach extra training. Reuse one synthetic pool, so seed variability does not measure variability between generated pools.

| Evaluation measure | Reason |
| --- | --- |
| **Macro F1 — primary** | Weights species equally and combines precision and recall |
| **Top-1 accuracy** | Reports the overall fraction of correct predictions |
| **Per-class precision, recall, F1** | Identifies which species benefit or deteriorate |
| **Generation and training time, separately** | Shows the computational cost of improvements; record hardware |

Select checkpoints on validation macro F1. After fixing experimental choices, evaluate on test; report mean ± standard deviation across seeds and paired macro-F1 changes in percentage points. Discuss seed variability and at least three real test failure examples.

**Optional extensions, outside the nine-run scope:** synthetic-to-real ratios 0.5:1 and 2:1; 5/20 real images per class; fine-tuning the final classifier block; filtered versus unfiltered synthetic data; varied prompts/backgrounds; deliberately imbalanced training classes. Change one factor at a time; keep synthetic counts equal when comparing quality or diversity.

## 4. Implementation plan

1. **Prepare data:** extend `src/load_data.py`; save split/subset IDs and build loaders with Hugging Face `datasets`, Pillow, PyTorch, and torchvision.
2. **Build training:** implement the shared classifier pipeline, checkpointing, configuration files, and runtime logging.
3. **Generate and review images:** use `diffusers`, `transformers`, and `accelerate`. First inspect 50 images across 10 species; finalize acceptance rules before generating the full pool. Save prompts, seeds, labels, and review decisions.
4. **Run and evaluate:** execute the comparison matrix; use scikit-learn for metrics, pandas for tables, and matplotlib for figures. Save predictions and reproducible run commands.

**Resources:** laptops for development; Colab T4 or Kaggle for GPU runs. Time a pilot before full execution. Store image archives/checkpoints in shared Google Drive; code, configurations, subset IDs, and small results in Git. Save outputs outside temporary notebook sessions.

## 5. Team responsibilities

### By 6 October — project-plan report

| Member | Responsibility | Status |
| --- | --- | --- |
| Karina | Question, ML task, hypothesis, additional research questions | Documented in README |
| Safina | Dataset description and EDA | Complete; loading code and saved EDA results available |
| Artem | Split, seeds, preprocessing, subset sampling, duplicate handling | Complete; see `docs/data_split.md` |
| Arina | Metrics and justification, experiments, implementation outline, work schedule | Documented in this plan |

### Following weeks — deadlines in 2026

| Deadline | Karina | Artem | Safina | Arina |
| --- | --- | --- | --- | --- |
| **13 Oct** | Training code and working baseline | Transforms, loaders, subset IDs | Generation pipeline and pilot | Metrics/logging; baseline evaluation |
| **20 Oct** | Standard augmentation and first synthetic comparison | Mixed-data loader; review half the synthetic pool | Generate pool; review other half | Check comparison settings; initial tables |
| **27 Oct** | Complete three real-only runs | Complete three standard-augmentation runs | Complete three synthetic-augmentation runs | Compare validation results; list remaining experiments |
| **3 Nov** | Freeze checkpoints; export test predictions | Verify reproducibility | Finalize generation quality/cost records | Final test metrics, runtime tables, figures |
| **10 Nov** | Classifier section/slides; demo | Data section/slides; README/environment | Generation section/slides; assemble presentation | Evaluation section/slides; edit two-page summary |

Each member analyzes one test failure, writes their report section, and rehearses the demo. Completed runs count toward the nine-run total. Review active workload weekly; assign optional experiments only if capacity remains.

## Readiness for 6 October

**Safina's dataset work and Artem's experimental setup are complete.** Then combine and check the report sections.

Working models, generation pilots, and final training parameters are later implementation tasks. The first working baseline is due on **13 October**.
