# Project Plan

## 1. Project scope

**Question:** Does diffusion-generated training data improve bird classification when real labeled data is limited?

**Task:** single-label image classification across all 200 CUB-200-2011 bird species. Compare real-only training, standard augmentation, and diffusion augmentation using the same classifier. We hypothesize that correctly labeled synthetic images improve generalization; inaccurate species labels may hurt it.

## 2. Dataset preparation

- **Dataset:** CUB-200-2011; input: bird image; target: species label. Safina completed the dataset review and EDA.
- **Split:** 4,952 train-pool / 1,000 validation / 5,794 test images. The test set is the official CUB test split and remains unchanged. Validation contains exactly five random images per class from the official training split (stratified, seed 42). This keeps validation balanced and leaves at least 20 training images per class for the 20-shot extension. Full reasoning: `docs/data_split.md`.
- **Limited-data subset:** sample 10 real train-pool images per class, giving 2,000 images per run. Manifests for seeds 0, 1, and 2 define the paired subsets used by all three approaches. Unused train-pool images remain outside the core experiments.
- **EDA findings:** no missing images; training classes are nearly balanced.
- **Duplicates and leakage:** 42 duplicate training images were removed, including train copies of test images. No duplicates remain across train, validation, and test; the official test set was not modified.
- **Preprocessing:** convert grayscale images to RGB and use ImageNet normalization to match the classifier's pretrained inputs. Baseline, evaluation, both comparison arms' real originals and generated images resize the shorter side to 256 (preserving aspect ratio) and center-crop to 224×224. Standard augmentation adds one fixed random resized crop/flip view per original. Real + synthetic uses no random crops/flips. **Update — 26 October 2026:** standard augmentation has 4,000 entries; real + reviewed synthetic has 3,440. Both share original-image preprocessing; the full-image real-only control is retained as a historical diagnostic.

Validation and test contain only real images, so evaluation measures performance on real data. Never use their images for generation or synthetic-image selection.

## 3. Experimental plan

**Classifier:** ImageNet-pretrained [ResNet-18](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html), a compact baseline, with a new 200-class head. Train only the head using cross-entropy; freeze the backbone and BatchNorm statistics. This limits trainable parameters and computation with only 10 examples per class. Results apply to this fixed-feature setting.

**Generator:** pretrained [Stable Diffusion 1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5), using species-name prompts and 512×512 images. Use FP16 and one image at a time to reduce GPU memory demand. Reusing the generator avoids diffusion-model training. Check species fidelity in the pilot: a realistic bird may still be the wrong species.

**Update — 26 October 2026:** the primary comparison retains 10 real images per species and the same real subset for each paired seed across all three approaches. Additional images do not change the real-data shot count.

| Approach | Training data | Purpose |
| --- | --- | --- |
| Real-only baseline | 2,000 real images | Reference performance |
| Real + standard augmentation | 2,000 real originals + 2,000 fixed crop/flip versions, one per real image | Tests ordinary augmentation in a 4,000-entry dataset |
| Real + reviewed synthetic | Same 2,000 real images + 1,440 accepted synthetic images; both sources use resize-to-256 and center-crop-to-224, with no random crops/flips | Tests the reviewed pool in a 3,440-entry dataset at an overall synthetic-to-real ratio of 0.72:1 |

**Reviewed pool — update, 26 October 2026:** retain `ok` (1,120) and `artifact` (320; correct species with visual defects) images, up to 10 per species. Keep available accepted images when fewer than 10 remain. Counts vary by species, including species with none; additional generation is postponed. Reuse this fixed pool across seeds.

**Budget — update, 26 October 2026:** the primary comparison remains **9 runs** (three approaches × three paired seeds); **7 are complete**, with 10-shot real-only seeds 1 and 2 remaining. The additional 20-shot real-only comparison comprises **3 completed runs**, for **12 planned comparison runs in total**. Earlier diagnostic runs and repeated runs are tracked separately. Pair real subsets and classifier initialization across the primary approaches; keep optimizer, batch size, and training-step budget fixed to avoid giving one approach extra training. Reusing one synthetic pool means seed variability does not measure variability between generated pools.

**20-shot reference — update, 26 October 2026:** compare against real-only training on 4,000 real images (20 per species), seeds 0–2. This measures performance with more real data and does not replace the matched 10-shot baseline or isolate the effect of augmentation.

**Preprocessing control:** an additional real-only run uses the same 2,000-image subset with
direct full-image resizing to 224 x 224 for training and no added images. All training settings
and validation preprocessing match the existing runs. This diagnoses the resize-versus-center-crop
change separately from the core three-arm comparison; run with `--full-image`.

| Evaluation measure | Reason |
| --- | --- |
| **Macro F1 — primary** | Weights species equally and combines precision and recall |
| **Top-1 accuracy** | Reports the overall fraction of correct predictions |
| **Per-class precision, recall, F1** | Identifies which species benefit or deteriorate |
| **Generation and training time, separately** | Shows the computational cost of improvements; record hardware |

Select checkpoints on validation macro F1. **Update — 26 October 2026:** complete the remaining 10-shot baseline runs, compare validation results, then freeze experimental choices and selected checkpoints before test evaluation. Report mean ± standard deviation across seeds and paired macro-F1 changes in percentage points. Discuss seed variability and at least three real test failure examples.

**Optional extensions — update, 26 October 2026:** the 20-shot reference is complete. Original versus reviewed synthetic data has been compared at seed 0, but both quality and quantity changed. Remaining optional extensions include controlled synthetic-to-real ratios, 5-shot training, fine-tuning the final classifier block, quality comparisons at matched counts and class coverage, varied prompts/backgrounds, and deliberately imbalanced training classes. Defer these until the primary comparison is complete; change one factor at a time.

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

**Schedule update — 26 October 2026:** complete 10-shot real-only seeds 1 and 2 by **3 November**, before final test evaluation. Proposed owner: **Karina**, consistent with the real-only training responsibility, pending team confirmation. Arina will compare the completed paired validation results before the team freezes experimental choices and checkpoints.

Each member analyzes one test failure, writes their report section, and rehearses the demo. Completed matching core runs count toward the nine-run primary comparison; reference and diagnostic runs are tracked separately. Review active workload weekly; assign optional experiments only if capacity remains.
