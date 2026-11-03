# Reproducibility verification — Artem, 3 November 2026

## Test archive update — 3 November 2026

The supplied archives provide 17 distinct test CSVs: **15 selected runs and two historical diagnostics**. All contain the exact 5,794 test IDs/labels, and metrics have been recomputed. Following Karina’s clarification, standard seed 0 now uses her complete validation/test/runtime bundle (`archive/standard_aug_10shot_seed0_original`); seeds 1–2 use Artem’s T4 runs. All 15 selected checkpoint files are available locally. No repeat export or training is required.

The selected seed-0 validation predictions, metrics and checkpoint selection were checked, and its weights match the saved configuration and validation F1. The source audit and CPU replay below retain their original scope: they cover T4 seed 0, not Karina’s different recorded source version. They must not be cited as source verification or replay of the replacement. The current report selection is recorded in `src/evaluation_selection.py`.

The archives lack original evaluation configurations/environment records. Checkpoint
metadata matching and CSV validation do not constitute inference replay. The updated
[evaluation README](../reports/evaluation_2026-11-03/README.md) and
[test verification](../reports/evaluation_2026-11-03/test_verification.json) document these limits.
The 2 November status below describes the files available before this import.

## Follow-up — Arina, 2 November 2026

After combining the merged PRs with the locally supplied baseline seeds 1–2, the audit
passes for **all 15 runs**, with no missing primary or reference runs. The explicit baseline
paths are `reports/classifier/baseline_seed{0,1,2}`. The three additional reviewed-synthetic
20-shot runs are now checked as well. Current evidence:
[follow-up audit](../reports/reproducibility_followup_2026-11-03.json).
`python src/check_reproducibility.py --require-complete` now succeeds.

The [local checkpoint inventory](../reports/evaluation_2026-11-03/checkpoint_inventory.json)
records SHA-256 and verifies embedded config, selected step and saved validation F1 for
12 checkpoints (nine primary and three real-only 20-shot references). The three synthetic
20-shot checkpoints are not available locally. This metadata check is not an inference replay
or a team freeze. No test predictions are available in the checked repository or local outputs.

The original audit below is retained as a dated record of what Artem verified against
`cd7d07a`; its missing-baseline statements describe that earlier snapshot only.

## Original audit — snapshot at `cd7d07a`

Audit performed on 2 November for the 3 November deadline, using the comparison artifacts
in `main` at `cd7d07a`. Machine-readable evidence, input checks and replay commands are in
[`reports/reproducibility_2026-11-03.json`](../reports/reproducibility_2026-11-03.json).
The offline audit checks saved validation results. A separate CPU checkpoint replay is
recorded below. Neither check retrains models or computes classifier test metrics.

## Repeat the audit

From the repository root, with Python 3.10 or newer:

```bash
python src/check_reproducibility.py --output outputs/reproducibility.json
python src/check_reproducibility.py --require-complete
```

The checker uses only Python's standard library and the tracked CSV/JSON artifacts.
It needs Git history for the original baseline source check; a shallow clone may need
`git fetch --unshallow`. The default exits successfully when all available planned runs
pass. Missing runs remain explicit in the report. `--require-complete` additionally exits
with status 1 until all nine primary runs are present and valid. It currently fails for
the two missing 10-shot baselines, seeds 1 and 2. This flag checks artifact completeness;
it does not verify external checkpoints or authorize final test evaluation.

For CPU replay and regression checks, create a Python 3.13 environment and install the
validated dependencies. Pillow is pinned because changing its package/build changed
five validation predictions in the initial local environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-replay.txt
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q -W error
```

On Windows, activate `.venv\Scripts\Activate.ps1`, set those environment variables in
PowerShell and run `python -m pytest -q -W error`. The thread settings bound CPU use;
disabling plugin autoload isolates the tests from unrelated globally installed plugins.
Worker tests explicitly use `spawn` to avoid forking a multithreaded PyTorch process.
The fixture selects the GNU MKL threading layer so Conda workers share PyTorch's OpenMP
runtime. The validated run completed with **82 passed, no skips and no warnings** under
`-W error`.
Tests run without downloads. Seven real-image checks are skipped if the processed data is
absent; to enable them, first run `python src/load_data.py`. They inspect loader behavior
and image transforms without computing test scores. The in-memory image fixture also
checks identical fixed augmentation pixels and shuffled batch IDs with 0 and 2 workers.

## Verified evidence

- All 12 split/subset manifests match `cub_splits.csv` and its saved per-class ranks.
  Train/validation/test contain 4,952/1,000/5,794 unique IDs, validation has five images
  per species, and all 42 dropped training duplicates remain excluded. All 5/10/20-shot
  subsets for seeds 0–2 are balanced and nested through their saved ranks.
- All ten available planned runs pass. Validation IDs, row order, labels and species
  mapping agree with `val.csv`. Accuracy, macro F1 and every species' precision, recall,
  F1 and support are recomputed from the saved predictions. Loss agrees with the selected
  history row; logits were not saved, so cross-entropy cannot be independently recomputed.
- Histories contain all ten validation points (steps 100–1,000). The earliest maximum
  validation macro F1 agrees with `runtime.json`'s selected step and saved validation
  metrics. Configurations use the same frozen ResNet-18/BatchNorm, ImageNet weights,
  AdamW hyperparameters, batch size, AMP and 1,000-step budget. Recorded input and source
  hashes match, with the historical baseline exception below.
- Paired standard/synthetic runs match the same seed's real-subset hash. The reviewed
  pool has 1,440 unique synthetic IDs, disjoint from all real IDs. Its recorded class
  counts and leakage report agree with the run configurations. The archived manifest
  and leakage report match the hashes of the original `data/synthetic/.../prepared_accepted/`
  files; the checker uses those explicit archive aliases when replaying the audit.
- Original and repeated standard-augmentation runs are kept separate. Only
  `standard_aug_10shot_seed{0,1,2}_t4` belongs to this audit's primary comparison;
  `archive/` and supplementary historical runs are excluded.

| Run | Selected step from history | Validation macro F1 | Validation accuracy |
| --- | ---: | ---: | ---: |
| baseline_seed0 (10-shot) | 700 | 53.1463% | 53.30% |
| standard_aug_10shot_seed0_t4 | 900 | 53.5311% | 53.70% |
| standard_aug_10shot_seed1_t4 | 800 | 52.9609% | 53.90% |
| standard_aug_10shot_seed2_t4 | 1,000 | 54.0340% | 54.50% |
| accepted_synthetic_10shot_seed0 | 600 | 48.9604% | 50.20% |
| accepted_synthetic_10shot_seed1 | 800 | 53.0621% | 53.90% |
| accepted_synthetic_10shot_seed2 | 900 | 51.8985% | 52.20% |
| baseline_20shot_seed0 (reference) | 1,000 | 60.0558% | 59.80% |
| baseline_20shot_seed1 (reference) | 900 | 60.1498% | 60.40% |
| baseline_20shot_seed2 (reference) | 1,000 | 58.8707% | 59.10% |

## Replay training

Each verified run's JSON entry contains a `training_command` with explicit seed, subset,
augmentation, training settings and a fresh `outputs/reproduction/<run>` directory.
These commands require processed CUB images, cached/downloadable ImageNet weights and a
CUDA GPU. Reviewed synthetic runs also require the accepted images under `data/synthetic/`;
the archived manifest can be passed directly to `--synthetic-manifest`. Existing output
directories are rejected by training; choose another output name for subsequent replays.

Historical baseline seed 0 stores raw Windows CRLF hashes for its data files. The audit
accepts LF/CRLF equivalents only, and still rejects content changes. Two source files
predate the current loader: their bytes match Git snapshot
`8f4050fee9c8265f3bf40b41b110001e99f81f23`. That version uses the same deterministic
resize-to-256/center-crop-to-224 baseline preprocessing. For an exact source replay:

```bash
git worktree add --detach ../cub-baseline-replay 8f4050fee9c8265f3bf40b41b110001e99f81f23
```

Restore processed data in that checkout and use the JSON command for `baseline_seed0` there.
Use the current checkout for the other commands, whose recorded training-source hashes
match it. The old snapshot has different loader/config bookkeeping, so a current-code
baseline rerun must be identified as a new run.

## CPU validation replay of Artem's checkpoints — resolved

All three original standard-augmentation `best.pt` files were available locally. Their
embedded configurations, selected steps and validation-selection scores match the tracked
T4 artifacts. Validation was evaluated again on CPU using `src/evaluate_classifier.py`.
The local validation Arrow data and metadata have identical SHA-256 hashes to the submitted
Kaggle input snapshot. The checkpoint hashes, environment, commands and prediction
comparisons are recorded in
[`reports/standard_aug_validation_replay_pillow12_2026-11-03.json`](../reports/standard_aug_validation_replay_pillow12_2026-11-03.json).

| Seed | Matching predictions | CPU macro F1 | Change from saved T4 macro F1 |
| --- | ---: | ---: | ---: |
| 0 | 1,000 / 1,000 | 53.5311% | 0 pp |
| 1 | 1,000 / 1,000 | 52.9609% | 0 pp |
| 2 | 1,000 / 1,000 | 54.0340% | 0 pp |

All **3,000 predictions**, accuracy, macro F1 and every per-class metric match the original
T4 results exactly. Accuracy is 53.7%, 53.9% and 54.5%. The maximum absolute loss difference
is `1.38e-7`, within the recorded `1e-6` tolerance for CPU/T4 floating-point reductions.

The initial Pillow 11.1.0 environment produced 1/1/3 different predictions. Replacing
only the Pillow package/build with 12.2.0 on the same CPU resolved all five differences;
checkpoints, image data and preprocessing source stayed identical. The initial evidence
is preserved in
[`reports/standard_aug_validation_replay_2026-11-03.json`](../reports/standard_aug_validation_replay_2026-11-03.json),
with a link to the resolution. `requirements.txt` now pins Pillow 12.2.0 and
`requirements-replay.txt` records the complete set of direct replay/check dependencies.
The precise original T4 decoding environment is unknown because its Pillow version was
not logged; the corrected environment is validated by the complete prediction comparison.

To repeat after restoring the checkpoint archive under `outputs/kaggle_standard_aug/runs/`:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python src/evaluate_classifier.py outputs/kaggle_standard_aug/runs/standard_aug_10shot_seed0/best.pt --split val --device cpu --output outputs/reproducibility_pillow12_seed0
```

Repeat for seeds 1 and 2, choosing fresh output directories. Compare the resulting
`val_predictions.csv` files with `reports/classifier/standard_aug_10shot_seed{seed}_t4/`.
The local replay environment was Python 3.13 with the versions listed in the JSON report.
Prediction CSVs must agree exactly; compare loss with the documented tolerance. This
confirms inference replay for these checkpoints, while training reproducibility remains
outside this verification.

## Limits and handoff

The primary comparison is **7/9 complete**; the three 20-shot reference runs are complete.
Karina's 10-shot baseline seeds 1 and 2 remain outstanding. Re-run the audit after adding
their six artifact files to `reports/classifier/baseline_10shot_seed{1,2}/`. Run filenames
are explicit so repeated runs cannot silently replace a selected comparison run.

This verification did not retrain any model, inspect baseline/synthetic checkpoints,
or repeat image-level MD5/embedding leakage detection. Saved IDs, hashes and leakage
records are consistent; that does not prove image-pixel integrity in an external archive.
Hardware/software records also differ across runs (T4, RTX 4060 Laptop and RTX 3070;
different Python/PyTorch/torchvision versions). `requirements.txt` has no exact environment
lock for the original training runs. The new replay requirements pin the validated direct
dependencies. Equal seeds alone do not guarantee identical GPU results across these environments.

Before final test evaluation, the checkpoint owner should finish the missing runs,
compare paired validation results, freeze the choices, restore all selected checkpoints,
and record their file hashes. Verify checkpoint steps/configurations against the audit
and reproduce validation predictions with `src/evaluate_classifier.py` before exporting
test predictions. Checkpoint freeze/export and final test metrics remain the respective
team owners' tasks in `PROJECT_PLAN.md`.
