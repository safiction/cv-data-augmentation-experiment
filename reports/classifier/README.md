# Classifier result artifacts

This directory stores small, reproducible validation artifacts. Model checkpoints remain in the Git-ignored `outputs/` directory and shared storage.

## Current results

| Directories | Experiment | Run responsibility/source |
| --- | --- | --- |
| `baseline_seed{0,1,2}` | Real-only, 10-shot | Karina; seeds 1–2 imported from `outputs (3).zip` on 27 October |
| `baseline_20shot_seed{0,1,2}` | Real-only, 20-shot | Karina |
| `archive/standard_aug_10shot_seed0_original` | Selected standard augmentation, 10-shot seed 0 | Karina; complete validation/test/runtime bundle |
| `standard_aug_10shot_seed{1,2}_t4` | Selected standard augmentation, 10-shot seeds 1–2 | Artem |
| `standard_aug_10shot_seed0_t4` | Alternative seed-0 rerun, excluded from current summary | Artem |
| `accepted_synthetic_10shot_seed0` | Reviewed synthetic augmentation, seed 0 | Karina |
| `accepted_synthetic_20shot_seed{0,1,2}` | Reviewed synthetic augmentation, 20-shot (4,000 real + 1,440 synthetic) | Safina |
| `accepted_synthetic_10shot_seed{1,2}` | Reviewed synthetic augmentation, seeds 1–2 | Safina |

Each run directory contains its configuration, training history, runtime, validation metrics, per-class metrics and image-level predictions. The files allow metrics and checkpoint selection to be checked without retraining.

- `results_summary.csv` lists all 15 current runs: nine matched 10-shot runs, three 20-shot real-only references and three 20-shot synthetic runs.
- `comparison_2026-10-27.csv` contains the 12 runs used in the 27 October progress report.
- `paired_comparison_2026-10-27.csv` contains standard-minus-baseline, synthetic-minus-baseline and synthetic-minus-standard changes for each matched seed; score changes are in percentage points.
- `archive/` contains superseded or historical seed-0 runs retained for reference. The exception is Karina’s standard seed-0 bundle, selected on 3 November for complete validation/test/runtime reporting; its original path is preserved.

Adding a result directory records the artifact in Git; it does not imply that the committer trained the model. Experiment ownership follows the table above and the project plan.

Separate [test results and provenance](../evaluation_2026-11-03/README.md) now cover all 15 selected models. Test exports are not included in this validation-only directory or its summaries. Karina’s seed-0 bundle is used consistently across validation, test and runtime; the alternative T4 seed-0 run remains available but is not included in the current summary.

The [reproducibility audit](../../docs/reproducibility.md) checks recorded manifests, source hashes, predictions and checkpoint selection. Its follow-up covers the original 15-run matrix with T4 standard seed 0; it does not verify the replacement’s historical source version. The replacement’s validation metrics/history and checkpoint metadata have been checked separately. External-checkpoint inference remains a separate check. Artem reproduced all 3,000 standard-augmentation validation predictions on CPU with Pillow 12.2.0.

Baseline seeds 1–2 checkpoints are stored locally at `outputs/baseline_10shot_seed{1,2}/best.pt`, outside Git. The imported files were byte-checked against the archive.

Regenerate the current run table with `python src/summarize_results.py`. The dated 27 October tables remain snapshots of the 12 runs available then; do not overwrite them with later experiments.
