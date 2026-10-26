# Classifier result artifacts

This directory stores small, reproducible validation artifacts. Model checkpoints remain in the Git-ignored `outputs/` directory and shared storage.

## Current results

| Directories | Experiment | Run responsibility/source |
| --- | --- | --- |
| `baseline_seed0` | Real-only, 10-shot, seed 0 | Karina |
| `baseline_20shot_seed{0,1,2}` | Real-only, 20-shot | Karina |
| `standard_aug_10shot_seed{0,1,2}_t4` | Standard augmentation, 10-shot | Artem, per the project plan and supplied archive |
| `accepted_synthetic_10shot_seed0` | Reviewed synthetic augmentation, seed 0 | Karina |
| `accepted_synthetic_10shot_seed{1,2}` | Reviewed synthetic augmentation, seeds 1–2 | Safina |

Each run directory contains its configuration, training history, runtime, validation metrics, per-class metrics and image-level predictions. The files allow metrics and checkpoint selection to be checked without retraining.

- `results_summary.csv` lists the current run directories plus the available matched 10-shot baseline.
- `comparison_2026-10-27.csv` contains the nine runs used in the 27 October progress report.
- `archive/` contains superseded or historical seed-0 runs retained for reference. They are excluded from `results_summary.csv`.

Adding a result directory records the artifact in Git; it does not imply that the committer trained the model. Experiment ownership follows the table above and the project plan.
