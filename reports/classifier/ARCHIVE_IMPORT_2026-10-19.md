# Seed-0 archive validation — 19 October 2026

Source: Karina's outputs.zip, supplied by Arina. Prepared for import; this report
and its neighboring files must be copied into the project to complete import.

The archive contains complete result sets for baseline_10shot_seed0,
standard_aug_10shot_seed0 and real_synthetic_10shot_seed0. JSON and CSV files
are preserved without edits. Existing baseline_seed0 reports remain the only
baseline row in results_summary.csv, which was rebuilt with summarize_results.py.

For all three runs, predictions match all 1,000 validation IDs, labels and class
names. Recomputed aggregate/per-class metrics and best-step history checks pass.
Configurations share seed 0, 1,000 steps, batch size 64, learning rate 0.001,
weight decay 0.0001, validation every 100 steps, AMP and matching real-subset,
validation and classifier-source hashes.

The synthetic run uses 2,000 real plus 2,000 synthetic entries, ten synthetic
images per species, without crop/flip. This is the original pre-review pool
reported in reports/augmentation_seed0.md. It is not a reviewed-pool run or
the optional diffusion_aug mode. Future runs with different pools must be
identified separately.

The archive does not include the prepared synthetic manifest, leakage report
or image pool. Their paths/hashes are in config.json; their contents were not
independently verified. Original Windows configuration paths are preserved.
Checkpoints were copied and hashed, not deserialized or evaluated.
No training or test-set evaluation was performed.

Keep outputs/ Git-ignored; version the small files under reports/classifier/.

## Checkpoint SHA-256 hashes

- `baseline_10shot_seed0/best.pt`: `a2411835f6ee139c6f896eaae38b6dc0105dc24b0d44e0a024184a79f1c8d9ed`
- `real_synthetic_10shot_seed0/best.pt`: `04a132a1486f3702989f959352a04d3dcd38c0466b5dfc3ca07fa5794e02890c`
- `standard_aug_10shot_seed0/best.pt`: `3d4d3a611c49ffb19dfce88c328d0cfdb45fa6114015a3f116fcb90f114fc4bc`
