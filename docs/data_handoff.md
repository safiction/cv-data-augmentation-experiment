# Data and environment handoff — 10 November 2026

Prepared and checked on 9 November. Scope: data report section, README and environment;
slides excluded as requested.

## Completed

- [Data section](data_report.md): report-ready text covering the final dataset,
  split, duplicate removal, paired subsets, transforms, synthetic class coverage and limitations.
  Includes the existing training-image preprocessing figure, links to evidence and an
  individual Red-winged Blackbird test-failure analysis checked against all nine selected
  10-shot prediction exports.
- [README](../README.md): setup, current experiment status, data download, checks, paired
  training commands, validation evaluation, generation and repository/artifact layout.
- [Environment guide](environment.md): pinned CPU replay dependencies, GPU setup boundaries,
  Windows activation, external archive restoration and troubleshooting.
- Corrected stale claims in [the detailed data protocol](data_split.md): synthetic training
  can optionally add fixed views, and unequal dataset sizes do not share an identical shuffled
  batch order. Distinguished held-out duplicate exclusion from species-quality selection.
- Corrected the synthetic replay instructions: archived CSV paths are relative to their
  original prepared directory. Restore the CSV and leakage report there before training;
  the archive alias used by the offline audit alone does not relocate image files.

## Verification

Using Python 3.13.5, a local `.venv` with system-site packages and Pillow 12.2.0 installed
inside it, without changing the global Pillow 11.1.0 package:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m pytest -q -W error
python src/check_reproducibility.py --require-complete --output outputs/reproducibility_2026-11-10.json
```

Tests: **95 passed in 25.23 s**, no skips or warnings. Processed-image checks were included.
After synchronization with `main` at `f0ceaec`, the offline audit with
`--require-complete` verifies **15 comparison runs, 0 errors**, with no missing primary runs.
The final report uses Karina’s complete standard seed-0 bundle and T4 seeds 1–2; the
source audit and earlier CPU replay retain their original T4 seed-0 scope. Environment versions and check outcomes are recorded
in [the handoff evidence](../reports/environment_verification.json).

This check reused installed dependencies rather than testing a fresh clean-room installation.
No training, new checkpoint inference, generation or classifier test scoring was performed.
The previous complete standard-checkpoint replay remains documented in
[the 3 November evidence](reproducibility.md).

## Integration with current main

The branch is based on `main` at `f0ceaec`, including the completed baseline seeds 1–2,
six supplementary 20-shot runs and final test exports. The earlier pre-sync 7/9 status
has been replaced throughout this deliverable. Test metrics were already supplied and
recomputed by the evaluation owner; no repeat export or model retuning is requested.
Original inference environment records remain unavailable, as documented in the evaluation
report. This documentation work does not claim independent checkpoint inference replay.

The individual test-failure discussion is included in
[the data section](data_report.md#individual-test-failure-analysis).
The image was visually inspected and all paired 10-shot predictions were checked against
the tracked gzip CSVs. The discussion separates observations from possible causes.
Demo rehearsal remains a team activity and is not claimed complete by repository changes.
