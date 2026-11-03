"""Reporting selection agreed on 3 November; original artifact paths are preserved."""
STANDARD_SEED0 = "archive/standard_aug_10shot_seed0_original"


def selected_run(run):
    # Karina supplied a complete val/test/runtime bundle for the same protocol.
    # Choose the bundle as a unit, not by comparing test scores between reruns.
    return STANDARD_SEED0 if run == "standard_aug_10shot_seed0_t4" else run
