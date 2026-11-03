"""Paired effects must use matching shot counts and seeds, with complete inputs."""
import pytest

from report_comparison import comparison


@pytest.fixture
def runs():
    groups = [(10, "baseline", .5), (10, "standard_aug", .5),
              (10, "real_synthetic", .4), (20, "baseline", .8),
              (20, "real_synthetic", .7)]
    return [{"run": f"{arm}_{shot}_{seed}", "k_shot": shot, "arm": arm, "seed": seed,
             "train_images": shot * 200, "macro_f1": start + .1 * seed,
             "accuracy": start + .1 * seed}
            for shot, arm, start in groups for seed in range(3)]


def test_pairs_match_shot_and_seed_independently_of_input_order(runs):
    summary, changes = comparison(list(reversed(runs)))
    baseline10 = next(r for r in summary if r["k_shot"] == 10 and r["arm"] == "baseline")
    assert baseline10["macro_f1_mean"] == pytest.approx(.6)
    assert baseline10["macro_f1_sample_sd"] == pytest.approx(.1)
    assert len(changes) == 12
    for row in changes:
        expected = 0 if row["comparison"] == "standard_aug_minus_baseline" else -10
        assert row["macro_f1_change_pp"] == pytest.approx(expected)
        assert row["first_run"].endswith(f"_{row['k_shot']}_{row['seed']}")
        assert row["second_run"].endswith(f"_{row['k_shot']}_{row['seed']}")


def test_incomplete_or_ambiguous_matrix_is_not_summarized(runs):
    for invalid in (runs[:-1], runs + [runs[0]], runs[1:] + [runs[1]]):
        with pytest.raises(ValueError, match="15 distinct selected runs"):
            comparison(invalid)
