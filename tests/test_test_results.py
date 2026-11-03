"""Hard-label metrics and incomplete-seed comparisons must not hide bad evidence."""
import pytest

from report_comparison import comparison
from report_test_results import score_predictions
from import_test_archives import checked_path


def example():
    names = ["a", "b", "c"]
    manifest = [{"image_id": str(i), "label": str(label), "class_name": names[label]}
                for i, label in enumerate([0, 0, 1, 2])]
    rows = [{**row, "prediction": str(pred), "predicted_class_name": names[pred]}
            for row, pred in zip(manifest, [0, 1, 1, 1])]
    return rows, manifest, names


def test_macro_metrics_include_species_never_predicted():
    metrics, per_class = score_predictions(*example())
    assert metrics["accuracy"] == .5
    assert metrics["macro_f1"] == pytest.approx(7 / 18)
    assert metrics["macro_precision"] == pytest.approx(4 / 9)
    assert metrics["macro_recall"] == .5
    assert per_class[2]["f1"] == 0 and per_class[2]["support"] == 1


@pytest.mark.parametrize("problem", ["duplicate", "order", "count", "mapping", "prediction"])
def test_invalid_test_predictions_are_rejected(problem):
    rows, manifest, names = example()
    if problem == "duplicate":
        rows[1] = rows[0].copy()
    elif problem == "order":
        rows[0], rows[1] = rows[1], rows[0]
    elif problem == "count":
        rows.pop()
    elif problem == "mapping":
        rows[0]["predicted_class_name"] = "c"
    else:
        rows[0]["prediction"] = "3"
    with pytest.raises(ValueError):
        score_predictions(rows, manifest, names)


def test_missing_standard_seed_is_excluded_from_paired_means():
    groups = [(10, "baseline"), (10, "standard_aug"), (10, "real_synthetic"),
              (20, "baseline"), (20, "real_synthetic")]
    rows = [{"run": f"{shot}_{arm}_{seed}", "k_shot": shot, "arm": arm, "seed": seed,
             "train_images": 200 * shot, "macro_f1": .5, "accuracy": .5}
            for shot, arm in groups for seed in range(3) if (arm, seed) != ("standard_aug", 0)]
    with pytest.raises(ValueError):
        comparison(rows)
    summary, paired = comparison(rows, split="test", allow_incomplete=True)
    assert next(r for r in summary if r["arm"] == "standard_aug")["n_seeds"] == 2
    assert all(r["split"] == "test" for r in summary + paired)
    assert len(paired) == 10
    assert {r["seed"] for r in paired if "standard_aug" in r["comparison"]} == {1, 2}


@pytest.mark.parametrize("path", ["../best.pt", "/tmp/best.pt", "run/../../best.pt", "run\\best.pt"])
def test_archive_paths_cannot_escape(path):
    with pytest.raises(ValueError):
        checked_path(path)
