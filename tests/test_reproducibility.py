"""Audit must reject altered evidence and keep missing experiments visible."""
import hashlib
import json
import shutil

import pytest

from check_reproducibility import ROOT, audit, check_splits, main, matching_hash, planned_runs

RUN = "standard_aug_10shot_seed0_t4"


@pytest.fixture
def snapshot(tmp_path):
    shutil.copytree(ROOT / "splits", tmp_path / "splits")
    shutil.copytree(ROOT / "src", tmp_path / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "reports/classifier" / RUN, tmp_path / "reports/classifier" / RUN)
    return tmp_path


def change_json(path, key, value):
    payload = json.loads(path.read_text())
    payload[key] = value
    path.write_text(json.dumps(payload))


def test_repository_artifacts_and_remaining_primary_runs():
    report = audit()
    assert not report["errors"]
    missing = [run for run, _, _, shot in planned_runs()
               if shot == 10 and not (ROOT / "reports/classifier" / run).exists()]
    assert report["missing_primary_runs"] == missing
    assert report["primary_complete"] == (not missing)
    baseline = next(r for r in report["runs"] if r["run"] == "baseline_seed0")
    assert baseline["input_hashes"]["src/preprocessing.py"]["status"] == "historical_match"
    assert main(["--require-complete"]) == int(bool(missing))
    assert main([]) == 0


def test_available_valid_run_is_distinct_from_complete_matrix(snapshot):
    report = audit(snapshot)
    assert not report["errors"]
    assert [r["run"] for r in report["runs"]] == [RUN]
    assert len(report["missing_primary_runs"]) == 8
    assert len(report["missing_reference_runs"]) == 3


@pytest.mark.parametrize("file,key,value", [
    ("config.json", "seed", 1),
    ("config.json", "lr", 0.01),
    ("config.json", "train_images", 2000),
    ("config.json", "train_preprocessing", "full_image_resize_224"),
    ("val_metrics.json", "accuracy", 0.0),
    ("runtime.json", "best_step", 100),
    ("runtime.json", "optimizer_steps", 999),
    ("runtime.json", "training_with_validation_seconds", float("nan")),
])
def test_rejects_inconsistent_run_records(snapshot, file, key, value):
    change_json(snapshot / "reports/classifier" / RUN / file, key, value)
    report = audit(snapshot)
    assert report["errors"]
    assert report["status"] == "failed"
    assert not report["runs"]


def test_rejects_prediction_rows_in_the_wrong_order(snapshot):
    path = snapshot / "reports/classifier" / RUN / "val_predictions.csv"
    rows = path.read_text().splitlines(keepends=True)
    rows[1], rows[2] = rows[2], rows[1]
    path.write_text("".join(rows))
    assert "predictions do not match" in audit(snapshot)["errors"][0]


def test_rejects_edited_per_species_metrics(snapshot):
    path = snapshot / "reports/classifier" / RUN / "val_per_class.json"
    payload = json.loads(path.read_text())
    payload["001.Black_footed_Albatross"]["support"] = 6
    path.write_text(json.dumps(payload))
    assert "incorrect per-class metrics" in audit(snapshot)["errors"][0]


def test_rejects_source_drift(snapshot):
    with (snapshot / "src/preprocessing.py").open("a") as stream:
        stream.write("\n# changed policy\n")
    assert "Hash mismatch: src/preprocessing.py" in audit(snapshot)["errors"][0]


def test_rejects_missing_artifact_and_missing_recorded_hash(snapshot):
    path = snapshot / "reports/classifier" / RUN
    config = json.loads((path / "config.json").read_text())
    del config["sha256"]["splits/manifests/test.csv"]
    (path / "config.json").write_text(json.dumps(config))
    assert "Missing recorded hashes" in audit(snapshot)["errors"][0]
    (path / "val_predictions.csv").unlink()
    assert "missing val_predictions.csv" in audit(snapshot)["errors"][0]


def test_rejects_incomplete_history(snapshot):
    path = snapshot / "reports/classifier" / RUN / "history.csv"
    rows = path.read_text().splitlines(keepends=True)
    path.write_text("".join([rows[0], *rows[2:]]))
    assert "Incomplete validation history" in audit(snapshot)["errors"][0]


def test_manifest_must_be_derived_from_saved_ranks(snapshot):
    path = snapshot / "splits/manifests/train_10shot_seed0.csv"
    path.write_text(path.read_text().replace("training/", "validation/", 1))
    with pytest.raises(ValueError, match="Manifest differs"):
        check_splits(snapshot)


def test_legacy_and_canonical_hashes_allow_only_line_ending_changes():
    lf = b"a,b\n1,2\n"
    windows_hash = hashlib.sha256(lf.replace(b"\n", b"\r\n")).hexdigest()
    assert matching_hash(lf, windows_hash)
    assert not matching_hash(lf.replace(b"1,2", b"1,3"), windows_hash)
