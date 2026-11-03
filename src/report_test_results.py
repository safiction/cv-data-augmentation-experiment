"""Recompute test metrics from supplied hard-label CSVs; do not run or tune models."""
import csv
import gzip
import hashlib
import io
import json
import statistics
from collections import Counter
from pathlib import Path

from check_reproducibility import ROOT, matching_hash, planned_runs
from report_comparison import comparison, figures, write_csv
from summarize_results import read_csv, read_json
from evaluation_selection import selected_run


def score_predictions(rows, manifest, names):
    if len(rows) != len(manifest) or len({r["image_id"] for r in rows}) != len(rows):
        raise ValueError("Wrong test count or duplicate image IDs")
    truth, predictions = [], []
    for row, expected in zip(rows, manifest):
        if any(row[key] != expected[key] for key in ("image_id", "label", "class_name")):
            raise ValueError("Test rows, labels or order differ from the frozen manifest")
        label, predicted = int(row["label"]), int(row["prediction"])
        if not (0 <= label < len(names) and 0 <= predicted < len(names)):
            raise ValueError("Label outside the species mapping")
        if names[label] != row["class_name"] or names[predicted] != row["predicted_class_name"]:
            raise ValueError("Prediction species mapping mismatch")
        truth.append(label)
        predictions.append(predicted)
    support, selected, correct = Counter(truth), Counter(predictions), Counter()
    for label, predicted in zip(truth, predictions):
        if label == predicted:
            correct[label] += 1
    per_class = []
    for label, name in enumerate(names):
        precision = correct[label] / selected[label] if selected[label] else 0.0
        recall = correct[label] / support[label] if support[label] else 0.0
        f1 = 2 * correct[label] / (support[label] + selected[label]) if support[label] + selected[label] else 0.0
        per_class.append({"class_name": name, "precision": precision, "recall": recall,
                          "f1": f1, "support": support[label]})
    metrics = {"images": len(rows), "accuracy": sum(correct.values()) / len(rows),
               "macro_f1": statistics.mean(r["f1"] for r in per_class),
               "macro_precision": statistics.mean(r["precision"] for r in per_class),
               "macro_recall": statistics.mean(r["recall"] for r in per_class)}
    return metrics, per_class


def main():
    output = ROOT / "reports/evaluation_2026-11-03"
    provenance = read_json(ROOT / "reports/test/import_provenance.json")
    manifest_path = ROOT / "splits/manifests/test.csv"
    manifest = read_csv(manifest_path)
    if len(manifest) != 5794:
        raise ValueError("Expected the official 5,794-image test split")
    names = read_json(ROOT / "reports/classifier/baseline_seed0/config.json")["class_names"]
    selected = {selected_run(run) for run, _, _, _ in planned_runs()}
    run_rows, species_rows, tables, species = [], [], {}, {}
    for record in provenance["runs"]:
        run = record["run"]
        digest = record["test_manifest_hash_from_checkpoint"]
        if digest and not matching_hash(manifest_path.read_bytes(), digest):
            raise ValueError(f"Checkpoint test manifest hash mismatch: {run}")
        if not matching_hash((ROOT / "splits/cub_splits.csv").read_bytes(), record["split_hash_from_checkpoint"]):
            raise ValueError(f"Checkpoint split hash mismatch: {run}")
        data = gzip.decompress((ROOT / record["predictions_path"]).read_bytes())
        if hashlib.sha256(data).hexdigest() != record["predictions_sha256"]:
            raise ValueError(f"Submitted prediction bytes changed: {run}")
        table = list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
        metrics, per_class = score_predictions(table, manifest, names)
        scope = "selected_comparison" if run in selected else "historical_diagnostic"
        run_rows.append({"run": run, "scope": scope, "split": "test", "k_shot": record["k_shot"],
                         "arm": record["arm"], "seed": record["seed"], "train_images": record["train_images"],
                         "checkpoint_step": record["checkpoint_step"], **metrics})
        species_rows.extend({"run": run, "scope": scope, **r} for r in per_class)
        tables[run], species[run] = table, per_class
    current = [r for r in run_rows if r["scope"] == "selected_comparison"]
    summary, paired = comparison(current, split="test", allow_incomplete=True)
    summary = [{**r, "seeds": ",".join(str(x["seed"]) for x in sorted(current, key=lambda t: t["seed"])
                                         if (x["k_shot"], x["arm"]) == (r["k_shot"], r["arm"]))} for r in summary]
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "test_runs.csv", run_rows)
    write_csv(output / "test_summary.csv", summary)
    write_csv(output / "test_paired.csv", paired)
    write_csv(output / "test_per_class.csv", species_rows)
    figures(summary, paired, current, output, split="test")

    species_changes = []
    for shot in (10, 20):
        for label, name in enumerate(names):
            bases = [f"baseline_seed{s}" if shot == 10 else f"baseline_20shot_seed{s}" for s in range(3)]
            synths = [f"accepted_synthetic_{shot}shot_seed{s}" for s in range(3)]
            deltas = [100 * (species[b][label]["f1"] - species[a][label]["f1"]) for a, b in zip(bases, synths)]
            species_changes.append({"k_shot": shot, "class_name": name,
                                    "baseline_mean_f1": statistics.mean(species[a][label]["f1"] for a in bases),
                                    "synthetic_mean_f1": statistics.mean(species[b][label]["f1"] for b in synths),
                                    "delta_seed0_pp": deltas[0], "delta_seed1_pp": deltas[1], "delta_seed2_pp": deltas[2],
                                    "mean_change_pp": statistics.mean(deltas), "sample_sd_pp": statistics.stdev(deltas)})
    write_csv(output / "test_species_changes.csv", species_changes)

    # Deterministic illustrative failures: first three distinct species for which all three
    # 10-shot baselines are right and all three synthetic models are wrong. No visual causes inferred.
    failures, used = [], set()
    for i, expected in enumerate(manifest):
        base = [tables[f"baseline_seed{s}"][i]["predicted_class_name"] for s in range(3)]
        synth = [tables[f"accepted_synthetic_10shot_seed{s}"][i]["predicted_class_name"] for s in range(3)]
        name = expected["class_name"]
        if name not in used and all(p == name for p in base) and all(p != name for p in synth):
            failures.append({"image_id": expected["image_id"], "true_species": name,
                             "baseline_correct_seeds": "0,1,2", **{f"synthetic_prediction_seed{s}": synth[s] for s in range(3)}})
            used.add(name)
            if len(failures) == 3:
                break
    if failures:
        write_csv(output / "test_failure_examples.csv", failures)
    status = {"verified_prediction_files": len(run_rows), "selected_test_runs": len(current),
              "expected_selected_runs": len(selected), "missing_selected_runs": sorted(selected - set(tables)),
              "historical_diagnostics": [r["run"] for r in run_rows if r["scope"] != "selected_comparison"],
              "test_images_per_run": len(manifest), "test_manifest_sha256": hashlib.sha256(manifest_path.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
              "metric_method": "Recomputed from supplied hard labels, all 200 species, zero_division=0",
              "limitations": provenance["limitations"]}
    (output / "test_verification.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))
    for r in summary:
        print(r["k_shot"], r["arm"], r["seeds"], f"F1={100*r['macro_f1_mean']:.4f}%", f"accuracy={100*r['accuracy_mean']:.4f}%")


if __name__ == "__main__":
    main()
