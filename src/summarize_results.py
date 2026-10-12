"""Collect classifier run metrics into one comparison table."""

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


FIELDS = [
    "run",
    "arm",
    "seed",
    "k_shot",
    "augment",
    "train_images",
    "val_images",
    "macro_f1",
    "accuracy",
    "val_loss",
    "best_step",
    "optimizer_steps",
    "training_seconds",
    "validated",
]


def read_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def close(actual, expected):
    return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12)


def validate_run(run_dir, config, metrics, runtime):
    names = config["class_names"]
    if len(names) != 200 or len(set(names)) != 200:
        raise ValueError(f"{run_dir}: expected 200 unique class names")

    predictions = read_csv(run_dir / "val_predictions.csv")
    history = read_csv(run_dir / "history.csv")
    per_class = read_json(run_dir / "val_per_class.json")
    project_root = Path(__file__).resolve().parents[1]
    manifest = read_csv(project_root / "splits" / "manifests" / "val.csv")

    if len(predictions) != len(manifest) or len(predictions) != metrics["images"]:
        raise ValueError(f"{run_dir}: prediction count does not match validation data")
    if len({row["image_id"] for row in predictions}) != len(predictions):
        raise ValueError(f"{run_dir}: repeated prediction image IDs")

    truth, predicted = [], []
    for row, expected in zip(predictions, manifest):
        for column in ("image_id", "label", "class_name"):
            if row[column] != expected[column]:
                raise ValueError(f"{run_dir}: predictions do not match val.csv")
        label, prediction = int(row["label"]), int(row["prediction"])
        if not (0 <= label < 200 and 0 <= prediction < 200):
            raise ValueError(f"{run_dir}: label outside 0..199")
        if row["class_name"] != names[label] or row["predicted_class_name"] != names[prediction]:
            raise ValueError(f"{run_dir}: class-name mapping mismatch")
        truth.append(label)
        predicted.append(prediction)

    accuracy = sum(a == b for a, b in zip(truth, predicted)) / len(truth)
    f1_scores = []
    supports = Counter(truth)
    for label, name in enumerate(names):
        tp = sum(a == label and b == label for a, b in zip(truth, predicted))
        fp = sum(a != label and b == label for a, b in zip(truth, predicted))
        fn = sum(a == label and b != label for a, b in zip(truth, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        saved = per_class[name]
        if not all(close(saved[key], value) for key, value in (
            ("precision", precision), ("recall", recall), ("f1-score", f1),
            ("support", supports[label]),
        )):
            raise ValueError(f"{run_dir}: incorrect per-class metrics for {name}")
        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / len(names)
    if not close(metrics["accuracy"], accuracy) or not close(metrics["macro_f1"], macro_f1):
        raise ValueError(f"{run_dir}: aggregate metrics do not match predictions")

    best = max(history, key=lambda row: float(row["val_macro_f1"]))
    if int(best["step"]) != runtime["best_step"]:
        raise ValueError(f"{run_dir}: best step does not match history")
    if not all((
        close(best["val_macro_f1"], metrics["macro_f1"]),
        close(best["val_accuracy"], metrics["accuracy"]),
        close(best["val_loss"], metrics["loss"]),
        close(runtime["best_val_macro_f1"], metrics["macro_f1"]),
    )):
        raise ValueError(f"{run_dir}: best history row does not match saved metrics")


def collect_run(run_dir):
    required = [
        "config.json", "history.csv", "runtime.json", "val_metrics.json",
        "val_per_class.json", "val_predictions.csv",
    ]
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise ValueError(f"{run_dir}: missing {', '.join(missing)}")

    config = read_json(run_dir / "config.json")
    metrics = read_json(run_dir / "val_metrics.json")
    runtime = read_json(run_dir / "runtime.json")
    validate_run(run_dir, config, metrics, runtime)
    return {
        "run": run_dir.name,
        "arm": config["arm"],
        "seed": config["seed"],
        "k_shot": config["k_shot"],
        "augment": config["augment"],
        "train_images": config["train_images"],
        "val_images": metrics["images"],
        "macro_f1": metrics["macro_f1"],
        "accuracy": metrics["accuracy"],
        "val_loss": metrics["loss"],
        "best_step": runtime["best_step"],
        "optimizer_steps": runtime["optimizer_steps"],
        "training_seconds": runtime["training_with_validation_seconds"],
        "validated": True,
    }


def print_markdown(rows):
    columns = ["run", "arm", "seed", "macro_f1", "accuracy", "best_step", "training_seconds"]
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            if column in {"macro_f1", "accuracy"}:
                value = f"{value:.4f}"
            elif column == "training_seconds":
                value = f"{value:.1f}"
            values.append(str(value))
        print("| " + " | ".join(values) + " |")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports/classifier"))
    parser.add_argument("--output", type=Path, default=Path("reports/classifier/results_summary.csv"))
    args = parser.parse_args()

    run_dirs = sorted(
        path for path in args.reports_dir.iterdir()
        if path.is_dir() and (path / "config.json").is_file()
    )
    if not run_dirs:
        raise SystemExit(f"No classifier runs found in {args.reports_dir}")

    rows = [collect_run(path) for path in run_dirs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print_markdown(rows)
    print(f"\nSaved {len(rows)} run(s) to {args.output}")


if __name__ == "__main__":
    main()
