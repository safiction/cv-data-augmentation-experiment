"""Validate and report three real-only, center-cropped 20-shot classifier runs."""
import argparse
import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from load_data import PROJECT_ROOT, canonical_sha256, load_manifest
from summarize_results import collect_run, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    rows, configs, per_class = [], [], []
    held_out = set(load_manifest("val").image_id) | set(load_manifest("test").image_id)
    pool = set(load_manifest("train").image_id)
    expected = dict(k_shot=20, augment=False, full_image=False, synthetic_manifest=None,
                    max_steps=1000, eval_every=100, batch_size=64, lr=0.001,
                    weight_decay=0.0001, amp=True, train_images=4000,
                    real_train_images=4000, synthetic_train_images=0,
                    augmented_train_images=0, train_preprocessing="resize_center_crop",
                    frozen_backbone=True, frozen_batchnorm=True, device="cuda")
    for seed in range(3):
        run = args.runs_dir / f"baseline_20shot_seed{seed}"
        config = read_json(run / "config.json")
        assert config["seed"] == seed
        for key, value in expected.items():
            assert config[key] == value, (run, key, config[key], value)
        manifest = load_manifest("train", 20, seed)
        assert len(manifest) == manifest.image_id.nunique() == 4000
        assert len(manifest.groupby("label")) == 200
        assert (manifest.groupby("label").size() == 20).all()
        assert set(manifest.image_id) <= pool
        assert not set(manifest.image_id) & held_out
        for relative, digest in config["sha256"].items():
            source = args.runs_dir / "source_snapshot" / relative
            assert canonical_sha256(source) == digest, (run, relative)
            assert canonical_sha256(PROJECT_ROOT / relative) == digest, relative
        row = collect_run(run)
        report = read_json(run / "val_per_class.json")
        row.update(macro_precision=report["macro avg"]["precision"],
                   macro_recall=report["macro avg"]["recall"])
        rows.append(row)
        configs.append(config)
        per_class.append(report)
    for key in ("class_names", "torch", "torchvision", "sklearn", "gpu", "weights",
                "architecture", "num_workers", "selection_metric"):
        assert all(c[key] == configs[0][key] for c in configs)
    metrics = ("macro_f1", "accuracy", "macro_precision", "macro_recall", "val_loss",
               "best_step", "training_seconds")
    summary = {key: {"mean": statistics.mean(row[key] for row in rows),
                     "std": statistics.stdev(row[key] for row in rows)} for key in metrics}
    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with prefix.with_name(prefix.name + "_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "sample_std", "n"])
        writer.writerows((key, value["mean"], value["std"], 3) for key, value in summary.items())
    with prefix.with_name(prefix.name + "_per_class.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "metric", "seed0", "seed1", "seed2", "mean", "sample_std"])
        for name in configs[0]["class_names"]:
            for key in ("precision", "recall", "f1-score"):
                values = [report[name][key] for report in per_class]
                writer.writerow([name, key, *values, statistics.mean(values), statistics.stdev(values)])
    prefix.with_suffix(".json").write_text(json.dumps(
        {"seeds": [0, 1, 2], "std_ddof": 1, "split": "val", "runs": rows, "summary": summary},
        indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for seed, ax_color in enumerate(("tab:blue", "tab:orange", "tab:green")):
        history = list(csv.DictReader((args.runs_dir / f"baseline_20shot_seed{seed}" / "history.csv").open()))
        steps = [int(row["step"]) for row in history]
        for ax, key, label in zip(axes[:2], ("val_macro_f1", "val_accuracy"), ("Macro F1 (%)", "Accuracy (%)")):
            ax.plot(steps, [100 * float(row[key]) for row in history], marker=".", color=ax_color, label=f"Seed {seed}")
            ax.set(xlabel="Optimizer step", ylabel=label)
            ax.grid(alpha=0.25)
    axes[0].legend()
    keys = ("macro_f1", "accuracy", "macro_precision", "macro_recall")
    axes[2].bar(range(4), [100 * summary[k]["mean"] for k in keys],
                yerr=[100 * summary[k]["std"] for k in keys], capsize=5, alpha=0.6)
    for seed, color in enumerate(("tab:blue", "tab:orange", "tab:green")):
        axes[2].scatter([i + (seed - 1) * 0.09 for i in range(4)],
                        [100 * rows[seed][key] for key in keys], color=color, s=20)
    axes[2].set(xticks=range(4), xticklabels=["F1", "Accuracy", "Precision", "Recall"],
                ylabel="Validation score (%)", title="Mean ± sample SD (n=3)")
    axes[2].grid(axis="y", alpha=0.25)
    fig.suptitle("Real-only 20-shot: center crop, seeds 0–2")
    fig.tight_layout()
    fig.savefig(prefix.with_suffix(".png"), dpi=180)
    plt.close(fig)
    table = ["| Seed | Macro F1 | Accuracy | Macro precision | Macro recall | Val loss | Best step | Time (min) |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        scores = " | ".join(f"{100 * row[key]:.2f}%" for key in keys)
        table.append(f"| {row['seed']} | {scores} | {row['val_loss']:.4f} | {row['best_step']} | {row['training_seconds']/60:.2f} |")
    scores = " | ".join(f"{100*summary[k]['mean']:.2f}% ± {100*summary[k]['std']:.2f} pp" for k in keys)
    table.append(f"| **Mean ± SD** | {scores} | {summary['val_loss']['mean']:.4f} ± {summary['val_loss']['std']:.4f} | — | {summary['training_seconds']['mean']/60:.2f} ± {summary['training_seconds']['std']/60:.2f} |")
    text = "\n".join(table)
    commands = "\n".join(f"python src/train_classifier.py --k-shot 20 --seed {s} --device cuda --amp --output {rows[s]['run']}" for s in range(3))
    # Use project-relative paths in the reproducible commands.
    commands = "\n".join(f"python src/train_classifier.py --k-shot 20 --seed {s} --device cuda --amp --output {(args.runs_dir / rows[s]['run']).relative_to(PROJECT_ROOT).as_posix()}" for s in range(3))
    document = f"""# Real-only 20-shot experiment, seeds 0–2

Run date: 23 October 2026. Three fresh runs; earlier seed-0 results are preserved.

{text}

SD is the sample standard deviation across three seeds (`ddof=1`). Score SDs are in percentage points (pp). Seeds change both the existing 20-shot subset and head initialization / shuffled training order. Macro precision, recall and F1 give equal weight to each of 200 species. Validation has five images per species, so macro recall equals accuracy.

![Training curves and mean with standard deviation]({prefix.name}.png)

Each run uses 4,000 distinct real images (20 per species). RGB conversion, aspect-ratio-preserving resize of the shorter side to 256, 224×224 center crop, tensor conversion and ImageNet normalization apply to training and validation. No generated images or additional augmented views are used.

Settings match the earlier 20-shot run: ImageNet ResNet-18 with frozen backbone and BatchNorm, new 200-class linear head, AdamW, learning rate 0.001, weight decay 0.0001, batch size 64, 1,000 optimizer steps, validation every 100 steps, CUDA with AMP, zero data-loader workers. Each checkpoint is selected independently by maximum macro F1 on the same 1,000-image validation split. Reported validation metrics use that selected checkpoint; test metrics were not computed. Three-seed variability is descriptive and is not a confidence interval or a test-set estimate.

```bash
{commands}
```

Outputs: `{args.runs_dir.relative_to(PROJECT_ROOT).as_posix()}/baseline_20shot_seed{{0,1,2}}/`, each containing `best.pt`, `config.json`, `history.csv`, `runtime.json`, `val_metrics.json`, `val_per_class.json`, `val_predictions.csv`, and `train.log`. Runtime includes training, periodic validation and checkpoint saving; excludes setup and final evaluation. The batch's `source_snapshot/` and `provenance.json` preserve source/input files and their hashes.

Files: [per-run CSV]({prefix.name}.csv), [mean and SD CSV]({prefix.name}_summary.csv), [JSON]({prefix.name}.json), [per-class scores]({prefix.name}_per_class.csv).

Verification passed: unique training IDs, 20 images per species, containment in the training pool, no validation/test overlap, shared class mapping and fixed settings, all recorded hashes against current files and the preserved snapshot, validation IDs in manifest order, aggregate and per-class metrics recomputed from predictions, and best checkpoint step / metrics matched to history.
"""
    prefix.with_suffix(".md").write_text(document, encoding="utf-8")
    print(text)
    print(f"Saved report: {prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()
