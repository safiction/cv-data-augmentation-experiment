"""Build Arina's validation comparison and runtime tables from audited artifacts.

This does not run inference or convert validation scores into test results.
"""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

from check_reproducibility import ROOT, audit
from summarize_results import read_json, collect_run
from evaluation_selection import STANDARD_SEED0, selected_run

GROUPS = [(10, "baseline"), (10, "standard_aug"), (10, "real_synthetic"),
          (20, "baseline"), (20, "real_synthetic")]
LABELS = ["Real-only\n10-shot", "Standard\n10-shot", "Synthetic\n10-shot",
          "Real-only\n20-shot", "Synthetic\n20-shot"]


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def comparison(rows, split="val", allow_incomplete=False):
    lookup = {(r["k_shot"], r["arm"], r["seed"]): r for r in rows}
    expected = {(shot, arm, seed) for shot, arm in GROUPS for seed in range(3)}
    valid_keys = set(lookup) <= expected if allow_incomplete else set(lookup) == expected
    if len(lookup) != len(rows) or not valid_keys:
        raise ValueError("Expected exactly the 15 distinct selected runs")
    summary, paired = [], []
    for shot, arm in GROUPS:
        group = [lookup[shot, arm, seed] for seed in range(3) if (shot, arm, seed) in lookup]
        if not group:
            raise ValueError(f"No runs for {shot}-shot {arm}")
        row = {"split": split, "k_shot": shot, "arm": arm, "n_seeds": len(group),
               "train_entries": group[0]["train_images"]}
        for metric in ("macro_f1", "accuracy"):
            values = [r[metric] for r in group]
            row[f"{metric}_mean"] = statistics.mean(values)
            row[f"{metric}_sample_sd"] = statistics.stdev(values) if len(values) > 1 else None
        summary.append(row)
    for shot, first, second in ((10, "standard_aug", "baseline"),
                                (10, "real_synthetic", "baseline"),
                                (10, "real_synthetic", "standard_aug"),
                                (20, "real_synthetic", "baseline")):
        for seed in range(3):
            if (shot, first, seed) not in lookup or (shot, second, seed) not in lookup:
                continue
            a, b = lookup[shot, first, seed], lookup[shot, second, seed]
            paired.append({"split": split, "k_shot": shot,
                           "comparison": f"{first}_minus_{second}", "seed": seed,
                           "first_run": a["run"], "second_run": b["run"],
                           "macro_f1_change_pp": 100 * (a["macro_f1"] - b["macro_f1"]),
                           "accuracy_change_pp": 100 * (a["accuracy"] - b["accuracy"])})
    return summary, paired


def checkpoint_inventory(rows):
    """Match local checkpoint metadata to the selected run; no model inference."""
    import torch
    candidates = list((ROOT / "outputs").rglob("best.pt"))
    inventory = []
    for row in rows:
        expected = read_json(ROOT / "reports/classifier" / row["run"] / "config.json")
        matches = []
        for path in candidates:
            config_path = path.parent / "config.json"
            if not config_path.exists() or read_json(config_path) != expected:
                continue
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if (checkpoint["config"] != expected or checkpoint["step"] != row["best_step"]
                    or abs(checkpoint["val_macro_f1"] - row["macro_f1"]) > 1e-12):
                raise ValueError(f"Checkpoint metadata disagrees with selected run: {path}")
            matches.append({"path": path.relative_to(ROOT).as_posix(),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        if len({m["sha256"] for m in matches}) > 1:
            raise ValueError(f"Multiple different checkpoints match {row['run']}")
        inventory.append({"run": row["run"], "selected_step": row["best_step"],
                          "status": "metadata_verified" if matches else "not_available_locally",
                          "files": matches})
    return {"scope": "Local file hashes and saved metadata only; not a team freeze or inference replay",
            "runs": inventory}


def figures(summary, paired, rows, output, split="val"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    legend = [Line2D([], [], marker="o", linestyle="", color=color, label=f"Seed {seed}")
              for seed, color in enumerate(("tab:blue", "tab:orange", "tab:green"))]
    title_split = "Validation" if split == "val" else "Test"
    prefix = "validation" if split == "val" else "test"
    colors = ("tab:blue", "tab:orange", "tab:green")
    labels = [label + (f"\n(n={row['n_seeds']}; incomplete)" if row['n_seeds'] != 3 else "")
              for label, row in zip(LABELS, summary)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric, title in zip(axes, ("macro_f1", "accuracy"), ("Macro F1", "Accuracy")):
        for i, ((shot, arm), row) in enumerate(zip(GROUPS, summary)):
            ax.errorbar(i, 100 * row[f"{metric}_mean"],
                        yerr=100 * row[f"{metric}_sample_sd"] if row[f"{metric}_sample_sd"] is not None else None,
                        fmt="o", color="black", capsize=5)
            group = sorted((r for r in rows if (r["k_shot"], r["arm"]) == (shot, arm)),
                           key=lambda r: r["seed"])
            ax.scatter([i + (r['seed'] - 1) * .12 for r in group], [100 * r[metric] for r in group],
                       c=[colors[r['seed']] for r in group], s=18, alpha=.8)
        ax.axvline(2.5, color="gray", linestyle=":")
        ax.set(xticks=range(5), xticklabels=labels, ylabel=f"{title_split} {title} (%)",
               title=f"{title}: mean ± sample SD")
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Validation-selected checkpoints; seeds 0–2 (not test results)" if split == "val"
                 else "Test results of validation-selected checkpoints; available seeds only")
    fig.legend(handles=legend, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, .08, 1, 1))
    fig.savefig(output / f"{prefix}_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    comparisons = list(dict.fromkeys((r["k_shot"], r["comparison"]) for r in paired))
    for i, key in enumerate(comparisons):
        group = [r for r in paired if (r["k_shot"], r["comparison"]) == key]
        values = [r["macro_f1_change_pp"] for r in group]
        ax.scatter([i + (r['seed'] - 1) * .08 for r in group], values, c=[colors[r['seed']] for r in group])
        ax.errorbar(i, statistics.mean(values), yerr=statistics.stdev(values) if len(values) > 1 else None, fmt="D",
                    color="black", capsize=5, markersize=4)
    ax.axhline(0, color="black", linewidth=.8)
    pair_labels = ["Standard − baseline\n10-shot", "Synthetic − baseline\n10-shot",
                   "Synthetic − standard\n10-shot", "Synthetic − baseline\n20-shot"]
    for i, key in enumerate(comparisons):
        seeds = [r['seed'] for r in paired if (r['k_shot'], r['comparison']) == key]
        if len(seeds) != 3:
            pair_labels[i] += "\nseeds " + ", ".join(map(str, seeds)) + " only"
    ax.set(xticks=range(4), xticklabels=pair_labels,
           ylabel=f"Paired {title_split.lower()} macro-F1 change (pp)", title="Paired seeds; mean ± sample SD")
    ax.grid(axis="y", alpha=.2)
    fig.legend(handles=legend, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, .08, 1, 1))
    fig.savefig(output / f"{prefix}_paired.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/evaluation_2026-11-03")
    parser.add_argument("--checkpoints", action="store_true", help="Also verify local best.pt metadata (requires torch)")
    args = parser.parse_args()
    checked = audit()
    if checked["errors"] or checked["missing_primary_runs"] or checked["missing_reference_runs"]:
        raise ValueError(f"Comparison audit incomplete: {checked['errors']}, "
                         f"{checked['missing_primary_runs']}, {checked['missing_reference_runs']}")
    # Keep Artem's source audit scoped to its original T4 matrix. Validate the
    # replacement bundle's saved predictions/history without claiming source replay.
    for item in checked["runs"]:
        if selected_run(item["run"]) != item["run"]:
            path = ROOT / "reports/classifier" / STANDARD_SEED0
            item["summary"] = {**collect_run(path, project_root=ROOT), "run": STANDARD_SEED0}
            config = read_json(path / "config.json")
            item["environment"] = {key: config[key] for key in item["environment"]}
    rows = [r["summary"] for r in checked["runs"]]
    summary, paired = comparison(rows)
    runtime = []
    for item in checked["runs"]:
        row, env = item["summary"], item["environment"]
        runtime.append({"run": row["run"], "k_shot": row["k_shot"], "arm": row["arm"],
                        "seed": row["seed"], "training_minutes": row["training_seconds"] / 60,
                        "optimizer_steps": row["optimizer_steps"], **env})
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "validation_summary.csv", summary)
    write_csv(args.output / "validation_paired.csv", paired)
    write_csv(args.output / "training_runtime.csv", runtime)
    figures(summary, paired, rows, args.output)
    if args.checkpoints:
        (args.output / "checkpoint_inventory.json").write_text(
            json.dumps(checkpoint_inventory(rows), indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(rows)} validation runs; wrote tables and figures to {args.output}")


if __name__ == "__main__":
    main()
