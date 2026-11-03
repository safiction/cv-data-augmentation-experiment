"""Validate and report three real + accepted synthetic 10-shot classifier runs (seeds 0-2).

Real-only references: the matched 10-shot baselines for seeds 0-2 (paired real subsets) and the
three-seed real-only 20-shot summary (4,000 real images, a similar training-set size).
The exploratory per-species breakdown below remains restricted to seed 0.

Usage: python src/report_synthetic_seeds.py \
    --runs outputs/accepted_synthetic_10shot_seed0_20261023/real_synthetic_10shot_seed0 \
           outputs/accepted_synthetic_10shot_seed1 outputs/accepted_synthetic_10shot_seed2 \
    --prepared reports/generation/prepared_sd15_wildlife_accepted \
    --output-prefix reports/accepted_synthetic_10shot_seeds0-2
"""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from summarize_results import collect_run, read_csv, read_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SYNTHETIC_MANIFEST = "data/synthetic/sd15_wildlife/prepared_accepted/manifest.csv"
LEAKAGE_REPORT = "data/synthetic/sd15_wildlife/prepared_accepted/leakage_report.json"
EXPECTED = dict(k_shot=10, augment=False, full_image=False, max_steps=1000, eval_every=100,
                batch_size=64, lr=0.001, weight_decay=0.0001, amp=True, num_workers=0,
                train_images=3440, real_train_images=2000, synthetic_train_images=1440,
                augmented_train_images=0, train_preprocessing="resize_center_crop",
                frozen_backbone=True, frozen_batchnorm=True, device="cuda", arm="real_synthetic")
# Settings the real-only baseline must share with the synthetic runs.
SHARED = ("k_shot", "augment", "full_image", "max_steps", "eval_every", "batch_size", "lr",
          "weight_decay", "amp", "num_workers", "train_preprocessing", "frozen_backbone",
          "frozen_batchnorm", "class_names")
# Synthetic images per species after review: none, few, partial, full.
BUCKETS = [("0", 0, 0), ("1-5", 1, 5), ("6-9", 6, 9), ("10", 10, 10)]
SCORES = ("macro_f1", "accuracy", "macro_precision", "macro_recall")


def with_macro(row, run):
    report = read_json(run / "val_per_class.json")
    row.update(macro_precision=report["macro avg"]["precision"],
               macro_recall=report["macro avg"]["recall"])
    return row, report


def check_hashes(config, run, prepared):
    """Split manifests and code match the repo; the synthetic pool matches the prepared files."""
    for relative, digest in config["sha256"].items():
        if relative == SYNTHETIC_MANIFEST:
            source = prepared / "manifest.csv"
        elif relative == LEAKAGE_REPORT:
            source = prepared / "leakage_report.json"
        else:
            source = PROJECT_ROOT / relative
        assert hash_matches(source, digest), (run, relative)


def hash_matches(path, expected, legacy=False):
    """Normalize text line endings; seed 0 also accepted raw Windows hashes."""
    raw = path.read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    candidates = (raw, lf, lf.replace(b"\n", b"\r\n")) if legacy else (lf,)
    return any(hashlib.sha256(data).hexdigest() == expected for data in candidates)


def mean_std(values):
    return {"mean": statistics.mean(values), "std": statistics.stdev(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, nargs=3, required=True, help="run directories for seeds 0, 1, 2")
    parser.add_argument("--prepared", type=Path, required=True, help="prepared_accepted directory used for training")
    parser.add_argument("--baseline", type=Path, default=PROJECT_ROOT / "reports" / "classifier" / "baseline_seed0")
    parser.add_argument("--baseline-seeds12", type=Path, nargs=2,
                        default=[PROJECT_ROOT / "reports" / "classifier" / f"baseline_seed{s}" for s in (1, 2)])
    parser.add_argument("--real20-summary", type=Path, default=PROJECT_ROOT / "reports" / "20shot_seeds0-2_summary.csv")
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    held_out = {row["image_id"] for split in ("val", "test")
                for row in read_csv(PROJECT_ROOT / "splits" / "manifests" / f"{split}.csv")}
    synthetic = read_csv(args.prepared / "manifest.csv")
    assert len(synthetic) == 1440 and not {row["image_id"] for row in synthetic} & held_out

    rows, configs, per_class = [], [], []
    for seed, run in enumerate(args.runs):
        config = read_json(run / "config.json")
        assert config["seed"] == seed, run
        for key, value in EXPECTED.items():
            assert config[key] == value, (run, key, config[key], value)
        check_hashes(config, run, args.prepared)
        row, report = with_macro(collect_run(run), run)
        row.update(run=run.resolve().relative_to(PROJECT_ROOT).as_posix(), gpu=config["gpu"])
        rows.append(row)
        configs.append(config)
        per_class.append(report)
    for key in ("class_names", "synthetic_per_class", "torch", "torchvision", "sklearn", "weights"):
        assert all(c[key] == configs[0][key] for c in configs), key
    assert len({c["sha256"][SYNTHETIC_MANIFEST] for c in configs}) == 1

    # The baseline config predates --full-image and train_preprocessing: then training always used
    # resize + center crop. Its raw Windows hashes also need the legacy line-ending check.
    defaults = {"full_image": False, "train_preprocessing": "resize_center_crop"}
    base_rows = []
    for seed, run in enumerate([args.baseline, *args.baseline_seeds12]):
        base_config = read_json(run / "config.json")
        assert base_config["seed"] == seed and base_config["arm"] == "baseline"
        assert base_config["train_images"] == 2000 and base_config.get("synthetic_manifest") is None
        for key in (*SHARED, "architecture", "weights", "selection_metric"):
            assert base_config.get(key, defaults.get(key)) == configs[seed][key], (run, key)
        for relative, digest in base_config["sha256"].items():
            # Seed 0 used an older training implementation; validate its data hashes here.
            if seed != 0 or relative.startswith("splits/"):
                assert hash_matches(PROJECT_ROOT / relative, digest, legacy=(seed == 0)), (run, relative)
        base_row, report = with_macro(collect_run(run), run)
        base_rows.append(base_row)
        if seed == 0:
            base_report = report
    base_row = base_rows[0]
    base_summary = {key: mean_std([r[key] for r in base_rows]) for key in SCORES}
    paired_deltas = [100 * (r["macro_f1"] - b["macro_f1"]) for r, b in zip(rows, base_rows)]

    real20 = {r["metric"]: (float(r["mean"]), float(r["sample_std"]))
              for r in csv.DictReader(args.real20_summary.open(encoding="utf-8"))}
    summary = {key: mean_std([row[key] for row in rows])
               for key in (*SCORES, "val_loss", "best_step", "training_seconds")}

    names = configs[0]["class_names"]
    counts = {names[int(label)]: n for label, n in configs[0]["synthetic_per_class"].items()}
    counts = {name: counts.get(name, 0) for name in names}
    f1 = {name: [report[name]["f1-score"] for report in per_class] for name in names}
    delta0 = {name: f1[name][0] - base_report[name]["f1-score"] for name in names}
    buckets = []
    for label, low, high in BUCKETS:
        members = [n for n in names if low <= counts[n] <= high]
        deltas = [delta0[n] for n in members]
        buckets.append({
            "synthetic_per_species": label, "species": len(members),
            "mean_f1_real_only_seed0": statistics.mean(base_report[n]["f1-score"] for n in members),
            "mean_f1_synthetic_seed0": statistics.mean(f1[n][0] for n in members),
            "mean_delta_f1_seed0_pp": 100 * statistics.mean(deltas),
            "species_better": sum(d > 0 for d in deltas), "species_same": sum(d == 0 for d in deltas),
            "species_worse": sum(d < 0 for d in deltas),
            "mean_f1_synthetic_seeds0_2": statistics.mean(statistics.mean(f1[n]) for n in members),
        })

    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fields = [*rows[0]]
    with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with prefix.with_name(prefix.name + "_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["metric", "mean", "sample_std", "n"])
        writer.writerows((key, value["mean"], value["std"], 3) for key, value in summary.items())
    with prefix.with_name(prefix.name + "_per_class.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["class_name", "synthetic_images", "metric", "seed0", "seed1", "seed2", "mean",
                         "sample_std", "real_only_seed0", "delta_seed0"])
        for name in names:
            for key in ("precision", "recall", "f1-score"):
                values = [report[name][key] for report in per_class]
                writer.writerow([name, counts[name], key, *values, statistics.mean(values),
                                 statistics.stdev(values), base_report[name][key],
                                 values[0] - base_report[name][key]])
    with prefix.with_name(prefix.name + "_by_synthetic_count.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(buckets[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(buckets)
    prefix.with_suffix(".json").write_text(json.dumps({
        "seeds": [0, 1, 2], "std_ddof": 1, "split": "val", "runs": rows, "summary": summary,
        "real_only_10shot_seed0": base_row, "real_only_20shot_seeds0_2": real20,
        "real_only_10shot_seeds0_2": base_rows, "real_only_10shot_summary": base_summary,
        "paired_macro_f1_change_pp": paired_deltas,
        "paired_macro_f1_change_summary_pp": mean_std(paired_deltas),
        "synthetic_manifest_sha256": configs[0]["sha256"][SYNTHETIC_MANIFEST],
        "by_synthetic_count_seed0": buckets,
    }, indent=2), encoding="utf-8")

    plot(args.runs, args.baseline, rows, summary, base_summary, real20, buckets, prefix)
    write_markdown(prefix, rows, summary, base_summary, paired_deltas, real20, buckets, counts, delta0)
    print(prefix.with_suffix(".md").read_text(encoding="utf-8").split("\n\n")[2])
    print(f"Saved report: {prefix.with_suffix('.md')}")


def plot(runs, baseline, rows, summary, base_summary, real20, buckets, prefix):
    colors = ("tab:blue", "tab:orange", "tab:green")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    for seed, (run, color) in enumerate(zip(runs, colors)):
        history = list(csv.DictReader((run / "history.csv").open(encoding="utf-8")))
        axes[0].plot([int(r["step"]) for r in history], [100 * float(r["val_macro_f1"]) for r in history],
                     marker=".", color=color, label=f"Real + synthetic, seed {seed}")
    history = list(csv.DictReader((baseline / "history.csv").open(encoding="utf-8")))
    axes[0].plot([int(r["step"]) for r in history], [100 * float(r["val_macro_f1"]) for r in history],
                 color="black", linestyle="--", label="Real-only 10-shot, seed 0")
    axes[0].set(xlabel="Optimizer step", ylabel="Validation macro F1 (%)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    labels = ["Real-only\n10-shot (seeds 0-2)", "Real + synthetic\n10-shot (seeds 0-2)", "Real-only\n20-shot (seeds 0-2)"]
    means = [100 * base_summary["macro_f1"]["mean"], 100 * summary["macro_f1"]["mean"], 100 * real20["macro_f1"][0]]
    errors = [100 * base_summary["macro_f1"]["std"], 100 * summary["macro_f1"]["std"], 100 * real20["macro_f1"][1]]
    axes[1].bar(range(3), means, yerr=errors, capsize=5, alpha=0.6, color=["gray", "tab:purple", "tab:gray"])
    for seed, color in enumerate(colors):
        axes[1].scatter([1 + (seed - 1) * 0.09], [100 * rows[seed]["macro_f1"]], color=color, s=20, zorder=3)
    axes[1].set(xticks=range(3), xticklabels=labels, ylabel="Validation macro F1 (%)",
                title="Mean ± sample SD", ylim=(40, 65))
    axes[1].grid(axis="y", alpha=0.25)

    axes[2].bar(range(len(buckets)), [b["mean_delta_f1_seed0_pp"] for b in buckets], alpha=0.7,
                color=["tab:red" if b["mean_delta_f1_seed0_pp"] < 0 else "tab:green" for b in buckets])
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set(xticks=range(len(buckets)),
                xticklabels=[f"{b['synthetic_per_species']}\n({b['species']} species)" for b in buckets],
                xlabel="Accepted synthetic images per species", ylabel="Mean per-species F1 change (pp)",
                title="Seed 0: real + synthetic minus real-only")
    axes[2].grid(axis="y", alpha=0.25)
    fig.suptitle("Real + accepted synthetic, 10-shot: center crop, seeds 0-2")
    fig.tight_layout()
    fig.savefig(prefix.with_suffix(".png"), dpi=180)
    plt.close(fig)


def pct(value):
    return f"{100 * value:.2f}%"


def write_markdown(prefix, rows, summary, base_summary, paired_deltas, real20, buckets, counts, delta0):
    table = ["| Seed | Macro F1 | Accuracy | Macro precision | Macro recall | Val loss | Best step | Time (min) | GPU |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in rows:
        scores = " | ".join(pct(row[key]) for key in SCORES)
        table.append(f"| {row['seed']} | {scores} | {row['val_loss']:.4f} | {row['best_step']} | "
                     f"{row['training_seconds'] / 60:.2f} | {row['gpu'].replace('NVIDIA GeForce ', '')} |")
    scores = " | ".join(f"{pct(summary[k]['mean'])} ± {100 * summary[k]['std']:.2f} pp" for k in SCORES)
    table.append(f"| **Mean ± SD** | {scores} | {summary['val_loss']['mean']:.4f} ± {summary['val_loss']['std']:.4f} "
                 f"| — | {summary['training_seconds']['mean'] / 60:.2f} ± {summary['training_seconds']['std'] / 60:.2f} | |")

    m, s = summary["macro_f1"]["mean"], summary["macro_f1"]["std"]
    r20, r20s = real20["macro_f1"]
    a20, a20s = real20["accuracy"]
    comparison = "\n".join([
        "| Training data | Real | Synthetic | Seeds | Validation macro F1 | Accuracy |",
        "| --- | ---: | ---: | --- | ---: | ---: |",
        f"| Real-only, 10-shot | 2,000 | 0 | 0-2 | {pct(base_summary['macro_f1']['mean'])} ± {100 * base_summary['macro_f1']['std']:.2f} pp | {pct(base_summary['accuracy']['mean'])} ± {100 * base_summary['accuracy']['std']:.2f} pp |",
        f"| **Real + accepted synthetic, 10-shot** | 2,000 | 1,440 | 0-2 | **{pct(m)} ± {100 * s:.2f} pp** | "
        f"**{pct(summary['accuracy']['mean'])} ± {100 * summary['accuracy']['std']:.2f} pp** |",
        f"| Real-only, 20-shot | 4,000 | 0 | 0-2 | {100 * r20:.2f}% ± {100 * r20s:.2f} pp | {100 * a20:.2f}% ± {100 * a20s:.2f} pp |",
    ])
    bucket_table = "\n".join([
        "| Synthetic images per species | Species | Real-only F1 | Real + synthetic F1 | Change | Better / same / worse |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
        *(f"| {b['synthetic_per_species']} | {b['species']} | {pct(b['mean_f1_real_only_seed0'])} | "
          f"{pct(b['mean_f1_synthetic_seed0'])} | {b['mean_delta_f1_seed0_pp']:+.2f} pp | "
          f"{b['species_better']} / {b['species_same']} / {b['species_worse']} |" for b in buckets),
    ])
    gap20 = 100 * (r20 - m)
    no_syn, full_syn = buckets[0], buckets[-1]
    commands = "\n".join(
        f"python src/train_classifier.py --k-shot 10 --seed {r['seed']} --device cuda --amp "
        f"--synthetic-manifest {SYNTHETIC_MANIFEST} --output {r['run']}" for r in rows)

    document = f"""# Real + accepted synthetic augmentation, 10-shot, seeds 0–2

Run dates: seed 0 on 23 October 2026 (RTX 4060 Laptop), seeds 1–2 on 24 October 2026 (RTX 3070).

Updated 27 October 2026 with Karina's real-only 10-shot seeds 1–2. The full three-approach comparison, including standard augmentation, is in the [weekly report](../docs/WEEKLY_REPORT_2026-10-27.md).

{chr(10).join(table)}

SD is the sample standard deviation across three seeds (`ddof=1`); score SDs are in percentage points (pp). A seed changes the 10-shot real subset, head initialization and shuffled training order; the synthetic pool is the same in all three runs.

## Comparison with real-only training

{comparison}

![Validation curves, real-only references and per-species change by synthetic count]({prefix.name}.png)

## Conclusions

1. **Synthetic images did not improve validation macro F1 in any paired seed.** Adding the reviewed pool changed macro F1 by **{', '.join(f'{d:+.2f}' for d in paired_deltas)} pp** for seeds 0, 1 and 2. The mean paired change is **{statistics.mean(paired_deltas):+.2f} ± {statistics.stdev(paired_deltas):.2f} pp** (sample SD). Real-only reaches {pct(base_summary['macro_f1']['mean'])} ± {100 * base_summary['macro_f1']['std']:.2f} pp; synthetic reaches {pct(m)} ± {100 * s:.2f} pp. This describes the tested pool and training setup, not synthetic data in general.
2. **The 20-shot real-only reference scores higher.** It gives {100 * r20:.2f}% ± {100 * r20s:.2f} pp, **{gap20:.1f} pp above** the synthetic arm. It uses twice as many distinct real images, so this comparison does not isolate the effect of augmentation (4,000 vs. 3,440 total entries).
3. **No species group gained on average, and species with incomplete synthetic sets lost most.** In seed 0, species with all 10 synthetic images changed by {full_syn['mean_delta_f1_seed0_pp']:+.2f} pp ({full_syn['species_better']} better, {full_syn['species_worse']} worse of {full_syn['species']}); species with 0–9 synthetic images changed by {min(b['mean_delta_f1_seed0_pp'] for b in buckets[:-1]):+.2f} to {max(b['mean_delta_f1_seed0_pp'] for b in buckets[:-1]):+.2f} pp. Their real-only F1 is similar ({min(b['mean_f1_real_only_seed0'] for b in buckets) * 100:.0f}–{max(b['mean_f1_real_only_seed0'] for b in buckets) * 100:.0f}%), so they were not simply harder. One explanation consistent with this: filtering left 10 to 20 training entries per species, and a classifier trained on this uneven mix favors species with more entries. With 5 validation images per species, one prediction moves a species' F1 by up to ~20 pp, so the pattern is single-seed and noisy.
4. **Generation and preparation are reproducible.** Seeds 1–2 were prepared and trained on another machine; the prepared manifest is byte-identical to the seed-0 one (same SHA-256), and all runs share code, split manifests and training settings.

Likely reasons, not tested here: the uneven per-species synthetic counts described above; manual review removed wrong-species images but cannot verify subtle field marks, so some accepted images still carry misleading features; at a fixed 1,000-step budget, synthetic images take {1440 / 3440:.0%} of presentations away from real ones; SD images differ from CUB photos in style (centered, clean backgrounds), which a frozen ImageNet backbone may encode differently.

## Setup

Each run uses its seed's `splits/manifests/train_10shot_seed{{s}}.csv` (2,000 real images, 10 per species) plus `{SYNTHETIC_MANIFEST}`: 1,440 accepted images (`ok` 1,120, `artifact` 320) from the 2,000-image `sd15_wildlife` pool, with zero matches against validation/test. 185 species have synthetic images; 118 have fewer than 10, including 15 with none (synthetic-to-real ratio 0.72:1 overall). Rejected images are not used to fill the gaps.

Both real and synthetic images use RGB conversion, aspect-ratio-preserving resize of the shorter side to 256, 224×224 center crop, tensor conversion and ImageNet normalization; no crop/flip variants (`--augment` omitted). Settings match all real-only runs: ImageNet ResNet-18 with frozen backbone and BatchNorm, new 200-class linear head, AdamW, learning rate 0.001, weight decay 0.0001, batch size 64, 1,000 optimizer steps, validation every 100 steps, CUDA with AMP, zero loader workers. Each checkpoint is selected by macro F1 on the same 1,000 real validation images; test metrics were not computed. Runtime covers training, periodic validation and checkpoint saves and depends on the GPU.

```bash
python src/prepare_synthetic.py --device cuda --review reports/generation/review_sd15_wildlife.csv --per-class 10 --output data/synthetic/sd15_wildlife/prepared_accepted
{commands}
python src/report_synthetic_seeds.py --runs {' '.join(r['run'] for r in rows)} --prepared reports/generation/prepared_sd15_wildlife_accepted --output-prefix {prefix.relative_to(PROJECT_ROOT).as_posix() if prefix.is_absolute() else prefix.as_posix()}
```

The prepared manifest and leakage report are in [generation/prepared_sd15_wildlife_accepted/](generation/prepared_sd15_wildlife_accepted/). Checkpoints (`best.pt`) are on the shared Drive.

Files: [per-run CSV]({prefix.name}.csv), [mean and SD CSV]({prefix.name}_summary.csv), [JSON]({prefix.name}.json), [per-class scores]({prefix.name}_per_class.csv), [change by synthetic count]({prefix.name}_by_synthetic_count.csv).

## Change by synthetic count, seed 0

{bucket_table}

## Limitations

Aggregate paired changes now cover seeds 0–2; the exploratory per-species count breakdown remains seed 0 only. Validation is also used for checkpoint selection, so scores may be optimistic. Three seeds give a descriptive SD, not a confidence interval. Filtering changed both image quality and class coverage, so these runs do not separate the effect of quality from the effect of the smaller, uneven pool. Hardware and library versions also differ between some runs.

Verification passed: fixed settings in every config, identical synthetic manifest and class mapping across seeds, recorded hashes against current split manifests, code and the prepared files, no synthetic/held-out overlap, baselines sharing settings and the corresponding seed's real subset, validation IDs in manifest order, aggregate and per-class metrics recomputed from predictions, and best checkpoint step / metrics matched to history. The older baseline seed-0 code hashes are retained as provenance; its data hashes and explicit settings were checked.
"""
    prefix.with_suffix(".md").write_text(document, encoding="utf-8")


if __name__ == "__main__":
    main()
