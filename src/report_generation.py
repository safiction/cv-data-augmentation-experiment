"""Validate and report generation cost and review quality of the synthetic pools.

Cost comes from the pool manifests: `gen_time_s` is the time of one pipeline call per image
(batch size 1), measured by generate.py; it excludes model download and loading. The first image
of a run includes CUDA warm-up, so the median is the per-image figure and the sum is the total.
Generation cost is reported separately from classifier training time, with the GPU.

Quality comes from the review tables (one decision per image, see docs/generation.md) and the
prepared training pool: how many images per species survived review and leakage filtering and
entered training. Validation and test images are used only by the leakage check.

Usage: python src/report_generation.py \
    --prepared reports/generation/prepared_sd15_wildlife_accepted \
    --output-prefix reports/generation/generation_records
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from generate import PILOT_CLASSES, POOL_SETTINGS
from load_data import PROJECT_ROOT, load_splits, manifest_path
from synthetic_data import ACCEPTED, DECISIONS

REPORTS_DIR = PROJECT_ROOT / "reports" / "generation"
POOLS = ("pilot_photo", "pilot_wildlife", "sd15_wildlife")
FULL_POOL = "sd15_wildlife"
TARGET = 10  # synthetic images per species in the training pool
# Settings that must be the same for every image of a pool.
CONSTANT = (*POOL_SETTINGS, "scheduler", "size", "batch_size", "device")
# Accepted images per species: none, few, partial, full (as in report_synthetic_seeds.py).
BUCKETS = [("0", 0, 0), ("1-5", 1, 5), ("6-9", 6, 9), ("10", 10, 10)]
# Pilot species: the first five are distinctive, the last five have look-alikes in CUB.
PILOT_GROUPS = {"distinctive": PILOT_CLASSES[:5], "hard": PILOT_CLASSES[5:]}


def load_pool(pool):
    """Manifest and review table of a pool, checked against each other and the leakage report."""
    manifest = pd.read_csv(manifest_path(f"synthetic_{pool}"))
    review = pd.read_csv(REPORTS_DIR / f"review_{pool}.csv", keep_default_na=False)
    assert manifest.image_id.is_unique and review.image_id.is_unique, pool
    assert set(manifest.image_id) == set(review.image_id), pool
    assert review.decision.isin(DECISIONS).all(), (pool, set(review.decision) - set(DECISIONS))
    assert manifest.gen_time_s.notna().all() and (manifest.gen_time_s > 0).all(), pool
    for key in CONSTANT:
        assert manifest[key].nunique() == 1, (pool, key, sorted(set(manifest[key])))
    leakage = json.loads((REPORTS_DIR / f"leakage_{pool}.json").read_text(encoding="utf-8"))
    assert leakage["pool"] == pool and leakage["images"] == len(manifest), pool
    leaked = {p["synthetic"] for pairs in leakage["pairs"].values() for p in pairs}
    assert leaked == set(review.loc[review.decision == "leak", "image_id"]), pool
    pool_df = manifest.merge(review[["image_id", "species", "decision"]], on="image_id")
    matches = {split: len(pairs) for split, pairs in leakage["pairs"].items()}
    return pool_df, matches


def pool_row(pool, pool_df, matches):
    first = pool_df.iloc[0]
    decisions = pool_df.decision.value_counts()
    accepted = int(pool_df.decision.isin(ACCEPTED).sum())
    total = pool_df.gen_time_s.sum()
    return {
        "pool": pool, "template": first["template"], "images": len(pool_df),
        "species": pool_df.class_name.nunique(), "gpu": first["device"], "dtype": first["dtype"],
        "scheduler": first["scheduler"], "steps": int(first["steps"]), "guidance": float(first["guidance"]),
        "size": int(first["size"]), "batch_size": int(first["batch_size"]),
        "median_seconds_per_image": pool_df.gen_time_s.median(),
        "mean_seconds_per_image": pool_df.gen_time_s.mean(), "total_minutes": total / 60,
        **{decision: int(decisions.get(decision, 0)) for decision in DECISIONS},
        "accepted": accepted, "accepted_share": accepted / len(pool_df),
        "seconds_per_accepted_image": total / accepted,
        "val_matches": matches["val"], "test_matches": matches["test"],
    }


def check_prepared(prepared, pool_df):
    """The prepared training pool keeps every accepted image up to TARGET per species, unchanged."""
    manifest = pd.read_csv(prepared / "manifest.csv")
    report = json.loads((prepared / "leakage_report.json").read_text(encoding="utf-8"))
    accepted = pool_df[pool_df.decision.isin(ACCEPTED)]
    assert report["pool_images"] == len(pool_df) and report["accepted_images"] == len(manifest)
    assert not report["matches"]["val"] and not report["matches"]["test"]
    assert set(manifest.image_id) <= set(accepted.image_id)
    md5 = dict(zip(pool_df.image_id, pool_df.md5))
    assert all(md5[i] == m for i, m in zip(manifest.image_id, manifest.md5))
    labels = load_splits()[["class_name", "label"]].drop_duplicates().set_index("class_name").label
    selected = manifest.class_name.value_counts()
    expected = accepted.class_name.value_counts().clip(upper=TARGET)
    for name, label in labels.items():
        n = int(selected.get(name, 0))
        assert n == int(expected.get(name, 0)) == report["accepted_per_class"][str(label)], name
    return manifest, labels


def per_class_rows(pool_df, selected, labels):
    rows = []
    for name, group in pool_df.groupby("class_name"):
        decisions = group.decision.value_counts()
        rows.append({
            "class_name": name, "label": int(labels[name]), "species": group.species.iloc[0],
            "images": len(group), **{d: int(decisions.get(d, 0)) for d in DECISIONS},
            "accepted": int(group.decision.isin(ACCEPTED).sum()),
            "selected_for_training": int(selected.get(name, 0)),
            "median_seconds_per_image": group.gen_time_s.median(),
        })
    return rows


def pilot_fidelity(pools):
    """Right-species images (ok + artifact) per pilot template and species group."""
    rows = []
    for pool in ("pilot_photo", "pilot_wildlife"):
        pool_df = pools[pool]
        for group, names in PILOT_GROUPS.items():
            members = pool_df[pool_df.class_name.isin(names)]
            rows.append({"pool": pool, "group": group, "images": len(members),
                         "right_species": int(members.decision.isin(ACCEPTED).sum()),
                         "ok": int((members.decision == "ok").sum())})
    return rows


def coverage(per_class):
    buckets = []
    for label, low, high in BUCKETS:
        members = [r for r in per_class if low <= r["selected_for_training"] <= high]
        buckets.append({"selected_per_species": label, "species": len(members),
                        "images": sum(r["selected_for_training"] for r in members)})
    return buckets


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepared", type=Path, required=True, help="prepared_accepted directory used for training")
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    pools, rows = {}, []
    for pool in POOLS:
        pool_df, matches = load_pool(pool)
        pools[pool] = pool_df
        rows.append(pool_row(pool, pool_df, matches))
    # The pilots are paired: same species, seeds and settings, only the template differs.
    photo, wildlife = pools["pilot_photo"], pools["pilot_wildlife"]
    assert list(photo.seed) == list(wildlife.seed) and set(photo.class_name) == set(PILOT_CLASSES)

    full = pools[FULL_POOL]
    assert len(full) == 2000 and (full.class_name.value_counts() == TARGET).all()
    prepared, labels = check_prepared(args.prepared, full)
    per_class = per_class_rows(full, prepared.class_name.value_counts(), labels)
    pilot = pilot_fidelity(pools)
    buckets = coverage(per_class)

    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    write_csv(prefix.with_name(prefix.name + "_by_pool.csv"), rows)
    write_csv(prefix.with_name(prefix.name + "_per_class.csv"), per_class)
    prefix.with_suffix(".json").write_text(json.dumps({
        "timing_scope": "Pipeline call per image, batch size 1; excludes model download and loading",
        "pools": rows, "pilot_fidelity": pilot, "training_pool": {
            "pool": FULL_POOL, "target_per_species": TARGET, "images": len(prepared),
            "species_with_images": sum(r["selected_for_training"] > 0 for r in per_class),
            "by_selected_count": buckets},
    }, indent=2), encoding="utf-8")

    plot(full, rows, per_class, prefix)
    write_markdown(prefix, args.prepared, rows, pilot, per_class, buckets)
    print(prefix.with_suffix(".md").read_text(encoding="utf-8").split("\n\n")[2])
    print(f"Saved report: {prefix.with_suffix('.md')}")


def plot(full, rows, per_class, prefix):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    median = full.gen_time_s.median()
    axes[0].hist(full.gen_time_s, bins=40, color="tab:blue", alpha=0.7)
    axes[0].axvline(median, color="black", linestyle="--", label=f"median {median:.2f} s")
    axes[0].set(xlabel="Seconds per image", ylabel="Images",
                title=f"{FULL_POOL}: generation time, {full.device.iloc[0].replace('NVIDIA GeForce ', '')}")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    colors = {"ok": "tab:green", "artifact": "tab:olive", "wrong_species": "tab:red",
              "not_a_bird": "tab:gray", "leak": "black"}
    bottom = [0.0] * len(rows)
    for decision in DECISIONS:
        shares = [100 * row[decision] / row["images"] for row in rows]
        axes[1].bar(range(len(rows)), shares, bottom=bottom, color=colors[decision], label=decision)
        bottom = [b + s for b, s in zip(bottom, shares)]
    axes[1].set(xticks=range(len(rows)), xticklabels=[f"{r['pool']}\n({r['images']} images)" for r in rows],
                ylabel="Share of images (%)", title="Review decisions", ylim=(0, 112))
    axes[1].legend(fontsize=7, loc="upper center", ncol=len(DECISIONS))

    counts = [r["selected_for_training"] for r in per_class]
    axes[2].bar(range(TARGET + 1), [counts.count(n) for n in range(TARGET + 1)], color="tab:purple", alpha=0.7)
    axes[2].set(xticks=range(TARGET + 1), xlabel="Accepted synthetic images per species",
                ylabel="Species", title="Training pool coverage")
    axes[2].grid(axis="y", alpha=0.25)
    fig.suptitle("Synthetic generation: cost and review quality")
    fig.tight_layout()
    fig.savefig(prefix.with_suffix(".png"), dpi=180)
    plt.close(fig)


def pct(value):
    return f"{100 * value:.1f}%"


def write_markdown(prefix, prepared, rows, pilot, per_class, buckets):
    cost = ["| Pool | Images | GPU | Median s/image | Mean s/image | Total (min) | s per accepted image |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    for r in rows:
        cost.append(f"| `{r['pool']}` | {r['images']:,} | {r['gpu'].replace('NVIDIA GeForce ', '')} | "
                    f"{r['median_seconds_per_image']:.2f} | {r['mean_seconds_per_image']:.2f} | "
                    f"{r['total_minutes']:.1f} | {r['seconds_per_accepted_image']:.2f} |")
    quality = ["| Pool | " + " | ".join(f"`{d}`" for d in DECISIONS) + " | Accepted | Val/test matches |",
               "| --- |" + " ---: |" * (len(DECISIONS) + 2)]
    for r in rows:
        quality.append(f"| `{r['pool']}` | " + " | ".join(f"{r[d]:,}" for d in DECISIONS) +
                       f" | {r['accepted']:,} ({pct(r['accepted_share'])}) | {r['val_matches'] + r['test_matches']} |")
    fidelity = ["| Template | Species group | Images | Right species (`ok` + `artifact`) | `ok` |",
                "| --- | --- | ---: | ---: | ---: |"]
    for r in pilot:
        fidelity.append(f"| {r['pool'].split('_')[1]} | {r['group']} | {r['images']} | "
                        f"{r['right_species']} | {r['ok']} |")
    coverage_table = ["| Accepted images per species | Species | Images |", "| --- | ---: | ---: |"]
    coverage_table += [f"| {b['selected_per_species']} | {b['species']} | {b['images']:,} |" for b in buckets]
    missing = [r for r in per_class if r["selected_for_training"] == 0]
    worst_table = ["| Species | `ok` | `artifact` | `wrong_species` | `not_a_bird` | Accepted |",
                   "| --- | ---: | ---: | ---: | ---: | ---: |"]
    worst_table += [f"| {r['species']} | {r['ok']} | {r['artifact']} | {r['wrong_species']} | "
                    f"{r['not_a_bird']} | {r['selected_for_training']} |" for r in missing]

    full = rows[-1]
    pilots = {r['pool']: r for r in rows[:2]}
    better = min(pilots, key=lambda pool: pilots[pool]['wrong_species'])
    right = {(r['pool'], r['group']): r['right_species'] for r in pilot}
    hard_ok = all(right[pool, 'hard'] >= right[pool, 'distinctive'] for pool in pilots)
    selected = sum(r["selected_for_training"] for r in per_class)
    covered = sum(r["selected_for_training"] > 0 for r in per_class)
    prepared_path = prepared.resolve().relative_to(PROJECT_ROOT).as_posix()
    output = prefix.resolve().relative_to(PROJECT_ROOT).as_posix()

    document = f"""# Synthetic generation: cost and quality records

Model `stable-diffusion-v1-5/stable-diffusion-v1-5`, text-to-image from species-name prompts, {full['dtype']}, {full['scheduler']} with {full['steps']} steps, guidance {full['guidance']}, {full['size']}×{full['size']}, batch size {full['batch_size']}. Each pool was generated on a single GPU.

{chr(10).join(cost)}

Time is one pipeline call per image as recorded in `gen_time_s`; it excludes model download and loading. The first image of each run includes CUDA warm-up, so the median is the representative per-image cost. "s per accepted image" divides the total time by the images that passed review. Generation is a one-off cost: the same pool is reused by every classifier seed, and classifier training time is reported separately.

## Review quality

{chr(10).join(quality)}

Accepted means `ok` or `artifact`: the labeled species with or without visual defects. `wrong_species`, `not_a_bird` and `leak` are dropped. No synthetic image matched a validation or test photo (MD5 or ResNet-18 embedding similarity).

### Pilot: template and species fidelity

{chr(10).join(fidelity)}

25 images per group: 5 species × 5 images, with the same seeds for both templates. `{better.split('_')[1]}` produced fewer wrong-species images ({pilots[better]['wrong_species']} vs. {min(r['wrong_species'] for r in pilots.values() if r is not pilots[better])}); the full pool uses `{full['template']}`. Hard species (gulls, a sparrow, a flycatcher, a warbler) were the right species {'at least as often as' if hard_ok else 'less often than'} distinctive ones in this small sample.

## Training pool

`{prepared_path}` holds {selected:,} images: all accepted images, at most {TARGET} per species, after a second leakage check against validation and test. {covered} of {len(per_class)} species have at least one synthetic image; the overall synthetic-to-real ratio with 2,000 real images is {selected / 2000:.2f}:1.

{chr(10).join(coverage_table)}

Species without synthetic images in training (only real images):

{chr(10).join(worst_table)}

![Generation time, review decisions and training pool coverage]({prefix.name}.png)

## Reproduce

```bash
python src/generate.py --pilot
python src/generate.py --pilot --template wildlife
python src/generate.py --template wildlife --per-class 10
python src/review_synthetic.py --pool sd15_wildlife
python src/prepare_synthetic.py --device cuda --review reports/generation/review_sd15_wildlife.csv --per-class 10 --output data/synthetic/sd15_wildlife/prepared_accepted
python src/report_generation.py --prepared {prepared_path} --output-prefix {output}
```

Files: [cost and decisions per pool]({prefix.name}_by_pool.csv), [decisions per species]({prefix.name}_per_class.csv), [JSON]({prefix.name}.json). Review tables and leakage reports: `reports/generation/review_{{pool}}.csv`, `reports/generation/leakage_{{pool}}.json`.

## Limitations

Timings come from a single GPU and include no review time, review decisions are manual. The pilot has 25 images per group, too few to compare species groups reliably.

Verification passed: review tables cover exactly the manifest images with valid decisions; generation settings and GPU are constant within each pool; `leak` decisions match the leakage reports; the pilots share species and seeds; the training pool contains only accepted images with unchanged MD5s, and its per-species counts match the review and the prepared leakage report.
"""
    prefix.with_suffix(".md").write_text(document, encoding="utf-8")


if __name__ == "__main__":
    main()
