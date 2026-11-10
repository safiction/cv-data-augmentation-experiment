"""Audit saved comparison artifacts offline, without images, weights or test evaluation.

Exit 1 on inconsistent artifacts. Missing planned runs are reported separately;
--require-complete also makes an incomplete primary comparison fail.
"""
import argparse
import hashlib
import json
import math
import shlex
import subprocess
from collections import Counter
from pathlib import Path

from summarize_results import collect_run, read_csv, read_json

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0, 1, 2)
FILES = ("config.json", "history.csv", "runtime.json", "val_metrics.json",
         "val_per_class.json", "val_predictions.csv")
COMMON = dict(architecture="resnet18", weights="IMAGENET1K_V1", frozen_backbone=True,
              frozen_batchnorm=True, selection_metric="val_macro_f1", max_steps=1000,
              eval_every=100, batch_size=64, lr=0.001, weight_decay=0.0001, amp=True)
SYNTHETIC = "reports/generation/prepared_sd15_wildlife_accepted"
ALIASES = {f"data/synthetic/sd15_wildlife/prepared_accepted/{name}": f"{SYNTHETIC}/{name}"
           for name in ("manifest.csv", "leakage_report.json")}
# The original baseline predates the augmented loader. Verify its recorded source
# bytes against the actual Git snapshot, rather than accepting arbitrary drift.
BASELINE_REV = "8f4050fee9c8265f3bf40b41b110001e99f81f23"
LEGACY_SOURCES = ("src/train_classifier.py", "src/preprocessing.py")
# Smoke helpers changed these files after the saved experiments. Accept only this
# exact source transition, and verify the recorded bytes in the pre-smoke snapshot.
PRE_SMOKE_REV = "466979b66f7bf3900e563e954f660f59ce2af189"
SMOKE_REV = "0b034b7efdb3c721694043fc4328765979585986"
SMOKE_SOURCES = (*LEGACY_SOURCES, "src/classifier.py", "src/prepare_synthetic.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def matching_hash(data, expected):
    lf = data.replace(b"\r\n", b"\n")
    return expected in {digest(value) for value in (data, lf, lf.replace(b"\n", b"\r\n"))}


def check_splits(root):
    rows = read_csv(root / "splits/cub_splits.csv")
    require(Counter(r["split"] for r in rows) == {"train": 4952, "val": 1000, "test": 5794},
            "Unexpected split sizes")
    require(len({r["image_id"] for r in rows}) == len(rows), "Duplicate IDs across splits")
    names = sorted({r["class_name"] for r in rows})
    require(len(names) == 200, "Expected 200 species")
    for row in rows:
        require(0 <= int(row["label"]) < 200 and names[int(row["label"])] == row["class_name"],
                "Invalid label/species mapping")
        prefix = "validation/" if row["split"] == "test" else "training/"
        require(row["image_id"].startswith(prefix), "Official split membership changed")
        require(row["image_id"].split("/")[1] == row["class_name"], "ID/species mismatch")
    dropped = read_csv(root / "splits/dropped_duplicates.csv")
    require(len(dropped) == len({r["image_id"] for r in dropped}) == 42, "Expected 42 dropped IDs")
    require(all(r["image_id"].startswith("training/") for r in dropped), "Dropped official test image")
    require(not {r["image_id"] for r in dropped} & {r["image_id"] for r in rows},
            "Dropped duplicate reappeared in splits")
    columns = ("image_id", "label", "class_name")
    expected = {}
    train = [r for r in rows if r["split"] == "train"]
    for split in ("train", "val", "test"):
        part = [r for r in rows if r["split"] == split]
        expected[f"{split}.csv"] = part
        if split == "val":
            require(Counter(r["label"] for r in part) == {str(i): 5 for i in range(200)},
                    "Validation is not five images per species")
    for seed in SEEDS:
        rank = f"rank_seed{seed}"
        for label in range(200):
            ranks = [int(r[rank]) for r in train if int(r["label"]) == label]
            require(sorted(ranks) == list(range(len(ranks))), "Invalid training ranks")
        require(all(int(r[rank]) == -1 for r in rows if r["split"] != "train"),
                "Held-out image has a training rank")
        for shot in (5, 10, 20):
            part = [r for r in train if int(r[rank]) < shot]
            require(Counter(r["label"] for r in part) == {str(i): shot for i in range(200)},
                    "Unbalanced k-shot subset")
            expected[f"train_{shot}shot_seed{seed}.csv"] = part
    hashes = {}
    for name, part in expected.items():
        path = root / "splits/manifests" / name
        derived = [{c: r[c] for c in columns} for r in sorted(part, key=lambda r: r["image_id"])]
        require(read_csv(path) == derived, f"Manifest differs from split/ranks: {name}")
        hashes[name] = digest(path.read_bytes().replace(b"\r\n", b"\n"))
    return {"images": dict(Counter(r["split"] for r in rows)), "classes": len(names),
            "dropped_duplicates": len(dropped), "manifest_sha256": hashes}, names


def planned_runs():
    for seed in SEEDS:
        yield f"baseline_seed{seed}", "baseline", seed, 10
        yield f"standard_aug_10shot_seed{seed}_t4", "standard_aug", seed, 10
        yield f"accepted_synthetic_10shot_seed{seed}", "real_synthetic", seed, 10
    for seed in SEEDS:
        yield f"baseline_20shot_seed{seed}", "baseline", seed, 20
        yield f"accepted_synthetic_20shot_seed{seed}", "real_synthetic", seed, 20


def check_hashes(root, run, config):
    hashes = config["sha256"]
    train = f"splits/manifests/train_{config['k_shot']}shot_seed{config['seed']}.csv"
    required = {train, "splits/manifests/val.csv", "splits/cub_splits.csv",
                "src/train_classifier.py", "src/classifier.py", "src/preprocessing.py"}
    if run != "baseline_seed0":
        required |= {"splits/manifests/test.csv", "src/synthetic_data.py"}
    if config["arm"] == "real_synthetic":
        required |= set(ALIASES) | {"src/prepare_synthetic.py"}
    require(required <= hashes.keys(), f"Missing recorded hashes: {sorted(required - hashes.keys())}")
    checked, warnings = {}, []
    for source, expected in hashes.items():
        path = root / ALIASES.get(source, source)
        require(path.is_relative_to(root) and ".." not in path.parts, f"Unsupported input path: {source}")
        require(path.is_file(), f"Missing recorded input: {source}")
        if matching_hash(path.read_bytes(), expected):
            checked[source] = {"status": "match", "path": path.relative_to(root).as_posix()}
            continue
        if run == "baseline_seed0" and source in LEGACY_SOURCES:
            revision = BASELINE_REV
        else:
            require(source in SMOKE_SOURCES, f"Hash mismatch: {source}")
            updated = subprocess.run(["git", "show", f"{SMOKE_REV}:{source}"], cwd=ROOT,
                                     capture_output=True, check=True).stdout
            require(matching_hash(path.read_bytes(), digest(updated)), f"Hash mismatch: {source}")
            revision = PRE_SMOKE_REV
        historical = subprocess.run(["git", "show", f"{revision}:{source}"], cwd=ROOT,
                                    capture_output=True, check=True).stdout
        require(matching_hash(historical, expected), f"Historical hash mismatch: {source}")
        checked[source] = {"status": "historical_match", "git_revision": revision}
        warnings.append(f"{source} matches historical commit {revision}, not current code")
    if "splits/manifests/test.csv" not in hashes:
        warnings.append("Legacy test manifest is verified through the saved cub_splits.csv hash")
    return checked, warnings


def training_command(run, config):
    command = ["python", "src/train_classifier.py", "--seed", str(config["seed"]),
               "--k-shot", str(config["k_shot"])]
    if config["augment"]:
        command.append("--augment")
    if config.get("synthetic_manifest"):
        command += ["--synthetic-manifest", f"{SYNTHETIC}/manifest.csv"]
    for key in ("max_steps", "eval_every", "log_every", "batch_size", "lr", "weight_decay", "num_workers"):
        command += [f"--{key.replace('_', '-')}", str(config[key])]
    command += ["--device", "cuda"]
    if config["amp"]:
        command.append("--amp")
    command += ["--output", f"outputs/reproduction/{run}"]
    return shlex.join(command)


def check_run(root, run, arm, seed, shot, names):
    path = root / "reports/classifier" / run
    summary = collect_run(path, project_root=root)
    config, runtime = read_json(path / "config.json"), read_json(path / "runtime.json")
    for key, value in {**COMMON, "arm": arm, "seed": seed, "k_shot": shot,
                       "class_names": names, "val_images": 1000, "augment": arm == "standard_aug"}.items():
        require(config.get(key) == value, f"Unexpected configuration {key}: {config.get(key)!r}")
    require(not config.get("full_image", False), "Full-image diagnostic used as a primary run")
    require(bool(config.get("synthetic_manifest")) == (arm == "real_synthetic"),
            "Synthetic manifest does not match arm")
    real, synthetic = 200 * shot, 1440 if arm == "real_synthetic" else 0
    augmented = real if arm == "standard_aug" else 0
    require(config["train_images"] == real + synthetic + augmented, "Wrong training-entry count")
    if run != "baseline_seed0":
        for key, value in dict(real_train_images=real, synthetic_train_images=synthetic,
                               augmented_train_images=augmented,
                               augmentation_policy="one_fixed_view_per_training_image" if augmented else None,
                               train_preprocessing="center_crop_originals_plus_fixed_crop_flip" if augmented
                               else "resize_center_crop").items():
            require(config.get(key) == value, f"Wrong {key}")
    history = read_csv(path / "history.csv")
    require([int(r["step"]) for r in history] == list(range(100, 1001, 100)), "Incomplete validation history")
    require(all(math.isfinite(float(r[k])) for r in history
                for k in ("train_loss", "val_loss", "val_accuracy", "val_macro_f1")), "Non-finite history")
    require(runtime["optimizer_steps"] == config["max_steps"], "Wrong optimizer-step budget")
    seconds = runtime["training_with_validation_seconds"]
    require(math.isfinite(seconds) and seconds > 0, "Invalid runtime")
    checked, warnings = check_hashes(root, run, config)
    if synthetic:
        pool = read_csv(root / SYNTHETIC / "manifest.csv")
        counts = Counter(r["label"] for r in pool)
        require(len(pool) == len({r["image_id"] for r in pool}) == synthetic, "Wrong synthetic pool size/IDs")
        require(all(0 <= int(r["label"]) < 200 and r["class_name"] == names[int(r["label"])]
                    for r in pool), "Invalid synthetic label mapping")
        real_ids = {r["image_id"] for r in read_csv(root / "splits/cub_splits.csv")}
        require(not real_ids & {r["image_id"] for r in pool}, "Synthetic IDs overlap real splits")
        require(config["synthetic_per_class"] == dict(counts), "Synthetic class counts changed")
        leakage = read_json(root / SYNTHETIC / "leakage_report.json")
        require(leakage["accepted_images"] == synthetic and
                leakage["accepted_per_class"] == {str(i): counts[str(i)] for i in range(200)},
                "Synthetic leakage record counts disagree")
        require(not leakage["matches"]["val"] and not leakage["matches"]["test"],
                "Recorded synthetic leakage matches require review")
    for key in ("python", "torch", "torchvision", "sklearn", "platform", "gpu"):
        require(bool(config.get(key)), f"Missing environment record: {key}")
    return {"run": run, "status": "passed", "summary": summary,
            "environment": {k: config[k] for k in ("python", "torch", "torchvision", "sklearn", "platform", "gpu")},
            "input_hashes": checked, "warnings": warnings,
            "artifact_sha256": {name: digest((path / name).read_bytes().replace(b"\r\n", b"\n")) for name in FILES},
            "training_command": training_command(run, config)}


def audit(root=ROOT):
    root = Path(root).resolve()
    report = {"scope": "Saved validation artifacts only; no retraining or checkpoint inference",
              "errors": [], "runs": [], "missing_primary_runs": [], "missing_reference_runs": []}
    try:
        report["splits"], names = check_splits(root)
    except (ValueError, KeyError, OSError) as error:
        report["errors"].append(f"splits: {error}")
        report["status"] = "failed"
        return report
    for run, arm, seed, shot in planned_runs():
        if not (root / "reports/classifier" / run).exists():
            report["missing_primary_runs" if shot == 10 else "missing_reference_runs"].append(run)
            continue
        try:
            report["runs"].append(check_run(root, run, arm, seed, shot, names))
        except (ValueError, KeyError, IndexError, TypeError, OSError, subprocess.CalledProcessError) as error:
            report["errors"].append(f"{run}: {error}")
    report["primary_complete"] = not report["missing_primary_runs"] and not report["errors"]
    report["limitations"] = [
        "Checkpoints in shared storage were not inspected; best_step is verified from history, not best.pt",
        "Image pixels and embedding/MD5 leakage detection were not rerun; only saved IDs, hashes and records were checked",
        "Loss is checked against history; predictions contain no logits to recompute cross-entropy",
        "Hardware/software differ between runs; identical seeds do not establish bitwise GPU reproducibility",
        "Historical training sources need their recorded Git snapshots for an exact source replay",
    ]
    report["status"] = "failed" if report["errors"] else "passed_with_limitations"
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write the audit as JSON")
    parser.add_argument("--require-complete", action="store_true", help="Fail if any primary run is missing")
    args = parser.parse_args(argv)
    report = audit()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{report['status']}: {len(report['runs'])} verified runs, {len(report['errors'])} errors")
    for key in ("errors", "missing_primary_runs", "missing_reference_runs"):
        for item in report[key]:
            print(f"{key}: {item}")
    return int(bool(report["errors"]) or (args.require_complete and not report.get("primary_complete", False)))


if __name__ == "__main__":
    raise SystemExit(main())
