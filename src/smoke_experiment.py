"""Generate images, smoke-train three paired CUB arms, and evaluate their test predictions.

Run from the project root: python src/smoke_experiment.py --device cuda --amp
Real data must already be saved by src/load_data.py. Defaults use two optimizer steps,
one generated image per species, and the complete real validation and test splits.
"""
import argparse
import json
import re
import subprocess
import sys
import uuid
from importlib.util import find_spec
from pathlib import Path

import pandas as pd
from datasets import load_from_disk
from tqdm.auto import tqdm

from load_data import PROJECT_ROOT, PROCESSED_DATA_DIR, SUBSET_SEEDS
from prompts import TEMPLATES

SRC_DIR = Path(__file__).resolve().parent
SETUPS = ("real_only", "real_generated", "real_standard_augmented")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "smoke",
                        help="New directory for all three runs and the comparison report")
    parser.add_argument("--seed", type=int, choices=SUBSET_SEEDS, default=0)
    parser.add_argument("--k-shot", type=int, choices=(5, 10, 20), default=10)
    parser.add_argument("--per-class", type=int, default=1,
                        help="Generated images per species (all 200 species; default 1)")
    parser.add_argument("--augment-per-class", type=int, default=1,
                        help="Standard augmented views per species (default 1: 200 added views)")
    parser.add_argument("--pool", help="Generation pool name; default is a new smoke_<unique ID> pool. "
                        "An existing pool resumes under generate.py's settings checks")
    parser.add_argument("--template", choices=sorted(TEMPLATES), default="wildlife")
    parser.add_argument("--generation-steps", type=int, default=25)
    parser.add_argument("--generation-batch-size", type=int, default=1)
    parser.add_argument("--allow-cpu-generation", action="store_true",
                        help="Allow slow CPU diffusion when CUDA is unavailable")
    parser.add_argument("--review", type=Path,
                        help="Optional existing review CSV for the selected generation pool")
    parser.add_argument("--max-steps", type=int, default=2,
                        help="Same optimizer-step budget for every setup (default 2)")
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="Classifier/leakage device: auto, cpu, cuda, cuda:<index>")
    parser.add_argument("--amp", action="store_true", help="Enable classifier AMP (requires CUDA)")
    args = parser.parse_args(argv)
    for name in ("per_class", "augment_per_class", "generation_steps", "generation_batch_size", "max_steps",
                 "eval_every", "batch_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers must be nonnegative")
    if args.augment_per_class > args.k_shot:
        parser.error("--augment-per-class cannot exceed --k-shot")
    if args.pool is not None and not re.fullmatch(r"[A-Za-z0-9_-]+", args.pool):
        parser.error("--pool must contain only letters, numbers, underscores or hyphens")
    return args


def run_script(script, arguments):
    """Separate processes release diffusion/classifier GPU memory between stages."""
    print(f"\nRunning {script}...", flush=True)
    subprocess.run([sys.executable, str(SRC_DIR / script), *map(str, arguments)],
                   cwd=PROJECT_ROOT, check=True)


def write_report(output):
    """Export actual misclassified test images; Arrow dataset IDs are not image paths."""
    test = load_from_disk(str(PROCESSED_DATA_DIR))["test"]
    row_of = {image_id: i for i, image_id in enumerate(test["image_id"])}
    summaries, details = [], []
    for setup in SETUPS:
        train_dir, test_dir = output / setup / "train", output / setup / "test"
        config = json.loads((train_dir / "config.json").read_text(encoding="utf-8"))
        metrics = json.loads((test_dir / "test_metrics.json").read_text(encoding="utf-8"))
        summaries.append({"setup": setup, **{key: config[key] for key in (
            "train_images", "real_train_images", "synthetic_train_images", "augmented_train_images")},
            **metrics})
        predictions = pd.read_csv(test_dir / "test_predictions.csv")
        wrong = predictions.loc[predictions.label != predictions.prediction].head(5).copy()
        image_paths, image_urls = [], []
        for rank, row in enumerate(wrong.itertuples(index=False), start=1):
            record = test[row_of[row.image_id]]
            if int(record["label"]) != int(row.label):
                raise ValueError(f"Test image label differs from the prediction row: {row.image_id}")
            path = test_dir / "wrong_predictions" / f"{rank:02d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            record["image"].convert("RGB").save(path)
            image_paths.append(path.relative_to(output).as_posix())
            image_urls.append(path.resolve().as_uri())
        wrong["image_path"], wrong["image_url"] = image_paths, image_urls
        wrong.to_csv(test_dir / "wrong_predictions.csv", index=False)
        details += [f"## {setup}", "",
                    f"[All test predictions]({setup}/test/test_predictions.csv) | "
                    f"[Per-class precision, recall and F1]({setup}/test/test_per_class.json) | "
                    f"[Wrong predictions CSV]({setup}/test/wrong_predictions.csv)", "",
                    f"Showing {len(wrong)} of {int((predictions.label != predictions.prediction).sum())} "
                    "wrong predictions (first five in test manifest order).", ""]
        print(f"\n{setup}: accuracy={metrics['accuracy']:.4f}, macro F1={metrics['macro_f1']:.4f}",
              flush=True)
        for row in wrong.itertuples(index=False):
            details.append(f"- [{row.image_id}]({row.image_path}): "
                           f"true `{row.class_name}`, predicted `{row.predicted_class_name}`")
            print(f"  {row.image_id}: true={row.class_name}, "
                  f"predicted={row.predicted_class_name}\n    {row.image_url}", flush=True)
        details.append("")

    pd.DataFrame(summaries).to_csv(output / "metrics_summary.csv", index=False)
    lines = ["# Three-setup smoke experiment", "",
             "The same real subset, seed, classifier, optimizer settings and step budget are used "
             "in every setup. Checkpoints are selected by validation macro F1 before test evaluation. "
             "Validation and test contain only real images and use their complete manifests.", "",
             "The default two-step run checks the pipeline; its scores are not a final experiment. "
             "Generated images undergo the existing integrity and val/test leakage checks. "
             "Manual species review is applied only when `--review` is provided.", "",
             "[Run settings](smoke_config.json) | [Metric summary CSV](metrics_summary.csv)", "",
             "| Setup | Training entries | Test images | Loss | Accuracy | Macro F1 |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in summaries:
        lines.append(f"| {row['setup']} | {row['train_images']} | {row['images']} | "
                     f"{row['loss']:.4f} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} |")
    (output / "summary.md").write_text("\n".join(lines + [""] + details), encoding="utf-8")
    print(f"\nReport: {(output / 'summary.md').resolve().as_uri()}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}. Choose a new --output directory.")
    if not PROCESSED_DATA_DIR.exists():
        raise FileNotFoundError("Real data is missing; run python src/load_data.py first")
    if args.review is not None:
        args.review = args.review.resolve()
        if not args.review.is_file():
            raise FileNotFoundError(f"Review table not found: {args.review}")
    missing = [name for name in ("diffusers", "transformers", "accelerate") if find_spec(name) is None]
    if missing:
        raise ModuleNotFoundError(f"Missing generation dependencies: {', '.join(missing)}. "
                                  "Install them with python -m pip install -r requirements.txt")
    from classifier import resolve_device
    import torch

    device = resolve_device(args.device)
    if args.amp and device.type != "cuda":
        raise ValueError("--amp requires a CUDA device")
    if not torch.cuda.is_available() and not args.allow_cpu_generation:
        raise ValueError("Generation requires CUDA; use --allow-cpu-generation for slow CPU diffusion")
    pool = args.pool or f"smoke_{uuid.uuid4().hex[:12]}"
    pool_dir = PROJECT_ROOT / "data" / "synthetic" / pool
    prepared_dir = output / "prepared_synthetic"
    synthetic_manifest = prepared_dir / "manifest.csv"
    output.mkdir(parents=True, exist_ok=False)
    settings = {key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()}
    settings.update(output=str(output), pool=pool, device=str(device), setups=list(SETUPS),
                    synthetic_manifest=str(synthetic_manifest),
                    checkpoint_selection="val_macro_f1", evaluation_split="test")
    (output / "smoke_config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")

    generation = ["--pool", pool, "--template", args.template, "--per-class", args.per_class,
                  "--steps", args.generation_steps, "--batch-size", args.generation_batch_size]
    if args.allow_cpu_generation:
        generation.append("--allow-cpu")
    stages = [("Generate images", "generate.py", generation)]
    preparation = ["--pool", pool_dir, "--output", prepared_dir, "--device", str(device),
                   "--batch-size", args.batch_size, "--expected-per-class", args.per_class,
                   "--per-class", args.per_class]
    if args.review is not None:
        preparation += ["--review", args.review]
    stages.append(("Check synthetic images for leakage", "prepare_synthetic.py", preparation))

    common = ["--seed", args.seed, "--k-shot", args.k_shot, "--max-steps", args.max_steps,
              "--eval-every", args.eval_every, "--log-every", 1, "--batch-size", args.batch_size,
              "--num-workers", args.num_workers, "--device", str(device)]
    if args.amp:
        common.append("--amp")
    # Training constructs the fixed crop/flip views lazily through the existing --augment loader.
    for setup in SETUPS:
        extra = (["--synthetic-manifest", synthetic_manifest] if setup == "real_generated" else
                 ["--augment", "--augment-per-class", args.augment_per_class]
                 if setup == "real_standard_augmented" else [])
        stages.append((f"Train {setup}", "train_classifier.py",
                       common + extra + ["--output", output / setup / "train"]))
    # Finish every training/validation decision before opening test predictions for any setup.
    for setup in SETUPS:
        stages.append((f"Test {setup}", "evaluate_classifier.py", [
            output / setup / "train" / "best.pt", "--split", "test", "--output", output / setup / "test",
            "--device", str(device), "--batch-size", args.batch_size, "--num-workers", args.num_workers]))
    with tqdm(total=len(stages) + 1, desc="Smoke experiment", unit="stage", dynamic_ncols=True) as progress:
        for index, (description, script, arguments) in enumerate(stages, start=1):
            # Child processes own their bars while running; keep the parent line out of their way.
            progress.clear()
            print(f"\n[{index}/{progress.total}] {description}", flush=True)
            run_script(script, arguments)
            progress.update(1)
        progress.clear()
        print(f"\n[{progress.total}/{progress.total}] Export error images and write report", flush=True)
        write_report(output)
        progress.update(1)


if __name__ == "__main__":
    main()
