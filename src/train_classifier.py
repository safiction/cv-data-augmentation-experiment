"""Train real-only, standard augmentation or real/synthetic CUB arms (with or without crops/flips);
evaluate validation."""
import argparse
import csv
import math
import platform
import time
from pathlib import Path

import torch
import torchvision
import sklearn
from torch import nn

from classifier import (
    BirdClassifier, class_names_from_manifest, evaluate, load_checkpoint,
    resolve_device, save_evaluation, save_json, seed_everything,
)
from load_data import (
    PROJECT_ROOT, SPLITS_CSV, SUBSET_SEEDS, canonical_sha256, load_manifest, manifest_path,
)
from preprocessing import get_loader
from synthetic_data import load_synthetic_manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SUBSET_SEEDS, default=0)
    subset = parser.add_mutually_exclusive_group()
    subset.add_argument("--k-shot", type=int, choices=(5, 10, 20), default=10)
    subset.add_argument("--all-train", dest="k_shot", action="store_const", const=None,
                        help="Use every image in the training split instead of a k-shot subset")
    parser.add_argument("--augment", action="store_true",
                        help="Append one fixed crop/flip version per training image (double dataset size); "
                             "with --synthetic-manifest this is the diffusion augmentation arm")
    parser.add_argument("--synthetic-manifest", type=Path,
                        help="Prepared synthetic CSV to append to real training images")
    parser.add_argument("--full-image", action="store_true",
                        help="Real-only control: resize the whole training image to 224 square")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:<index>")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA")
    parser.add_argument("--output", type=Path, help="New output directory; must not already exist")
    args = parser.parse_args(argv)
    if args.full_image and (args.augment or args.synthetic_manifest is not None):
        parser.error("--full-image selects the real-only control; omit --augment and --synthetic-manifest")
    for name in ("max_steps", "eval_every", "log_every", "batch_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers must be nonnegative")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("--weight-decay must be finite and nonnegative")
    return args


def train_loop(model, train_loader, val_loader, config, output, device):
    """Use a fixed optimizer-step budget and select the best validation macro F1."""
    if not len(train_loader):
        raise ValueError("Cannot train on an empty loader")
    optimizer = torch.optim.AdamW(
        model.head.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=config["amp"])
    iterator = iter(train_loader)
    running_loss, seen = 0.0, 0
    best_f1, best_step = -1.0, 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with (output / "history.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "step", "train_loss", "val_loss", "val_accuracy", "val_macro_f1"])
        writer.writeheader()
        for step in range(1, config["max_steps"] + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            model.train()  # BirdClassifier.train keeps the backbone in evaluation mode.
            images = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=config["amp"]):
                loss = criterion(model(images), labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at step {step}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)
            if step % config["log_every"] == 0:
                print(f"Step {step}/{config['max_steps']}: train loss={running_loss / seen:.4f}",
                      flush=True)
            if step % config["eval_every"] == 0 or step == config["max_steps"]:
                print(f"Validating at step {step}...", flush=True)
                metrics, _, _ = evaluate(model, val_loader, device)
                if not all(math.isfinite(metrics[key]) for key in ("loss", "accuracy", "macro_f1")):
                    raise RuntimeError("Non-finite validation metric")
                writer.writerow({
                    "step": step, "train_loss": running_loss / seen,
                    **{f"val_{key}": metrics[key] for key in ("loss", "accuracy", "macro_f1")},
                })
                stream.flush()
                print(f"Val macro F1={metrics['macro_f1']:.4f}; accuracy={metrics['accuracy']:.4f}",
                      flush=True)
                if metrics["macro_f1"] > best_f1:
                    best_f1, best_step = metrics["macro_f1"], step
                    # The full backbone is saved so evaluation never needs another download.
                    torch.save({
                        "format_version": 1, "model": model.state_dict(), "config": config,
                        "step": step, "val_macro_f1": best_f1,
                    }, output / "best.pt")
                running_loss, seen = 0.0, 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {
        "training_with_validation_seconds": time.perf_counter() - started,
        "best_step": best_step, "best_val_macro_f1": best_f1,
        "optimizer_steps": config["max_steps"],
        "scope": "Training, periodic validation and saves; excludes setup and final evaluation",
    }


def main(argv=None):
    args = parse_args(argv)
    device = resolve_device(args.device)
    if args.amp and device.type != "cuda":
        raise ValueError("--amp requires a CUDA device")
    if args.synthetic_manifest:
        arm = "diffusion_aug" if args.augment else "real_synthetic"
    else:
        arm = "standard_aug" if args.augment else "baseline_full_image" if args.full_image else "baseline"
    subset_name = "all_train" if args.k_shot is None else f"{args.k_shot}shot"
    output = args.output or PROJECT_ROOT / "outputs" / f"{arm}_{subset_name}_seed{args.seed}"
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}. Choose a new --output directory.")
    seed_everything(args.seed)
    train_manifest = load_manifest("train", args.k_shot, args.seed)
    val_manifest = load_manifest("val")
    names = class_names_from_manifest(train_manifest)
    if class_names_from_manifest(val_manifest) != names:
        raise ValueError("Train and validation class mappings differ")
    if args.k_shot is not None and not (train_manifest.groupby("label").size() == args.k_shot).all():
        raise ValueError("Train manifest does not contain k images per class")
    if set(train_manifest["image_id"]) & set(val_manifest["image_id"]):
        raise ValueError("Training and validation images overlap")
    synthetic = (load_synthetic_manifest(args.synthetic_manifest, names)
                 if args.synthetic_manifest else None)
    if synthetic is not None:
        held_out_ids = set(val_manifest["image_id"]) | set(load_manifest("test")["image_id"])
        if set(synthetic["image_id"]) & (held_out_ids | set(train_manifest["image_id"])):
            raise ValueError("Synthetic IDs overlap real data")
    train_loader = get_loader(
        "train", k_shot=args.k_shot, seed=args.seed, augment=args.augment,
        batch_size=args.batch_size, num_workers=args.num_workers,
        synthetic_manifest=args.synthetic_manifest, full_image=args.full_image)
    val_loader = get_loader("val", batch_size=args.batch_size, num_workers=args.num_workers)
    print(f"Device: {device}; train: {len(train_loader.dataset)}; val: {len(val_loader.dataset)}",
          flush=True)
    print("Loading ImageNet ResNet-18 weights...", flush=True)
    model = BirdClassifier().to(device)
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update(
        output=str(output.resolve()), device=str(device), arm=arm,
        architecture="resnet18", weights="IMAGENET1K_V1", class_names=names,
        frozen_backbone=True, frozen_batchnorm=True, selection_metric="val_macro_f1",
        train_images=len(train_loader.dataset), real_train_images=len(train_manifest),
        augmented_train_images=(len(train_manifest) + (0 if synthetic is None else len(synthetic))
                                if args.augment else 0),
        augmentation_policy="one_fixed_view_per_training_image" if args.augment else None,
        synthetic_train_images=0 if synthetic is None else len(synthetic),
        train_preprocessing=("full_image_resize_224" if args.full_image else
                             "center_crop_originals_plus_fixed_crop_flip" if args.augment else
                             "resize_center_crop"),
        val_images=len(val_manifest),
        python=platform.python_version(), platform=platform.platform(),
        torch=str(torch.__version__), torchvision=str(torchvision.__version__),
        sklearn=str(sklearn.__version__),
        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    )
    sources = [manifest_path("train", args.k_shot, args.seed), manifest_path("val"), manifest_path("test"), SPLITS_CSV,
               Path(__file__), PROJECT_ROOT / "src" / "classifier.py",
               PROJECT_ROOT / "src" / "preprocessing.py", PROJECT_ROOT / "src" / "synthetic_data.py"]
    if synthetic is not None:
        sources += [args.synthetic_manifest.resolve(), PROJECT_ROOT / "src" / "prepare_synthetic.py"]
        leakage_report = args.synthetic_manifest.resolve().with_name("leakage_report.json")
        if leakage_report.exists():
            sources.append(leakage_report)
        config["synthetic_manifest"] = str(args.synthetic_manifest.resolve())
        config["synthetic_per_class"] = {str(k): int(v) for k, v in synthetic.groupby("label").size().items()}
    config["sha256"] = {(path.relative_to(PROJECT_ROOT).as_posix() if path.is_relative_to(PROJECT_ROOT)
                        else str(path)): canonical_sha256(path)
                        for path in sources}
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "config.json", config)
    runtime = train_loop(model, train_loader, val_loader, config, output, device)
    model, _ = load_checkpoint(output / "best.pt", device)
    save_evaluation(output, "val", val_manifest, names, evaluate(model, val_loader, device))
    save_json(output / "runtime.json", runtime)
    print(f"Best step: {runtime['best_step']}. Results: {output}", flush=True)


if __name__ == "__main__":
    main()
