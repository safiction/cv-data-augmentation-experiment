"""Evaluate a saved classifier; use test only after experimental choices are fixed."""
import argparse
import hashlib
from pathlib import Path

from classifier import (
    class_names_from_manifest, evaluate, load_checkpoint, resolve_device,
    save_evaluation, save_json,
)
from load_data import load_manifest, manifest_path
from preprocessing import get_loader


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.num_workers < 0:
        parser.error("Batch size must be positive; worker count must be nonnegative")
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    device = resolve_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    manifest = load_manifest(args.split)
    names = checkpoint["config"]["class_names"]
    if class_names_from_manifest(manifest) != names:
        raise ValueError("Evaluation class mapping differs from the checkpoint")
    # Validation selection must remain reproducible against the saved manifest.
    path = manifest_path(args.split)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = checkpoint["config"].get("sha256", {}).get(f"splits/manifests/{args.split}.csv")
    if expected is not None and digest != expected:
        raise ValueError("Evaluation manifest has changed since training")
    loader = get_loader(args.split, batch_size=args.batch_size, num_workers=args.num_workers)
    print(f"Evaluating {args.split}: {len(loader.dataset)} images on {device}", flush=True)
    result = evaluate(model, loader, device)
    args.output.mkdir(parents=True, exist_ok=False)
    save_evaluation(args.output, args.split, manifest, names, result)
    save_json(args.output / "evaluation_config.json", {
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_step": checkpoint["step"],
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "split": args.split, "manifest_sha256": digest, "device": str(device),
        "batch_size": args.batch_size, "num_workers": args.num_workers,
    })
    print(result[0], flush=True)


if __name__ == "__main__":
    main()
