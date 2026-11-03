"""Import supplied test CSVs and identify their checkpoints by saved training config.

Weights stay in ignored outputs/. Tracked gzip files preserve the original CSV bytes.
Requires torch, and libarchive-c only for RAR inputs. Never executes archive code.
"""
import argparse
import gzip
import hashlib
import io
import json
import stat
import zipfile
from pathlib import Path, PurePosixPath

from check_reproducibility import ROOT
from summarize_results import read_json


def sha(data):
    return hashlib.sha256(data).hexdigest()


def checked_path(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"Unsafe archive path: {name}")
    return path


def members(path):
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            for item in archive.infolist():
                name = checked_path(item.filename)
                if stat.S_ISLNK(item.external_attr >> 16):
                    raise ValueError(f"Archive symlink: {name}")
                if not item.is_dir():
                    yield name, archive.read(item)
    else:
        import libarchive
        with libarchive.file_reader(str(path)) as archive:
            for item in archive:
                name = checked_path(item.pathname)
                if item.isdir:
                    continue
                if not item.isfile:
                    raise ValueError(f"Non-file archive entry: {name}")
                yield name, b"".join(item.get_blocks())


def write_new_or_identical(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Refusing to overwrite different data: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", type=Path, nargs="+")
    args = parser.parse_args()
    artifacts = ROOT / "reports/classifier"
    configs = {p.parent.relative_to(artifacts).as_posix(): read_json(p)
               for p in artifacts.rglob("config.json")}
    known = {}
    inventory_path = ROOT / "reports/evaluation_2026-11-03/checkpoint_inventory.json"
    for run in read_json(inventory_path)["runs"]:
        for file in run["files"]:
            path = ROOT / file["path"]
            if path.exists() and sha(path.read_bytes()) == file["sha256"]:
                known[file["sha256"]] = path
    entries, archives = {}, []
    for archive in args.archives:
        archives.append({"name": archive.name, "sha256": sha(archive.read_bytes())})
        pending = {}
        for path, data in members(archive):
            if path.name not in {"best.pt", "test_predictions.csv"}:
                raise ValueError(f"Unexpected file (not executed): {path}")
            key = path.parent.as_posix()
            pending.setdefault(key, {})[path.name] = data
            if len(pending[key]) != 2:
                continue
            files = pending.pop(key)
            checkpoint = torch.load(io.BytesIO(files["best.pt"]), map_location="cpu", weights_only=True)
            config = checkpoint["config"]
            matches = [run for run, expected in configs.items() if expected == config]
            if len(matches) != 1:
                # The supplied all-train diagnostic has no six-file validation artifact folder.
                if matches or path.parent.name != "baseline_all_train_seed0" or config["arm"] != "baseline" or config["train_images"] != 4952:
                    raise ValueError(f"Cannot uniquely identify checkpoint: {key}, {matches}")
                run = "diagnostics/baseline_all_train_seed0"
            else:
                run = matches[0]
                runtime = read_json(artifacts / run / "runtime.json")
                if checkpoint["step"] != runtime["best_step"] or abs(checkpoint["val_macro_f1"] - runtime["best_val_macro_f1"]) > 1e-12:
                    raise ValueError(f"Checkpoint is not the validation-selected one: {run}")
            weight_hash, prediction_hash = sha(files["best.pt"]), sha(files["test_predictions.csv"])
            source = {"archive": archive.name, "directory": key}
            if run in entries:
                if (entries[run]["checkpoint_sha256"], entries[run]["predictions_sha256"]) != (weight_hash, prediction_hash):
                    raise ValueError(f"Conflicting submissions for {run}")
                entries[run]["sources"].append(source)
                continue
            checkpoint_path = known.get(weight_hash, ROOT / "outputs/test_submission_2026-11-03" / run / "best.pt")
            if weight_hash not in known:
                write_new_or_identical(checkpoint_path, files["best.pt"])
                write_new_or_identical(checkpoint_path.parent / "config.json", (json.dumps(config, indent=2) + "\n").encode())
            raw = ROOT / "outputs/test_submission_2026-11-03" / run / "test_predictions.csv"
            write_new_or_identical(raw, files["test_predictions.csv"])
            compressed = ROOT / "reports/test" / run / "test_predictions.csv.gz"
            write_new_or_identical(compressed, gzip.compress(files["test_predictions.csv"], mtime=0))
            entries[run] = {"run": run, "sources": [source], "checkpoint_sha256": weight_hash,
                            "checkpoint_path": checkpoint_path.relative_to(ROOT).as_posix(),
                            "checkpoint_step": checkpoint["step"], "checkpoint_val_macro_f1": checkpoint["val_macro_f1"],
                            "predictions_sha256": prediction_hash,
                            "predictions_path": compressed.relative_to(ROOT).as_posix(),
                            "k_shot": config["k_shot"], "arm": config["arm"], "seed": config["seed"],
                            "train_images": config["train_images"],
                            "test_manifest_hash_from_checkpoint": config["sha256"].get("splits/manifests/test.csv"),
                            "split_hash_from_checkpoint": config["sha256"]["splits/cub_splits.csv"]}
        if pending:
            raise ValueError(f"Incomplete checkpoint/prediction pair: {list(pending)}")
    payload = {"archives": archives, "runs": list(entries.values()),
               "limitations": ["Original evaluation_config.json and metric JSON files were not supplied",
                               "Matching co-delivered weights does not independently prove which environment/checkpoint produced each CSV",
                               "Scores will be recomputed from hard labels; loss and confidence are unavailable"]}
    destination = ROOT / "reports/test/import_provenance.json"
    write_new_or_identical(destination, (json.dumps(payload, indent=2) + "\n").encode())
    print(f"Imported {len(entries)} unique runs; duplicate submissions matched byte-for-byte.")


if __name__ == "__main__":
    main()
