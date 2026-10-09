"""Review material for a synthetic pool: contact sheets, a decision table and a leakage check.

Contact sheet, one per species: the top row shows real photos the classifier trains on (the
10-shot seed-0 subset), the rows below show the pool's synthetic images, numbered by index. Real
photos come from the train pool only; val/test stay unseen until evaluation.

Review table: one row per synthetic image with an empty `decision` column, filled in by hand with
one of DECISIONS (see docs/generation.md for the rules). Re-running the script adds rows for new
images and keeps the decisions already made.

Leakage: SD 1.5 was trained on web images and may reproduce CUB photos. Every synthetic image is
compared with val and test using the detectors of leakage.py (md5 and ResNet-18 embedding
similarity >= DUP_MIN_SIM, labels ignored); a match must be dropped from the pool.

Usage: python src/review_synthetic.py --pool pilot_photo
  (needs data/processed from load_data.py)
Output:
  data/synthetic/{pool}/sheets/{class_name}.jpg   contact sheets (shared with the pool via Drive)
  reports/generation/review_{pool}.csv            decision table (in git)
  reports/generation/leakage_{pool}.json          synthetic images matching val/test (in git)
"""
import argparse
import hashlib
import json

import pandas as pd
from datasets import load_from_disk
from PIL import Image, ImageDraw, ImageOps

from generate import SYNTHETIC_DIR, synthetic_path
from leakage import duplicate_pairs, embeddings, image_features
from load_data import PROCESSED_DATA_DIR, PROJECT_ROOT, load_manifest
from prompts import species_name

REPORTS_DIR = PROJECT_ROOT / "reports" / "generation"
DECISIONS = ("ok", "wrong_species", "artifact", "not_a_bird", "leak")
THUMB = 192
PER_ROW = 5
CAPTION = 18  # pixels of text above each thumbnail


def load_pool(pool):
    return load_manifest(f"synthetic_{pool}")


def real_examples(class_names, per_class=PER_ROW):
    """{class_name: [PIL images]} from the 10-shot seed-0 subset (train only)."""
    subset = load_manifest("train", k_shot=10, seed=0)
    subset = subset[subset["class_name"].isin(class_names)].groupby("class_name").head(per_class)
    ds = load_from_disk(PROCESSED_DATA_DIR)["train"]
    row_of = {image_id: i for i, image_id in enumerate(ds["image_id"])}
    rows = ds.select([row_of[i] for i in subset["image_id"]])
    examples = {}
    for class_name, ex in zip(subset["class_name"], rows):
        examples.setdefault(class_name, []).append(ex["image"])
    return examples


def contact_sheet(title, real, synthetic):
    """Grid image: a title, one row of real photos, then rows of numbered synthetic images."""
    tiles = [("real", img) for img in real] + [(f"#{k}", img) for k, img in synthetic]
    n_rows = 1 + -(-len(synthetic) // PER_ROW)
    cell = THUMB + CAPTION
    sheet = Image.new("RGB", (PER_ROW * THUMB, CAPTION + n_rows * cell), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((4, 2), title, fill="black")
    for n, (caption, img) in enumerate(tiles):
        row, col = (0, n) if caption == "real" else (1 + (n - len(real)) // PER_ROW, (n - len(real)) % PER_ROW)
        x, y = col * THUMB, CAPTION + row * cell
        draw.text((x + 4, y + 2), caption, fill="black")
        sheet.paste(ImageOps.pad(img.convert("RGB"), (THUMB, THUMB), color="white"), (x, y + CAPTION))
    return sheet


def write_sheets(pool, pool_df):
    out_dir = SYNTHETIC_DIR / pool / "sheets"
    out_dir.mkdir(parents=True, exist_ok=True)
    real = real_examples(pool_df["class_name"].unique())
    for class_name, group in pool_df.groupby("class_name"):
        synthetic = [(int(i.rsplit("_", 1)[1]), Image.open(synthetic_path(i))) for i in group["image_id"]]
        title = f"{class_name}  |  prompt: {group['prompt'].iloc[0]}"
        contact_sheet(title, real.get(class_name, []), synthetic).save(out_dir / f"{class_name}.jpg", quality=90)
    return out_dir


def write_review_table(pool, pool_df):
    """Adds new images to reports/generation/review_{pool}.csv, keeping existing decisions."""
    path = REPORTS_DIR / f"review_{pool}.csv"
    table = pool_df[["image_id", "class_name"]].assign(species=pool_df["class_name"].map(species_name),
                                                       decision="", note="")
    if path.exists():
        old = pd.read_csv(path, keep_default_na=False)
        table = pd.concat([old, table[~table["image_id"].isin(old["image_id"])]])
    bad = set(table["decision"]) - set(DECISIONS) - {""}
    if bad:
        print(f"Warning: unknown decisions {sorted(bad)}; allowed: {DECISIONS}")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    table.sort_values("image_id").to_csv(path, index=False)
    return table, path


def check_leakage(pool, pool_df):
    """Synthetic images that duplicate a val or test photo, by md5 or embedding similarity."""
    paths = [synthetic_path(i) for i in pool_df["image_id"]]
    syn_md5 = [hashlib.md5(p.read_bytes()).hexdigest() for p in paths]
    syn_emb = embeddings([Image.open(p) for p in paths])
    dataset = load_from_disk(PROCESSED_DATA_DIR)

    report = {"pool": pool, "images": len(paths), "pairs": {}}
    for split in ("val", "test"):
        md5, emb = image_features(dataset[split])
        ids = dataset[split]["image_id"]
        pairs = duplicate_pairs(syn_md5, syn_emb, pool_df["label"].tolist(), md5, emb, labels_b=None)
        report["pairs"][split] = [{"synthetic": pool_df["image_id"].iloc[i], "real": ids[j],
                                   "similarity": round(s, 4)} for i, j, s in pairs]
    path = REPORTS_DIR / f"leakage_{pool}.json"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    return report, path


def summarize(table):
    """Share of each decision per species, for the reviewed images only."""
    reviewed = table[table["decision"] != ""]
    if reviewed.empty:
        return None
    return pd.crosstab(reviewed["species"], reviewed["decision"], margins=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", required=True, help="e.g. pilot_photo (see splits/manifests/synthetic_*.csv)")
    parser.add_argument("--no-leakage", action="store_true", help="skip the val/test duplicate check")
    args = parser.parse_args()

    pool_df = load_pool(args.pool)
    print("Contact sheets:", write_sheets(args.pool, pool_df).relative_to(PROJECT_ROOT))
    table, path = write_review_table(args.pool, pool_df)
    print("Review table:", path.relative_to(PROJECT_ROOT))
    if (summary := summarize(table)) is not None:
        print(summary)
    if not args.no_leakage:
        report, path = check_leakage(args.pool, pool_df)
        print("Matches with val/test:", {s: len(p) for s, p in report["pairs"].items()},
              "->", path.relative_to(PROJECT_ROOT))
