"""Check that a synthetic pool unpacked from the shared archive matches its manifest in git.

Every image of splits/manifests/synthetic_{pool}.csv must exist under data/synthetic/{pool}/ with the
md5 recorded at generation time.

Usage: python src/check_synthetic.py --pool sd15_photo
"""
import argparse
import hashlib

from generate import synthetic_path
from load_data import load_manifest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", required=True)
    args = parser.parse_args()

    manifest = load_manifest(f"synthetic_{args.pool}")
    missing = [i for i in manifest["image_id"] if not synthetic_path(i).exists()]
    changed = [i for i, m in zip(manifest["image_id"], manifest["md5"])
               if i not in missing and hashlib.md5(synthetic_path(i).read_bytes()).hexdigest() != m]
    print(f"{args.pool}: {len(manifest)} images, {len(missing)} missing, {len(changed)} with another md5")
    for i in (missing + changed)[:10]:
        print("  ", i)
    if missing or changed:
        raise SystemExit(1)
