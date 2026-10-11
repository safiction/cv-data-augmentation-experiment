"""Save a grid that shows what the classifier sees: original, eval transform, augmented samples.

A visual check that the eval center crop and the 50-100% random crops keep the bird in the frame.
One row per image: the first image of a few random classes in train_10shot_seed0.

Usage: python src/preview_transforms.py  ->  reports/figures/transforms_preview.jpg
"""
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from datasets import load_from_disk  # noqa: E402

from load_data import PROCESSED_DATA_DIR, PROJECT_ROOT, load_manifest  # noqa: E402
from preprocessing import IMAGENET_MEAN, IMAGENET_STD, eval_transform, to_rgb, train_aug_transform  # noqa: E402

OUT_PATH = PROJECT_ROOT / "reports" / "figures" / "transforms_preview.jpg"
N_IMAGES = 6
N_AUGMENTED = 4
SEED = 0


def to_image(tensor):
    """Undo ImageNet normalization: (3, H, W) tensor -> (H, W, 3) array in [0, 1]."""
    mean, std = torch.tensor(IMAGENET_MEAN)[:, None, None], torch.tensor(IMAGENET_STD)[:, None, None]
    return (tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def preview(out_path=OUT_PATH):
    manifest = load_manifest("train", 10, SEED)
    rng = np.random.default_rng(SEED)
    labels = rng.choice(manifest["label"].unique(), size=N_IMAGES, replace=False)
    rows = manifest[manifest["label"].isin(labels)].groupby("label").head(1)

    ds = load_from_disk(PROCESSED_DATA_DIR)["train"]
    row_of = {image_id: i for i, image_id in enumerate(ds["image_id"])}
    torch.manual_seed(SEED)
    evaluate, augment = eval_transform(), train_aug_transform()

    n_cols = 2 + N_AUGMENTED
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(2.2 * n_cols, 2.2 * len(rows)))
    titles = ["original", "eval (val/test)"] + [f"augmented {i + 1}" for i in range(N_AUGMENTED)]
    for r, (image_id, class_name) in enumerate(rows[["image_id", "class_name"]].to_numpy()):
        img = to_rgb(ds[row_of[image_id]]["image"])
        panels = [np.asarray(img), to_image(evaluate(img))] + [to_image(augment(img)) for _ in range(N_AUGMENTED)]
        for c, panel in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(panel)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=9)
        axes[r, 0].set_ylabel(class_name.split(".", 1)[1].replace("_", " "), fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=90, pil_kwargs={"quality": 85})
    return out_path


if __name__ == "__main__":
    print("Saved", preview())
