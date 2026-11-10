"""Image comparisons and live training figures for notebooks/smoke_experiment.ipynb."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import display
from PIL import Image

from load_data import DATA_DIR
from preprocessing import (
    FixedAugmentedImages, IMAGENET_MEAN, IMAGENET_STD, eval_transform, select_augmentation_source, to_rgb,
)

SETUP_NAMES = {
    "real_only": "Real only",
    "real_generated": "Real + generated",
    "real_standard_augmented": "Real + standard augmentation",
}


def species_name(name):
    return name.split(".", 1)[-1].replace("_", " ")


def to_image(pixels):
    """Undo normalization for display without changing the training tensor."""
    pixels = pixels.detach().cpu()
    mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
    std = torch.tensor(IMAGENET_STD)[:, None, None]
    return (pixels * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def save_figure(fig, path):
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=140, bbox_inches="tight")
    return fig


def example_indices(real, n, seed):
    """Choose distinct species from the actual real training subset."""
    first = {}
    for index, label in enumerate(real["label"]):
        first.setdefault(int(label), index)
    labels = np.random.default_rng(seed).choice(sorted(first), min(n, len(first)), replace=False)
    return [first[int(label)] for label in labels]


def show_generated_pairs(real, metadata, names, n=6, seed=0, save_path=None):
    indices = example_indices(real, n, seed)
    fig, axes = plt.subplots(len(indices), 2, figsize=(8, 3 * len(indices)), squeeze=False)
    generated = metadata.sort_values("image_id").groupby("label").first()
    for row, index in enumerate(indices):
        record = real[index]
        label = int(record["label"])
        axes[row, 0].imshow(to_rgb(record["image"]))
        image_id = generated.loc[label, "image_id"]
        with Image.open(DATA_DIR / f"{image_id}.jpg") as image:
            axes[row, 1].imshow(image.convert("RGB"))
        axes[row, 0].set_title(f"{species_name(names[label])}\nReal training image")
        axes[row, 1].set_title(f"{species_name(names[label])}\nGenerated image")
        for ax in axes[row]:
            ax.axis("off")
    fig.suptitle("Real vs. generated: same species (before synthetic filtering)", fontsize=14)
    fig.tight_layout()
    return save_figure(fig, save_path)


def show_augmentations(real, names, n=6, seed=0, save_path=None, augment_per_class=None):
    real = select_augmentation_source(real, augment_per_class)
    indices = example_indices(real, n, seed)
    augmented = FixedAugmentedImages(real, seed)
    baseline = eval_transform()
    fig, axes = plt.subplots(len(indices), 3, figsize=(12, 3 * len(indices)), squeeze=False)
    for row, index in enumerate(indices):
        record = real[index]
        image = to_rgb(record["image"])
        panels = [image, to_image(baseline(image)), to_image(augmented[index]["pixel_values"])]
        titles = ["Original real image", "Baseline training view", "Fixed crop/flip training view"]
        for ax, panel, title in zip(axes[row], panels, titles):
            ax.imshow(panel)
            ax.set_title(f"{species_name(names[int(record['label'])])}\n{title}")
            ax.axis("off")
    fig.suptitle("Standard augmentation: the exact views used by the training loader", fontsize=14)
    fig.tight_layout()
    return save_figure(fig, save_path)


def plot_history(history, title, axes=None):
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    else:
        fig = axes[0].figure
    for ax in axes:
        ax.clear()
        ax.set_xlabel("Optimizer step")
        ax.grid(alpha=0.25)
    axes[0].set_title("Loss")
    axes[1].set_title("Validation accuracy")
    axes[2].set_title("Validation macro F1")
    axes[1].set_ylim(0, 1)
    axes[2].set_ylim(0, 1)
    if len(history):
        axes[0].plot(history.step, history.train_loss, "o-", label="Train (interval mean)")
        axes[0].plot(history.step, history.val_loss, "o-", label="Validation")
        axes[0].legend()
        axes[1].plot(history.step, history.val_accuracy, "o-", color="tab:green")
        axes[2].plot(history.step, history.val_macro_f1, "o-", color="tab:purple")
        best = history.loc[history.val_macro_f1.idxmax()]
        axes[2].scatter([best.step], [best.val_macro_f1], marker="*", s=160, color="tab:orange",
                        label=f"Best checkpoint: step {int(best.step)}", zorder=3)
        axes[2].legend(fontsize=8)
    else:
        for ax in axes:
            ax.text(0.5, 0.5, "Waiting for first validation checkpoint", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


class LiveTrainingPlot:
    """Update one notebook display after each validation without clearing progress or messages."""

    def __init__(self, title, save_path=None):
        self.title, self.save_path = title, save_path
        self.records = []
        self.fig, self.axes = plt.subplots(1, 3, figsize=(14, 3.8))
        plot_history(pd.DataFrame(), title, self.axes)
        self.handle = display(self.fig, display_id=True)
        plt.close(self.fig)  # Avoid an extra automatic inline figure at the end of the cell.

    def update(self, record):
        self.records.append(dict(record))
        plot_history(pd.DataFrame(self.records), self.title, self.axes)
        if self.handle is not None:
            self.handle.update(self.fig)

    def finish(self):
        save_figure(self.fig, self.save_path)
        plt.close(self.fig)


def show_test_comparison(summary, save_path=None):
    labels = [SETUP_NAMES[setup] for setup in summary.setup]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric, title in zip(axes, ("accuracy", "macro_f1"), ("Test accuracy", "Test macro F1")):
        bars = ax.bar(labels, summary[metric], color=["#4c78a8", "#f58518", "#54a24b"])
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", labelrotation=12)
        ax.bar_label(bars, fmt="%.3f", padding=3)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return save_figure(fig, save_path)


def show_wrong_predictions(run_dir, setup, save_path=None):
    run_dir = Path(run_dir)
    wrong = pd.read_csv(run_dir / setup / "test" / "wrong_predictions.csv")
    if wrong.empty:
        print(f"{SETUP_NAMES[setup]}: no wrong test predictions.")
        return None
    fig, axes = plt.subplots(1, len(wrong), figsize=(3.2 * len(wrong), 4), squeeze=False)
    for ax, row in zip(axes[0], wrong.itertuples(index=False)):
        with Image.open(run_dir / row.image_path) as image:
            ax.imshow(image.convert("RGB"))
        ax.set_title(f"True: {species_name(row.class_name)}\nPredicted: {species_name(row.predicted_class_name)}",
                     fontsize=9, wrap=True)
        ax.axis("off")
    fig.suptitle(f"{SETUP_NAMES[setup]}: up to five wrong test predictions")
    fig.tight_layout()
    return save_figure(fig, save_path)
