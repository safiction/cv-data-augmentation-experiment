"""Notebook cells compile, and visualizations preserve species/image/checkpoint alignment."""
import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import Dataset, Image as ImageFeature
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from preprocessing import FixedAugmentedImages
from smoke_visualization import LiveTrainingPlot, show_augmentations, show_generated_pairs, to_image


class SmokeNotebookTests(unittest.TestCase):
    def test_notebook_code_cells_compile(self):
        notebook = json.loads((ROOT / "notebooks" / "smoke_experiment.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"notebook_cell_{index}", "exec")

    def test_setup_recovers_stale_kernel_modules_and_refreshes_dependent_loaders(self):
        # Use a separate process: reloading dataset classes must not affect other tests' imports.
        script = textwrap.dedent('''
            import json
            import sys
            from pathlib import Path
            import matplotlib
            matplotlib.use("Agg")
            notebook = json.loads(Path("notebooks/smoke_experiment.ipynb").read_text(encoding="utf-8"))
            setup = "".join(next(cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code"))
            namespace = {}
            exec(compile(setup, "notebook_setup", "exec"), namespace)

            import preprocessing
            import synthetic_data
            import smoke_visualization
            import train_classifier
            import evaluate_classifier
            import prepare_synthetic
            old_source = preprocessing.select_augmentation_source
            old_loader = preprocessing.get_loader
            old_dataset_class = preprocessing.FixedAugmentedImages
            del preprocessing.select_augmentation_source
            assert not hasattr(preprocessing, "select_augmentation_source")

            # This repeats the user's first cell in an already-running kernel.
            exec(compile(setup, "notebook_setup_rerun", "exec"), namespace)
            source = namespace["select_augmentation_source"]
            assert callable(source)
            assert source is preprocessing.select_augmentation_source
            assert source is not old_source
            assert smoke_visualization.select_augmentation_source is source
            assert namespace["show_augmentations"] is smoke_visualization.show_augmentations
            assert preprocessing.get_loader is not old_loader
            assert train_classifier.get_loader is preprocessing.get_loader
            assert evaluate_classifier.get_loader is preprocessing.get_loader
            assert prepare_synthetic.get_loader is preprocessing.get_loader
            assert preprocessing.SyntheticImages is synthetic_data.SyntheticImages
            assert preprocessing.FixedAugmentedImages is not old_dataset_class
            assert smoke_visualization.FixedAugmentedImages is preprocessing.FixedAugmentedImages

            from datasets import Dataset
            real = Dataset.from_dict({"image_id": ["train/b", "train/a"], "label": [0, 0]})
            assert source(real, 1)["image_id"] == ["train/a"]
            assert "generate" not in sys.modules  # Setup does not load or run diffusion.
        ''')
        result = subprocess.run([sys.executable, "-X", "utf8", "-c", script], cwd=ROOT,
                                capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_generated_gallery_pairs_species_by_label_instead_of_row_position(self):
        names = ["001.Red_Bird", "002.Green_Bird", "003.Blue_Bird"]
        colors = [(220, 30, 10), (20, 210, 15), (10, 25, 225)]
        real = Dataset.from_dict({"label": [0, 1, 2], "image_id": ["train/r", "train/g", "train/b"],
                                  "image": [Image.new("RGB", (20, 20), color) for color in colors]
                                  }).cast_column("image", ImageFeature())
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            data_dir = Path(directory)
            records = []
            for label in (2, 0, 1):
                image_id = f"synthetic/{label}"
                path = data_dir / f"{image_id}.jpg"
                path.parent.mkdir(exist_ok=True)
                Image.new("RGB", (20, 20), colors[label]).save(path, quality=100, subsampling=0)
                records.append({"image_id": image_id, "label": label})
            with patch("smoke_visualization.DATA_DIR", data_dir):
                fig = show_generated_pairs(real, pd.DataFrame(records), names, n=3)
            self.addCleanup(plt.close, fig)
            for left, right in zip(fig.axes[::2], fig.axes[1::2]):
                self.assertEqual(left.get_title().split("\n")[0], right.get_title().split("\n")[0])
                np.testing.assert_allclose(left.images[0].get_array(), right.images[0].get_array(), atol=2)

    def test_augmentation_gallery_shows_the_exact_fixed_training_view_without_changing_rng(self):
        image = Image.new("RGB", (448, 224), "blue")
        image.paste("red", (0, 0, 100, 224))
        real = Dataset.from_dict({"label": [0], "image_id": ["train/red"], "image": [image]
                                  }).cast_column("image", ImageFeature())
        fixed = FixedAugmentedImages(real, seed=2)[0]["pixel_values"]
        import torch
        before = torch.get_rng_state().clone()
        fig = show_augmentations(real, ["001.Red_Bird"], n=1, seed=2, augment_per_class=1)
        self.addCleanup(plt.close, fig)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        np.testing.assert_array_equal(fig.axes[2].images[0].get_array(), to_image(fixed))
        self.assertEqual(len(fig.axes), 3)

    def test_live_plot_updates_one_display_and_marks_best_validation_checkpoint(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "training.png"
            handle = Mock()
            with patch("smoke_visualization.display", return_value=handle) as display:
                plot = LiveTrainingPlot("Real only", save_path=path)
                for step, f1 in ((5, 0.6), (10, 0.4)):
                    plot.update({"step": step, "train_loss": 1.0, "val_loss": 1.5,
                                 "val_accuracy": 0.5, "val_macro_f1": f1})
                plot.finish()
            self.assertEqual(display.call_count, 1)
            self.assertEqual(handle.update.call_count, 2)
            self.assertTrue(path.is_file())
            np.testing.assert_allclose(plot.axes[2].collections[0].get_offsets(), [[5, 0.6]])
            self.assertEqual(plot.axes[0].lines[0].get_xdata().tolist(), [5, 10])


if __name__ == "__main__":
    unittest.main()
