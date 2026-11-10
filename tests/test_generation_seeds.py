"""Fresh pools use new diffusion seeds; legacy and resumed pools keep their recorded seeds."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from generate import SEED_OFFSET, generate


class GenerationSeedTests(unittest.TestCase):
    def test_new_offsets_reach_diffusion_and_resuming_cannot_mix_seed_offsets(self):
        names = ["001.Test_Bird", "018.Other_Bird"]
        splits = pd.DataFrame({"class_name": names, "label": [0, 17]})
        calls = []

        def images(prompts, **kwargs):
            seeds = [generator.initial_seed() for generator in kwargs["generator"]]
            calls.append(seeds)
            return SimpleNamespace(images=[Image.new("RGB", (8, 8), (seed % 255, 10, 90)) for seed in seeds])

        pipeline = Mock(side_effect=images)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            data_dir = Path(directory)
            synthetic = data_dir / "synthetic"
            with patch("generate.DATA_DIR", data_dir), patch("generate.SYNTHETIC_DIR", synthetic), \
                    patch("generate.load_splits", return_value=splits), \
                    patch("generate.torch.cuda.is_available", return_value=False), \
                    patch("generate.device_name", return_value="test CPU"), \
                    patch("generate.load_pipeline", return_value=pipeline) as load:
                # The existing default seed formula and metadata schema remain compatible.
                generate("legacy", names, 1, "wildlife", batch_size=2, allow_cpu=True)
                generate("fresh_a", names, 1, "wildlife", batch_size=2, allow_cpu=True, seed_offset=1_000_000)
                generate("fresh_b", names, 1, "wildlife", batch_size=2, allow_cpu=True, seed_offset=2_000_000)
                self.assertEqual(calls, [[SEED_OFFSET, SEED_OFFSET + 17_000],
                                         [1_000_000, 1_017_000], [2_000_000, 2_017_000]])
                generate("legacy", names, 1, "wildlife", allow_cpu=True)
                generate("fresh_a", names, 1, "wildlife", allow_cpu=True, seed_offset=1_000_000)
                self.assertEqual(load.call_count, 3)  # Completed pools do not load the model again.
                with self.assertRaisesRegex(ValueError, "another seed offset"):
                    generate("fresh_a", names, 1, "wildlife", allow_cpu=True, seed_offset=2_000_000)
                self.assertEqual(load.call_count, 3)
                metadata = pd.read_csv(synthetic / "fresh_a" / "metadata.csv")
                self.assertEqual(metadata.seed.tolist(), [1_000_000, 1_017_000])

    def test_negative_offset_is_rejected_before_generation(self):
        with patch("generate.load_pipeline") as load:
            with self.assertRaisesRegex(ValueError, "nonnegative"):
                generate("invalid", [], 1, "wildlife", seed_offset=-1)
        load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
