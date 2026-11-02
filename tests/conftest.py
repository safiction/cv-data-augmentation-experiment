import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from load_data import PROCESSED_DATA_DIR  # noqa: E402

requires_data = pytest.mark.skipif(
    not PROCESSED_DATA_DIR.exists(), reason="run `python src/load_data.py` to download the images"
)


@pytest.fixture(autouse=True)
def spawn_loader_workers(monkeypatch):
    """Exercise portable workers without forking a multithreaded PyTorch process."""
    import preprocessing
    from torch.utils.data import DataLoader

    # Conda's MKL and PyTorch's OpenMP must select the same worker runtime.
    monkeypatch.setenv("MKL_THREADING_LAYER", "GNU")

    def loader(*args, **kwargs):
        if kwargs.get("num_workers", 0):
            kwargs["multiprocessing_context"] = "spawn"
        return DataLoader(*args, **kwargs)

    monkeypatch.setattr(preprocessing, "DataLoader", loader)
