import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from load_data import PROCESSED_DATA_DIR  # noqa: E402

requires_data = pytest.mark.skipif(
    not PROCESSED_DATA_DIR.exists(), reason="run `python src/load_data.py` to download the images"
)
