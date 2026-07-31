"""Marker panels bundled with kurtorank."""
from importlib.resources import files
from pathlib import Path


def default_markers_csv() -> Path:
    """Return the path to the default markers-v6.csv shipped with kurtorank."""
    return Path(str(files(__name__) / "data" / "markers-v6.csv"))
