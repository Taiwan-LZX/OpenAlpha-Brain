import importlib.resources
from pathlib import Path


def get_data_path(filename: str) -> Path:
    return Path(str(importlib.resources.files("openalpha_brain.data") / filename))


VEC_STORE_DIR = get_data_path("vec_store")
