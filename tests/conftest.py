import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def load_module():
    """Load a module from a repo-relative path (skills/ 不是 package,只能用路徑載入)."""
    def _load(rel_path: str, name: str):
        spec = importlib.util.spec_from_file_location(name, ROOT / rel_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    return _load
