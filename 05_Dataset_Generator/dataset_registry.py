"""
dataset_registry.py
=====================
Unified dataset lookup: maps a short name (deepship / shipsear / ds3500 /
npol / synthetic) to a real-data folder configured in config.yaml's
`datasets:` section, and returns a ready-to-sample `RealAudioPool` (see
real_data_loader.py) for it.

Adding a new dataset requires touching exactly one place:
    1. add a `my_new_dataset: /path/to/folder` line under `datasets:` in
       config.yaml (or leave it null and pass --data-dir explicitly)
No other code changes are needed -- train.py, dataset_builder.py, and
this file all key off the same name.
"""
from __future__ import annotations
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "13_Scripts"))

from config_loader import load_config, get as cfg_get, resolve_path  # noqa: E402

# Known dataset names this suite ships awareness of. "synthetic" is always
# available and needs no folder (it's the built-in generator). Adding a
# name here is optional -- any key under config.yaml's `datasets:` section
# works automatically via list_datasets()/get_dataset_dir(), this list is
# only used for the friendly CLI description text.
KNOWN_DATASETS = {
    "deepship": "DeepShip -- 4-class underwater acoustic ship dataset (public)",
    "shipsear": "ShipsEar -- underwater ship/ambient noise database (public)",
    "ds3500": "DS3500 -- underwater acoustic dataset",
    "ocean_ambient": "NOAA SanctSound-derived ocean ambient noise (anthropogenic/animal/sonar clips)",
    "npol": "Future/official NPOL dataset (path set once released)",
    "synthetic": "Built-in synthetic generator -- always available, no path needed",
}


def list_datasets() -> dict:
    """Return {name: configured_path_or_None} for every dataset entry in config.yaml,
    merged with any KNOWN_DATASETS name not yet present (as None)."""
    cfg = load_config()
    configured = dict(cfg_get(cfg, "datasets", {}) or {})
    for name in KNOWN_DATASETS:
        configured.setdefault(name, None)
    return configured


def get_dataset_dir(name: str) -> str | None:
    """Resolve a configured dataset's folder path, or None if unset/synthetic."""
    name = name.lower()
    cfg = load_config()
    path = cfg_get(cfg, f"datasets.{name}", None)
    if path in (None, "built-in", ""):
        return None
    return resolve_path(path)


def get_dataset_pool(name: str, target_fs: float = 16000.0, seed: int | None = None):
    """Return a RealAudioPool for the named dataset. Raises a clear error if the
    dataset isn't configured yet (points the user at config.yaml)."""
    if name.lower() == "synthetic":
        raise ValueError("'synthetic' has no RealAudioPool -- use "
                          "04_Signal_Generator/synthetic_signal_generator.py directly.")
    path = get_dataset_dir(name)
    if path is None:
        raise FileNotFoundError(
            f"Dataset '{name}' has no folder configured. Edit config.yaml -> datasets.{name}: "
            f"<path to your {name} folder>, or pass --real-data-dir explicitly to train.py.")
    from real_data_loader import RealAudioPool
    return RealAudioPool(path, target_fs=target_fs, seed=seed)


if __name__ == "__main__":
    print("Configured datasets:")
    for name, path in list_datasets().items():
        status = path if path else "(not configured -- built-in)" if name == "synthetic" else "(not configured)"
        print(f"  {name:<12s} {status}")
