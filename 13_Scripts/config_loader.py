"""
config_loader.py
==================
Single source of truth for reading `config.yaml` (project root). Every
module in the suite calls `load_config()` (cached) to get its defaults,
so parameters live in one human-editable file instead of being scattered
across scripts as literals.

Design notes:
    - Falls back to a built-in default dict if config.yaml is missing or
      PyYAML isn't installed, so the suite never hard-crashes just
      because someone deleted/renamed the config file -- it logs a
      warning and keeps going with sane defaults instead.
    - `get(cfg, "model.confidence_threshold", default)` gives safe
      dotted-path access without every caller writing nested `.get()`
      chains.
    - CLI flags in each script still win: the pattern used throughout
      this suite is `argparse.add_argument(..., default=get(cfg, "..."))`,
      so `python train.py --lr 1e-4` still overrides config.yaml for that
      one run, exactly like before.
"""
from __future__ import annotations
import copy
import logging
import os
import threading
from typing import Any

logger = logging.getLogger("npol.config")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config.yaml")

_lock = threading.Lock()
_cache: dict | None = None
_cache_path: str | None = None

# Minimal built-in fallback (mirrors config.yaml) so the suite still runs
# if the YAML file is missing/corrupt/unreadable.
_FALLBACK_CONFIG: dict = {
    "signal": {
        "sampling_rate_hz": 16000.0, "fft_size": 1024, "hop_length": 256,
        "window": "hann", "normalization_mode": "peak",
        "bandpass": {"low_hz": 50.0, "high_hz": 4000.0, "order": 4},
        "pcen": {"alpha": 0.98, "delta": 2.0, "r": 0.5, "eps": 1e-6, "t_const": 0.4},
        "chunk_duration_s": 4.0,
    },
    "array": {"n_hydrophones": 8, "spacing_m": 0.2, "sound_speed_mps": 1500.0},
    "doa": {"scan_min_deg": -90.0, "scan_max_deg": 90.0, "scan_step_deg": 2.0},
    "model": {
        "active": "segformer", "weights_dir": "11_Weights",
        "segformer_weights": "11_Weights/active/segformer_b0_tonal.pt",
        "unet_weights": "11_Weights/unet_tonal.pt",
        "onnx_fp32": "11_Weights/segformer_b0_tonal.onnx",
        "onnx_int8": "11_Weights/segformer_b0_tonal_int8.onnx",
        "torchscript": "11_Weights/segformer_b0_tonal.torchscript.pt",
        "confidence_threshold": 0.5, "device": "cpu",
    },
    "cfar": {"num_train": 16, "num_guard": 4, "pfa": 1e-2},
    "morphology": {"open_size": [2, 2], "close_size": [3, 3], "min_area": 4, "min_elongation": 1.3},
    "tracker": {"gate_hz": 15.0, "confirm_hits": 3, "max_misses": 4, "process_var": 0.5, "meas_var": 4.0},
    "detection": {"freq_tolerance_hz": 15.0, "harmonic_tolerance_hz": 8.0, "max_harmonic": 6},
    "training": {
        "model": "segformer", "epochs": 50, "steps_per_epoch": 200, "batch_size": 8,
        "lr": 3e-4, "duration_s": 4.0, "snr_lo_db": -12.0, "snr_hi_db": 2.0,
        "hard_snr_prob": 0.7, "hard_snr_range_db": [-12.0, -7.0], "num_workers": 0,
        "noise_profiles": ["shipping", "environmental", "biological", "background"], "doppler_prob": 0.3,
        "real_data_dir": None, "real_noise_prob": 0.5,
        "validation_split": 0.2, "seed": 42,
        "output_checkpoint": "11_Weights/active/segformer_b0_tonal.pt",
        "save_history": True, "save_metrics": True,
        "f_min_hz": 150.0, "f_max_hz": 3500.0,
        "freq_bands_hz": [[150, 500], [500, 1000], [1000, 1750], [1750, 2500], [2500, 3500]],
        "snr_eval_grid_db": [2.0, 0.0, -2.0, -4.0, -6.0, -8.0, -10.0, -12.0],
    },
    "signal_generator": {
        "default_type": "multi_tone", "duration_s": 4.0, "snr_db": -8.0,
        "snr_range_db": [-15.0, 20.0], "f_min_hz": 150.0, "f_max_hz": 3500.0,
    },
    "datasets": {"deepship": None, "shipsear": None, "ds3500": None, "uatd": None, "npol": None, "synthetic": "built-in"},
    "logging": {"logs_dir": "logs", "csv_log": "logs/detections_log.csv",
                "txt_log": "logs/detections_log.txt", "level": "INFO"},
    "reports": {"reports_dir": "reports", "legacy_results_dir": "06_Model_Evaluation/results"},
    "gui": {"chunk_s": 4.0, "refresh_ms": 150, "clock_refresh_ms": 1000,
            "resource_refresh_ms": 2000, "figsize": [9, 7]},
    "ethernet": {"host": "0.0.0.0", "port": 50000, "protocol": "udp"},
    "array_ethernet": {"host": "0.0.0.0", "port": 50001, "protocol": "udp"},
    "api": {"cors_origins": ["http://localhost:5173"], "host": "127.0.0.1",
            "generator_port": 8001, "detection_port": 8000},
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def project_root() -> str:
    return _PROJECT_ROOT


def resolve_path(relative_or_absolute: str) -> str:
    """Resolve a config path relative to the project root, unless already absolute."""
    if relative_or_absolute is None:
        return relative_or_absolute
    if os.path.isabs(relative_or_absolute):
        return relative_or_absolute
    return os.path.join(_PROJECT_ROOT, relative_or_absolute)


def load_config(path: str | None = None, force_reload: bool = False) -> dict:
    """Load (and cache) config.yaml. Returns a plain nested dict."""
    global _cache, _cache_path
    target = path or _DEFAULT_CONFIG_PATH
    with _lock:
        if _cache is not None and not force_reload and _cache_path == target:
            return _cache

        cfg = copy.deepcopy(_FALLBACK_CONFIG)
        try:
            import yaml  # PyYAML
            if os.path.exists(target):
                with open(target, "r") as f:
                    user_cfg = yaml.safe_load(f) or {}
                cfg = _deep_merge(cfg, user_cfg)
            else:
                logger.warning("config.yaml not found at %s -- using built-in defaults", target)
        except ImportError:
            logger.warning("PyYAML not installed (pip install pyyaml) -- using built-in defaults")
        except Exception as e:  # noqa: BLE001
            logger.warning("failed to parse %s (%s) -- using built-in defaults", target, e)

        _cache, _cache_path = cfg, target
        return cfg


def get(cfg: dict, dotted_key: str, default: Any = None) -> Any:
    """Safe dotted-path lookup, e.g. get(cfg, 'model.confidence_threshold', 0.5)."""
    node = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return default if node is None and default is not None else node


def save_config(cfg: dict, path: str | None = None) -> str:
    """Write a (possibly edited) config dict back to config.yaml."""
    import yaml
    target = path or _DEFAULT_CONFIG_PATH
    with open(target, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=None)
    global _cache, _cache_path
    with _lock:
        _cache, _cache_path = copy.deepcopy(cfg), target
    return target


if __name__ == "__main__":
    import json
    c = load_config()
    print(json.dumps(c, indent=2))
