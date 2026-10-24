"""
test_config.py
=================
Configuration loading and consistency (freeze spec Part 59, items 1-3).
"""
from __future__ import annotations
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "13_Scripts"))

from config_loader import load_config, get as cfg_get, resolve_path  # noqa: E402


def test_config_loads():
    cfg = load_config()
    assert isinstance(cfg, dict)
    assert "array" in cfg and "doa" in cfg and "array_ethernet" in cfg


def test_array_ethernet_port_differs_from_legacy_ethernet_port():
    cfg = load_config()
    legacy_port = cfg_get(cfg, "ethernet.port")
    array_port = cfg_get(cfg, "array_ethernet.port")
    assert legacy_port != array_port, \
        "the legacy single-channel and new multi-channel wire formats must never share a port"


def test_checkpoint_path_resolves_to_active_folder():
    cfg = load_config()
    path = cfg_get(cfg, "model.segformer_weights")
    assert "active" in path, "checkpoint path should point into 11_Weights/active/ per the checkpoint reorg"
    resolved = resolve_path(path)
    assert os.path.exists(resolved), f"configured checkpoint does not exist at {resolved}"


def test_training_snr_range_matches_freeze_spec():
    cfg = load_config()
    lo = cfg_get(cfg, "training.snr_lo_db")
    hi = cfg_get(cfg, "training.snr_hi_db")
    assert lo <= -12.0 and hi >= 2.0, f"training SNR range ({lo}, {hi}) does not cover the requested -12..+2 dB"


def test_training_frequency_range_matches_signal_generator_range():
    """These two ranges were independently maintained before -- verify they
    haven't drifted apart again."""
    cfg = load_config()
    assert cfg_get(cfg, "training.f_min_hz") == cfg_get(cfg, "signal_generator.f_min_hz")
    assert cfg_get(cfg, "training.f_max_hz") == cfg_get(cfg, "signal_generator.f_max_hz")


def test_array_geometry_single_source_of_truth():
    """array_simulator/generator_backend/detection_backend should all be
    ABLE to read the same geometry from config -- this test checks the
    config values themselves are internally consistent (n_hydrophones/
    spacing_m/sound_speed_mps all present and sane)."""
    cfg = load_config()
    n = cfg_get(cfg, "array.n_hydrophones")
    spacing = cfg_get(cfg, "array.spacing_m")
    c = cfg_get(cfg, "array.sound_speed_mps")
    assert n >= 2
    assert 0 < spacing < 10
    assert 1000 < c < 2000
