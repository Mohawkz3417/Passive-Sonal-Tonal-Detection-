"""
test_contracts.py
====================
Frozen data contracts (18_Contracts/data_models.py + input_adapter.py) and
API-level behavior built on them (freeze spec Part 59, items 4-6, 10-21).
"""
from __future__ import annotations
import json
import os
import sys
import tempfile

import numpy as np
import pytest
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "18_Contracts"))
sys.path.append(os.path.join(ROOT, "13_Scripts"))
sys.path.append(os.path.join(ROOT, "15_Multichannel_IO"))
sys.path.append(os.path.join(ROOT, "04_Signal_Generator"))
sys.path.append(os.path.join(ROOT, "14_Array_Processing"))
sys.path.append(os.path.join(ROOT, "17_Array_Inference"))

from data_models import (ArrayFrame, MonoFrame, ValidationError, DetectionResult,  # noqa: E402
                          json_safe, TRACK_STATES, PIPELINE_STATES)
import input_adapter  # noqa: E402


# --------------------------------------------------------------------- ArrayFrame/MonoFrame shape contract
def test_array_frame_rejects_wrong_orientation():
    """[n_samples, n_channels] must be rejected, not silently transposed --
    silent transposition is exactly the kind of ambiguity Part 6 forbids."""
    with pytest.raises(ValueError):
        ArrayFrame(samples=np.zeros((1000, 8)), sample_rate_hz=16000.0, n_channels=8, source="test")


def test_mono_frame_to_array_frame_is_single_channel():
    mf = MonoFrame(samples=np.zeros(1000), sample_rate_hz=16000.0, source="mp3")
    af = mf.to_array_frame()
    assert af.n_channels == 1
    assert af.samples.shape == (1, 1000)


# --------------------------------------------------------------------- WAV/NPZ/MP3 adapters, incl. failure paths
def test_wav_channel_mismatch_raises_clean_error_not_crash():
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        sf.write(tmp.name, np.random.randn(16000, 4) * 0.1, 16000)
        with pytest.raises(ValidationError) as exc_info:
            input_adapter.wav_to_frame(tmp.name, expected_n_channels=8, expected_sample_rate_hz=16000.0)
        assert exc_info.value.code == "INCOMPATIBLE_CHANNEL_COUNT"


def test_wav_sample_rate_mismatch_raises_clean_error():
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        sf.write(tmp.name, np.random.randn(8000, 1) * 0.1, 8000)
        with pytest.raises(ValidationError) as exc_info:
            input_adapter.wav_to_frame(tmp.name, expected_sample_rate_hz=16000.0)
        assert exc_info.value.code == "UNSUPPORTED_SAMPLE_RATE"


def test_npz_missing_keys_raises_clean_error():
    with tempfile.NamedTemporaryFile(suffix=".npz") as tmp:
        np.savez(tmp.name, something_else=np.zeros(10))
        with pytest.raises(ValidationError) as exc_info:
            input_adapter.npz_to_array_frame(tmp.name)
        assert exc_info.value.code == "UNSUPPORTED_FORMAT"


def test_mp3_never_produces_a_bearing():
    """Part 41: explicitly verify no fabricated bearing appears for MP3 --
    checked at the frame level (MonoFrame can never carry a bearing) and
    at the pipeline level (see test_end_to_end.py's Ethernet/array tests
    for the multi-channel counterpart)."""
    mf = MonoFrame(samples=np.zeros(16000), sample_rate_hz=16000.0, source="mp3")
    assert not hasattr(mf, "estimated_bearing_deg")  # MonoFrame has no such field at all -- structurally impossible


# --------------------------------------------------------------------- DetectionResult contract
def test_detection_result_json_safe_with_numpy_fields():
    r = DetectionResult(detected=True, timestamp="t", source="test", n_channels=8,
                         bearing_available=True, estimated_bearing_deg=np.float64(37.0),
                         frequency_hz=np.float32(443.0), requested_snr_db=-7.0, measured_snr_db=np.float64(1.2),
                         segformer_confidence=np.float32(0.9), classification="SHIP_LIKE",
                         classification_confidence=np.float64(0.8), track_id=np.int64(3),
                         track_status="CONFIRMED", bandwidth_hz=8.0, n_harmonics=np.int32(2),
                         array_processing_ms=1.0, model_inference_ms=2.0, classification_ms=0.5, total_ms=3.5)
    payload = r.to_json()
    json.dumps(payload)  # must not raise
    assert payload["track_status"] in TRACK_STATES + ("NONE",)


def test_requested_and_measured_snr_are_independent_fields():
    r = DetectionResult(detected=False, timestamp="t", source="test", n_channels=1,
                         bearing_available=False, estimated_bearing_deg=None,
                         frequency_hz=None, requested_snr_db=-7.0, measured_snr_db=2.3,
                         segformer_confidence=None, classification=None, classification_confidence=None,
                         track_id=None, track_status="NONE", bandwidth_hz=None, n_harmonics=None,
                         array_processing_ms=0.0, model_inference_ms=0.0, classification_ms=0.0, total_ms=0.0)
    assert r.requested_snr_db != r.measured_snr_db
    d = r.to_json()
    assert "requested_snr_db" in d and "measured_snr_db" in d


def test_pipeline_states_are_from_fixed_vocabulary():
    assert PIPELINE_STATES == ("READY", "RUNNING", "WARNING", "ERROR", "UNAVAILABLE")


# --------------------------------------------------------------------- integration: array pipeline honors bearing_available
def test_single_channel_never_sets_bearing_available():
    from array_inference_pipeline import ArrayTonalDetector
    from config_loader import load_config, get as cfg_get, resolve_path
    cfg = load_config()
    det = ArrayTonalDetector(checkpoint_path=resolve_path(cfg_get(cfg, "model.segformer_weights")))
    result = det.process_array_chunk(np.random.randn(1, 16000) * 0.1, fs=16000.0)
    assert result.bearing_available is False
    assert result.estimated_bearing_deg is None
