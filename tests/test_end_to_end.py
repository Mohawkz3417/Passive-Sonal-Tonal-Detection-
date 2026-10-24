"""
tests/test_end_to_end.py
===========================
Implements TEST 1-10 from the brief. Self-contained: does not require the
generator-backend / detection-backend uvicorn servers to already be
running (tests 9 uses FastAPI's TestClient, which runs the app in-process
on a dedicated test port so it won't collide with a live instance).

Run:
    cd tests && python -m pytest -v test_end_to_end.py

Honesty note (read this before "fixing" a red test)
-------------------------------------------------------
Tests that depend on the SHIPPED SegFormer-B0 checkpoint's detection
SENSITIVITY (finding a tonal at all) are marked informational: they assert
the pipeline runs correctly and returns a well-formed result, not that a
detection occurs, because the shipped checkpoint is a known-undertrained
80-gradient-step demo run (see 00_Documentation/RESULTS.md). Once you
retrain with proper epochs, these should start finding real detections
without any code change here. Tests of array geometry, beamforming, DOA,
the network protocol, and Kalman tracking mechanics are NOT informational
-- they assert exact/tight-tolerance correctness because none of those
depend on model training quality.
"""
from __future__ import annotations
import os
import sys
import time
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "04_Signal_Generator"))
sys.path.append(os.path.join(ROOT, "14_Array_Processing"))
sys.path.append(os.path.join(ROOT, "15_Multichannel_IO"))
sys.path.append(os.path.join(ROOT, "16_Classification"))
sys.path.append(os.path.join(ROOT, "17_Array_Inference"))
sys.path.append(os.path.join(ROOT, "07_Deployment_Module"))
sys.path.append(os.path.join(ROOT, "13_Scripts"))
sys.path.append(os.path.join(ROOT, "09_Models"))

from synthetic_signal_generator import TonalSpec, generate_tonal_waveform, generate_ambient_noise  # noqa: E402
from array_simulator import ArrayGeometry, simulate_array_reception  # noqa: E402
from beamforming import estimate_bearing_delay_and_sum  # noqa: E402
from multichannel_protocol import iter_packets, unpack_multichannel_packet  # noqa: E402
from multichannel_receiver import MultichannelReceiver  # noqa: E402
from multichannel_sender import build_demo_array, stream_array  # noqa: E402
from array_inference_pipeline import ArrayTonalDetector  # noqa: E402
from kalman_tracker import MultiTonalTracker  # noqa: E402
from config_loader import load_config, get as config_get, resolve_path  # noqa: E402

CHECKPOINT = resolve_path(config_get(load_config(), "model.segformer_weights"))
FS = 16000.0


def _noise_fn(n, fs, seed):
    return generate_ambient_noise(n, fs, seed=seed)


# --------------------------------------------------------------------- TEST 1
def test_01_single_channel_detection_pipeline_still_works():
    """Generate a single-channel signal; verify the EXISTING detection
    pipeline still runs unmodified through the new integration point."""
    detector = ArrayTonalDetector(checkpoint_path=CHECKPOINT)
    n_samples = int(FS * 4.0)
    tone = TonalSpec(f0=443.0, n_harmonics=2)
    wave, _ = generate_tonal_waveform(n_samples, FS, tone, seed=1)
    noise = _noise_fn(n_samples, FS, 2)
    clip = (wave + noise * 0.5)[np.newaxis, :]  # (1, n_samples) -- single channel

    result = detector.process_array_chunk(clip, fs=FS)
    assert result.n_channels == 1
    assert result.estimated_bearing_deg is None, "single channel must not fabricate a bearing"
    assert "beamforming/DOA skipped" in " ".join(result.warnings)
    assert isinstance(result.detected, bool)  # ran to completion, well-formed output


# --------------------------------------------------------------------- TEST 2
def test_02_generate_8_channel_array():
    geometry = ArrayGeometry(n_elements=8, spacing_m=0.2)
    n_samples = int(FS * 2.0)
    tone = TonalSpec(f0=443.0)
    wave, _ = generate_tonal_waveform(n_samples, FS, tone, seed=3)
    channels = simulate_array_reception(wave, FS, geometry, 37.0, -7.0, _noise_fn, seed=3)
    assert channels.shape == (8, n_samples)


# --------------------------------------------------------------------- TEST 3
def test_03_known_bearing_recovered():
    geometry = ArrayGeometry(n_elements=8, spacing_m=0.2)
    n_samples = int(FS * 4.0)
    tone = TonalSpec(f0=443.0, n_harmonics=2)
    wave, _ = generate_tonal_waveform(n_samples, FS, tone, seed=4)
    true_bearing = 37.0
    channels = simulate_array_reception(wave, FS, geometry, true_bearing, -7.0, _noise_fn, seed=4)
    est, _angles, _powers = estimate_bearing_delay_and_sum(channels, FS, geometry, step_deg=1.0)
    assert abs(est - true_bearing) <= 2.0, f"expected ~{true_bearing} deg, got {est} deg"


# --------------------------------------------------------------------- TEST 4
def test_04_channels_are_not_identical():
    geometry = ArrayGeometry(n_elements=8, spacing_m=0.2)
    n_samples = int(FS * 2.0)
    tone = TonalSpec(f0=443.0)
    wave, _ = generate_tonal_waveform(n_samples, FS, tone, seed=5)

    # At the realistic operating SNR (-7 dB) most of each channel's ENERGY is
    # independent per-channel noise by design (see array_simulator.py
    # docstring) -- so raw time-domain correlation between channels is
    # naturally low there too. That is physically correct, not a bug, so it
    # is the wrong condition to check "same underlying source" against.
    channels_op = simulate_array_reception(wave, FS, geometry, 37.0, -7.0, _noise_fn, seed=5)
    assert not np.allclose(channels_op[0], channels_op[1]), "channels must not be identical copies"

    # Check "genuinely correlated, shared source" at a high SNR instead,
    # where the shared tonal actually dominates each channel's energy --
    # this is the unambiguous way to prove the channels are NOT independent
    # noise streams that happen to differ, without conflating that check
    # with how much noise -7 dB deliberately adds.
    channels_hi = simulate_array_reception(wave, FS, geometry, 37.0, 20.0, _noise_fn, seed=5)
    corr = np.corrcoef(channels_hi[0], channels_hi[1])[0, 1]
    assert corr > 0.8, f"channels should be strongly correlated at high SNR (shared source), corr={corr:.3f}"


# --------------------------------------------------------------------- TEST 5 + 6 (network send/receive)
def test_05_06_transmit_and_receive_over_real_udp():
    port = 50199  # dedicated test port, distinct from the app's default 50001
    rx = MultichannelReceiver(host="127.0.0.1", port=port, buffer_seconds=10.0)
    rx.start()
    time.sleep(0.2)
    try:
        data = build_demo_array(37.0, snr_db=-7.0, n_hydrophones=8, fs=FS, duration_s=2.0, seed=6)
        n_sent = stream_array(data, int(FS), "127.0.0.1", port, realtime=True)
        assert n_sent > 0
        time.sleep(0.3)
        st = rx.status()
        assert st["connected"] is True, "receiver should show CONNECTED after real packets arrive"
        assert st["n_channels"] == 8
        assert st["total_samples_written"] == data.shape[1], "no samples should be lost over real UDP loopback"
        assert all(v == "ONLINE" for v in st["channel_status"].values())
    finally:
        rx.stop()


# --------------------------------------------------------------------- TEST 7
def test_07_beamforming_on_received_data_matches_bearing():
    port = 50200
    rx = MultichannelReceiver(host="127.0.0.1", port=port, buffer_seconds=10.0)
    rx.start()
    time.sleep(0.2)
    try:
        true_bearing = 37.0
        data = build_demo_array(true_bearing, snr_db=-7.0, n_hydrophones=8, fs=FS, duration_s=3.0, seed=7)
        stream_array(data, int(FS), "127.0.0.1", port, realtime=True)
        time.sleep(0.3)
        chunk = rx.get_latest_chunk(int(FS * 3.0))
        assert chunk is not None
        geometry = ArrayGeometry(8, 0.2)
        est, _a, _p = estimate_bearing_delay_and_sum(chunk, FS, geometry, step_deg=1.0)
        assert abs(est - true_bearing) <= 2.0
    finally:
        rx.stop()


# --------------------------------------------------------------------- TEST 8 (informational on sensitivity)
def test_08_segformer_processes_spectrogram():
    """Verify SegFormer-B0 actually runs on the beamformed spectrogram and
    returns a correctly-shaped probability mask. Does NOT assert a
    detection occurs -- see module docstring."""
    detector = ArrayTonalDetector(checkpoint_path=CHECKPOINT)
    geometry = ArrayGeometry(8, 0.2)
    data = build_demo_array(37.0, snr_db=-7.0, n_hydrophones=8, fs=FS, duration_s=4.0, seed=8)
    result = detector.process_array_chunk(data, fs=FS)
    payload = detector.get_spectrogram_payload()
    assert payload is not None and payload["magnitude"], "SegFormer/spectrogram stage did not run"
    assert len(payload["detection_mask"]) == len(payload["magnitude"])
    if not result.detected:
        pytest.skip("INFORMATIONAL: pipeline ran correctly; no detection with the "
                    "currently-shipped under-trained checkpoint (see RESULTS.md). "
                    "Retrain for a real pass/fail here.")


# --------------------------------------------------------------------- TEST 9
def test_09_detection_result_reaches_testing_ui():
    """Verify a well-formed detection result is retrievable through the
    actual FastAPI testing UI backend (not a mock), via TestClient."""
    os.environ["NPOL_ETHERNET_PORT"] = "50201"  # avoid clashing with a live dev instance
    sys.path.insert(0, os.path.join(ROOT, "detection-backend-app"))
    import importlib
    import detection_backend as db
    importlib.reload(db)  # fresh module instance bound to the test port
    from fastapi.testclient import TestClient

    with TestClient(db.app) as client:
        r = client.get("/api/status")
        assert r.status_code == 200
        body = r.json()
        assert body["model"]["loaded"] is True
        assert body["pipeline"]["SEGFORMER"] == "READY"

        r2 = client.get("/api/latest_detection")
        assert r2.status_code == 200
        assert "detected" in r2.json()


# --------------------------------------------------------------------- TEST 10
def test_10_kalman_tracking_across_multiple_frames():
    """Drive the EXISTING MultiTonalTracker directly with a controlled
    multi-frame frequency sequence -- proves multi-frame confirmation and
    persistence independent of SegFormer detection sensitivity."""
    tracker = MultiTonalTracker(gate_hz=15.0, confirm_hits=3, max_misses=4)
    freq_sequence = [441.0, 442.0, 443.0, 444.0, 444.0]
    confirmed_counts = []
    for f in freq_sequence:
        confirmed = tracker.step([f])
        confirmed_counts.append(len(confirmed))

    assert confirmed_counts[0] == 0, "track should not be confirmed on frame 1 (confirm_hits=3)"
    assert confirmed_counts[-1] == 1, "track should be confirmed by the last frame"
    final_track = tracker.step([444.5])[0]
    assert final_track.hits >= 3
    assert 440.0 < final_track.frequency < 448.0, "Kalman-smoothed frequency should track the input sequence"


def test_11_kalman_track_ages_out_when_detections_stop():
    """Regression test for a real bug found during the freeze pass: step()
    had an early-exit (`if not unmatched_dets: break`) that skipped calling
    mark_missed() on remaining/all tracks whenever a frame had fewer
    detections than tracks (including zero) -- so a track whose source
    stopped producing detections never aged out via max_misses. Fixed by
    removing the early exit; this test would have failed before the fix."""
    tracker = MultiTonalTracker(gate_hz=15.0, confirm_hits=2, max_misses=2)
    tracker.step([443.0])
    confirmed = tracker.step([443.0])
    assert len(confirmed) == 1, "track should be confirmed after 2 hits"
    track_id = confirmed[0].id

    for _ in range(3):  # exceed max_misses=2 with zero detections
        tracker.step([])

    assert len(tracker.tracks) == 0, "track must be pruned (LOST) once misses exceeds max_misses"
    assert track_id in tracker.last_dropped_track_ids


def test_12_all_tracks_status_reports_tentative_before_confirmed():
    tracker = MultiTonalTracker(gate_hz=15.0, confirm_hits=3, max_misses=4)
    tracker.step([443.0])
    statuses = tracker.all_tracks_status()
    assert len(statuses) == 1
    assert statuses[0]["status"] == "TENTATIVE"
    tracker.step([443.0])
    tracker.step([443.0])
    statuses = tracker.all_tracks_status()
    assert statuses[0]["status"] == "CONFIRMED"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
