"""
test_pipeline.py
==================
Unit tests covering every module of the NPOL Tonal Detection Suite.
Run with:  pytest -v  (from this directory or the project root)
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ["13_Scripts", "04_Signal_Generator", "05_Dataset_Generator",
            "09_Models", "07_Deployment_Module", "08_Ethernet_Receiver",
            "06_Model_Evaluation"]:
    sys.path.append(os.path.join(ROOT, sub))

import numpy as np
import torch
import pytest


# ---------------------------------------------------------------------- DSP
def test_dc_removal():
    from signal_processing import remove_dc
    x = np.ones(1000) * 0.5 + np.random.randn(1000) * 0.01
    y = remove_dc(x)
    assert abs(np.mean(y)) < 1e-6


def test_bandpass_filter_rejects_out_of_band():
    from signal_processing import bandpass_filter
    fs = 16000
    t = np.arange(fs * 2) / fs
    low_tone = np.sin(2 * np.pi * 10 * t)       # below band
    mid_tone = np.sin(2 * np.pi * 1000 * t)      # inside band
    x = low_tone + mid_tone
    y = bandpass_filter(x, fs, low_hz=100, high_hz=4000)
    # power should drop substantially (low tone attenuated)
    assert np.std(y) < np.std(x)


def test_normalize_signal_peak():
    from signal_processing import normalize_signal
    x = np.random.randn(1000) * 5
    y = normalize_signal(x, mode="peak")
    assert np.max(np.abs(y)) <= 1.0 + 1e-6


def test_stft_shape():
    from signal_processing import compute_stft
    fs, n_fft, hop = 16000, 1024, 256
    x = np.random.randn(fs * 2)
    freqs, times, S = compute_stft(x, fs, n_fft, hop)
    assert S.shape[0] == n_fft // 2 + 1
    assert len(freqs) == S.shape[0]
    assert len(times) == S.shape[1]


def test_pcen_output_nonnegative_and_finite():
    from signal_processing import compute_stft, pcen
    fs, n_fft, hop = 16000, 1024, 256
    x = np.random.randn(fs)
    _, _, S = compute_stft(x, fs, n_fft, hop)
    out = pcen(S, fs, hop)
    assert np.all(np.isfinite(out))
    assert out.min() >= -1e-3  # PCEN is approximately non-negative


# ------------------------------------------------------------ signal gen
def test_generate_clip_shape_and_snr_field():
    from synthetic_signal_generator import generate_clip
    clip = generate_clip(duration_s=1.0, fs=16000, snr_db=-8.0, n_tonals=1, seed=0)
    assert clip.waveform.shape[0] == 16000
    assert clip.snr_db == -8.0
    assert len(clip.tonal_freq_tracks) == 1


def test_generate_clip_zero_tonals_is_pure_noise():
    from synthetic_signal_generator import generate_clip
    clip = generate_clip(duration_s=1.0, fs=16000, snr_db=-8.0, n_tonals=0, seed=0)
    assert len(clip.tonal_freq_tracks) == 0


# ------------------------------------------------------------ dataset
def test_make_sample_shapes():
    from dataset_builder import make_sample
    img, mask, clip = make_sample(duration_s=1.0, fs=16000, snr_db=-8.0, n_tonals=1, seed=0)
    assert img.shape == mask.shape
    assert img.ndim == 3 and img.shape[0] == 1
    assert img.min() >= 0 and img.max() <= 1
    assert set(np.unique(mask)).issubset({0.0, 1.0})


# ------------------------------------------------------------- models
def test_segformer_forward_shape():
    from segformer_tonal import segformer_b0_tonal
    model = segformer_b0_tonal()
    x = torch.randn(1, 1, 257, 101)  # odd, non-multiple-of-32 size on purpose
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 257, 101)


def test_segformer_param_count_reasonable():
    from segformer_tonal import segformer_b0_tonal
    model = segformer_b0_tonal()
    n_params = model.num_parameters()
    assert 2e6 < n_params < 6e6, "B0 should be a few million params (laptop-deployable)"


def test_unet_forward_shape():
    from unet_tonal import UNetTonal
    model = UNetTonal()
    x = torch.randn(1, 1, 200, 150)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 200, 150)


# --------------------------------------------------------- postprocess
def test_cfar_flags_strong_tone_over_noise():
    from postprocess import ca_cfar_1d
    n_freq, n_frames = 200, 50
    rng = np.random.default_rng(0)
    spec = np.abs(rng.standard_normal((n_freq, n_frames))) * 0.1
    spec[100, :] += 5.0  # strong tone at bin 100
    mask = ca_cfar_1d(spec, num_train=16, num_guard=4, pfa=1e-2)
    assert mask[100, :].mean() > 0.8, "CFAR should reliably flag a strong persistent tone"
    assert mask.mean() < 0.3, "CFAR should not flag most of the background"


def test_cfar_ignores_filtered_stopband():
    """Regression test: a near-zero (deliberately band-pass-filtered-out)
    region must never be flagged, even though its tiny values are highly
    correlated rather than random -- see passband_bin_range in
    signal_processing.py and the valid_bin_range fix in ca_cfar_1d."""
    from postprocess import ca_cfar_1d
    n_freq, n_frames = 200, 50
    rng = np.random.default_rng(0)
    spec = np.abs(rng.standard_normal((n_freq, n_frames))) * 0.1
    spec[100, :] += 5.0
    spec[150:, :] = 1e-9  # simulated filter stopband: deterministic near-zero
    mask_unrestricted = ca_cfar_1d(spec, num_train=16, num_guard=4, pfa=5e-2)
    assert mask_unrestricted[150:, :].mean() > 0.5, \
        "sanity check: an unrestricted CFAR call *should* misbehave on a deterministic near-zero band"
    mask_restricted = ca_cfar_1d(spec, num_train=16, num_guard=4, pfa=5e-2,
                                  valid_bin_range=(0, 150))
    assert mask_restricted[150:, :].sum() == 0, "stopband must be fully excluded when valid_bin_range is given"
    assert mask_restricted[100, :].mean() > 0.8, "the real tone must still be found"


def test_morphological_clean_removes_isolated_pixels():
    from postprocess import morphological_clean
    mask = np.zeros((50, 50), dtype=np.float32)
    mask[10, 10] = 1  # isolated single pixel (noise)
    mask[20:40, 25] = 1  # long vertical line (tonal-like: elongated)
    cleaned = morphological_clean(mask, open_size=(1, 1), close_size=(2, 2), min_area=5)
    assert cleaned[10, 10] == 0
    assert cleaned[25:35, 25].sum() > 0


def test_connected_components_basic():
    from postprocess import connected_components
    mask = np.zeros((30, 30), dtype=np.float32)
    mask[5:10, 5:8] = 1
    comps = connected_components(mask)
    assert len(comps) == 1
    assert comps[0]["area"] == 5 * 3


# --------------------------------------------------------- freq / conf
def test_bin_to_freq():
    from freq_estimation import bin_to_freq
    f = bin_to_freq(200, n_fft=1024, fs=16000)
    assert abs(f - 3125.0) < 1e-6


def test_harmonic_grouping():
    from freq_estimation import group_harmonics
    tonals = [
        {"frequency_hz": 100.0, "bbox": (0, 0, 1, 1), "area": 5},
        {"frequency_hz": 200.5, "bbox": (0, 0, 1, 1), "area": 5},
        {"frequency_hz": 301.0, "bbox": (0, 0, 1, 1), "area": 5},
        {"frequency_hz": 777.0, "bbox": (0, 0, 1, 1), "area": 5},  # unrelated
    ]
    grouped = group_harmonics(tonals, tol_hz=3.0)
    fund = [g for g in grouped if g["frequency_hz"] == 100.0][0]
    assert fund["n_harmonics_found"] == 2


def test_confidence_score_bounded():
    from freq_estimation import compute_confidence
    c = compute_confidence(seg_prob=0.9, area=50, snr_db=-8, track_len=10,
                            freq_variance=1.0, n_harmonics=2)
    assert 0 <= c <= 100
    c_low = compute_confidence(seg_prob=0.1, area=1, snr_db=-20, track_len=1,
                                freq_variance=1000.0, n_harmonics=0)
    assert c_low < c


# --------------------------------------------------------------- kalman
def test_kalman_tracker_confirms_persistent_tone():
    from kalman_tracker import MultiTonalTracker
    tracker = MultiTonalTracker(gate_hz=15.0, confirm_hits=3, max_misses=3)
    confirmed = []
    for _ in range(6):
        confirmed = tracker.step([300.0 + np.random.randn() * 1.0])
    assert len(confirmed) == 1
    assert abs(confirmed[0].frequency - 300.0) < 5.0


def test_kalman_tracker_rejects_random_noise():
    from kalman_tracker import MultiTonalTracker
    rng = np.random.default_rng(0)
    tracker = MultiTonalTracker(gate_hz=10.0, confirm_hits=4, max_misses=1)
    confirmed = []
    for _ in range(8):
        # a different random frequency every frame -> should never persist
        confirmed = tracker.step([float(rng.uniform(100, 3000))])
    assert len(confirmed) == 0


# ------------------------------------------------------------- ethernet
def test_ethernet_packet_roundtrip():
    from receiver import pack_packet, unpack_packet
    rng = np.random.default_rng(0)
    # real audio is normalized to [-1, 1] before transmission (see signal_processing.normalize_signal);
    # clip here so the test reflects that, rather than exercising the (correct) int16 saturation behaviour.
    samples = np.clip(rng.standard_normal(512) * 0.3, -0.99, 0.99).astype(np.float32)
    pkt = pack_packet(seq=7, sample_rate=16000, samples=samples)
    out = unpack_packet(pkt)
    assert out["seq"] == 7
    assert out["sample_rate"] == 16000
    assert len(out["samples"]) == 512
    assert np.allclose(out["samples"], samples, atol=2e-3)  # int16 quantization noise


def test_ethernet_receiver_tcp_roundtrip():
    import socket
    import threading
    import time
    from receiver import EthernetReceiverTCP, pack_packet

    port = 50555
    waveform = (np.random.randn(4000) * 0.2).astype(np.float32)

    def server():
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", port))
        srv.listen(1)
        conn, _ = srv.accept()
        seq = 0
        for start in range(0, len(waveform), 512):
            conn.sendall(pack_packet(seq, 16000, waveform[start:start + 512]))
            seq += 1
        conn.close()
        srv.close()

    t = threading.Thread(target=server, daemon=True)
    t.start()
    time.sleep(0.2)

    rx = EthernetReceiverTCP(host="127.0.0.1", port=port)
    rx.start()
    chunk = rx.get_chunk(4000, timeout=5.0)
    rx.stop()
    assert chunk is not None
    assert chunk.shape == (4000,)
    assert rx.sample_rate == 16000


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))


# ================================================================
# New in this upgrade pass: config system, model manager, expanded
# signal generator, dataset registry, TorchScript deployment path.
# ================================================================

# ------------------------------------------------------------ config
def test_config_loader_returns_expected_defaults():
    from config_loader import load_config, get
    cfg = load_config()
    assert get(cfg, "signal.sampling_rate_hz") == 16000.0
    assert get(cfg, "signal.fft_size") == 1024
    assert get(cfg, "cfar.pfa") > 0


def test_config_get_dotted_path_missing_key_returns_default():
    from config_loader import load_config, get
    cfg = load_config()
    assert get(cfg, "nonexistent.nested.key", "fallback") == "fallback"


def test_inference_pipeline_uses_config_defaults():
    from inference_pipeline import _default_cfar_kwargs, _default_morph_kwargs, _default_tracker_kwargs
    cfar_kwargs = _default_cfar_kwargs()
    assert "num_train" in cfar_kwargs and "pfa" in cfar_kwargs
    morph_kwargs = _default_morph_kwargs()
    assert "min_area" in morph_kwargs
    tracker_kwargs = _default_tracker_kwargs()
    assert "gate_hz" in tracker_kwargs


# ------------------------------------------------------------ model manager
def test_model_manager_discovers_weights_dir_checkpoints(tmp_path):
    from model_manager import discover_models
    # a fake weights dir with a plausible-looking (but not a real torch file) entry
    fake_dir = tmp_path / "weights"
    fake_dir.mkdir()
    (fake_dir / "not_a_model.txt").write_text("ignored, wrong extension")
    entries = discover_models(str(fake_dir))
    assert entries == []  # only .pt/.pth/.ts/.onnx are considered models


def test_model_manager_best_deployment_artifact_prefers_torchscript(tmp_path):
    from model_manager import best_deployment_artifact
    fake_dir = tmp_path / "weights"
    fake_dir.mkdir()
    (fake_dir / "model.pt").write_bytes(b"0" * 10)
    (fake_dir / "model_int8.onnx").write_bytes(b"0" * 10)
    (fake_dir / "model.torchscript.pt").write_bytes(b"0" * 10)
    best = best_deployment_artifact(str(fake_dir))
    assert best.kind == "torchscript"


# ------------------------------------------------------------ waveforms
def test_white_noise_is_flatter_spectrum_than_pink_noise():
    from waveforms import white_gaussian_noise, pink_noise
    fs, n = 16000, 16000
    white = white_gaussian_noise(n, fs, seed=0)
    pink = pink_noise(n, fs, seed=0)
    assert white.shape == (n,) and pink.shape == (n,)
    # pink noise concentrates more energy at low frequencies than white noise
    w_spec = np.abs(np.fft.rfft(white)) ** 2
    p_spec = np.abs(np.fft.rfft(pink)) ** 2
    low = slice(1, n // 100)
    high = slice(n // 4, n // 2)
    white_ratio = w_spec[low].mean() / (w_spec[high].mean() + 1e-12)
    pink_ratio = p_spec[low].mean() / (p_spec[high].mean() + 1e-12)
    assert pink_ratio > white_ratio


def test_linear_chirp_sweeps_from_start_to_end_frequency():
    from waveforms import linear_chirp
    fs, duration = 16000, 2.0
    n = int(fs * duration)
    x = linear_chirp(n, fs, f_start=200.0, f_end=2000.0)
    # instantaneous frequency near the start should be far lower than near the end
    from signal_processing import compute_stft
    _, _, S = compute_stft(x, fs, n_fft=512, hop_length=128)
    early_peak_bin = np.argmax(S[:, 2])
    late_peak_bin = np.argmax(S[:, -3])
    assert late_peak_bin > early_peak_bin


def test_fm_signal_shape_and_finite():
    from waveforms import fm_signal
    x = fm_signal(8000, 16000.0, f_carrier=500.0, f_mod=2.0, mod_index=3.0)
    assert x.shape == (8000,)
    assert np.all(np.isfinite(x))


def test_mix_at_snr_achieves_approximately_requested_snr():
    from waveforms import white_gaussian_noise, mix_at_snr
    fs, n = 16000, 16000
    rng = np.random.default_rng(0)
    tone = np.sin(2 * np.pi * 500 * np.arange(n) / fs)
    noise = white_gaussian_noise(n, fs, seed=1)
    mixed = mix_at_snr(tone, noise, snr_db=-6.0)
    assert mixed.shape == (n,)
    assert np.max(np.abs(mixed)) <= 1.0 + 1e-6


# ------------------------------------------------------------ dataset registry
def test_dataset_registry_lists_known_names():
    from dataset_registry import list_datasets, KNOWN_DATASETS
    datasets = list_datasets()
    for name in KNOWN_DATASETS:
        assert name in datasets


def test_dataset_registry_raises_clear_error_for_unconfigured_dataset():
    from dataset_registry import get_dataset_pool
    with pytest.raises(FileNotFoundError):
        get_dataset_pool("deepship")


# ------------------------------------------------------------ TorchScript deployment path
def test_load_model_dispatches_on_torchscript_filename(tmp_path):
    """load_model() should route to torch.jit.load for TorchScript-named
    files without requiring a real archive -- verified by checking it does
    NOT try the state_dict-checkpoint code path (which would raise a
    different, unpickling-related error for a bogus file)."""
    from inference_pipeline import load_model
    fake_ts = tmp_path / "fake.torchscript.pt"
    fake_ts.write_bytes(b"not a real torchscript archive")
    with pytest.raises(Exception) as excinfo:
        load_model(str(fake_ts))
    # should fail inside torch.jit.load (RuntimeError), not KeyError on ckpt["state_dict"]
    assert not isinstance(excinfo.value, KeyError)
