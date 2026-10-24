"""
real_audio_mixing.py
=======================
Shared, single-source-of-truth noise injection for mixing a REAL recording
with SYNTHETIC noise at a controlled, requested SNR. Originally written
inline inside 06_Model_Evaluation/run_real_validation.py; pulled out here
so the Signal Generator app, the Detection Backend's real-dataset analysis
endpoint, and run_real_validation.py all call the exact same implementation
instead of three copies silently drifting apart.

This does power-matched additive mixing, not level-clamping or any other
approximation: the returned waveform's ACTUAL SNR should still be measured
by the caller (e.g. signal_processing.estimate_snr_db) rather than trusted
to equal `snr_db` exactly -- see the note in inject_noise_at_snr().
"""
from __future__ import annotations
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(_ROOT, "04_Signal_Generator"))
sys.path.append(os.path.join(_ROOT, "05_Dataset_Generator"))
sys.path.append(os.path.join(_ROOT, "13_Scripts"))


def inject_noise_at_snr(waveform: np.ndarray, fs: float, snr_db: float, noise_profile: str,
                         seed: int = 0) -> np.ndarray:
    """Mix `waveform` (e.g. a real recording, or a real-recording-derived
    background) with SYNTHETIC noise of the requested profile
    (shipping/environmental/biological/background) at the requested SNR,
    reusing the same generate_ambient_noise() the training generator uses.
    Returns the mixed waveform; the caller should still measure the actual
    resulting SNR from the pipeline's own estimator rather than trusting
    this function's target value exactly (the noise-floor estimator and
    this injection's power-matching can differ slightly in practice)."""
    from synthetic_signal_generator import generate_ambient_noise
    sig_power = float(np.mean(waveform.astype(np.float64) ** 2)) + 1e-20
    noise = generate_ambient_noise(len(waveform), fs, seed=seed, profile=noise_profile)
    noise_power = float(np.mean(noise ** 2)) + 1e-20
    target_noise_power = sig_power / (10 ** (snr_db / 10.0))
    noise = noise * np.sqrt(target_noise_power / noise_power)
    return waveform.astype(np.float64) + noise


def mix_real_background_per_channel(real_buffer: np.ndarray, n_channels: int, fs: float,
                                     duration_s: float, seed: int | None = None,
                                     synthetic_dither_db: float = -30.0) -> list[np.ndarray]:
    """Turn ONE mono real recording into `n_channels` independent-ish noise
    backgrounds for the array simulator.

    Honesty note: public underwater datasets (ShipsEar/DeepShip/DS3500/...)
    are mono, single-sensor recordings -- there is no real 8-hydrophone
    array capture to draw from. Using the SAME samples (even at different
    gains) on every channel is not just an approximation, it is actively
    wrong for array processing: with no delay diversity across channels,
    the beamformer sees a perfectly correlated, zero-delay signal on every
    element, which is exactly what a plane wave arriving from broadside
    (bearing 0 deg) looks like -- so a loud enough shared real-noise
    component silently drags the DOA estimate toward 0 deg regardless of
    where the actual synthetic tonal is placed. (Found by testing this
    exact function's first version, which did that.)

    Fix: `real_buffer` should be LONGER than `duration_s` (ideally several
    times longer -- most real recordings are minutes long, easily enough).
    Each channel gets an INDEPENDENTLY, randomly time-offset `duration_s`
    crop of the same recording, so channels are decorrelated by genuine
    time-shift rather than by an added synthetic dither on top of identical
    content. If the buffer isn't long enough for distinct offsets it falls
    back to wrapping (still better than zero offset), and a small
    synthetic dither (default -30 dB relative to the recording's own
    power) is layered on top for extra decorrelation on short buffers.
    This is reported to the caller/UI as "real recording background
    (approximated per-channel)" -- never claimed to be a genuine multi-
    hydrophone capture.
    """
    from synthetic_signal_generator import generate_ambient_noise
    rng = np.random.default_rng(seed)
    n_target = int(round(duration_s * fs))
    buf = real_buffer.astype(np.float64)
    if len(buf) < n_target:
        reps = int(np.ceil(n_target / max(1, len(buf))))
        buf = np.tile(buf, reps)

    clip_power = float(np.mean(buf ** 2)) + 1e-20
    dither_power = clip_power * (10 ** (synthetic_dither_db / 10.0))

    out = []
    for ch in range(n_channels):
        max_start = len(buf) - n_target
        start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        crop = buf[start:start + n_target].copy()
        gain = 10 ** (rng.normal(0.0, 0.5) / 20)
        dither_seed = None if seed is None else int(seed) + 5000 + ch
        dither = generate_ambient_noise(n_target, fs, seed=dither_seed, profile="background")
        dither_p = float(np.mean(dither ** 2)) + 1e-20
        dither = dither * np.sqrt(dither_power / dither_p)
        out.append(crop * gain + dither)
    return out
