"""
verify_array_doa.py
======================
Real correctness check, not a demo: generate an 8-channel array signal
with a KNOWN, PROGRAM-SET bearing and SNR (reusing this project's existing
tonal + ambient-noise generators unchanged), run the new Delay-and-Sum DOA
scan on it, and print how close the estimate came to ground truth -- plus
the actual measured array gain from beamforming.

Run:
    cd 14_Array_Processing && python verify_array_doa.py
"""
from __future__ import annotations
import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "04_Signal_Generator"))
sys.path.append(os.path.dirname(__file__))

from synthetic_signal_generator import TonalSpec, generate_tonal_waveform, generate_ambient_noise  # noqa: E402
from array_simulator import ArrayGeometry, simulate_array_reception  # noqa: E402
from beamforming import estimate_bearing_delay_and_sum, measure_array_gain_db  # noqa: E402


def noise_fn(n_samples, fs, seed):
    return generate_ambient_noise(n_samples, fs, seed=seed)


def run_trial(true_bearing_deg: float, snr_db: float, fs: float = 16000.0,
              duration_s: float = 4.0, f0: float = 443.0, n_elements: int = 8,
              seed: int = 0):
    n_samples = int(fs * duration_s)
    tone = TonalSpec(f0=f0, n_harmonics=2, drift_hz=0.3)
    source_wave, _freq_track = generate_tonal_waveform(n_samples, fs, tone, seed=seed)

    geometry = ArrayGeometry(n_elements=n_elements, spacing_m=0.2)

    channels = simulate_array_reception(
        source_wave, fs, geometry, true_bearing_deg, snr_db, noise_fn, seed=seed)

    # Separate, clean captures for a trustworthy (not corrupted) gain measurement:
    # signal-only (effectively no noise) and noise-only (silent source), both at
    # the same bearing/geometry.
    silent_source = np.zeros_like(source_wave)
    signal_only = simulate_array_reception(
        source_wave, fs, geometry, true_bearing_deg, 60.0, noise_fn, seed=seed)
    noise_only = simulate_array_reception(
        silent_source, fs, geometry, true_bearing_deg, 0.0, noise_fn, seed=seed + 500)

    est_bearing, angles, powers = estimate_bearing_delay_and_sum(channels, fs, geometry)
    gain_db = measure_array_gain_db(signal_only, noise_only, fs, geometry, true_bearing_deg)

    return {
        "true_bearing_deg": true_bearing_deg,
        "snr_db": snr_db,
        "estimated_bearing_deg": est_bearing,
        "bearing_error_deg": abs(est_bearing - true_bearing_deg),
        "measured_array_gain_db": gain_db,
        "aliasing_limit_hz": geometry.max_alias_free_freq_hz(),
    }


if __name__ == "__main__":
    true_bearing = 37.0
    print(f"Array: 8 elements, 0.2 m spacing -> alias-free up to "
          f"{ArrayGeometry(8, 0.2).max_alias_free_freq_hz():.0f} Hz")
    print(f"True bearing: {true_bearing} deg,  source: 443 Hz ship-like tonal + 2 harmonics\n")
    print(f"{'SNR (dB)':>10} | {'Est. bearing':>13} | {'Error (deg)':>12} | {'Array gain (dB)':>16}")
    print("-" * 62)
    for snr_db in [0, -3, -5, -7, -10]:
        r = run_trial(true_bearing, snr_db, seed=42)
        print(f"{r['snr_db']:>10.1f} | {r['estimated_bearing_deg']:>13.1f} | "
              f"{r['bearing_error_deg']:>12.2f} | {r['measured_array_gain_db']:>16.2f}")
