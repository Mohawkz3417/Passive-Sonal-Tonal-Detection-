"""
beamforming.py
================
Delay-and-Sum (broadband) and Bartlett (narrowband, covariance-based)
beamformers for a Uniform Linear Array, plus a DOA angle scan built on top.

Delay-and-Sum: steer every channel by the delay that WOULD align a source
at the scanned angle, sum coherently, measure output power across the
whole band at once. Cheap, broadband, the natural first beamformer.

Bartlett (a.k.a. conventional / classic beamformer): build the spatial
covariance matrix R = E[x x^H] from frequency-domain snapshots inside a
chosen band, then P(theta) = a(theta)^H R a(theta) for each candidate
steering vector a(theta). Narrowband by construction; gives a genuine
angular power spectrum, which is what a real sonar console plots.
"""
from __future__ import annotations
import numpy as np
from array_simulator import ArrayGeometry


def steer_and_sum(channels: np.ndarray, fs: float, geometry: ArrayGeometry,
                   scan_deg: float) -> np.ndarray:
    """
    Apply the NEGATIVE of each element's expected delay for the scanned
    angle (i.e. steer/compensate), then sum coherently. If scan_deg equals
    the true bearing, all channels realign in phase and the signal adds
    constructively; noise, independent per channel, adds incoherently.
    That difference is the array gain.
    """
    n_elem, n_samples = channels.shape
    delays = geometry.element_delays(scan_deg)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    acc = np.zeros(len(freqs), dtype=complex)
    for i in range(n_elem):
        spec = np.fft.rfft(channels[i])
        acc += spec * np.exp(2j * np.pi * freqs * delays[i])
    acc /= n_elem
    return np.fft.irfft(acc, n=n_samples)


def delay_and_sum_power(channels: np.ndarray, fs: float, geometry: ArrayGeometry,
                         scan_deg: float) -> float:
    beamformed = steer_and_sum(channels, fs, geometry, scan_deg)
    return float(np.mean(beamformed ** 2))


def estimate_bearing_delay_and_sum(channels: np.ndarray, fs: float,
                                    geometry: ArrayGeometry,
                                    scan_range_deg: tuple = (-90.0, 90.0),
                                    step_deg: float = 1.0):
    """Scan +/-90 deg only -- a ULA can't resolve front/back (see module docstring
    in array_simulator.py); returns (best_bearing_deg, angles, power_curve)."""
    angles = np.arange(scan_range_deg[0], scan_range_deg[1] + step_deg, step_deg)
    powers = np.array([delay_and_sum_power(channels, fs, geometry, a) for a in angles])
    best_idx = int(np.argmax(powers))
    return float(angles[best_idx]), angles, powers


def bartlett_spectrum(channels: np.ndarray, fs: float, geometry: ArrayGeometry,
                       angles_deg: np.ndarray, band_hz: tuple) -> np.ndarray:
    n_elem, n_samples = channels.shape
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    band_mask = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    specs = np.fft.rfft(channels, axis=1)[:, band_mask]
    band_freqs = freqs[band_mask]

    powers = np.zeros(len(angles_deg))
    for bin_idx, f in enumerate(band_freqs):
        x = specs[:, bin_idx][:, None]
        R = x @ x.conj().T
        for a_idx, ang in enumerate(angles_deg):
            delays = geometry.element_delays(ang)
            a = np.exp(-2j * np.pi * f * delays)[:, None]
            powers[a_idx] += np.real((a.conj().T @ R @ a)[0, 0])
    return powers / max(len(band_freqs), 1)


def measure_array_gain_db(signal_only_channels: np.ndarray, noise_only_channels: np.ndarray,
                           fs: float, geometry: ArrayGeometry, true_bearing_deg: float) -> float:
    """
    Real, measured (not theoretical-formula) array gain. Requires two
    SEPARATE captures at the same bearing/geometry: one with the source
    present and no added noise (signal_only_channels), one with the
    source silent (noise_only_channels). Mixing signal and noise together
    before measuring corrupts the ratio -- this is deliberately kept
    decomposed so the number reported is trustworthy.

    array_gain_db = 10*log10( (P_sig_out/P_noise_out) / (P_sig_in/P_noise_in) )
    """
    in_sig_p = float(np.mean([np.mean(signal_only_channels[i] ** 2)
                               for i in range(signal_only_channels.shape[0])]))
    in_noise_p = float(np.mean([np.mean(noise_only_channels[i] ** 2)
                                 for i in range(noise_only_channels.shape[0])]))
    in_snr = in_sig_p / (in_noise_p + 1e-30)

    bf_sig = steer_and_sum(signal_only_channels, fs, geometry, true_bearing_deg)
    bf_noise = steer_and_sum(noise_only_channels, fs, geometry, true_bearing_deg)
    out_snr = float(np.mean(bf_sig ** 2)) / (float(np.mean(bf_noise ** 2)) + 1e-30)

    return 10 * np.log10(out_snr / in_snr)
