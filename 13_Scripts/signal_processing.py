"""
signal_processing.py
=====================
Core DSP building blocks for the NPOL Tonal Detection Suite.

Implements, in order, the pipeline described in the project proposal:
    Raw signal -> DC removal -> band-pass filter -> normalization
    -> noise power estimation -> STFT -> PCEN

All functions operate on 1-D numpy float64/float32 arrays (mono hydrophone
channel) unless stated otherwise.
"""
from __future__ import annotations
import numpy as np
from scipy import signal as sps


# ---------------------------------------------------------------------------
# 1. DC Removal
# ---------------------------------------------------------------------------
def remove_dc(x: np.ndarray) -> np.ndarray:
    """Subtract the mean to remove sensor / electronics DC offset."""
    x = np.asarray(x, dtype=np.float64)
    return x - np.mean(x)


# ---------------------------------------------------------------------------
# 2. Band-pass Filtering
# ---------------------------------------------------------------------------
def bandpass_filter(x: np.ndarray, fs: float, low_hz: float = 50.0,
                     high_hz: float = 4000.0, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter (SOS form for stability)."""
    nyq = fs / 2.0
    low = max(low_hz / nyq, 1e-5)
    high = min(high_hz / nyq, 0.999)
    sos = sps.butter(order, [low, high], btype="bandpass", output="sos")
    return sps.sosfiltfilt(sos, x)


# ---------------------------------------------------------------------------
# 3. Signal Normalization
# ---------------------------------------------------------------------------
def normalize_signal(x: np.ndarray, mode: str = "peak") -> np.ndarray:
    """Scale signal to roughly [-1, 1] (peak) or zero-mean/unit-var (zscore)."""
    x = np.asarray(x, dtype=np.float64)
    if mode == "peak":
        peak = np.max(np.abs(x))
        return x / peak if peak > 1e-12 else x
    elif mode == "zscore":
        std = np.std(x)
        return (x - np.mean(x)) / std if std > 1e-12 else x - np.mean(x)
    raise ValueError(f"unknown normalization mode: {mode}")


# ---------------------------------------------------------------------------
# 4. Noise Power Estimation
# ---------------------------------------------------------------------------
def estimate_noise_power(x: np.ndarray) -> float:
    """Simple average-energy noise power estimate: P = (1/N) * sum(x^2)."""
    x = np.asarray(x, dtype=np.float64)
    return float(np.mean(x ** 2))


def estimate_noise_floor_spectrum(S_mag: np.ndarray, percentile: float = 20.0) -> np.ndarray:
    """
    Minimum-statistics style noise floor estimate per frequency bin.
    Uses a low percentile across time as a robust floor estimate
    (cheap proxy for Martin's minimum-statistics / IMCRA).

    S_mag : (n_freq, n_frames) magnitude spectrogram
    returns: (n_freq,) noise floor per bin
    """
    return np.percentile(S_mag, percentile, axis=1)


def estimate_snr_db(signal_power: float, noise_power: float) -> float:
    """SNR(dB) = 10*log10(P_signal / P_noise)."""
    noise_power = max(noise_power, 1e-20)
    return 10.0 * np.log10(max(signal_power, 1e-20) / noise_power)


# ---------------------------------------------------------------------------
# 5. STFT
# ---------------------------------------------------------------------------
def compute_stft(x: np.ndarray, fs: float, n_fft: int = 1024,
                  hop_length: int = 256, window: str = "hann"):
    """
    Returns (freqs, times, S_mag) where S_mag is the linear magnitude
    spectrogram of shape (n_freq_bins, n_frames).
    """
    win = sps.get_window(window, n_fft)
    freqs, times, Zxx = sps.stft(
        x, fs=fs, window=win, nperseg=n_fft,
        noverlap=n_fft - hop_length, boundary=None, padded=False
    )
    S_mag = np.abs(Zxx)
    return freqs, times, S_mag


# ---------------------------------------------------------------------------
# 6. PCEN (Per-Channel Energy Normalization)
# ---------------------------------------------------------------------------
def pcen(S_mag: np.ndarray, sr: float, hop_length: int,
          alpha: float = 0.98, delta: float = 2.0, r: float = 0.5,
          eps: float = 1e-6, t_const: float = 0.4) -> np.ndarray:
    """
    PCEN(t,f) = ( E / (eps + M)^alpha + delta )^r - delta^r

    where M(t,f) is an exponential moving average (EMA) of the energy E(t,f)
    across time, estimating the slowly varying background.

    S_mag       : (n_freq, n_frames) linear magnitude spectrogram
    sr, hop_length : used to derive the EMA smoothing coefficient from t_const (s)
    """
    E = S_mag ** 2  # energy
    frame_rate = sr / hop_length
    # smoothing coefficient s.t. time constant ~= t_const seconds
    b = 1.0 - np.exp(-1.0 / (t_const * frame_rate))

    M = np.empty_like(E)
    M[:, 0] = E[:, 0]
    for t in range(1, E.shape[1]):
        M[:, t] = (1 - b) * M[:, t - 1] + b * E[:, t]

    smooth = (eps + M) ** alpha
    out = (E / smooth + delta) ** r - delta ** r
    return out


def passband_bin_range(n_fft: int, fs: float, low_hz: float = 50.0, high_hz: float = 4000.0):
    """
    STFT bin range [lo, hi) that falls inside the band-pass filter's
    passband. Detection stages (CFAR etc.) should be restricted to this
    range: outside it, the signal is deliberately attenuated to near-zero
    by bandpass_filter(), which is a *deterministic*, highly-correlated
    near-silence -- not the random noise CFAR's statistics assume -- and
    will otherwise cause spurious "always-on" false detections right at
    the edge of the passband/stopband transition.
    """
    freq_res = fs / n_fft
    lo_bin = max(0, int(np.floor(low_hz / freq_res)))
    hi_bin = min(n_fft // 2 + 1, int(np.ceil(high_hz / freq_res)) + 1)
    return lo_bin, hi_bin


def preprocess_pipeline(x: np.ndarray, fs: float, low_hz: float = 50.0,
                         high_hz: float = 4000.0, n_fft: int = 1024,
                         hop_length: int = 256):
    """Full chain: DC removal -> band-pass -> normalize -> STFT -> PCEN."""
    x = remove_dc(x)
    x = bandpass_filter(x, fs, low_hz, high_hz)
    x = normalize_signal(x, mode="peak")
    noise_power = estimate_noise_power(x)
    freqs, times, S_mag = compute_stft(x, fs, n_fft, hop_length)
    S_pcen = pcen(S_mag, fs, hop_length)
    lo_bin, hi_bin = passband_bin_range(n_fft, fs, low_hz, high_hz)
    return {
        "x_clean": x,
        "noise_power": noise_power,
        "freqs": freqs,
        "times": times,
        "S_mag": S_mag,
        "S_pcen": S_pcen,
        "passband_bins": (lo_bin, hi_bin),
    }
