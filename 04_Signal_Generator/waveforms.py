"""
waveforms.py
=============
Additional standalone waveform generators that round out the Signal
Generator module: linear chirp, FM signal, white Gaussian noise, and an
explicit pink-noise generator (1/f power spectrum, the canonical
definition -- distinct from `synthetic_signal_generator.generate_ambient_noise`,
which is a specific *shaped ambient-ocean* noise model, not generic pink
noise). Pure/dual/multi-tone and harmonics are already covered by
`synthetic_signal_generator.generate_clip(n_tonals=..., ...)` and are not
duplicated here.

Kept in a separate file (rather than folded into synthetic_signal_generator.py)
so the original, already-tested generator is left untouched.
"""
from __future__ import annotations
import numpy as np


def white_gaussian_noise(n_samples: int, fs: float, seed: int | None = None) -> np.ndarray:
    """Flat power spectral density noise, normalized to unit std."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n_samples)
    return (x / (np.std(x) + 1e-12)).astype(np.float64)


def pink_noise(n_samples: int, fs: float, seed: int | None = None) -> np.ndarray:
    """
    Canonical pink noise: power spectral density falls as 1/f (amplitude
    spectrum falls as 1/sqrt(f)), generated via spectral shaping of white
    noise (Voss-McCartney-equivalent result, simpler to implement exactly
    in the frequency domain for a fixed-length clip).
    """
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n_samples)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
    shaping = 1.0 / np.sqrt(freqs)
    shaping /= np.max(shaping)
    pink = np.fft.irfft(spec * shaping, n=n_samples)
    return (pink / (np.std(pink) + 1e-12)).astype(np.float64)


def linear_chirp(n_samples: int, fs: float, f_start: float = 200.0, f_end: float = 2000.0,
                  amplitude: float = 1.0) -> np.ndarray:
    """Linearly-swept-frequency tone from f_start to f_end Hz across the clip."""
    t = np.arange(n_samples) / fs
    duration = n_samples / fs
    k = (f_end - f_start) / max(duration, 1e-9)   # chirp rate (Hz/s)
    phase = 2 * np.pi * (f_start * t + 0.5 * k * t ** 2)
    return (amplitude * np.sin(phase)).astype(np.float64)


def fm_signal(n_samples: int, fs: float, f_carrier: float = 800.0, f_mod: float = 3.0,
              mod_index: float = 4.0, amplitude: float = 1.0) -> np.ndarray:
    """
    Sinusoidally frequency-modulated tone:
        x(t) = A * sin(2*pi*f_carrier*t + mod_index * sin(2*pi*f_mod*t))
    A simple, standard narrowband-FM model -- useful for exercising the
    detector against a tonal whose instantaneous frequency wobbles
    continuously (unlike the slow random-walk drift used by TonalSpec).
    """
    t = np.arange(n_samples) / fs
    phase = 2 * np.pi * f_carrier * t + mod_index * np.sin(2 * np.pi * f_mod * t)
    return (amplitude * np.sin(phase)).astype(np.float64)


def mix_at_snr(tonal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Scale `tonal` so that tonal-power / noise-power = 10^(snr_db/10), then mix."""
    tonal_power = float(np.mean(tonal ** 2)) + 1e-20
    noise_power = float(np.mean(noise ** 2)) + 1e-20
    target_power = noise_power * (10 ** (snr_db / 10.0))
    scale = np.sqrt(target_power / tonal_power)
    mixed = noise + tonal * scale
    peak = np.max(np.abs(mixed))
    return (mixed / peak if peak > 1e-12 else mixed).astype(np.float32)


WAVEFORM_TYPES = ("pure_tone", "dual_tone", "multi_tone", "harmonics", "linear_chirp",
                   "fm_signal", "white_noise", "pink_noise", "ambient_ocean")
