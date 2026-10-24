"""
array_simulator.py
====================
Multi-hydrophone (Uniform Linear Array) receiver simulator.

Physical model
--------------
A far-field plane wave arrives at a ULA of N sensors spaced `d` metres
apart, from bearing `theta_deg` measured from broadside (0 deg =
perpendicular to the array line, +/-90 deg = endfire).

Inter-sensor delay for sensor index n (0-based, referenced to sensor 0):

    tau_n = n * d * sin(theta) / c

c = speed of sound in water (~1500 m/s).

Each sensor receives:

    x_n(t) = s(t - tau_n) + w_n(t)

s(t) is the coherent source waveform (the tonal target -- it has to be
correlated across sensors for a beamformer to have anything to gain on).
w_n(t) is an INDEPENDENT ambient-noise realization per sensor. That
independence is what gives a beamformer array gain; if every channel got
literally the same noise, delay-and-sum would add noise coherently too and
beamforming would do nothing.

Spatial-aliasing note
----------------------
To avoid grating lobes (ambiguous / wrapped-around DOA estimates), element
spacing must satisfy

    d <= c / (2 * f_max)

for the highest frequency component you care about. With c = 1500 m/s and
a tonal band up to ~3.5 kHz (this project's existing f_max), that's
d <= ~0.214 m. The default spacing below respects that with margin.

Known limitation (by design, not a bug)
----------------------------------------
A ULA cannot distinguish bearing theta from (180 - theta) -- front/back
ambiguity. Real towed arrays resolve this with platform motion or a second
line; this simulator/estimator does not attempt to, and the DOA scan below
is restricted to +/-90 deg (one side) for that reason.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional

C_SOUND_WATER = 1500.0  # m/s


def fractional_delay(waveform: np.ndarray, fs: float, delay_s: float) -> np.ndarray:
    """
    Delay a waveform by a (possibly fractional-sample) amount using a
    frequency-domain linear-phase shift: exact for band-limited signals,
    unlike rounding to the nearest integer sample (which would make DOA
    estimates suspiciously perfect and hide real quantization error).
    """
    n = len(waveform)
    spec = np.fft.rfft(waveform)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    phase = np.exp(-2j * np.pi * freqs * delay_s)
    return np.fft.irfft(spec * phase, n=n)


@dataclass
class ArrayGeometry:
    n_elements: int = 8
    spacing_m: float = 0.2
    sound_speed_mps: float = C_SOUND_WATER

    def element_delays(self, bearing_deg: float) -> np.ndarray:
        theta = np.deg2rad(bearing_deg)
        n = np.arange(self.n_elements)
        return n * self.spacing_m * np.sin(theta) / self.sound_speed_mps

    def max_alias_free_freq_hz(self) -> float:
        return self.sound_speed_mps / (2 * self.spacing_m)

    def aperture_m(self) -> float:
        return (self.n_elements - 1) * self.spacing_m


def simulate_array_reception(source_waveform: np.ndarray, fs: float,
                              geometry: ArrayGeometry, bearing_deg: float,
                              per_channel_snr_db: float,
                              noise_fn: Callable[[int, float, Optional[int]], np.ndarray],
                              gain_jitter_db: float = 0.5,
                              seed: Optional[int] = None) -> np.ndarray:
    """
    Build (n_elements, n_samples) array data from one clean source waveform
    (the coherent tonal target) arriving at `bearing_deg`.

    noise_fn(n_samples, fs, seed) -> np.ndarray, called once per channel
    with a different seed each time (independent realizations).
    """
    rng = np.random.default_rng(seed)
    delays = geometry.element_delays(bearing_deg)
    n = len(source_waveform)
    channels = np.zeros((geometry.n_elements, n), dtype=np.float64)

    sig_power = float(np.mean(source_waveform ** 2)) + 1e-12
    snr_lin = 10 ** (per_channel_snr_db / 10)
    noise_power_target = sig_power / snr_lin

    for i in range(geometry.n_elements):
        delayed = fractional_delay(source_waveform, fs, delays[i])

        # Real hydrophones are not perfectly matched -- small sensitivity jitter.
        gain_db = rng.normal(0.0, gain_jitter_db)
        delayed = delayed * (10 ** (gain_db / 20))

        chan_seed = None if seed is None else int(seed) + 1000 + i
        noise = noise_fn(n, fs, chan_seed)
        noise = noise / (np.std(noise) + 1e-12) * np.sqrt(noise_power_target)

        channels[i] = delayed + noise

    return channels
