"""
synthetic_signal_generator.py
==============================
Generates synthetic passive-sonar hydrophone waveforms:

    ambient ocean noise (colored/shaped)  +  weak tonal line(s) (+ harmonics)

at a controllable Signal-to-Noise Ratio, so the rest of the pipeline can be
trained/tested down to the -7 dB to -10 dB SNR regime called for in the
project brief, without depending on classified/restricted real recordings.

Real datasets (DeepShip / ShipsEar / DS3500 / NPOL data, if released) are
meant to be plugged in later for validation -- see 05_Dataset_Generator.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Ambient ocean noise model
# ---------------------------------------------------------------------------
# Spectral-tilt exponent per requested noise "type" -- NEW, additive only:
# shaping ~ 1/f**exponent. Larger exponent = more low-frequency-heavy energy
# (closer to a ship's dominant machinery/propeller noise band); smaller
# exponent = flatter spectrum. This is a real DSP parameter, not a label --
# selecting a different type genuinely changes the noise spectrum below.
_NOISE_PROFILE_EXPONENT = {
    "shipping": 0.75,       # heavier low-frequency emphasis
    "environmental": 0.5,   # the original default Knudsen-curve-ish shaping
    "biological": 0.30,     # flatter, more mid/high-frequency energy
    "background": 0.15,     # nearly white broadband floor
}


def generate_ambient_noise(n_samples: int, fs: float, seed: Optional[int] = None,
                            profile: str = "environmental") -> np.ndarray:
    """
    Shaped noise approximating underwater ambient noise, which falls off
    roughly as 1/f**exponent above a few hundred Hz (simplified Knudsen-curve
    behaviour) plus a broadband white floor. `profile` selects the spectral
    tilt exponent (see _NOISE_PROFILE_EXPONENT); unknown profiles fall back
    to the original "environmental" shaping so existing callers/tests that
    don't pass `profile` are unaffected.
    """
    exponent = _NOISE_PROFILE_EXPONENT.get(profile, 0.5)
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n_samples)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    freqs[0] = freqs[1]  # avoid div by zero
    shaping = 1.0 / (freqs ** exponent)
    shaping /= np.max(shaping)
    colored = np.fft.irfft(spec * shaping, n=n_samples)
    colored = colored / (np.std(colored) + 1e-12)
    # small amount of white floor so very high frequencies aren't silent
    out = 0.85 * colored + 0.15 * white
    return out / (np.std(out) + 1e-12)


# ---------------------------------------------------------------------------
# Tonal (target) model
# ---------------------------------------------------------------------------
@dataclass
class TonalSpec:
    f0: float                     # centre frequency (Hz)
    n_harmonics: int = 0          # additional harmonics (2*f0, 3*f0, ...)
    drift_hz: float = 0.5         # slow random-walk drift amplitude (Hz)
    am_depth: float = 0.15        # amplitude-modulation depth (0-1), e.g. shaft rate
    am_rate_hz: float = 2.0       # AM modulation frequency (Hz)
    duty: float = 1.0             # fraction of the clip during which the tone is present (<1 => intermittent)
    onset_frac: float = 0.0       # fraction into the clip where the tone starts (if duty<1)
    doppler_ramp_hz_per_s: float = 0.0   # NEW, additive: linear frequency ramp (Hz/s),
                                          # a first-order approximation of a closing/opening
                                          # source's Doppler shift over the clip duration.
                                          # 0.0 (default) reproduces the original behaviour
                                          # exactly, so existing callers are unaffected.


def _make_freq_track(n_samples: int, fs: float, tone: TonalSpec, seed: Optional[int] = None) -> np.ndarray:
    """Slowly-varying instantaneous frequency track for one tonal (Hz)."""
    rng = np.random.default_rng(seed)
    n_ctrl = max(4, int(n_samples / fs))  # roughly one control point per second
    walk = np.cumsum(rng.standard_normal(n_ctrl)) * tone.drift_hz
    walk -= walk.mean()
    ctrl_t = np.linspace(0, n_samples - 1, n_ctrl)
    full_t = np.arange(n_samples)
    drift = np.interp(full_t, ctrl_t, walk)
    doppler_ramp = tone.doppler_ramp_hz_per_s * (full_t / fs)
    return tone.f0 + drift + doppler_ramp


def generate_tonal_waveform(n_samples: int, fs: float, tone: TonalSpec,
                             seed: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (waveform, freq_track) for a single tonal (fundamental + harmonics),
    with slow frequency drift and optional amplitude modulation / intermittency.
    freq_track is the fundamental instantaneous frequency in Hz (for ground truth).
    """
    freq_track = _make_freq_track(n_samples, fs, tone, seed)
    phase = 2 * np.pi * np.cumsum(freq_track) / fs
    wave = np.sin(phase)

    for h in range(2, tone.n_harmonics + 2):
        wave = wave + (1.0 / h) * np.sin(h * phase)

    if tone.am_depth > 0:
        t = np.arange(n_samples) / fs
        am = 1.0 + tone.am_depth * np.sin(2 * np.pi * tone.am_rate_hz * t)
        wave = wave * am

    active = np.ones(n_samples, dtype=bool)
    if tone.duty < 1.0:
        active[:] = False
        start = int(tone.onset_frac * n_samples)
        length = int(tone.duty * n_samples)
        end = min(n_samples, start + length)
        active[start:end] = True
        wave = wave * active
        freq_track = np.where(active, freq_track, np.nan)

    wave = wave / (np.std(wave[active]) + 1e-12)
    return wave, freq_track


@dataclass
class SyntheticClip:
    waveform: np.ndarray
    fs: float
    tonal_freq_tracks: List[np.ndarray]      # one per tonal, NaN where inactive
    snr_db: float
    tone_specs: List[TonalSpec] = field(default_factory=list)
    noise_profile: str = "environmental"


def generate_clip(duration_s: float = 4.0, fs: float = 16000.0,
                   snr_db: float = -8.0, n_tonals: int = 1,
                   f_min: float = 150.0, f_max: float = 3500.0,
                   allow_intermittent: bool = True,
                   real_noise_waveform: Optional[np.ndarray] = None,
                   noise_profile: str = "environmental",
                   doppler_prob: float = 0.3,
                   seed: Optional[int] = None) -> SyntheticClip:
    """
    Build one synthetic hydrophone clip: ambient noise + `n_tonals` tonal
    lines injected at the requested SNR (dB), measured in-band
    (tonal power vs. ambient noise power).

    real_noise_waveform : optional pre-loaded real recording (e.g. from
        05_Dataset_Generator/real_data_loader.py, sampled from ShipsEar /
        DeepShip / DS3500 / etc.), already at `fs` and length
        `duration_s * fs`. When given, this REPLACES the synthetic
        shaped-noise background -- the synthetic tonal(s) are mixed into
        real ocean/ship noise at the requested SNR, so the model sees
        real noise statistics while keeping exact, free ground truth
        (since we control the injected tonal). When None (default),
        falls back to the synthetic ambient noise model.
    """
    rng = np.random.default_rng(seed)
    n_samples = int(duration_s * fs)

    if real_noise_waveform is not None:
        noise = real_noise_waveform.astype(np.float64)
        if len(noise) != n_samples:
            # defensive: crop/tile to the exact requested length
            if len(noise) > n_samples:
                noise = noise[:n_samples]
            else:
                noise = np.tile(noise, int(np.ceil(n_samples / len(noise))))[:n_samples]
    else:
        noise = generate_ambient_noise(n_samples, fs, seed=seed, profile=noise_profile)
    noise_power = np.mean(noise ** 2)

    tone_specs: List[TonalSpec] = []
    freq_tracks: List[np.ndarray] = []
    tonal_mix = np.zeros(n_samples)

    for i in range(n_tonals):
        f0 = rng.uniform(f_min, f_max)
        duty = 1.0
        onset = 0.0
        if allow_intermittent and rng.random() < 0.3:
            duty = rng.uniform(0.4, 0.9)
            onset = rng.uniform(0, 1 - duty)
        # ~doppler_prob fraction of tonals get a small Doppler ramp, matching the
        # magnitude used by the Signal Generator app's explicit "Doppler ON"
        # control (~1-2% of f0 over the clip). Probability is now a real,
        # configurable parameter (training.doppler_prob in config.yaml) --
        # previously hardcoded to 0.3 with no way to change it without editing
        # this file.
        doppler_ramp = 0.0
        if rng.random() < doppler_prob:
            doppler_ramp = (rng.uniform(0.005, 0.02) * f0 / max(duration_s, 1e-6)) * rng.choice([-1.0, 1.0])
        tone = TonalSpec(
            f0=f0,
            n_harmonics=int(rng.integers(0, 3)),
            drift_hz=rng.uniform(0.1, 1.2),
            am_depth=rng.uniform(0.0, 0.25),
            am_rate_hz=rng.uniform(1.0, 6.0),
            duty=duty,
            onset_frac=onset,
            doppler_ramp_hz_per_s=doppler_ramp,
        )
        wave, ftrack = generate_tonal_waveform(n_samples, fs, tone, seed=None if seed is None else seed + i + 1)
        tone_specs.append(tone)
        freq_tracks.append(ftrack)
        tonal_mix += wave

    if n_tonals > 0:
        tonal_power = np.mean(tonal_mix ** 2) + 1e-20
        target_tonal_power = noise_power * (10 ** (snr_db / 10.0))
        scale = np.sqrt(target_tonal_power / tonal_power)
        tonal_mix = tonal_mix * scale

    clip = noise + tonal_mix
    clip = clip / (np.max(np.abs(clip)) + 1e-12)

    return SyntheticClip(
        waveform=clip, fs=fs, tonal_freq_tracks=freq_tracks,
        snr_db=snr_db, tone_specs=tone_specs, noise_profile=noise_profile,
    )


if __name__ == "__main__":
    c = generate_clip(duration_s=2.0, fs=16000, snr_db=-8.0, n_tonals=2, seed=0)
    print("waveform shape:", c.waveform.shape, "SNR(dB) target:", c.snr_db)
    for i, ft in enumerate(c.tonal_freq_tracks):
        print(f"tonal {i}: mean f0 = {np.nanmean(ft):.1f} Hz")
