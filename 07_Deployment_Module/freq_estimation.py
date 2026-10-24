"""
freq_estimation.py
====================
Module 8 (proposal): convert pixel/bin coordinates of each detected blob
into physical units (Hz, seconds), plus harmonic grouping and a combined
confidence score (Module: "Confidence Score" enhancement).
"""
from __future__ import annotations
import numpy as np


def bin_to_freq(bin_idx: float, n_fft: int, fs: float) -> float:
    """f = k * Fs / N"""
    return bin_idx * fs / n_fft


def frame_to_time(frame_idx: float, hop_length: int, fs: float) -> float:
    return frame_idx * hop_length / fs


def estimate_tonal_params(component: dict, n_fft: int, fs: float, hop_length: int) -> dict:
    """Augment a connected-component descriptor with physical units."""
    freq_hz = bin_to_freq(component["freq_bin_centroid"], n_fft, fs)
    time_s = frame_to_time(component["time_frame_centroid"], hop_length, fs)
    bandwidth_hz = bin_to_freq(component["height_bins"], n_fft, fs)
    duration_s = component["width_frames"] * hop_length / fs
    out = dict(component)
    out.update({
        "frequency_hz": freq_hz,
        "time_s": time_s,
        "bandwidth_hz": bandwidth_hz,
        "duration_s": duration_s,
    })
    return out


def group_harmonics(tonals: list[dict], tol_hz: float = 5.0, max_harmonic: int = 6) -> list[dict]:
    """
    Greedy harmonic grouping: for each candidate fundamental (lowest
    frequency in an unassigned group), check whether other detected tonals
    are close to integer multiples of it. Detected harmonic series raise
    confidence (real machinery tonals are frequently accompanied by
    harmonics; isolated noise blobs are not).
    """
    freqs = sorted(tonals, key=lambda t: t["frequency_hz"])
    used = set()
    groups = []
    for i, fund in enumerate(freqs):
        if id(fund) in used:
            continue
        group = [fund]
        used.add(id(fund))
        for h in range(2, max_harmonic + 1):
            target = fund["frequency_hz"] * h
            for cand in freqs:
                if id(cand) in used:
                    continue
                if abs(cand["frequency_hz"] - target) <= tol_hz:
                    group.append(cand)
                    used.add(id(cand))
                    break
        groups.append(group)

    out = []
    for group in groups:
        fund = group[0]
        fund = dict(fund)
        fund["n_harmonics_found"] = len(group) - 1
        fund["harmonic_freqs_hz"] = [g["frequency_hz"] for g in group[1:]]
        out.append(fund)
    return out


def frequency_stability(freq_history: list[float]) -> float:
    """Variance of recent frequency estimates (Hz^2) -- low = stable target."""
    if len(freq_history) < 2:
        return float("inf")
    return float(np.var(freq_history))


def compute_confidence(seg_prob: float, area: int, snr_db: float,
                        track_len: int, freq_variance: float,
                        n_harmonics: int = 0) -> float:
    """
    Combine several independent cues into a single 0-100% confidence score:
      - seg_prob        : mean SegFormer sigmoid probability over the blob
      - area            : blob size (bigger & more persistent => more confident)
      - snr_db          : estimated in-band SNR
      - track_len       : number of consecutive frames the Kalman tracker has confirmed
      - freq_variance   : variance of the tracked frequency (Hz^2); lower = more stable
      - n_harmonics     : number of harmonics found (machinery tonals often have some)
    """
    c_prob = np.clip(seg_prob, 0, 1)
    c_area = np.clip(area / 40.0, 0, 1)
    c_snr = np.clip((snr_db + 12.0) / 22.0, 0, 1)          # maps ~-12dB..+10dB -> 0..1
    c_track = np.clip(track_len / 10.0, 0, 1)
    c_stability = np.clip(1.0 - freq_variance / 200.0, 0, 1) if np.isfinite(freq_variance) else 0.3
    c_harm = np.clip(n_harmonics / 3.0, 0, 1)

    weights = {"prob": 0.30, "area": 0.10, "snr": 0.15, "track": 0.20,
               "stability": 0.20, "harm": 0.05}
    score = (weights["prob"] * c_prob + weights["area"] * c_area + weights["snr"] * c_snr +
             weights["track"] * c_track + weights["stability"] * c_stability +
             weights["harm"] * c_harm)
    return float(np.clip(score * 100.0, 0, 100))
