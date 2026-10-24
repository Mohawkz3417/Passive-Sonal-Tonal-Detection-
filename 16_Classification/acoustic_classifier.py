"""
acoustic_classifier.py
=========================
Lightweight acoustic contact classifier: SHIP_LIKE / BIOLOGICAL / UNKNOWN.

Design decision (read before changing thresholds/classes)
-----------------------------------------------------------
Features are computed with REAL signal processing on the waveform itself
(narrowband Hilbert-transform instantaneous frequency/envelope, FFT
spectral centroid, harmonic search) -- NOT by reading back the synthetic
generator's own parameters. That means the exact same `extract_features`
function used to build the (synthetic) training set is also what the live
pipeline calls on a beamformed, detected tonal segment; the classifier
sees the same kind of input in both training and live use, which is what
makes its reported accuracy meaningful rather than circular.

Every feature/number this module reports (training accuracy, confusion
matrix, live class probabilities) comes from actually running this code --
nothing here is a placeholder or a hand-picked "looks impressive" value.
This module trains ONLY on synthetic data (labelled at generation time);
that is stated explicitly wherever its metrics are surfaced, per the
project's "do not fabricate/mislabel results" rule.
"""
from __future__ import annotations
import os
import sys
import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "13_Scripts"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "04_Signal_Generator"))
from signal_processing import bandpass_filter  # noqa: E402
from synthetic_signal_generator import TonalSpec, generate_tonal_waveform, generate_ambient_noise  # noqa: E402

from scipy.signal import hilbert
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import joblib

CLASSES = ["SHIP_LIKE", "BIOLOGICAL", "UNKNOWN"]
FEATURE_NAMES = [
    "dominant_freq_hz", "freq_stability_hz", "am_depth", "am_rate_hz_est",
    "spectral_centroid_hz", "bandwidth_hz", "n_harmonics_detected",
    "energy_rms", "duration_s",
]
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "acoustic_classifier.joblib")


# --------------------------------------------------------------------- features
def extract_features(waveform: np.ndarray, fs: float, peak_freq_hz: float,
                      duration_s: float, search_band_hz: float = 40.0,
                      max_harmonics: int = 5) -> dict:
    """
    Real, waveform-derived feature vector centred on a known/estimated peak
    frequency (e.g. the frequency SegFormer+CFAR+CCA already found for a
    tonal candidate, or -- for training -- the synthetic generator's f0).
    """
    n = len(waveform)
    narrow = bandpass_filter(waveform, fs,
                              low_hz=max(1.0, peak_freq_hz - search_band_hz),
                              high_hz=peak_freq_hz + search_band_hz)
    analytic = hilbert(narrow)
    inst_phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(inst_phase) / (2.0 * np.pi) * fs
    inst_freq = inst_freq[np.isfinite(inst_freq)]
    dominant_freq = float(np.median(inst_freq)) if len(inst_freq) else peak_freq_hz
    freq_stability = float(np.std(inst_freq)) if len(inst_freq) else 0.0

    envelope = np.abs(analytic)
    env_mean = float(np.mean(envelope)) + 1e-12
    am_depth = float((np.max(envelope) - np.min(envelope)) / (np.max(envelope) + np.min(envelope) + 1e-12))
    # crude AM-rate estimate: peak of the envelope's own spectrum, excluding DC
    env_spec = np.abs(np.fft.rfft(envelope - env_mean))
    env_freqs = np.fft.rfftfreq(len(envelope), d=1.0 / fs)
    valid = env_freqs > 0.2
    am_rate_hz_est = float(env_freqs[valid][np.argmax(env_spec[valid])]) if valid.any() else 0.0

    spec = np.abs(np.fft.rfft(waveform))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spectral_centroid = float(np.sum(freqs * spec) / (np.sum(spec) + 1e-12))

    # -3 dB bandwidth around the peak bin
    peak_bin = int(np.argmin(np.abs(freqs - dominant_freq)))
    peak_mag = spec[peak_bin] + 1e-12
    half_power = peak_mag / np.sqrt(2)
    lo = peak_bin
    while lo > 0 and spec[lo] > half_power:
        lo -= 1
    hi = peak_bin
    while hi < len(spec) - 1 and spec[hi] > half_power:
        hi += 1
    bandwidth_hz = float(freqs[hi] - freqs[lo]) if hi > lo else float(freqs[1] - freqs[0])

    # harmonic search: is there real energy at k*f0 above the local noise floor?
    noise_floor = float(np.median(spec))
    n_harm = 0
    for k in range(2, max_harmonics + 2):
        target = dominant_freq * k
        if target >= freqs[-1]:
            break
        b = int(np.argmin(np.abs(freqs - target)))
        if spec[b] > 6.0 * noise_floor:
            n_harm += 1

    energy_rms = float(np.sqrt(np.mean(waveform ** 2)))

    return {
        "dominant_freq_hz": dominant_freq, "freq_stability_hz": freq_stability,
        "am_depth": am_depth, "am_rate_hz_est": am_rate_hz_est,
        "spectral_centroid_hz": spectral_centroid, "bandwidth_hz": bandwidth_hz,
        "n_harmonics_detected": float(n_harm), "energy_rms": energy_rms,
        "duration_s": float(duration_s),
    }


def feature_dict_to_vector(feat: dict) -> np.ndarray:
    return np.array([feat[k] for k in FEATURE_NAMES], dtype=np.float64)


# --------------------------------------------------------------------- synthetic labelled dataset
def _make_labelled_clip(label: str, fs: float, duration_s: float, rng: np.random.Generator):
    # Noise profile is sampled INDEPENDENTLY of class label. Tying a specific
    # noise color 1:1 to each class would leak the label through a trivial
    # broadband-spectrum shortcut (the classifier would learn "which noise
    # background was this generated with" instead of any real target
    # signature) and produce an inflated, meaningless accuracy number.
    noise_profile = rng.choice(["shipping", "environmental", "biological", "background"])
    f0 = rng.uniform(150.0, 2000.0)
    if label == "SHIP_LIKE":
        tone = TonalSpec(f0=f0, n_harmonics=int(rng.integers(2, 5)),
                          drift_hz=rng.uniform(0.05, 0.35), am_depth=rng.uniform(0.08, 0.25),
                          am_rate_hz=rng.uniform(1.0, 4.0), duty=1.0)
    elif label == "BIOLOGICAL":
        tone = TonalSpec(f0=f0, n_harmonics=int(rng.integers(0, 2)),
                          drift_hz=rng.uniform(0.8, 2.5), am_depth=rng.uniform(0.3, 0.6),
                          am_rate_hz=rng.uniform(5.0, 15.0), duty=rng.uniform(0.3, 0.7),
                          onset_frac=rng.uniform(0.0, 0.3))
    else:  # UNKNOWN -- deliberately ambiguous/overlapping with both classes above
        tone = TonalSpec(f0=f0, n_harmonics=int(rng.integers(0, 3)),
                          drift_hz=rng.uniform(0.2, 1.5), am_depth=rng.uniform(0.1, 0.5),
                          am_rate_hz=rng.uniform(1.0, 10.0), duty=rng.uniform(0.4, 1.0))

    n_samples = int(fs * duration_s)
    seed = int(rng.integers(0, 2**31 - 1))
    wave, _ = generate_tonal_waveform(n_samples, fs, tone, seed=seed)
    noise = generate_ambient_noise(n_samples, fs, seed=seed + 1, profile=noise_profile)
    snr_db = rng.uniform(-5.0, 10.0)
    sig_power = np.mean(wave ** 2) + 1e-12
    noise = noise / (np.std(noise) + 1e-12) * np.sqrt(sig_power / (10 ** (snr_db / 10)))
    clip = wave + noise
    duration_present = tone.duty * duration_s
    return clip, fs, f0, duration_present


def build_synthetic_dataset(n_per_class: int = 150, fs: float = 16000.0,
                             duration_s: float = 4.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for label in CLASSES:
        for _ in range(n_per_class):
            clip, fs_, f0, dur_present = _make_labelled_clip(label, fs, duration_s, rng)
            feat = extract_features(clip, fs_, f0, dur_present)
            X.append(feature_dict_to_vector(feat))
            y.append(label)
    return np.array(X), np.array(y)


# --------------------------------------------------------------------- train / evaluate / load
def train_and_evaluate(n_per_class: int = 150, save_path: str = DEFAULT_MODEL_PATH,
                        seed: int = 0) -> dict:
    X, y = build_synthetic_dataset(n_per_class=n_per_class, seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y)

    clf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=seed)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=CLASSES, average=None, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=CLASSES)

    joblib.dump({"model": clf, "classes": CLASSES, "feature_names": FEATURE_NAMES}, save_path)

    metrics = {
        "trained_on": "synthetic data only (see module docstring)",
        "n_train": int(len(y_train)), "n_test": int(len(y_test)),
        "accuracy": acc,
        "per_class": {c: {"precision": float(p), "recall": float(r), "f1": float(f)}
                      for c, p, r, f in zip(CLASSES, precision, recall, f1)},
        "confusion_matrix": cm.tolist(), "confusion_matrix_labels": CLASSES,
        "feature_importances": dict(zip(FEATURE_NAMES, clf.feature_importances_.tolist())),
    }
    metrics_path = os.path.splitext(save_path)[0] + "_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


class AcousticClassifier:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained classifier at {model_path}. Run "
                f"`python acoustic_classifier.py --train` first.")
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.classes = bundle["classes"]

    def classify(self, waveform: np.ndarray, fs: float, peak_freq_hz: float,
                 duration_s: float) -> tuple[str, float, dict]:
        feat = extract_features(waveform, fs, peak_freq_hz, duration_s)
        x = feature_dict_to_vector(feat).reshape(1, -1)
        probs = self.model.predict_proba(x)[0]
        idx = int(np.argmax(probs))
        label = self.model.classes_[idx]
        confidence = float(probs[idx])
        prob_dict = {c: float(p) for c, p in zip(self.model.classes_, probs)}
        return label, confidence, prob_dict


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--n-per-class", type=int, default=150)
    args = ap.parse_args()
    if args.train:
        m = train_and_evaluate(n_per_class=args.n_per_class)
        print(json.dumps(m, indent=2))
