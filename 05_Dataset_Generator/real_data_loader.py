"""
real_data_loader.py
=====================
Loads real recordings from any WAV/FLAC-based dataset folder -- ShipsEar,
DeepShip, DS3500, an "Underwater Noise Dataset", or an official NPOL
dataset if released -- with no assumptions about internal folder layout
beyond "audio files somewhere under this directory".

This does NOT assume per-file tonal-frequency ground truth exists (most
public ship-noise datasets don't have that). What it provides is a pool
of REAL background recordings that can be used two ways (see
dataset_builder.py and pseudo_label_real_tonals.py):

  1. Real-noise injection: use a real recording as the *background*,
     and mix a synthetic tonal (with known, exact ground truth) on top
     of it at a controlled SNR. You get real ocean/ship noise statistics
     with perfect labels, no manual annotation needed. This is the
     recommended default and is what dataset_builder.TonalSegDataset
     does when you pass --real-data-dir.

  2. Pseudo-labeling real tonals directly out of the recordings
     themselves (see pseudo_label_real_tonals.py) for a small,
     human-reviewed fine-tuning set of genuinely real tonal examples.
"""
from __future__ import annotations
import os
import random
from dataclasses import dataclass

import numpy as np
from scipy.signal import resample_poly
from math import gcd

AUDIO_EXTS = (".wav", ".WAV", ".flac", ".FLAC")

try:
    import soundfile as sf
    _HAVE_SF = True
except ImportError:
    _HAVE_SF = False
    import scipy.io.wavfile as wavfile


def find_audio_files(root_dir: str) -> list[str]:
    files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.endswith(AUDIO_EXTS):
                files.append(os.path.join(dirpath, fn))
    return sorted(files)


def _read_audio(path: str):
    """Returns (waveform float32 mono [-1,1], sample_rate)."""
    if _HAVE_SF:
        data, sr = sf.read(path, always_2d=False, dtype="float32")
    else:
        sr, data = wavfile.read(path)
        if np.issubdtype(data.dtype, np.integer):
            data = data.astype(np.float32) / np.iinfo(data.dtype).max
        else:
            data = data.astype(np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    g = gcd(int(sr_in), int(sr_out))
    up, down = sr_out // g, sr_in // g
    return resample_poly(x, up, down).astype(np.float32)


def load_and_resample(path: str, target_fs: float, duration_s: float | None = None,
                       offset_s: float = 0.0, seed: int | None = None) -> np.ndarray:
    """Load ONE specific audio file (mono-mixed), resample to `target_fs`, and
    return a `duration_s`-long crop (looping short files, exactly like
    RealAudioPool.sample_clip does) starting at `offset_s` seconds -- or a
    random offset if `offset_s` is None. Unlike RealAudioPool (which picks a
    random file from a whole folder), this is for "the user picked this exact
    file in the UI" -- same resample/loop/normalize behavior, just pinned to
    one path instead of drawn from a pool. DC-removed and peak-normalized,
    same as RealAudioPool.sample_clip, so the two are interchangeable as a
    noise/background source."""
    data, sr = _read_audio(path)
    data = _resample(data.astype(np.float32), sr, int(target_fs))
    if duration_s is not None:
        n_target = int(duration_s * target_fs)
        if len(data) < n_target:
            reps = int(np.ceil(n_target / max(1, len(data))))
            data = np.tile(data, reps)
        if offset_s is None:
            rng = random.Random(seed)
            max_start = max(0, len(data) - n_target)
            start = rng.randint(0, max_start) if max_start > 0 else 0
        else:
            start = int(offset_s * target_fs) % max(1, len(data) - n_target + 1) if len(data) > n_target else 0
        data = data[start:start + n_target]
    data = data - np.mean(data)
    peak = np.max(np.abs(data))
    if peak > 1e-8:
        data = data / peak
    return data.astype(np.float32)


@dataclass
class AudioFileInfo:
    path: str
    n_samples: int
    sample_rate: int

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.sample_rate


class RealAudioPool:
    """
    Indexes a real dataset folder once (fast -- reads headers only where
    possible), then serves random fixed-duration crops resampled to the
    target sample rate, ready to drop straight into the signal-processing
    pipeline (13_Scripts/signal_processing.py) or the synthetic generator
    as a real-noise background (04_Signal_Generator).
    """

    def __init__(self, root_dir: str, target_fs: float = 16000.0,
                 max_files: int | None = None, min_duration_s: float = 2.0,
                 seed: int | None = None):
        self.root_dir = root_dir
        self.target_fs = target_fs
        self._rng = random.Random(seed)

        paths = find_audio_files(root_dir)
        if not paths:
            raise FileNotFoundError(
                f"No .wav/.flac files found under {root_dir!r}. "
                "Point --real-data-dir at the folder containing your "
                "ShipsEar / DeepShip / DS3500 / etc. audio files (any "
                "subfolder layout is fine).")
        if max_files:
            paths = paths[:max_files]

        self.files: list[AudioFileInfo] = []
        for p in paths:
            try:
                if _HAVE_SF:
                    info = sf.info(p)
                    n_samples, sr = info.frames, info.samplerate
                else:
                    sr, data = wavfile.read(p)
                    n_samples = len(data)
                if n_samples / sr >= min_duration_s:
                    self.files.append(AudioFileInfo(p, n_samples, sr))
            except Exception as e:  # noqa: BLE001 -- real datasets sometimes have a few corrupt files
                print(f"[RealAudioPool] skipping unreadable file {p}: {e}")

        if not self.files:
            raise RuntimeError(
                f"Found audio files under {root_dir!r} but none were readable "
                f"or long enough (>= {min_duration_s}s). Check the files/format.")
        print(f"[RealAudioPool] indexed {len(self.files)} usable files from {root_dir}")

    def __len__(self):
        return len(self.files)

    def sample_clip(self, duration_s: float) -> np.ndarray:
        """Return a random `duration_s`-long mono clip, resampled to target_fs,
        DC-removed and peak-normalized. Short files are looped to fill the length."""
        info = self._rng.choice(self.files)
        try:
            n_needed_in = int(duration_s * info.sample_rate)
            if _HAVE_SF:
                if info.n_samples > n_needed_in:
                    start = self._rng.randint(0, info.n_samples - n_needed_in)
                    with sf.SoundFile(info.path) as f:
                        f.seek(start)
                        data = f.read(n_needed_in, dtype="float32", always_2d=False)
                else:
                    data, _ = sf.read(info.path, dtype="float32", always_2d=False)
            else:
                _, data = wavfile.read(info.path)
                if np.issubdtype(data.dtype, np.integer):
                    data = data.astype(np.float32) / np.iinfo(data.dtype).max
                if info.n_samples > n_needed_in:
                    start = self._rng.randint(0, info.n_samples - n_needed_in)
                    data = data[start:start + n_needed_in]
            if data.ndim > 1:
                data = data.mean(axis=1)
        except Exception as e:  # noqa: BLE001
            print(f"[RealAudioPool] read failed for {info.path} ({e}); retrying with another file")
            self.files.remove(info)
            if not self.files:
                raise
            return self.sample_clip(duration_s)

        data = _resample(data.astype(np.float32), info.sample_rate, int(self.target_fs))
        n_target = int(duration_s * self.target_fs)
        if len(data) < n_target:
            reps = int(np.ceil(n_target / max(1, len(data))))
            data = np.tile(data, reps)
        data = data[:n_target]

        data = data - np.mean(data)
        peak = np.max(np.abs(data))
        if peak > 1e-8:
            data = data / peak
        return data.astype(np.float32)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python real_data_loader.py <path_to_dataset_folder>")
        raise SystemExit(1)
    pool = RealAudioPool(sys.argv[1], target_fs=16000.0)
    clip = pool.sample_clip(4.0)
    print("sampled real-noise clip:", clip.shape, "peak", np.max(np.abs(clip)))
