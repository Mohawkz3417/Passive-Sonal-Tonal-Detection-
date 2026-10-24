"""
dataset_builder.py
====================
Turns synthetic hydrophone clips into (spectrogram, binary tonal mask)
training pairs, and exposes a PyTorch Dataset that generates data on the
fly (infinite, no disk I/O needed) as well as a materializer that writes
a fixed-size dataset of .npz pairs to 10_Datasets/ for repeatable
experiments.

Ground-truth mask construction:
    For every tonal, the known instantaneous frequency track is converted
    to (time-frame, freq-bin) coordinates and a mask row of +/- `line_width_bins`
    around that bin is marked as 1 (tonal) for every active frame.
"""
from __future__ import annotations
import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "13_Scripts"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "04_Signal_Generator"))

from signal_processing import preprocess_pipeline          # noqa: E402
from synthetic_signal_generator import generate_clip        # noqa: E402


def build_mask(freqs: np.ndarray, times: np.ndarray, fs: float, hop_length: int,
                tonal_freq_tracks, n_samples: int, line_width_bins: int = 1) -> np.ndarray:
    """
    freqs : (n_freq,) STFT bin centre frequencies
    times : (n_frames,) STFT frame centre times (s)
    tonal_freq_tracks : list of (n_samples,) arrays, NaN where inactive
    returns mask (n_freq, n_frames) of {0,1}
    """
    n_freq, n_frames = len(freqs), len(times)
    mask = np.zeros((n_freq, n_frames), dtype=np.float32)
    freq_res = freqs[1] - freqs[0]

    for track in tonal_freq_tracks:
        # sample the (per-sample) freq track at the STFT frame centre times
        sample_idx = np.clip((times * fs).astype(int), 0, n_samples - 1)
        f_at_frame = track[sample_idx]
        for t_idx, f in enumerate(f_at_frame):
            if np.isnan(f):
                continue
            bin_idx = int(round(f / freq_res))
            lo = max(0, bin_idx - line_width_bins)
            hi = min(n_freq, bin_idx + line_width_bins + 1)
            if lo < hi:
                mask[lo:hi, t_idx] = 1.0
    return mask


def make_sample(duration_s: float = 4.0, fs: float = 16000.0, snr_db: float = -8.0,
                 n_tonals: int = 1, n_fft: int = 1024, hop_length: int = 256,
                 real_noise_waveform: np.ndarray | None = None,
                 noise_profile: str = "environmental",
                 f_min: float = 150.0, f_max: float = 3500.0,
                 doppler_prob: float = 0.3,
                 seed: int | None = None):
    """Generate one (image, mask) pair as float32 numpy arrays, shape (1, F, T).
    If `real_noise_waveform` is given (see real_data_loader.RealAudioPool),
    it is used as the noise background instead of the synthetic noise model
    -- see synthetic_signal_generator.generate_clip for details. `noise_profile`
    selects the synthetic ambient-noise spectral shape (ignored when
    real_noise_waveform is given) -- shipping/environmental/biological/background.
    `f_min`/`f_max`/`doppler_prob` are passed straight through to generate_clip()
    -- previously this function silently relied on generate_clip()'s own
    hardcoded defaults, so changing config.yaml's training.f_min_hz/f_max_hz/
    doppler_prob had NO effect on actual training data. Fixed."""
    clip = generate_clip(duration_s=duration_s, fs=fs, snr_db=snr_db,
                          n_tonals=n_tonals, real_noise_waveform=real_noise_waveform,
                          noise_profile=noise_profile, f_min=f_min, f_max=f_max,
                          doppler_prob=doppler_prob, seed=seed)
    proc = preprocess_pipeline(clip.waveform, fs, n_fft=n_fft, hop_length=hop_length)
    S_pcen = proc["S_pcen"]

    # per-clip min-max normalize to [0,1] for stable network input
    lo, hi = np.percentile(S_pcen, 1), np.percentile(S_pcen, 99)
    img = np.clip((S_pcen - lo) / (hi - lo + 1e-8), 0, 1).astype(np.float32)

    mask = build_mask(proc["freqs"], proc["times"], fs, hop_length,
                       clip.tonal_freq_tracks, len(clip.waveform))
    return img[None, ...], mask[None, ...], clip  # add channel dim


try:
    import torch
    from torch.utils.data import Dataset

    class TonalSegDataset(Dataset):
        """
        Infinite on-the-fly synthetic dataset. Each __getitem__ draws a fresh
        random clip -- SNR is sampled uniformly from `snr_range` (dB) so the
        model sees the full difficulty spectrum, with extra emphasis on the
        -7..-10 dB target band via `hard_snr_prob`.

        Real-data integration: pass `real_noise_dir` (a folder containing
        ShipsEar / DeepShip / DS3500 / any WAV or FLAC dataset -- any
        subfolder layout) to mix real recordings in as the noise background
        for a fraction (`real_noise_prob`) of samples. The synthetic tonal(s)
        are still injected on top with known ground truth, so you get real
        ocean/ship noise statistics with zero manual labeling effort. See
        05_Dataset_Generator/real_data_loader.py and
        00_Documentation/REAL_DATA_GUIDE.md for details and for how to
        additionally fine-tune on pseudo-labeled *real* tonals.
        """

        def __init__(self, length: int = 1000, duration_s: float = 4.0, fs: float = 16000.0,
                     n_fft: int = 1024, hop_length: int = 256,
                     snr_range=(-10.0, 20.0), hard_snr_range=(-10.0, -7.0),
                     hard_snr_prob: float = 0.4, max_tonals: int = 2, seed: int | None = None,
                     real_noise_dir: str | None = None, real_noise_prob: float = 0.5,
                     noise_profiles: list | None = None,
                     f_min: float = 150.0, f_max: float = 3500.0, doppler_prob: float = 0.3):
            self.length = length
            self.duration_s = duration_s
            self.fs = fs
            self.n_fft = n_fft
            self.hop_length = hop_length
            self.snr_range = snr_range
            self.hard_snr_range = hard_snr_range
            self.hard_snr_prob = hard_snr_prob
            self.max_tonals = max_tonals
            self.base_seed = seed
            self.real_noise_prob = real_noise_prob if real_noise_dir else 0.0
            self._real_pool = None
            # NEW: one of these is picked at random PER SAMPLE (uniformly) when no
            # real_noise_waveform is used for that sample -- previously every sample
            # used the single default ("environmental") profile regardless of what
            # the Signal Generator app exposed, found by check_training_data.py.
            self.noise_profiles = noise_profiles or ["shipping", "environmental", "biological", "background"]
            # NEW: previously generate_clip()'s hardcoded f_min=150/f_max=3500 and
            # doppler_prob=0.3 defaults were used no matter what config.yaml said --
            # these are now real constructor parameters threaded all the way from
            # train.py's config-sourced CLI args.
            self.f_min = f_min
            self.f_max = f_max
            self.doppler_prob = doppler_prob
            if real_noise_dir:
                from real_data_loader import RealAudioPool
                self._real_pool = RealAudioPool(real_noise_dir, target_fs=fs, seed=seed)

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            rng = np.random.default_rng(None if self.base_seed is None else self.base_seed + idx)
            if rng.random() < self.hard_snr_prob:
                snr_db = rng.uniform(*self.hard_snr_range)
            else:
                snr_db = rng.uniform(*self.snr_range)
            n_tonals = int(rng.integers(0, self.max_tonals + 1))  # allow 0 => pure-noise negatives
            real_noise = None
            if self._real_pool is not None and rng.random() < self.real_noise_prob:
                real_noise = self._real_pool.sample_clip(self.duration_s)
            noise_profile = rng.choice(self.noise_profiles)
            img, mask, _ = make_sample(self.duration_s, self.fs, snr_db, n_tonals,
                                        self.n_fft, self.hop_length,
                                        real_noise_waveform=real_noise,
                                        noise_profile=noise_profile,
                                        f_min=self.f_min, f_max=self.f_max,
                                        doppler_prob=self.doppler_prob,
                                        seed=int(rng.integers(0, 2**31 - 1)))
            return torch.from_numpy(img), torch.from_numpy(mask)

except ImportError:
    TonalSegDataset = None  # torch not installed; numpy-only usage still works


def materialize_dataset(out_dir: str, n_samples: int = 200, **kwargs):
    """Write a fixed dataset of .npz(img, mask, snr_db) files for repeatable eval."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(123)
    for i in range(n_samples):
        snr_db = rng.uniform(kwargs.get("snr_lo", -10), kwargs.get("snr_hi", 20))
        n_tonals = int(rng.integers(0, kwargs.get("max_tonals", 2) + 1))
        img, mask, clip = make_sample(snr_db=snr_db, n_tonals=n_tonals, seed=int(i))
        np.savez_compressed(os.path.join(out_dir, f"sample_{i:05d}.npz"),
                             img=img, mask=mask, snr_db=snr_db)
    print(f"wrote {n_samples} samples to {out_dir}")


if __name__ == "__main__":
    img, mask, clip = make_sample(snr_db=-8.0, n_tonals=1, seed=0)
    print("img", img.shape, img.dtype, "mask", mask.shape, "positive px:", mask.sum())
