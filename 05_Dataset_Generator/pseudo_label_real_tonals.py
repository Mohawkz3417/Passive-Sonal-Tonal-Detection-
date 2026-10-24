"""
pseudo_label_real_tonals.py
=============================
Real datasets (ShipsEar, DeepShip, DS3500, ...) give you real recordings
but essentially never give you per-frequency-bin tonal ground truth --
that is normally hand-annotated by a sonar analyst. This tool builds a
*starting point* for that: a sensitive classical detector that finds
candidate persistent narrowband lines in a real recording, so you can
review/correct a small set and fine-tune on genuinely real tonals
(rather than only synthetic tonals on real noise, which is what
dataset_builder.TonalSegDataset's real_noise_dir option gives you).

Method (deliberately classical/simple, not the SegFormer model, so it
doesn't just circularly re-label whatever the network already believes):
    1. STFT + long-term average spectrum per file
    2. CA-CFAR (07_Deployment_Module/postprocess.py) run per-frame
    3. A bin is kept only if it is CFAR-flagged for at least
       `persistence` fraction of frames in a sliding time window
       (real tonals are persistent; noise spikes are not)
    4. Standard morphological cleanup + connected components

This is a RECALL-ORIENTED detector (finds candidates), not a precision
detector -- it is expected to include some false positives. Always
visually review the output overlay PNG before trusting a mask as ground
truth; treat unreviewed candidates as a WEAK label (see --weak-loss-note
in train.py's docstring / REAL_DATA_GUIDE.md) rather than a verified one.

Usage:
    python pseudo_label_real_tonals.py --input /path/to/recording.wav \
        --out-dir ../10_Datasets/pseudo_labeled --persistence 0.6
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "13_Scripts"))
sys.path.append(os.path.join(ROOT, "07_Deployment_Module"))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from signal_processing import preprocess_pipeline               # noqa: E402
from postprocess import ca_cfar_1d, morphological_clean, connected_components  # noqa: E402
from real_data_loader import _read_audio, _resample              # noqa: E402


def persistent_cfar_mask(S_mag: np.ndarray, num_train=16, num_guard=4, pfa=5e-2,
                          persistence=0.6, window_frames=50,
                          valid_bin_range: tuple[int, int] | None = None) -> np.ndarray:
    """CFAR per frame, then keep only bins that fire in >= `persistence`
    fraction of frames within a sliding `window_frames` window -- this is
    what separates a real, persistent tonal candidate from a one-off
    noise spike that a single-frame CFAR pass would also flag."""
    cfar = ca_cfar_1d(S_mag, num_train=num_train, num_guard=num_guard, pfa=pfa,
                       valid_bin_range=valid_bin_range)
    # fraction of "on" frames within a centered rolling window, per bin
    kernel = np.ones(window_frames) / window_frames
    rolling = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 1, cfar)
    return (rolling >= persistence).astype(np.float32)


def process_file(wav_path: str, out_dir: str, fs: float = 16000.0, n_fft: int = 1024,
                  hop_length: int = 256, persistence: float = 0.6, chunk_s: float = 10.0):
    os.makedirs(out_dir, exist_ok=True)
    data, sr_in = _read_audio(wav_path)
    data = _resample(data, sr_in, int(fs))

    n_chunk = int(chunk_s * fs)
    base = os.path.splitext(os.path.basename(wav_path))[0]
    n_written = 0

    for start in range(0, len(data) - n_chunk + 1, n_chunk):
        chunk = data[start:start + n_chunk]
        proc = preprocess_pipeline(chunk, fs, n_fft=n_fft, hop_length=hop_length)
        S_mag, S_pcen = proc["S_mag"], proc["S_pcen"]

        candidate = persistent_cfar_mask(S_mag, persistence=persistence,
                                          valid_bin_range=proc["passband_bins"])
        cleaned = morphological_clean(candidate, min_area=8, min_elongation=2.0)
        comps = connected_components(cleaned, weight_img=S_pcen)

        if not comps:
            continue  # no persistent candidate tonal in this chunk -- skip

        lo, hi = np.percentile(S_pcen, 1), np.percentile(S_pcen, 99)
        img = np.clip((S_pcen - lo) / (hi - lo + 1e-8), 0, 1).astype(np.float32)

        tag = f"{base}_chunk{start // n_chunk:04d}"
        np.savez_compressed(os.path.join(out_dir, f"{tag}.npz"),
                              img=img[None], mask=cleaned[None].astype(np.float32),
                              source_file=wav_path, n_candidates=len(comps))

        # review overlay: spectrogram with candidate mask outlined
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(S_pcen, aspect="auto", origin="lower", cmap="magma",
                   extent=[0, proc["times"][-1], proc["freqs"][0], proc["freqs"][-1]])
        ax.contour(cleaned, levels=[0.5], colors="cyan", linewidths=1.0,
                    extent=[0, proc["times"][-1], proc["freqs"][0], proc["freqs"][-1]])
        ax.set_title(f"{tag}  ({len(comps)} candidate(s)) -- REVIEW BEFORE TRUSTING")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Hz")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{tag}_review.png"), dpi=120)
        plt.close(fig)

        n_written += 1

    print(f"[pseudo-label] {wav_path}: wrote {n_written} candidate chunk(s) to {out_dir}")
    return n_written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="a single audio file, or a directory to scan recursively")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "10_Datasets", "pseudo_labeled"))
    ap.add_argument("--fs", type=float, default=16000.0)
    ap.add_argument("--persistence", type=float, default=0.6,
                     help="fraction of frames in the rolling window a bin must be CFAR-flagged "
                          "in to be kept as a candidate tonal (higher = fewer, more confident candidates)")
    ap.add_argument("--chunk-s", type=float, default=10.0)
    args = ap.parse_args()

    if os.path.isdir(args.input):
        from real_data_loader import find_audio_files
        files = find_audio_files(args.input)
    else:
        files = [args.input]

    total = 0
    for f in files:
        try:
            total += process_file(f, args.out_dir, fs=args.fs, persistence=args.persistence,
                                    chunk_s=args.chunk_s)
        except Exception as e:  # noqa: BLE001
            print(f"[pseudo-label] failed on {f}: {e}")

    print(f"\n[pseudo-label] done. {total} candidate chunks written to {args.out_dir}.")
    print("IMPORTANT: open the *_review.png files and check the cyan outlines actually trace "
          "real tonal lines before using these .npz files for training -- this detector is "
          "tuned for high recall (it WILL include some false positives).")


if __name__ == "__main__":
    main()
