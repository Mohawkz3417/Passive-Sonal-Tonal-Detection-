"""
generator_cli.py
==================
Signal Generator, expanded per the suite spec: produces any of

    pure_tone, dual_tone, multi_tone, harmonics, linear_chirp, fm_signal,
    white_noise, pink_noise, ambient_ocean

at a chosen SNR (supports the full +20 dB .. -15 dB range), and
automatically saves a complete bundle for every generated clip:

    <tag>.wav              -- the audio itself
    <tag>_spectrogram.png  -- raw magnitude spectrogram
    <tag>_pcen.png         -- PCEN spectrogram
    <tag>_mask.png         -- ground-truth tonal mask (where applicable)
    <tag>_meta.json        -- SNR, waveform type, frequency, duration, fs, seed, ...

This is menu option 4 ("Signal Generator") in main.py. For building a
full training dataset of many samples at once, see menu option 5
("Dataset Generator" -> 05_Dataset_Generator/dataset_builder.py), which
uses the same underlying generate_clip() machinery.

Usage:
    python generator_cli.py --type multi_tone --snr-db -8 --out-dir ../10_Datasets/generated
    python generator_cli.py --type linear_chirp --snr-db 5 --duration 3
    python generator_cli.py --type pink_noise --duration 10   # pure noise, no tonal
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "13_Scripts"))
sys.path.append(os.path.join(ROOT, "05_Dataset_Generator"))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from signal_processing import preprocess_pipeline                     # noqa: E402
from synthetic_signal_generator import generate_clip, generate_ambient_noise  # noqa: E402
from waveforms import (white_gaussian_noise, pink_noise, linear_chirp,        # noqa: E402
                        fm_signal, mix_at_snr, WAVEFORM_TYPES)
from config_loader import load_config, get as cfg_get, resolve_path   # noqa: E402


def build_waveform(wave_type: str, duration_s: float, fs: float, snr_db: float,
                    f_min: float = 150.0, f_max: float = 3500.0,
                    seed: int | None = None):
    """
    Returns (waveform, freq_tracks, info_dict). freq_tracks is a list of
    per-sample instantaneous-frequency arrays (NaN where inactive) so the
    same ground-truth-mask code path used by dataset_builder.py can build
    a mask for chirps/FM/tones alike; it's empty for pure-noise types.
    """
    rng = np.random.default_rng(seed)
    n_samples = int(duration_s * fs)
    info = {"waveform_type": wave_type, "duration_s": duration_s, "fs": fs, "snr_db": snr_db}

    if wave_type in ("pure_tone", "dual_tone", "multi_tone", "harmonics"):
        n_tonals = {"pure_tone": 1, "dual_tone": 2, "multi_tone": 3}.get(wave_type, 1)
        n_harm = 3 if wave_type == "harmonics" else 0
        clip = generate_clip(duration_s=duration_s, fs=fs, snr_db=snr_db, n_tonals=n_tonals,
                              f_min=f_min, f_max=f_max, seed=seed)
        if wave_type == "harmonics":
            # regenerate with forced harmonics for a clean demonstration sample
            from synthetic_signal_generator import TonalSpec, generate_tonal_waveform
            f0 = rng.uniform(f_min, f_max / (n_harm + 1))
            tone = TonalSpec(f0=f0, n_harmonics=n_harm, drift_hz=0.2, am_depth=0.0)
            wave, ftrack = generate_tonal_waveform(n_samples, fs, tone, seed=seed)
            noise = generate_ambient_noise(n_samples, fs, seed=seed)
            mixed = mix_at_snr(wave, noise, snr_db)
            info["fundamental_hz"] = f0
            info["n_harmonics"] = n_harm
            return mixed, [ftrack], info
        info["tonal_freqs_hz"] = [float(np.nanmean(t)) for t in clip.tonal_freq_tracks]
        return clip.waveform.astype(np.float32), clip.tonal_freq_tracks, info

    noise = generate_ambient_noise(n_samples, fs, seed=seed)

    if wave_type == "linear_chirp":
        f_start, f_end = f_min, min(f_max, f_min + (f_max - f_min) * 0.6)
        tone = linear_chirp(n_samples, fs, f_start=f_start, f_end=f_end)
        mixed = mix_at_snr(tone, noise, snr_db)
        freq_track = np.linspace(f_start, f_end, n_samples)
        info.update({"f_start_hz": f_start, "f_end_hz": f_end})
        return mixed, [freq_track], info

    if wave_type == "fm_signal":
        f_carrier = rng.uniform(f_min, f_max)
        tone = fm_signal(n_samples, fs, f_carrier=f_carrier, f_mod=3.0, mod_index=4.0)
        mixed = mix_at_snr(tone, noise, snr_db)
        t = np.arange(n_samples) / fs
        inst_freq = f_carrier + 4.0 * 3.0 * np.cos(2 * np.pi * 3.0 * t)  # d/dt of phase / 2pi
        info.update({"carrier_hz": f_carrier, "mod_rate_hz": 3.0, "mod_index": 4.0})
        return mixed, [inst_freq], info

    if wave_type == "white_noise":
        wn = white_gaussian_noise(n_samples, fs, seed=seed)
        return (wn / (np.max(np.abs(wn)) + 1e-12)).astype(np.float32), [], info

    if wave_type == "pink_noise":
        pn = pink_noise(n_samples, fs, seed=seed)
        return (pn / (np.max(np.abs(pn)) + 1e-12)).astype(np.float32), [], info

    if wave_type == "ambient_ocean":
        return (noise / (np.max(np.abs(noise)) + 1e-12)).astype(np.float32), [], info

    raise ValueError(f"unknown waveform type '{wave_type}' (choose from {WAVEFORM_TYPES})")


def save_clip_bundle(waveform: np.ndarray, freq_tracks: list, info: dict, out_dir: str,
                      tag: str | None = None, fs: float = 16000.0,
                      n_fft: int = 1024, hop_length: int = 256) -> dict:
    """Save WAV + spectrogram PNG + PCEN PNG + mask PNG + metadata JSON. Returns the
    dict of written file paths."""
    os.makedirs(out_dir, exist_ok=True)
    tag = tag or f"{info.get('waveform_type', 'clip')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    paths = {}

    # WAV
    wav_path = os.path.join(out_dir, f"{tag}.wav")
    try:
        import soundfile as sf
        sf.write(wav_path, waveform, int(fs))
    except ImportError:
        import scipy.io.wavfile as wavfile
        pcm16 = np.clip(waveform * 32767, -32768, 32767).astype(np.int16)
        wavfile.write(wav_path, int(fs), pcm16)
    paths["wav"] = wav_path

    # DSP for spectrogram / PCEN / mask
    proc = preprocess_pipeline(waveform, fs, n_fft=n_fft, hop_length=hop_length)
    extent = [0, proc["times"][-1] if len(proc["times"]) else 1, proc["freqs"][0], proc["freqs"][-1]]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.imshow(20 * np.log10(proc["S_mag"] + 1e-8), aspect="auto", origin="lower", extent=extent, cmap="viridis")
    ax.set_title(f"{tag} -- Raw Spectrogram (dB)"); ax.set_xlabel("Time (s)"); ax.set_ylabel("Hz")
    fig.tight_layout(); spec_path = os.path.join(out_dir, f"{tag}_spectrogram.png")
    fig.savefig(spec_path, dpi=130); plt.close(fig)
    paths["spectrogram_png"] = spec_path

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.imshow(proc["S_pcen"], aspect="auto", origin="lower", extent=extent, cmap="magma")
    ax.set_title(f"{tag} -- PCEN Spectrogram"); ax.set_xlabel("Time (s)"); ax.set_ylabel("Hz")
    fig.tight_layout(); pcen_path = os.path.join(out_dir, f"{tag}_pcen.png")
    fig.savefig(pcen_path, dpi=130); plt.close(fig)
    paths["pcen_png"] = pcen_path

    # Ground-truth mask (only meaningful when we have a known freq track)
    if freq_tracks:
        from dataset_builder import build_mask
        mask = build_mask(proc["freqs"], proc["times"], fs, hop_length, freq_tracks, len(waveform))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.imshow(mask, aspect="auto", origin="lower", extent=extent, cmap="gray")
        ax.set_title(f"{tag} -- Ground-truth Tonal Mask"); ax.set_xlabel("Time (s)"); ax.set_ylabel("Hz")
        fig.tight_layout(); mask_path = os.path.join(out_dir, f"{tag}_mask.png")
        fig.savefig(mask_path, dpi=130); plt.close(fig)
        paths["mask_png"] = mask_path
        np.savez_compressed(os.path.join(out_dir, f"{tag}_mask.npz"), mask=mask.astype(np.float32))
        paths["mask_npz"] = os.path.join(out_dir, f"{tag}_mask.npz")

    # Metadata
    meta = dict(info)
    meta.update({"tag": tag, "n_fft": n_fft, "hop_length": hop_length,
                 "n_samples": len(waveform), "generated_at": datetime.now().isoformat(timespec="seconds")})
    meta_path = os.path.join(out_dir, f"{tag}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else str(o))
    paths["metadata_json"] = meta_path

    return paths


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--type", choices=WAVEFORM_TYPES,
                     default=cfg_get(cfg, "signal_generator.default_type", "multi_tone"))
    ap.add_argument("--duration", type=float, default=cfg_get(cfg, "signal_generator.duration_s", 4.0))
    ap.add_argument("--fs", type=float, default=cfg_get(cfg, "signal.sampling_rate_hz", 16000.0))
    ap.add_argument("--snr-db", type=float, default=cfg_get(cfg, "signal_generator.snr_db", -8.0),
                     help="target SNR in dB (supports the full +20..-15 dB range for pure-tone types)")
    ap.add_argument("--f-min", type=float, default=cfg_get(cfg, "signal_generator.f_min_hz", 150.0))
    ap.add_argument("--f-max", type=float, default=cfg_get(cfg, "signal_generator.f_max_hz", 3500.0))
    ap.add_argument("--out-dir", type=str, default=os.path.join(ROOT, "10_Datasets", "generated"))
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if not (-15.0 <= args.snr_db <= 20.0):
        print(f"[generator] warning: --snr-db {args.snr_db} is outside the suite's supported "
              f"+20..-15 dB range; proceeding anyway, but results may be unrepresentative.")

    waveform, freq_tracks, info = build_waveform(
        args.type, args.duration, args.fs, args.snr_db, args.f_min, args.f_max, seed=args.seed)

    paths = save_clip_bundle(waveform, freq_tracks, info, resolve_path(args.out_dir),
                              tag=args.tag, fs=args.fs)

    print(f"[generator] type={args.type}  SNR={args.snr_db:+.1f} dB  duration={args.duration:.1f}s")
    for k, v in paths.items():
        print(f"  {k:16s} {v}")


if __name__ == "__main__":
    main()
