"""
check_training_data.py
=========================
Run BEFORE training (Part 32 of the freeze spec): actually samples
04_Signal_Generator.generate_clip() the same way 05_Dataset_Generator/
dataset_builder.TonalSegDataset does internally (same hard-SNR-band
weighting), and buckets the REAL ground truth returned on each clip
(clip.snr_db, clip.tone_specs[*].f0) into SNR bins and frequency bands.
This does not assume coverage from the config values alone -- it checks
what the generator actually produces.

Usage:
    python 01_Training_Module/check_training_data.py
    python 01_Training_Module/check_training_data.py --n-samples 500
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "13_Scripts"))
sys.path.append(os.path.join(ROOT, "04_Signal_Generator"))

from config_loader import load_config, get as cfg_get  # noqa: E402
from synthetic_signal_generator import generate_clip  # noqa: E402


def _sample_snr_like_training(rng: np.random.Generator, snr_lo: float, snr_hi: float,
                               hard_lo: float, hard_hi: float, hard_prob: float) -> float:
    """Mirrors TonalSegDataset.__getitem__'s exact sampling logic (see
    05_Dataset_Generator/dataset_builder.py) so this check reflects the
    real training distribution, not a different one invented here."""
    if rng.random() < hard_prob:
        return float(rng.uniform(hard_lo, hard_hi))
    return float(rng.uniform(snr_lo, snr_hi))


def check_snr_frequency_coverage(cfg: dict, n_samples: int, seed: int = 0):
    snr_lo = float(cfg_get(cfg, "training.snr_lo_db", -12.0))
    snr_hi = float(cfg_get(cfg, "training.snr_hi_db", 2.0))
    hard_lo, hard_hi = cfg_get(cfg, "training.hard_snr_range_db", [-12.0, -7.0])
    hard_prob = float(cfg_get(cfg, "training.hard_snr_prob", 0.7))
    doppler_prob = float(cfg_get(cfg, "training.doppler_prob", 0.3))
    eval_grid = cfg_get(cfg, "training.snr_eval_grid_db", [2, 0, -2, -4, -6, -8, -10, -12])
    freq_bands = [tuple(b) for b in cfg_get(cfg, "training.freq_bands_hz",
                                             [[150, 500], [500, 1000], [1000, 1750], [1750, 2500], [2500, 3500]])]
    f_min = float(cfg_get(cfg, "training.f_min_hz", 150.0))
    f_max = float(cfg_get(cfg, "training.f_max_hz", 3500.0))
    duration_s = float(cfg_get(cfg, "training.duration_s", 4.0))

    rng = np.random.default_rng(seed)
    noise_profiles = cfg_get(cfg, "training.noise_profiles",
                              ["shipping", "environmental", "biological", "background"])
    snr_bins_seen = {f"{g:+.0f} dB": 0 for g in eval_grid}
    freq_bands_seen = {b: 0 for b in freq_bands}
    noise_profile_counts = {p: 0 for p in noise_profiles}
    below_range = above_range = 0
    n_zero_tonal = 0
    harmonics_seen = drift_seen = doppler_seen = duty_lt_1_seen = 0

    for i in range(n_samples):
        snr_db = _sample_snr_like_training(rng, snr_lo, snr_hi, hard_lo, hard_hi, hard_prob)
        n_tonals = int(rng.integers(0, 3))  # matches TonalSegDataset's max_tonals=2 default -> range(0,3)
        profile = str(rng.choice(noise_profiles))
        clip = generate_clip(duration_s=duration_s, fs=16000.0, snr_db=snr_db, n_tonals=n_tonals,
                              f_min=f_min, f_max=f_max, noise_profile=profile, doppler_prob=doppler_prob,
                              seed=int(rng.integers(0, 2**31 - 1)))
        noise_profile_counts[profile] = noise_profile_counts.get(profile, 0) + 1

        nearest = min(eval_grid, key=lambda g: abs(g - clip.snr_db))
        snr_bins_seen[f"{nearest:+.0f} dB"] += 1

        if n_tonals == 0:
            n_zero_tonal += 1
        for spec in clip.tone_specs:
            placed = False
            for lo, hi in freq_bands:
                if lo <= spec.f0 < hi:
                    freq_bands_seen[(lo, hi)] += 1
                    placed = True
                    break
            if not placed:
                if spec.f0 < freq_bands[0][0]:
                    below_range += 1
                else:
                    above_range += 1
            if spec.n_harmonics > 0:
                harmonics_seen += 1
            if spec.drift_hz > 0:
                drift_seen += 1
            if getattr(spec, "doppler_ramp_hz_per_s", 0.0) != 0.0:
                doppler_seen += 1
            if spec.duty < 1.0:
                duty_lt_1_seen += 1

    return {
        "snr_range_configured": (snr_lo, snr_hi), "hard_range_configured": (hard_lo, hard_hi),
        "hard_prob": hard_prob, "snr_eval_grid": eval_grid, "snr_bins_seen": snr_bins_seen,
        "freq_bands_configured": freq_bands, "freq_bands_seen": freq_bands_seen,
        "n_below_freq_range": below_range, "n_above_freq_range": above_range,
        "n_zero_tonal_negatives": n_zero_tonal, "n_harmonics_seen": harmonics_seen,
        "n_drift_seen": drift_seen, "n_doppler_seen": doppler_seen, "n_intermittent_seen": duty_lt_1_seen,
        "noise_profile_counts": noise_profile_counts,
        "n_samples_checked": n_samples,
    }


def check_exact_evaluation_grid(cfg: dict, n_per_point: int = 3, seed: int = 1000):
    """Distinct from check_snr_frequency_coverage() above: that function
    reports what a RANDOM continuous training sample happens to land near
    (statistical training-range coverage) -- a bin could show n=0 there
    simply because chance didn't produce a sample near that exact point in
    a given run, even though the training range legitimately covers it.

    This function instead DELIBERATELY generates samples AT the EXACT SNR
    values in training.snr_eval_grid_db (no sampling/bucketing -- the same
    way 06_Model_Evaluation/evaluate.py's run_pixel_eval/run_detection_eval
    actually call generate_clip/make_sample with an exact snr_db), and
    verifies each one produces a valid, processable clip. This is what
    "the evaluation grid is actually usable" means, and is checked
    independently of whatever the training distribution happens to look
    like this run."""
    from synthetic_signal_generator import generate_clip
    from signal_processing import preprocess_pipeline

    eval_grid = cfg_get(cfg, "training.snr_eval_grid_db", [2, 0, -2, -4, -6, -8, -10, -12])
    f_min = float(cfg_get(cfg, "training.f_min_hz", 150.0))
    f_max = float(cfg_get(cfg, "training.f_max_hz", 3500.0))
    duration_s = float(cfg_get(cfg, "training.duration_s", 4.0))

    results = {}
    for g in eval_grid:
        ok_count = 0
        errors = []
        for k in range(n_per_point):
            try:
                clip = generate_clip(duration_s=duration_s, fs=16000.0, snr_db=float(g), n_tonals=1,
                                      f_min=f_min, f_max=f_max, seed=seed + k)
                if clip.snr_db != float(g):
                    errors.append(f"requested {g} dB but clip.snr_db={clip.snr_db}")
                    continue
                if not np.all(np.isfinite(clip.waveform)) or len(clip.waveform) == 0:
                    errors.append("waveform is empty or contains non-finite values")
                    continue
                proc = preprocess_pipeline(clip.waveform, 16000.0)  # mirrors evaluate.py's actual usage
                if not np.all(np.isfinite(proc["S_pcen"])):
                    errors.append("S_pcen contains non-finite values")
                    continue
                ok_count += 1
            except Exception as e:
                errors.append(str(e))
        results[g] = {"ok": ok_count, "attempted": n_per_point, "errors": errors[:3]}  # cap stored errors
    return results


def check_noise_variety():
    from synthetic_signal_generator import _NOISE_PROFILE_EXPONENT
    return {name: True for name in _NOISE_PROFILE_EXPONENT}


def check_classifier_class_balance():
    metrics_path = os.path.join(ROOT, "16_Classification", "acoustic_classifier_metrics.json")
    if not os.path.exists(metrics_path):
        return None
    import json
    with open(metrics_path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config()
    print("=" * 60)
    print("TRAINING DATA COVERAGE")
    print("=" * 60)

    r = check_snr_frequency_coverage(cfg, args.n_samples, seed=args.seed)
    print(f"\nSampled {r['n_samples_checked']} clips the SAME way train.py's TonalSegDataset does "
          f"(snr_range={r['snr_range_configured']}, hard_range={r['hard_range_configured']}, "
          f"hard_prob={r['hard_prob']})\n")

    print("--- TRAINING RANGE COVERAGE (statistical -- what random sampling actually produced this run) ---")
    print("SNR (nearest configured eval-grid point)")
    for g in r["snr_eval_grid"]:
        label = f"{g:+.0f} dB"
        n = r["snr_bins_seen"].get(label, 0)
        mark = "\u2713" if n > 0 else "\u2717 NOT SEEN"
        print(f"  {label:>8s}   {mark:<12s}  (n={n})")
    print("  NOTE: a bin showing low/zero n here reflects chance in THIS run's random sampling, not")
    print("  necessarily a gap in the configured range -- see EXACT EVALUATION GRID COVERAGE below for")
    print("  a deliberate, non-statistical check of the same SNR points.")

    print("\nFREQUENCY BANDS (actual tonal.f0 from generated clips)")
    for band, n in r["freq_bands_seen"].items():
        mark = "\u2713" if n > 0 else "\u2717 NOT SEEN"
        print(f"  {band[0]:>5d}-{band[1]:<5d} Hz   {mark:<12s}  (n={n})")
    if r["n_below_freq_range"] or r["n_above_freq_range"]:
        print(f"  NOTE: {r['n_below_freq_range']} below configured range, {r['n_above_freq_range']} above")

    print("\nTONAL CHARACTERISTIC VARIETY (observed in this sample, not just capability)")
    print(f"  {'zero-tonal negatives':<28s} {'OK' if r['n_zero_tonal_negatives']>0 else 'NOT SEEN'}  (n={r['n_zero_tonal_negatives']})")
    print(f"  {'harmonics present':<28s} {'OK' if r['n_harmonics_seen']>0 else 'NOT SEEN'}  (n={r['n_harmonics_seen']})")
    print(f"  {'frequency drift present':<28s} {'OK' if r['n_drift_seen']>0 else 'NOT SEEN'}  (n={r['n_drift_seen']})")
    print(f"  {'intermittent (duty<1)':<28s} {'OK' if r['n_intermittent_seen']>0 else 'NOT SEEN'}  (n={r['n_intermittent_seen']})")
    print(f"  {'Doppler ramp present':<28s} {'OK' if r['n_doppler_seen']>0 else 'WARNING: not seen this run'}  "
          f"(n={r['n_doppler_seen']}, configured doppler_prob={cfg_get(cfg, 'training.doppler_prob', 0.3)})")

    print("\nNOISE PROFILES (actual counts from generated training samples)")
    for name, n in r["noise_profile_counts"].items():
        mark = "OK" if n > 0 else "NOT SEEN"
        print(f"  {name:<28s} {mark:<10s}  (n={n})")

    print("\n--- EXACT EVALUATION GRID COVERAGE (deliberate -- not statistical) ---")
    print("Generating samples AT EACH exact SNR point (same way evaluate.py does), independent of")
    print("the random training distribution above:")
    grid_results = check_exact_evaluation_grid(cfg)
    grid_all_ok = True
    for g, res in grid_results.items():
        ok = res["ok"] == res["attempted"]
        grid_all_ok = grid_all_ok and ok
        mark = "PASS" if ok else "FAIL"
        print(f"  {g:+.0f} dB   {mark}   ({res['ok']}/{res['attempted']} generated + preprocessed cleanly)")
        for err in res["errors"]:
            print(f"      error: {err}")

    print("\nCLASSIFICATION CLASS BALANCE (from last classifier training run, not re-run here)")
    cls_metrics = check_classifier_class_balance()
    if cls_metrics is None:
        print("  NOT TRAINED YET -- run 16_Classification/acoustic_classifier.py --train first")
    else:
        print(f"  n_train={cls_metrics['n_train']}, n_test={cls_metrics['n_test']}, accuracy={cls_metrics['accuracy']:.3f}")
        for cls_name, m in cls_metrics["per_class"].items():
            print(f"  {cls_name:<15s} precision={m['precision']:.2f} recall={m['recall']:.2f} f1={m['f1']:.2f}")
        print("  NOTE: synthetic-data-only metrics -- see 16_Classification/acoustic_classifier.py docstring")

    print("\n" + "=" * 60)
    missing_snr = [f"{g:+.0f} dB" for g in r["snr_eval_grid"] if r["snr_bins_seen"].get(f"{g:+.0f} dB", 0) == 0]
    missing_freq = [f"{b[0]}-{b[1]}" for b, n in r["freq_bands_seen"].items() if n == 0]
    missing_noise = [p for p, n in r["noise_profile_counts"].items() if n == 0]
    if missing_snr or missing_freq or missing_noise or r["n_doppler_seen"] == 0 or not grid_all_ok:
        print("COVERAGE CHECK: WARNING")
        if missing_snr:
            print(f"  training-range statistical coverage missed: {missing_snr} this run "
                  f"-- likely just sampling chance (see note above); increase --n-samples to confirm")
        if missing_freq:
            print(f"  missing freq bands: {missing_freq} -- increase --n-samples or check f_min_hz/f_max_hz")
        if missing_noise:
            print(f"  missing noise profiles: {missing_noise} -- increase --n-samples or check training.noise_profiles")
        if r["n_doppler_seen"] == 0:
            print("  training data has no Doppler-shifted tonals -- check training.doppler_prob")
        if not grid_all_ok:
            print("  EXACT EVALUATION GRID COVERAGE FAILED for one or more points -- this is a real problem, "
                  "not sampling chance -- see errors printed above")
        return 1
    print("COVERAGE CHECK: PASS -- proceed to training")
    return 0


if __name__ == "__main__":
    sys.exit(main())
