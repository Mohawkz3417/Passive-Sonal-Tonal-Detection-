"""
run_real_validation.py
=========================
Real-data validation (Parts 18-30 of the freeze spec). Reuses the
EXISTING dataset registry (05_Dataset_Generator/dataset_registry.py --
already wired to config.yaml's `datasets:` section) rather than building
a second, parallel way of finding datasets. Validates real audio through
the SAME array_inference_pipeline.ArrayTonalDetector every other input
source uses -- no per-dataset AI pipeline.

Ground truth reality check for this project, as-is: public underwater
acoustic datasets (ShipsEar, DeepShip, DS3500, UATD) give file-level or
no labels, essentially never per-frequency tonal ground truth or an
array-geometry ground truth bearing. So per Part 27, this script reports:
  - segmentation/tonal metrics: NOT COMPUTED (no ground truth available)
  - bearing_error: null (these are mono recordings; DOA is not attempted)
  - classification: reported as inference output, accuracy only computed
    if you supply a --label-map for a dataset with usable file-level labels
  - qualitative pass-through: did the file load, did the pipeline run,
    what did it output -- this is what Part 42 calls a "smoke" validation,
    and it is genuinely what this class of dataset supports without manual
    annotation effort (see 00_Documentation/REAL_DATA_GUIDE.md).

Usage:
    python run_real_validation.py --smoke              # a few files per configured dataset
    python run_real_validation.py --dataset shipsear   # all files in one configured dataset
    python run_real_validation.py --input /path/to/file.wav --dataset shipsear
    python run_real_validation.py --input /path/to/file.wav --snr -5 --noise shipping
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "13_Scripts"))
sys.path.append(os.path.join(ROOT, "05_Dataset_Generator"))
sys.path.append(os.path.join(ROOT, "17_Array_Inference"))
sys.path.append(os.path.join(ROOT, "18_Contracts"))
sys.path.append(os.path.join(ROOT, "04_Signal_Generator"))
sys.path.append(os.path.join(ROOT, "07_Deployment_Module"))

from config_loader import load_config, get as cfg_get, resolve_path  # noqa: E402
from dataset_registry import list_datasets, get_dataset_dir  # noqa: E402
from array_inference_pipeline import ArrayTonalDetector  # noqa: E402
import input_adapter  # noqa: E402
from data_models import ValidationError  # noqa: E402
from kalman_tracker import MultiTonalTracker  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "real_validation")
os.makedirs(OUT_DIR, exist_ok=True)

AUDIO_EXTS = (".wav", ".WAV", ".flac", ".FLAC")


def _find_audio_files(folder: str, limit: int | None = None) -> list:
    found = []
    for root, _dirs, files in os.walk(folder):
        for f in files:
            if f.endswith(AUDIO_EXTS):
                found.append(os.path.join(root, f))
                if limit and len(found) >= limit:
                    return found
    return found


def _get_detector() -> ArrayTonalDetector:
    cfg = load_config()
    ckpt = resolve_path(cfg_get(cfg, "model.segformer_weights"))
    return ArrayTonalDetector(checkpoint_path=ckpt)


# Moved to 13_Scripts/real_audio_mixing.py (Part 24's noise injection) so the
# Signal Generator app and the Detection Backend's real-dataset analysis
# endpoint call the exact same implementation instead of each keeping a
# separate copy that could silently drift apart.
from real_audio_mixing import inject_noise_at_snr as _inject_noise_at_snr  # noqa: E402


def validate_file(detector: ArrayTonalDetector, path: str, dataset_name: str,
                   requested_snr_db: float | None = None, noise_type: str | None = None) -> dict:
    """
    `detector` is reused across files (its loaded model weights are
    expensive to reload), but its TRACKER is reset to a fresh instance at
    the top of every call -- NOT left shared/persistent across files.

    BUG FIX (same class of bug previously found and fixed in evaluate.py's
    run_detection_eval -- see TRAINING_AND_EVAL_REPORT.md Section 1, Bug B):
    this used to call one already-built detector once per file across the
    whole batch with no tracker reset. Because MultiTonalTracker persists
    hits/misses across calls, a track confirmed on file N could carry over
    and silently produce `detected=True` on file N+1 even if file N+1's
    content had nothing to do with it -- especially likely here since files
    in the same class/sub-folder (e.g. ShipsEar's `3_1/*.wav`) are often
    spectrally similar. For real recordings with no ground truth, that
    cross-file leakage would be actively misleading, not just imprecise.

    Fix: give this file a fresh tracker (same confirm_hits/gate/etc.
    parameters, zero accumulated state), and feed it the SAME file's frame
    `confirm_hits` times in a row -- this measures "would this file's own
    content, if it were a continuously-present signal (which is what the
    tracker's design assumes), get confirmed" -- a file-local, leakage-free
    question -- rather than "did some earlier unrelated file's track happen
    to still be alive."
    """
    old_tracker = detector.tonal_pipeline.tracker
    detector.tonal_pipeline.tracker = MultiTonalTracker(
        gate_hz=old_tracker.gate_hz, confirm_hits=old_tracker.confirm_hits,
        max_misses=old_tracker.max_misses, process_var=old_tracker.process_var,
        meas_var=old_tracker.meas_var)
    detector.last_full_result = None
    row = {
        "dataset": dataset_name, "file": path, "original_format": os.path.splitext(path)[1],
        "sample_rate_hz": None, "channels": None, "duration_s": None,
        "source_label": "UNMAPPED", "mapped_label": "UNMAPPED",
        "noise_condition": noise_type or "none (clean real recording)",
        "requested_snr_db": requested_snr_db, "measured_snr_db": None,
        "detected": None, "frequency_hz": None, "bearing_deg": None,
        "classification": None, "confidence": None, "track_id": None, "track_status": None,
        "latency_ms": None, "raw_blobs_found": None, "max_segmentation_probability": None,
        "notes": "",
    }
    try:
        import soundfile as sf
        info = sf.info(path)
        row["sample_rate_hz"] = info.samplerate
        row["channels"] = info.channels
        row["duration_s"] = round(info.frames / info.samplerate, 2)

        t0 = time.time()
        if not path.lower().endswith(".wav"):
            row["notes"] = "FLAC not read by this script's adapter path (wav_to_frame is WAV-specific); skipped"
            return row

        if requested_snr_db is not None and noise_type is not None:
            # ACTUALLY mix in the requested noise at the requested SNR (Part 24) --
            # read raw samples ourselves rather than going through wav_to_frame
            # first, since that adapter has no noise-injection step (by design --
            # it's a format adapter, not a validation-experiment tool).
            raw, fs = sf.read(path, always_2d=True)  # (n_samples, n_channels)
            channels = raw.T.astype(np.float64)
            mixed = np.stack([_inject_noise_at_snr(channels[c], fs, requested_snr_db, noise_type, seed=c)
                               for c in range(channels.shape[0])])
            from data_models import ArrayFrame, MonoFrame
            frame = (MonoFrame(samples=mixed[0], sample_rate_hz=float(fs), source="wav_mono")
                     if mixed.shape[0] == 1 else
                     ArrayFrame(samples=mixed, sample_rate_hz=float(fs), n_channels=mixed.shape[0], source="wav"))
        else:
            frame = input_adapter.wav_to_frame(path)

        # Feed this file's OWN frame to its OWN fresh tracker `confirm_hits`
        # times in a row (see docstring) instead of a single call on a
        # detector shared across the whole batch.
        confirm_hits = detector.tonal_pipeline.tracker.confirm_hits
        result = None
        for _ in range(confirm_hits):
            result = detector.process_frame(frame, requested_snr_db=requested_snr_db)
        row["latency_ms"] = round((time.time() - t0) * 1000, 1)
        row["detected"] = result.detected
        row["frequency_hz"] = result.frequency_hz
        row["bearing_deg"] = result.estimated_bearing_deg  # null for mono real recordings, correctly
        row["measured_snr_db"] = result.measured_snr_db
        row["classification"] = result.classification
        row["confidence"] = result.classification_confidence
        row["track_id"] = result.track_id
        row["track_status"] = result.track_status
        raw = detector.last_full_result
        row["raw_blobs_found"] = len(raw["tonals_raw"]) if raw else None
        row["max_segmentation_probability"] = float(np.max(raw["prob_mask"])) if raw else None
        row["notes"] = "; ".join(result.warnings) if result.warnings else "OK"
    except ValidationError as e:
        row["notes"] = f"VALIDATION_ERROR[{e.code}]: {e}"
    except Exception as e:
        row["notes"] = f"ERROR: {e}"
    return row


def run_smoke(files_per_dataset: int = 3):
    configured = list_datasets()
    print("Configured datasets (from config.yaml 'datasets:' + dataset_registry.py):")
    for name, path in configured.items():
        if name == "synthetic":
            print(f"  {name:<12s} -> built-in (always available, not a real dataset)")
            continue
        exists = path and os.path.isdir(path)
        print(f"  {name:<12s} {'-> ' + path if path else '(not configured)'}"
              f"{'  [EXISTS]' if exists else '  [MISSING]' if path else ''}")

    real_datasets = {k: v for k, v in configured.items() if k != "synthetic" and v and os.path.isdir(v)}
    if not real_datasets:
        print("\nNo real datasets are configured with an existing local folder.")
        print("This is an honest result, not a failure of this script: 0 of the datasets in "
              "config.yaml's 'datasets:' section (deepship/shipsear/ds3500/uatd/npol) have a path set, "
              "or the configured path doesn't exist on this machine.")
        print("To validate against real data: edit config.yaml -> datasets.<name>: <path to your folder>, "
              "then re-run with --smoke or --dataset <name>.")
        summary = {"datasets_configured": list(configured.keys()), "datasets_with_local_data": [],
                   "n_files_validated": 0, "note": "no real datasets available on this machine"}
        with open(os.path.join(OUT_DIR, "validation_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        return summary

    detector = _get_detector()
    rows = []
    for name, folder in real_datasets.items():
        files = _find_audio_files(folder, limit=files_per_dataset)
        print(f"\n{name}: {len(files)} representative file(s)")
        for path in files:
            row = validate_file(detector, path, name)
            rows.append(row)
            print(f"  {os.path.basename(path):<40s} detected={row['detected']} notes={row['notes']}")

    _write_report(rows)
    return {"datasets_with_local_data": list(real_datasets.keys()), "n_files_validated": len(rows)}


def _write_report(rows: list):
    if not rows:
        return
    results_path = os.path.join(OUT_DIR, "validation_results.csv")
    with open(results_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = {
        "n_files": len(rows),
        "n_detected": sum(1 for r in rows if r["detected"]),
        "datasets": sorted(set(r["dataset"] for r in rows)),
        "note": "Segmentation-level metrics (precision/recall/F1/IoU) are NOT computed here -- "
                "these datasets do not ship per-frequency tonal ground truth. bearing_deg is null "
                "for all rows because these are mono recordings (no array). This is a qualitative "
                "pipeline-execution validation, not a statistical performance benchmark -- see module docstring.",
    }
    with open(os.path.join(OUT_DIR, "validation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {results_path} and validation_summary.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2-5 representative files per configured dataset")
    ap.add_argument("--dataset", type=str, default=None, help="validate every file in one configured dataset")
    ap.add_argument("--input", type=str, default=None, help="validate a single file")
    ap.add_argument("--snr", type=float, default=None, help="reserved for future real+noise mixing (Part 25/29)")
    ap.add_argument("--noise", type=str, default=None)
    ap.add_argument("--files-per-dataset", type=int, default=3)
    args = ap.parse_args()

    if args.input:
        detector = _get_detector()
        row = validate_file(detector, args.input, args.dataset or "adhoc",
                             requested_snr_db=args.snr, noise_type=args.noise)
        print(json.dumps(row, indent=2, default=str))
        _write_report([row])
        return

    if args.dataset:
        folder = get_dataset_dir(args.dataset)
        if folder is None or not os.path.isdir(folder):
            print(f"Dataset '{args.dataset}' is not configured with an existing local folder. "
                  f"Set config.yaml -> datasets.{args.dataset}: <path>.")
            return 1
        detector = _get_detector()
        # BUG FIX: --files-per-dataset was silently ignored here (only honored by
        # run_smoke()'s default path), so `--dataset X --files-per-dataset 20`
        # would walk and validate EVERY file in X regardless of the flag -- fine
        # for a small folder, but a real dataset can have thousands of files and
        # this would run for a very long time without the limit actually applying.
        # None means "no cap" (the old default behavior, still available
        # explicitly via --files-per-dataset 0).
        limit = args.files_per_dataset if args.files_per_dataset and args.files_per_dataset > 0 else None
        files = _find_audio_files(folder, limit=limit)
        print(f"Validating {len(files)} file(s) from '{args.dataset}'"
              f"{' (capped by --files-per-dataset)' if limit else ' (ALL files -- no cap)'}...", flush=True)
        rows = []
        for i, p in enumerate(files):
            rows.append(validate_file(detector, p, args.dataset))
            if (i + 1) % 10 == 0 or (i + 1) == len(files):
                print(f"  ...{i + 1}/{len(files)} done", flush=True)
        _write_report(rows)
        return

    # default: smoke test
    run_smoke(files_per_dataset=args.files_per_dataset)


if __name__ == "__main__":
    sys.exit(main() or 0)
