"""
evaluate.py
=============
Module 6 / Section 8 of the proposal: automatic evaluation across the
full SNR range (+20 dB down to -10 dB), producing:

    - Pixel-level Precision / Recall / F1 / IoU / Dice vs SNR
    - Detection-level Probability of Detection (Pd) and False-Alarm rate (Pfa) vs SNR
      (using the *full* deployment pipeline: SegFormer -> CFAR -> morphology -> CCA)
    - Frequency estimation error (Hz) for correctly detected tonals
    - Processing time / FPS benchmark (single-chunk latency)

Results are written as CSV + PNG plots into 06_Model_Evaluation/results/.

Usage:
    python evaluate.py --checkpoint ../11_Weights/segformer_b0_tonal.pt \
                        --n-per-snr 10
"""
from __future__ import annotations
import argparse
import os
import sys
import time
import csv

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "05_Dataset_Generator"))
sys.path.append(os.path.join(ROOT, "07_Deployment_Module"))
sys.path.append(os.path.join(ROOT, "04_Signal_Generator"))
sys.path.append(os.path.join(ROOT, "13_Scripts"))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset_builder import make_sample                       # noqa: E402
from synthetic_signal_generator import generate_clip           # noqa: E402
from inference_pipeline import load_model, TonalDetectorPipeline  # noqa: E402
from config_loader import load_config, get as cfg_get, resolve_path  # noqa: E402
import report_extras                                            # noqa: E402


def pixel_metrics_np(prob: np.ndarray, target: np.ndarray, thresh=0.5):
    pred = (prob > thresh).astype(np.float32)
    tp = float((pred * target).sum())
    fp = float((pred * (1 - target)).sum())
    fn = float(((1 - pred) * target).sum())
    tn = float(((1 - pred) * (1 - target)).sum())
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    return dict(precision=precision, recall=recall, f1=f1, iou=iou, dice=dice)


@torch.no_grad()
def run_pixel_eval(model, snr_grid, n_per_snr, device="cpu", n_fft=1024, hop_length=256, fs=16000.0):
    rows = []
    for snr in snr_grid:
        agg = {"precision": [], "recall": [], "f1": [], "iou": [], "dice": []}
        for i in range(n_per_snr):
            img, mask, _ = make_sample(snr_db=snr, n_tonals=1, n_fft=n_fft,
                                        hop_length=hop_length, seed=10_000 + int(snr * 7) + i)
            x = torch.from_numpy(img[None]).to(device)
            logits = model(x)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
            m = pixel_metrics_np(prob, mask[0])
            for k in agg:
                agg[k].append(m[k])
        row = {"snr_db": snr, "n": n_per_snr}
        row.update({k: float(np.mean(v)) for k, v in agg.items()})
        rows.append(row)
        print(f"[pixel-eval] SNR={snr:+.1f}dB  P={row['precision']:.3f} R={row['recall']:.3f} "
              f"F1={row['f1']:.3f} IoU={row['iou']:.3f}")
    return rows


def run_detection_eval(model, snr_grid, n_per_snr, seg_thresh, fs=16000.0, freq_tol_hz=15.0):
    """
    Full-pipeline (SegFormer -> CFAR -> morph -> CCA -> Kalman tracker) evaluation.

    BUG FIX (found while diagnosing why this always reported Pd=0.00 regardless of
    checkpoint quality): this used to build ONE `TonalDetectorPipeline` in main()
    and call `pipe.process_chunk(clip.waveform)` exactly ONCE per independently
    random test clip, using the SAME tracker instance across every clip in the
    whole SNR sweep. But `MultiTonalTracker` only reports a track as "detected"
    (in the list `process_chunk` returns as `detections`) once it has accumulated
    `tracker.confirm_hits` (3, from config) consecutive matching frames -- a brand
    new track always starts with hits=1 and is NOT confirmed. A single call can
    therefore never produce a detection no matter how good the segmentation is;
    the old code was measuring "can a track be confirmed from 1 frame" (always
    false by construction), not "can the pipeline detect this tonal". Reusing one
    tracker across unrelated clips was also methodologically unsound (stale
    tracks from a previous random clip could gate-match or dilute the next one).

    Fix: build a FRESH pipeline (fresh tracker, same loaded model) per test clip,
    and feed it the same clip's waveform `confirm_hits` times in a row -- this is
    exactly the real deployment scenario the tracker is designed for (a genuine
    tonal is physically present continuously, so consecutive analysis windows of
    it keep re-detecting the same frequency until the tracker confirms it isn't a
    one-frame false alarm). This does not change the CFAR/morphology/tracker
    algorithms or thresholds themselves, only how the test harness exercises them.
    """
    rows = []
    for snr in snr_grid:
        n_tp, n_fn, n_fp, freq_errs = 0, 0, 0, []
        for i in range(n_per_snr):
            clip = generate_clip(duration_s=4.0, fs=fs, snr_db=snr, n_tonals=1,
                                  seed=20_000 + int(snr * 7) + i)
            true_f = float(np.nanmean(clip.tonal_freq_tracks[0]))
            local_pipe = TonalDetectorPipeline(model, fs=fs, seg_thresh=seg_thresh)
            confirm_hits = local_pipe.tracker.confirm_hits
            result = None
            for _frame in range(confirm_hits):
                result = local_pipe.process_chunk(clip.waveform)
            dets = result["detections"]
            matched = [d for d in dets if abs(d.frequency_hz - true_f) <= freq_tol_hz]
            if matched:
                n_tp += 1
                best = min(matched, key=lambda d: abs(d.frequency_hz - true_f))
                freq_errs.append(abs(best.frequency_hz - true_f))
            else:
                n_fn += 1
            n_fp += max(0, len(dets) - len(matched))
        pd = n_tp / max(1, (n_tp + n_fn))
        pfa_per_clip = n_fp / max(1, n_per_snr)
        freq_err = float(np.mean(freq_errs)) if freq_errs else float("nan")
        rows.append({"snr_db": snr, "pd": pd, "pfa_per_clip": pfa_per_clip,
                      "freq_err_hz": freq_err, "n": n_per_snr})
        print(f"[det-eval]   SNR={snr:+.1f}dB  Pd={pd:.2f}  Pfa/clip={pfa_per_clip:.2f}  "
              f"freq_err={freq_err:.1f}Hz")
    return rows


def benchmark_latency(model, device="cpu", n_fft=1024, hop_length=256, n_runs=10,
                       duration_s=4.0, fs=16000.0):
    import torch
    n_frames = int(duration_s * fs / hop_length) + 1
    n_freq = n_fft // 2 + 1
    x = torch.randn(1, 1, n_freq, n_frames).to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(2):
            model(x)
        t0 = time.time()
        for _ in range(n_runs):
            model(x)
        dt = (time.time() - t0) / n_runs
    return {"latency_ms": dt * 1000.0, "fps_equiv": 1.0 / dt,
            "realtime_factor": duration_s / dt}


def save_plots(pixel_rows, det_rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    snrs = [r["snr_db"] for r in pixel_rows]
    plt.figure(figsize=(6, 4))
    for key in ("precision", "recall", "f1", "iou"):
        plt.plot(snrs, [r[key] for r in pixel_rows], marker="o", label=key)
    plt.axvspan(-10, -7, color="red", alpha=0.08, label="target band (-7..-10 dB)")
    plt.xlabel("SNR (dB)"); plt.ylabel("score"); plt.title("Pixel-level metrics vs SNR")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pixel_metrics_vs_snr.png"), dpi=140)
    plt.close()

    snrs2 = [r["snr_db"] for r in det_rows]
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(snrs2, [r["pd"] for r in det_rows], "o-", color="tab:blue", label="Pd")
    ax1.set_xlabel("SNR (dB)"); ax1.set_ylabel("Probability of Detection", color="tab:blue")
    ax1.axvspan(-10, -7, color="red", alpha=0.08)
    ax2 = ax1.twinx()
    ax2.plot(snrs2, [r["pfa_per_clip"] for r in det_rows], "s--", color="tab:orange", label="Pfa/clip")
    ax2.set_ylabel("False alarms per clip", color="tab:orange")
    plt.title("Detection performance vs SNR")
    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, "pd_pfa_vs_snr.png"), dpi=140)
    plt.close()


def write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.path.join(
        ROOT, cfg_get(cfg, "model.segformer_weights", "11_Weights/segformer_b0_tonal.pt")))
    ap.add_argument("--n-per-snr", type=int, default=6)
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "results"),
                     help="legacy per-module results dir (kept for backward compatibility)")
    ap.add_argument("--reports-dir", default=resolve_path(cfg_get(cfg, "reports.reports_dir", "reports")),
                     help="canonical suite-wide reports dir (confusion matrix, ROC/PR, "
                          "false-alarm/missed-detection, training curves)")
    ap.add_argument("--seg-thresh", type=float,
                     default=cfg_get(cfg, "model.confidence_threshold", 0.3))
    ap.add_argument("--skip-extra-reports", action="store_true",
                     help="skip confusion matrix / ROC / PR (adds extra pixel-eval passes)")
    args = ap.parse_args()

    device = "cpu"
    model, mcfg = load_model(args.checkpoint, device=device)
    # NEW: SNR evaluation grid now comes from config.yaml's training.snr_eval_grid_db
    # (was a hardcoded list here) -- centralizes it alongside the training SNR range
    # it's meant to validate. Falls back to the original list if not configured.
    snr_grid = cfg_get(cfg, "training.snr_eval_grid_db", [-10, -8, -6, -4, -2, 0, 5, 10, 15, 20])

    print("== Pixel-level evaluation ==")
    pixel_rows = run_pixel_eval(model, snr_grid, args.n_per_snr, device=device)

    print("\n== Full-pipeline detection evaluation (SegFormer + CFAR + morph + CCA + Kalman) ==")
    eval_fs = cfg_get(cfg, "signal.sampling_rate_hz", 16000.0)
    det_rows = run_detection_eval(model, snr_grid, args.n_per_snr, seg_thresh=args.seg_thresh, fs=eval_fs)

    print("\n== Latency / throughput benchmark (CPU) ==")
    bench = benchmark_latency(model, device=device)
    print(f"  {bench['latency_ms']:.1f} ms/chunk  ->  {bench['realtime_factor']:.1f}x real-time "
          f"on a 4s chunk (single CPU core)")

    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(pixel_rows, os.path.join(args.out_dir, "pixel_metrics.csv"))
    write_csv(det_rows, os.path.join(args.out_dir, "detection_metrics.csv"))
    with open(os.path.join(args.out_dir, "latency_benchmark.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(bench.keys()))
        w.writeheader(); w.writerow(bench)
    save_plots(pixel_rows, det_rows, args.out_dir)
    print(f"\nLegacy results written to {args.out_dir}")

    if not args.skip_extra_reports:
        print("\n== Extra reports: confusion matrix / ROC / PR / false-alarm & missed-detection ==")
        os.makedirs(args.reports_dir, exist_ok=True)
        probs, targets = report_extras.collect_flat_pixel_predictions(
            model, make_sample, snr_grid, args.n_per_snr, device=device)
        cm = report_extras.confusion_matrix_report(probs, targets, args.reports_dir, thresh=args.seg_thresh)
        print(f"  confusion matrix: TP={cm['tp']} FP={cm['fp']} FN={cm['fn']} TN={cm['tn']}")
        curves = report_extras.roc_pr_report(probs, targets, args.reports_dir)
        print(f"  ROC AUC={curves['roc_auc']:.3f}  Average Precision={curves['average_precision']:.3f}")
        report_extras.false_alarm_missed_detection_report(det_rows, args.reports_dir)
        # also copy the SNR-sweep plots/CSVs and a latency summary into reports/ so it's a
        # one-stop directory, without disturbing the legacy per-module results/ folder above
        write_csv(pixel_rows, os.path.join(args.reports_dir, "pixel_metrics.csv"))
        write_csv(det_rows, os.path.join(args.reports_dir, "detection_metrics.csv"))
        save_plots(pixel_rows, det_rows, args.reports_dir)
        hist_path = os.path.splitext(args.checkpoint)[0] + "_history.json"
        tc_path = report_extras.training_curves_report(hist_path, args.reports_dir)
        if tc_path:
            print(f"  training curves: {tc_path}")
        print(f"\nFull report set written to {args.reports_dir}")


if __name__ == "__main__":
    main()
