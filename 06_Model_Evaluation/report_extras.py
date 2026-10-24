"""
report_extras.py
==================
Additional reports called for by the suite spec, layered on top of
evaluate.py's existing pixel-level / detection-level evaluation:

    - Confusion matrix (pixel-level, at the operating threshold)
    - ROC curve (pixel-level, threshold-swept)
    - Precision-Recall curve (pixel-level, threshold-swept)
    - False Alarm Rate / Missed Detection Rate explicit summary
      (detection-level, i.e. after the full CFAR+morphology+CCA+Kalman chain)

Kept separate from evaluate.py so the already-validated pixel/detection
sweep loops there are untouched; this module only adds new outputs
written to the shared `reports/` directory (see config.yaml -> reports.*).
"""
from __future__ import annotations
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def collect_flat_pixel_predictions(model, make_sample_fn, snr_grid, n_per_snr,
                                    device="cpu", n_fft=1024, hop_length=256):
    """Run the model over the same sweep evaluate.py uses, but keep every
    pixel's (probability, ground-truth) pair, flattened, for ROC/PR/confusion-matrix.
    Downsamples per-image (every 4th pixel) to keep memory bounded on long sweeps."""
    import torch
    probs_all, targets_all = [], []
    with torch.no_grad():
        for snr in snr_grid:
            for i in range(n_per_snr):
                img, mask, _ = make_sample_fn(snr_db=snr, n_tonals=1, n_fft=n_fft,
                                               hop_length=hop_length, seed=30_000 + int(snr * 7) + i)
                x = torch.from_numpy(img[None]).to(device)
                logits = model(x)
                prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
                probs_all.append(prob.flatten()[::4])
                targets_all.append(mask[0].flatten()[::4])
    return np.concatenate(probs_all), np.concatenate(targets_all)


def confusion_matrix_report(probs: np.ndarray, targets: np.ndarray, out_dir: str, thresh: float = 0.5):
    from sklearn.metrics import confusion_matrix
    pred = (probs > thresh).astype(int)
    cm = confusion_matrix(targets.astype(int), pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                 color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["background", "tonal"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["background", "tonal"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Ground truth")
    ax.set_title(f"Pixel-level Confusion Matrix (thresh={thresh})")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=140)
    plt.close(fig)

    with open(os.path.join(out_dir, "confusion_matrix.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["", "pred_background", "pred_tonal"])
        w.writerow(["true_background", tn, fp])
        w.writerow(["true_tonal", fn, tp])
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def roc_pr_report(probs: np.ndarray, targets: np.ndarray, out_dir: str):
    from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
    fpr, tpr, _ = roc_curve(targets.astype(int), probs)
    prec, rec, _ = precision_recall_curve(targets.astype(int), probs)
    auc = roc_auc_score(targets.astype(int), probs)
    ap = average_precision_score(targets.astype(int), probs)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="tab:blue", label=f"ROC (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.8)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Pixel-level ROC Curve"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rec, prec, color="tab:orange", label=f"PR (AP={ap:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Pixel-level Precision-Recall Curve"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "pr_curve.png"), dpi=140); plt.close(fig)

    with open(os.path.join(out_dir, "roc_pr_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["roc_auc", auc])
        w.writerow(["average_precision", ap])
    return {"roc_auc": float(auc), "average_precision": float(ap)}


def false_alarm_missed_detection_report(det_rows: list[dict], out_dir: str):
    """Re-expresses evaluate.py's Pd/Pfa detection sweep with the explicit
    'False Alarm Rate' / 'Missed Detection Rate' naming called for in the spec."""
    rows = []
    for r in det_rows:
        rows.append({
            "snr_db": r["snr_db"],
            "false_alarm_rate_per_clip": r["pfa_per_clip"],
            "missed_detection_rate": 1.0 - r["pd"],
            "probability_of_detection": r["pd"],
            "freq_err_hz": r["freq_err_hz"],
            "n": r["n"],
        })
    path = os.path.join(out_dir, "false_alarm_missed_detection.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    fig, ax1 = plt.subplots(figsize=(6, 4))
    snrs = [r["snr_db"] for r in rows]
    ax1.plot(snrs, [r["missed_detection_rate"] for r in rows], "o-", color="tab:red",
             label="Missed Detection Rate")
    ax1.set_xlabel("SNR (dB)"); ax1.set_ylabel("Missed Detection Rate", color="tab:red")
    ax1.axvspan(-10, -7, color="red", alpha=0.06)
    ax2 = ax1.twinx()
    ax2.plot(snrs, [r["false_alarm_rate_per_clip"] for r in rows], "s--", color="tab:purple",
             label="False Alarm Rate (per clip)")
    ax2.set_ylabel("False Alarm Rate (per clip)", color="tab:purple")
    plt.title("False Alarm Rate & Missed Detection Rate vs SNR")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "false_alarm_missed_detection.png"), dpi=140)
    plt.close(fig)
    return path


def training_curves_report(history_json_path: str, out_dir: str):
    """Plot training loss / validation metrics vs epoch from a *_history.json
    file written by 01_Training_Module/train.py, if one exists."""
    import json
    if not os.path.exists(history_json_path):
        return None
    with open(history_json_path) as f:
        hist = json.load(f)
    if not hist:
        return None
    epochs = [h["epoch"] for h in hist]

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(epochs, [h["train_loss"] for h in hist], color="tab:blue", label="train loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss", color="tab:blue")
    ax2 = ax1.twinx()
    for key, color in (("f1", "tab:green"), ("precision", "tab:orange"), ("recall", "tab:red")):
        if key in hist[0]:
            ax2.plot(epochs, [h.get(key, np.nan) for h in hist], color=color, label=f"val {key}")
    ax2.set_ylabel("Validation score")
    fig.legend(loc="upper right", fontsize=8)
    plt.title("Training Curves")
    fig.tight_layout()
    out_path = os.path.join(out_dir, "training_curves.png")
    fig.savefig(out_path, dpi=140); plt.close(fig)
    return out_path
