"""
test_model_performance.py
===========================
Regression guard for the "SegFormer always outputs zero" issue that was
found and fixed in this pass:

  1. The active checkpoint used to be the model's ORIGINAL, essentially
     untrained state (~80 gradient steps, carried through many prior
     engineering passes unchanged because training was explicitly deferred).
     It produced a flat probability mask and 0.000 precision/recall/F1/IoU
     at every SNR in evaluate.py, no matter how correct the rest of the
     pipeline was.
  2. Separately, evaluate.py's full-pipeline (tracker-confirmed) detection
     metric was ALSO always zero, for an unrelated reason: it called
     TonalDetectorPipeline.process_chunk() exactly once per test clip, but
     MultiTonalTracker only reports a track as detected after
     `confirm_hits` (3) consecutive matching frames -- a single call can
     never satisfy that by construction, regardless of model quality.

Both are now fixed (see 01_Training_Module/train.py --resume,
11_Weights/active/segformer_b0_tonal.pt, and
06_Model_Evaluation/evaluate.py's run_detection_eval). These tests are
intentionally cheap (a handful of samples) -- they are not a substitute for
the full 06_Model_Evaluation/evaluate.py sweep, only a fast tripwire so a
future change can't silently reintroduce either failure mode.
"""
from __future__ import annotations
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("13_Scripts", "05_Dataset_Generator", "07_Deployment_Module",
            "09_Models", "04_Signal_Generator"):
    sys.path.append(os.path.join(ROOT, sub))

from config_loader import load_config, get as cfg_get, resolve_path  # noqa: E402
from dataset_builder import make_sample                                # noqa: E402
from inference_pipeline import load_model, TonalDetectorPipeline       # noqa: E402
from synthetic_signal_generator import generate_clip                   # noqa: E402

CFG = load_config()
CKPT = resolve_path(cfg_get(CFG, "model.segformer_weights"))

# The original never-trained checkpoint's exact weights, for a "did this
# regress back to the shipped-but-untrained state" tripwire. Kept as a
# constant rather than re-hashing the archived file so this test doesn't
# depend on that archive path continuing to exist.
_ORIGINAL_UNTRAINED_MD5 = "49fed1a79409d548401c235e8d6c9d0c"


def _file_md5(path: str) -> str:
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def test_active_checkpoint_is_not_the_original_untrained_weights():
    assert os.path.isfile(CKPT), f"active checkpoint missing at {CKPT}"
    assert _file_md5(CKPT) != _ORIGINAL_UNTRAINED_MD5, (
        "active checkpoint is byte-identical to the original ~80-step, "
        "never-really-trained weights -- training regressed")


def test_active_checkpoint_achieves_nontrivial_pixel_f1():
    """Cheap smoke check: a handful of samples at a middling SNR should
    give a clearly-better-than-random pixel F1. This is not the full
    evaluate.py sweep, just a fast regression tripwire."""
    device = "cpu"
    model, _ = load_model(CKPT, device=device)
    model.eval()
    n_fft, hop = 1024, 256
    tp = fp = fn = 0
    with torch.no_grad():
        for i in range(6):
            img, mask, _clip = make_sample(duration_s=4.0, fs=16000.0, snr_db=-6.0, n_tonals=1,
                                            n_fft=n_fft, hop_length=hop, seed=70_000 + i)
            x = torch.from_numpy(img[None]).to(device)
            logits = model(x)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
            target = mask[0]
            pred = (prob > 0.5).astype(np.float32)
            tp += float((pred * target).sum())
            fp += float((pred * (1 - target)).sum())
            fn += float(((1 - pred) * target).sum())
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    assert f1 > 0.3, (
        f"pixel F1={f1:.3f} at -6dB is at/near the zero-signal floor -- "
        f"the checkpoint may have reverted to an untrained state")


def test_full_pipeline_confirms_a_persistent_tonal_within_confirm_hits_frames():
    """The tracker requires `confirm_hits` consecutive matching frames
    before it reports a detection (by design, to reject one-frame false
    alarms) -- this must NOT be mistaken for the model not working. Feed a
    fresh pipeline the same persistent clip `confirm_hits` times, exactly
    as 06_Model_Evaluation/evaluate.py's run_detection_eval and
    13_Scripts/system_health_check.py's integration checks now do."""
    device = "cpu"
    model, _ = load_model(CKPT, device=device)
    fs = cfg_get(CFG, "signal.sampling_rate_hz", 16000.0)
    clip = generate_clip(duration_s=4.0, fs=fs, snr_db=-6.0, n_tonals=1, seed=71_000)
    true_f = float(np.nanmean(clip.tonal_freq_tracks[0]))

    pipe = TonalDetectorPipeline(model, fs=fs)
    confirm_hits = pipe.tracker.confirm_hits
    result = None
    for _ in range(confirm_hits):
        result = pipe.process_chunk(clip.waveform)

    dets = result["detections"]
    matched = [d for d in dets if abs(d.frequency_hz - true_f) <= 15.0]
    assert matched, (
        f"expected a confirmed detection near {true_f:.1f} Hz after "
        f"{confirm_hits} consecutive frames of a persistent tonal, got "
        f"detections={[d.frequency_hz for d in dets]}")
