"""
inference_pipeline.py
=======================
Ties every module together into the single streaming detector object used
by both the Live GUI (03_Live_Detector) and the evaluation harness
(06_Model_Evaluation):

    raw waveform chunk
        -> preprocessing (DC removal, band-pass, normalize, noise estimate)
        -> STFT + PCEN
        -> SegFormer-B0 / U-Net forward pass (probability mask)
        -> CFAR verification
        -> morphological cleanup
        -> connected component analysis
        -> frequency & harmonic estimation
        -> Kalman multi-target tracking (persistent across chunks)
        -> confidence scoring
        -> list of confirmed TonalDetection objects
"""
from __future__ import annotations
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "13_Scripts"))
sys.path.append(os.path.join(ROOT, "09_Models"))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from signal_processing import preprocess_pipeline, estimate_noise_floor_spectrum, estimate_snr_db  # noqa: E402
from segformer_tonal import segformer_b0_tonal                        # noqa: E402
from unet_tonal import UNetTonal                                      # noqa: E402
from postprocess import ca_cfar_1d, morphological_clean, connected_components  # noqa: E402
from freq_estimation import estimate_tonal_params, group_harmonics, compute_confidence  # noqa: E402
from kalman_tracker import MultiTonalTracker                          # noqa: E402
from config_loader import load_config, get as cfg_get                 # noqa: E402

CFG = load_config()


def _default_cfar_kwargs() -> dict:
    return {"num_train": cfg_get(CFG, "cfar.num_train", 16),
            "num_guard": cfg_get(CFG, "cfar.num_guard", 4),
            "pfa": cfg_get(CFG, "cfar.pfa", 1e-2)}


def _default_morph_kwargs() -> dict:
    open_size = cfg_get(CFG, "morphology.open_size", [2, 2])
    close_size = cfg_get(CFG, "morphology.close_size", [3, 3])
    return {"open_size": tuple(open_size), "close_size": tuple(close_size),
            "min_area": cfg_get(CFG, "morphology.min_area", 4),
            "min_elongation": cfg_get(CFG, "morphology.min_elongation", 1.3)}


def _default_tracker_kwargs() -> dict:
    return {"gate_hz": cfg_get(CFG, "tracker.gate_hz", 15.0),
            "confirm_hits": cfg_get(CFG, "tracker.confirm_hits", 3),
            "max_misses": cfg_get(CFG, "tracker.max_misses", 4),
            "process_var": cfg_get(CFG, "tracker.process_var", 0.5),
            "meas_var": cfg_get(CFG, "tracker.meas_var", 4.0)}


@dataclass
class TonalDetection:
    frequency_hz: float
    confidence: float
    snr_db: float
    track_id: int
    track_hits: int
    bandwidth_hz: float = 0.0
    n_harmonics: int = 0
    harmonic_freqs_hz: list = field(default_factory=list)


def load_model(checkpoint_path: str, device="cpu"):
    """
    Load a trained model for inference. Supports:
      - a regular PyTorch training checkpoint (.pt with a "state_dict" dict, as
        written by 01_Training_Module/train.py)
      - a TorchScript archive (.torchscript.pt / .ts, as written by
        07_Deployment_Module/export_torchscript.py) -- these are already-traced
        graphs, so they're simply loaded and returned ready for inference; no
        training code is imported in this path (see deploy.py).
    Returns (model, config_dict). config_dict is {} for TorchScript archives,
    since they don't carry the training args dict that .pt checkpoints do.
    """
    ext = os.path.splitext(checkpoint_path)[1].lower()
    if ext in (".ts",) or "torchscript" in os.path.basename(checkpoint_path).lower():
        model = torch.jit.load(checkpoint_path, map_location=device)
        model.eval()
        return model, {}

    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, torch.jit.ScriptModule):  # defensive: a TorchScript file with a misleading name
        ckpt.eval()
        return ckpt, {}
    name = ckpt.get("model_name", "segformer")
    model = segformer_b0_tonal() if name == "segformer" else UNetTonal()
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt.get("config", {})


class TonalDetectorPipeline:
    def __init__(self, model: torch.nn.Module,
                 fs: float = None, n_fft: int = None,
                 hop_length: int = None, device: str = "cpu",
                 seg_thresh: float = None, use_cfar: bool = True,
                 cfar_kwargs: dict | None = None, morph_kwargs: dict | None = None,
                 tracker_kwargs: dict | None = None):
        # Unset (None) numeric args fall back to config.yaml, so the pipeline
        # is config-driven by default but still fully overridable by callers
        # (GUI, evaluate.py, deploy.py) for experiments / sweeps.
        fs = fs if fs is not None else cfg_get(CFG, "signal.sampling_rate_hz", 16000.0)
        n_fft = n_fft if n_fft is not None else cfg_get(CFG, "signal.fft_size", 1024)
        hop_length = hop_length if hop_length is not None else cfg_get(CFG, "signal.hop_length", 256)
        seg_thresh = seg_thresh if seg_thresh is not None else cfg_get(CFG, "model.confidence_threshold", 0.5)

        try:
            self.model = model.to(device).eval()
        except AttributeError:
            # TorchScript modules loaded via torch.jit.load already support .eval()/.to()
            self.model = model.to(device)
            self.model.eval()
        self.device = device
        self.fs = fs
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.seg_thresh = seg_thresh
        self.use_cfar = use_cfar
        self.cfar_kwargs = cfar_kwargs or _default_cfar_kwargs()
        self.morph_kwargs = morph_kwargs or _default_morph_kwargs()
        self.tracker = MultiTonalTracker(**(tracker_kwargs or _default_tracker_kwargs()))
        self.last_latency_s = None

    @torch.no_grad()
    def process_chunk(self, waveform: np.ndarray) -> dict:
        """
        Run the full chain on one waveform chunk (e.g. a few seconds of
        hydrophone audio) and return a dict with intermediate products
        (for GUI visualisation) plus the list of confirmed TonalDetections.
        """
        t0 = time.time()
        proc = preprocess_pipeline(waveform, self.fs, n_fft=self.n_fft, hop_length=self.hop_length)
        S_mag, S_pcen = proc["S_mag"], proc["S_pcen"]

        lo, hi = np.percentile(S_pcen, 1), np.percentile(S_pcen, 99)
        img = np.clip((S_pcen - lo) / (hi - lo + 1e-8), 0, 1).astype(np.float32)

        x = torch.from_numpy(img[None, None, ...]).to(self.device)
        logits = self.model(x)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
        seg_mask = (prob > self.seg_thresh).astype(np.float32)

        if self.use_cfar:
            cfar_mask = ca_cfar_1d(S_mag, valid_bin_range=proc["passband_bins"], **self.cfar_kwargs)
            combined = seg_mask * cfar_mask
        else:
            combined = seg_mask

        cleaned = morphological_clean(combined, **self.morph_kwargs)
        comps = connected_components(cleaned, weight_img=S_pcen)
        tonals = [estimate_tonal_params(c, self.n_fft, self.fs, self.hop_length) for c in comps]
        tonals = group_harmonics(tonals, tol_hz=8.0)

        # per-blob mean segmentation probability (for confidence)
        for t in tonals:
            minr, minc, maxr, maxc = t["bbox"]
            t["seg_prob"] = float(prob[minr:maxr, minc:maxc].mean()) if maxr > minr and maxc > minc else 0.0

        # per-bin noise floor (robust, low-percentile across time) -> per-tonal local SNR
        noise_floor_mag = estimate_noise_floor_spectrum(S_mag, percentile=50.0)
        for t in tonals:
            minr, minc, maxr, maxc = t["bbox"]
            bin_c = int(np.clip(round(t["freq_bin_centroid"]), 0, S_mag.shape[0] - 1))
            peak_mag = float(S_mag[minr:maxr, minc:maxc].max()) if maxr > minr and maxc > minc else float(S_mag[bin_c].max())
            floor_power = max(noise_floor_mag[bin_c] ** 2, 1e-20)
            t["snr_db"] = estimate_snr_db(peak_mag ** 2, floor_power)

        scene_snr_db = float(np.mean([t["snr_db"] for t in tonals])) if tonals else \
            estimate_snr_db(np.mean(S_mag ** 2), np.mean(noise_floor_mag ** 2))

        confirmed_tracks = self.tracker.step([t["frequency_hz"] for t in tonals])

        detections = []
        for tr in confirmed_tracks:
            match = min(tonals, key=lambda t: abs(t["frequency_hz"] - tr.frequency), default=None)
            n_harm = match["n_harmonics_found"] if match else 0
            harm_freqs = match["harmonic_freqs_hz"] if match else []
            bw = match["bandwidth_hz"] if match else 0.0
            seg_prob = match["seg_prob"] if match else 0.5
            area = match["area"] if match else 10
            det_snr_db = match["snr_db"] if match else scene_snr_db
            conf = compute_confidence(seg_prob, area, det_snr_db, tr.hits, tr.variance, n_harm)
            detections.append(TonalDetection(
                frequency_hz=tr.frequency, confidence=conf, snr_db=det_snr_db,
                track_id=tr.id, track_hits=tr.hits, bandwidth_hz=bw,
                n_harmonics=n_harm, harmonic_freqs_hz=harm_freqs,
            ))

        self.last_latency_s = time.time() - t0
        return {
            "waveform": proc["x_clean"], "freqs": proc["freqs"], "times": proc["times"],
            "S_mag": S_mag, "S_pcen": S_pcen, "prob_mask": prob, "seg_mask": seg_mask,
            "cleaned_mask": cleaned, "tonals_raw": tonals, "detections": detections,
            "snr_db": scene_snr_db, "latency_s": self.last_latency_s,
        }
