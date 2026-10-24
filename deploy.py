"""
deploy.py
==========
Deployment entry point: run the trained tonal detector standalone, with
NO dependency on 01_Training_Module, 05_Dataset_Generator's training
path, or losses.py -- only the inference-side modules (07_Deployment_Module,
09_Models, 13_Scripts) are imported. This is the script you hand to
someone who just needs to *run* the detector on a laptop, not train it.

Supports both deployment artifact families:
    - PyTorch checkpoint / TorchScript archive  -> runs through
      07_Deployment_Module/inference_pipeline.py (full pipeline: DSP + CFAR
      + morphology + CCA + Kalman tracking + confidence, exactly like the
      Live GUI).
    - ONNX (FP32 or INT8)                        -> runs the SegFormer/U-Net
      forward pass through onnxruntime for the speed/footprint benefit;
      still uses the same 13_Scripts/07_Deployment_Module DSP and
      post-processing code around it.

Usage:
    # auto-picks the best available artifact in 11_Weights/ and opens the GUI
    python deploy.py

    # explicit model + headless (no GUI, just prints detections + logs them)
    python deploy.py --checkpoint 11_Weights/segformer_b0_tonal.torchscript.pt --headless \
                      --source synthetic --snr-db -8

    # ONNX INT8 deployment, headless
    python deploy.py --checkpoint 11_Weights/segformer_b0_tonal_int8.onnx --headless
"""
from __future__ import annotations
import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT, "13_Scripts"))
sys.path.append(os.path.join(ROOT, "09_Models"))
sys.path.append(os.path.join(ROOT, "07_Deployment_Module"))
sys.path.append(os.path.join(ROOT, "04_Signal_Generator"))
sys.path.append(os.path.join(ROOT, "08_Ethernet_Receiver"))

from config_loader import load_config, get as cfg_get, resolve_path  # noqa: E402
from model_manager import discover_models, best_deployment_artifact, ModelEntry  # noqa: E402
from log_utils import DetectionLogger                                  # noqa: E402


class OnnxTonalRunner:
    """Minimal ONNX-backed stand-in exposing the same .process_chunk() contract
    as TonalDetectorPipeline, for ONNX (FP32/INT8) deployment artifacts."""

    def __init__(self, onnx_path: str, fs: float = None, n_fft: int = None, hop_length: int = None,
                 seg_thresh: float = None, use_cfar: bool = True,
                 cfar_kwargs: dict | None = None, morph_kwargs: dict | None = None,
                 tracker_kwargs: dict | None = None):
        import onnxruntime as ort
        from signal_processing import preprocess_pipeline, estimate_noise_floor_spectrum, estimate_snr_db
        from postprocess import ca_cfar_1d, morphological_clean, connected_components
        from freq_estimation import estimate_tonal_params, group_harmonics, compute_confidence
        from kalman_tracker import MultiTonalTracker
        from inference_pipeline import (TonalDetection, _default_cfar_kwargs,
                                         _default_morph_kwargs, _default_tracker_kwargs)

        cfg = load_config()
        self.fs = fs or cfg_get(cfg, "signal.sampling_rate_hz", 16000.0)
        self.n_fft = n_fft or cfg_get(cfg, "signal.fft_size", 1024)
        self.hop_length = hop_length or cfg_get(cfg, "signal.hop_length", 256)
        self.seg_thresh = seg_thresh if seg_thresh is not None else cfg_get(cfg, "model.confidence_threshold", 0.5)
        self.use_cfar = use_cfar
        self.cfar_kwargs = cfar_kwargs or _default_cfar_kwargs()
        self.morph_kwargs = morph_kwargs or _default_morph_kwargs()
        self.tracker = MultiTonalTracker(**(tracker_kwargs or _default_tracker_kwargs()))
        self.last_latency_s = None

        self._preprocess_pipeline = preprocess_pipeline
        self._estimate_noise_floor_spectrum = estimate_noise_floor_spectrum
        self._estimate_snr_db = estimate_snr_db
        self._ca_cfar_1d = ca_cfar_1d
        self._morphological_clean = morphological_clean
        self._connected_components = connected_components
        self._estimate_tonal_params = estimate_tonal_params
        self._group_harmonics = group_harmonics
        self._compute_confidence = compute_confidence
        self._TonalDetection = TonalDetection

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])
        self._input_name = self.sess.get_inputs()[0].name

    def process_chunk(self, waveform: np.ndarray) -> dict:
        t0 = time.time()
        proc = self._preprocess_pipeline(waveform, self.fs, n_fft=self.n_fft, hop_length=self.hop_length)
        S_mag, S_pcen = proc["S_mag"], proc["S_pcen"]

        lo, hi = np.percentile(S_pcen, 1), np.percentile(S_pcen, 99)
        img = np.clip((S_pcen - lo) / (hi - lo + 1e-8), 0, 1).astype(np.float32)

        x = img[None, None, ...]
        try:
            logits = self.sess.run(None, {self._input_name: x})[0]
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "Got invalid dimensions" in msg or "InvalidArgument" in msg:
                raise RuntimeError(
                    "ONNX model has a fixed input shape from export time (see "
                    "export_onnx.py/--n-freq/--n-frames), but this chunk's spectrogram is a "
                    f"different size. Either use --chunk-s to match the shape the ONNX model was "
                    f"exported with (default 4.0s @ 16kHz -> 513x247), or re-export with "
                    f"export_onnx.py --n-frames <value> for your chunk size. Original error: {msg}"
                ) from e
            raise
        prob = 1.0 / (1.0 + np.exp(-logits))[0, 0]
        seg_mask = (prob > self.seg_thresh).astype(np.float32)

        if self.use_cfar:
            cfar_mask = self._ca_cfar_1d(S_mag, valid_bin_range=proc["passband_bins"], **self.cfar_kwargs)
            combined = seg_mask * cfar_mask
        else:
            combined = seg_mask

        cleaned = self._morphological_clean(combined, **self.morph_kwargs)
        comps = self._connected_components(cleaned, weight_img=S_pcen)
        tonals = [self._estimate_tonal_params(c, self.n_fft, self.fs, self.hop_length) for c in comps]
        tonals = self._group_harmonics(tonals, tol_hz=8.0)

        for t in tonals:
            minr, minc, maxr, maxc = t["bbox"]
            t["seg_prob"] = float(prob[minr:maxr, minc:maxc].mean()) if maxr > minr and maxc > minc else 0.0

        noise_floor_mag = self._estimate_noise_floor_spectrum(S_mag, percentile=50.0)
        for t in tonals:
            minr, minc, maxr, maxc = t["bbox"]
            bin_c = int(np.clip(round(t["freq_bin_centroid"]), 0, S_mag.shape[0] - 1))
            peak_mag = float(S_mag[minr:maxr, minc:maxc].max()) if maxr > minr and maxc > minc else float(S_mag[bin_c].max())
            floor_power = max(noise_floor_mag[bin_c] ** 2, 1e-20)
            t["snr_db"] = self._estimate_snr_db(peak_mag ** 2, floor_power)

        scene_snr_db = float(np.mean([t["snr_db"] for t in tonals])) if tonals else \
            self._estimate_snr_db(np.mean(S_mag ** 2), np.mean(noise_floor_mag ** 2))

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
            conf = self._compute_confidence(seg_prob, area, det_snr_db, tr.hits, tr.variance, n_harm)
            detections.append(self._TonalDetection(
                frequency_hz=tr.frequency, confidence=conf, snr_db=det_snr_db,
                track_id=tr.id, track_hits=tr.hits, bandwidth_hz=bw,
                n_harmonics=n_harm, harmonic_freqs_hz=harm_freqs))

        self.last_latency_s = time.time() - t0
        return {"waveform": proc["x_clean"], "freqs": proc["freqs"], "times": proc["times"],
                "S_mag": S_mag, "S_pcen": S_pcen, "prob_mask": prob, "seg_mask": seg_mask,
                "cleaned_mask": cleaned, "tonals_raw": tonals, "detections": detections,
                "snr_db": scene_snr_db, "latency_s": self.last_latency_s}


def build_pipeline(entry: ModelEntry, device: str = "cpu"):
    """Build a ready-to-run pipeline object from a discovered ModelEntry,
    dispatching to the PyTorch/TorchScript path or the ONNX path."""
    if entry.kind in ("pytorch", "torchscript"):
        from inference_pipeline import load_model, TonalDetectorPipeline
        model, _ = load_model(entry.path, device=device)
        return TonalDetectorPipeline(model, device=device)
    elif entry.kind in ("onnx", "onnx_int8"):
        return OnnxTonalRunner(entry.path)
    raise ValueError(f"unsupported artifact kind for deployment: {entry.kind}")


def run_headless(pipeline, source, n_chunks: int, chunk_s: float, logger: DetectionLogger):
    print(f"[deploy] running headless -- {n_chunks} chunk(s) of {chunk_s:.1f}s each "
          f"(Ctrl+C to stop early)\n")
    try:
        for i in range(n_chunks):
            chunk = source.next_chunk()
            if chunk is None:
                print("[deploy] no data from source, stopping.")
                break
            result = pipeline.process_chunk(chunk)
            cpu_pct, ram_mb = _resource_usage()
            for d in result["detections"]:
                print(f"  chunk {i:04d}  f={d.frequency_hz:8.1f} Hz  conf={d.confidence:5.1f}%  "
                      f"SNR={d.snr_db:6.1f} dB  harmonics={d.n_harmonics}  "
                      f"latency={result['latency_s'] * 1000:.1f} ms")
                logger.log_detection(frequency_hz=d.frequency_hz, confidence_pct=d.confidence,
                                      snr_db=d.snr_db, bandwidth_hz=d.bandwidth_hz,
                                      n_harmonics=d.n_harmonics, track_id=d.track_id,
                                      inference_time_ms=result["latency_s"] * 1000,
                                      cpu_percent=cpu_pct)
            if not result["detections"]:
                print(f"  chunk {i:04d}  no confirmed tonals  (scene SNR est. {result['snr_db']:.1f} dB, "
                      f"latency {result['latency_s'] * 1000:.1f} ms)")
    except KeyboardInterrupt:
        print("\n[deploy] stopped by user.")


def _resource_usage():
    try:
        import psutil
        p = psutil.Process(os.getpid())
        return psutil.cpu_percent(interval=None), p.memory_info().rss / 1e6
    except ImportError:
        return None, None


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None,
                     help="explicit model artifact path (.pt / .torchscript.pt / .onnx). "
                          "If omitted, the best available artifact in 11_Weights/ is auto-selected.")
    ap.add_argument("--list-models", action="store_true", help="list discovered models and exit")
    ap.add_argument("--headless", action="store_true",
                     help="run without the Tkinter GUI -- prints/logs detections to the console + logs/")
    ap.add_argument("--n-chunks", type=int, default=20, help="headless mode: number of chunks to process")
    ap.add_argument("--source", choices=["synthetic", "wav", "ethernet"], default="synthetic")
    ap.add_argument("--wav-path", type=str, default=None)
    ap.add_argument("--snr-db", type=float, default=cfg_get(cfg, "signal_generator.snr_db", -8.0))
    ap.add_argument("--chunk-s", type=float, default=cfg_get(cfg, "gui.chunk_s", 4.0))
    ap.add_argument("--eth-host", type=str, default=cfg_get(cfg, "ethernet.host", "0.0.0.0"))
    ap.add_argument("--eth-port", type=int, default=cfg_get(cfg, "ethernet.port", 50000))
    ap.add_argument("--eth-protocol", choices=["udp", "tcp"], default=cfg_get(cfg, "ethernet.protocol", "udp"))
    args = ap.parse_args()

    if args.list_models:
        from model_manager import list_models_pretty
        print(list_models_pretty())
        return

    entry = None
    if args.checkpoint:
        from model_manager import ModelEntry
        lower = args.checkpoint.lower()
        if "torchscript" in lower or lower.endswith(".ts"):
            kind = "torchscript"
        elif lower.endswith(".onnx"):
            kind = "onnx_int8" if "int8" in lower else "onnx"
        else:
            kind = "pytorch"
        entry = ModelEntry(kind, "unknown", resolve_path(args.checkpoint),
                            os.path.getsize(resolve_path(args.checkpoint)) / 1e6
                            if os.path.exists(resolve_path(args.checkpoint)) else 0.0)
    else:
        entry = best_deployment_artifact()

    if entry is None:
        print("No deployable model found in 11_Weights/. Train one first: "
              "python main.py -> option 1, or python 01_Training_Module/train.py")
        return

    print(f"[deploy] using model: {entry.name}  (kind={entry.kind}, {entry.size_mb:.2f} MB)")
    pipeline = build_pipeline(entry)

    logger = DetectionLogger()

    if args.headless:
        sys.path.append(os.path.join(ROOT, "03_Live_Detector"))
        from live_detector import ChunkSource
        source = ChunkSource(args.source, cfg_get(cfg, "signal.sampling_rate_hz", 16000.0),
                              args.chunk_s, snr_db=args.snr_db, wav_path=args.wav_path,
                              eth_host=args.eth_host, eth_port=args.eth_port,
                              eth_protocol=args.eth_protocol)
        run_headless(pipeline, source, args.n_chunks, args.chunk_s, logger)
    else:
        # Reuse the full Live GUI, just pre-loaded with the deployment-selected model
        # (no training-module import happens on this path either).
        import tkinter as tk
        sys.path.append(os.path.join(ROOT, "03_Live_Detector"))
        from live_detector import LiveDetectorApp, ChunkSource
        source = ChunkSource(args.source, cfg_get(cfg, "signal.sampling_rate_hz", 16000.0),
                              args.chunk_s, snr_db=args.snr_db, wav_path=args.wav_path,
                              eth_host=args.eth_host, eth_port=args.eth_port,
                              eth_protocol=args.eth_protocol)
        root = tk.Tk()
        app = LiveDetectorApp(root, pipeline, source, logger.csv_path, args.chunk_s,
                               model_name=entry.name)
        root.protocol("WM_DELETE_WINDOW", lambda: (app.shutdown(), root.destroy()))
        root.mainloop()


if __name__ == "__main__":
    main()
