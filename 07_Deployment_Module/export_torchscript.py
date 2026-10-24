"""
export_torchscript.py
========================
Deployment Module: export a trained PyTorch checkpoint to TorchScript
(via tracing). A TorchScript archive is a self-contained, serialized
graph -- `deploy.py` and `inference_pipeline.load_model()` can load and
run it with just `torch.jit.load()`, with no dependency on
09_Models/segformer_tonal.py, 09_Models/unet_tonal.py, or any training
code being importable. This is the "run the detector standalone"
counterpart to the FP32/INT8 ONNX export in export_onnx.py -- use
whichever deployment format suits your target machine:

    - TorchScript : stays inside the PyTorch ecosystem, easiest if the
                    deployment machine already has `torch` installed.
    - ONNX/ONNX INT8 : smaller, usually faster on CPU via onnxruntime,
                    and the INT8 build has the smallest footprint --
                    preferred when `torch` itself is not wanted on the
                    deployment machine (only `onnxruntime` is needed).

Usage:
    python export_torchscript.py --checkpoint ../11_Weights/segformer_b0_tonal.pt \
                                  --out ../11_Weights/segformer_b0_tonal.torchscript.pt
"""
from __future__ import annotations
import argparse
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "07_Deployment_Module"))
sys.path.append(os.path.join(ROOT, "13_Scripts"))
from inference_pipeline import load_model      # noqa: E402
from config_loader import load_config, get as cfg_get  # noqa: E402


def export_torchscript(checkpoint: str, out_path: str, n_freq: int = 513, n_frames: int = 247,
                        method: str = "trace") -> str:
    model, cfg = load_model(checkpoint, device="cpu")
    model.eval()
    dummy = torch.randn(1, 1, n_freq, n_frames)

    with torch.no_grad():
        if method == "trace":
            ts_model = torch.jit.trace(model, dummy, strict=False)
        elif method == "script":
            ts_model = torch.jit.script(model)
        else:
            raise ValueError(f"unknown method: {method} (expected 'trace' or 'script')")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ts_model.save(out_path)
    print(f"[export_torchscript] wrote {out_path} "
          f"({os.path.getsize(out_path) / 1e6:.2f} MB, method={method})")
    return out_path


def verify(out_path: str, n_freq: int = 513, n_frames: int = 247, n_runs: int = 10) -> float:
    """Load the exported archive back and sanity-check it runs + benchmark latency."""
    ts_model = torch.jit.load(out_path, map_location="cpu")
    ts_model.eval()
    x = torch.randn(1, 1, n_freq, n_frames)
    with torch.no_grad():
        y = ts_model(x)
        assert y.shape == x.shape, f"unexpected output shape {y.shape} for input {x.shape}"
        for _ in range(2):
            ts_model(x)
        t0 = time.time()
        for _ in range(n_runs):
            ts_model(x)
        dt = (time.time() - t0) / n_runs
    print(f"[export_torchscript] verified OK -- output shape {tuple(y.shape)}, "
          f"{dt * 1000:.1f} ms/inference on CPU")
    return dt * 1000.0


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.path.join(
        ROOT, cfg_get(cfg, "model.segformer_weights", "11_Weights/segformer_b0_tonal.pt")))
    ap.add_argument("--out", default=os.path.join(
        ROOT, cfg_get(cfg, "model.torchscript", "11_Weights/segformer_b0_tonal.torchscript.pt")))
    ap.add_argument("--n-freq", type=int, default=513)
    ap.add_argument("--n-frames", type=int, default=247)
    ap.add_argument("--method", choices=["trace", "script"], default="trace",
                     help="trace: faster/more reliable for this model (default). "
                          "script: preserves data-dependent control flow but requires the whole "
                          "model to be TorchScript-compatible.")
    args = ap.parse_args()

    out_path = export_torchscript(args.checkpoint, args.out, args.n_freq, args.n_frames, args.method)
    verify(out_path, args.n_freq, args.n_frames)


if __name__ == "__main__":
    main()
