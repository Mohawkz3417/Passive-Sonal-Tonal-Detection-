"""
export_onnx.py
================
Deployment Module: export the trained PyTorch checkpoint to ONNX, then
apply dynamic INT8 quantization (weights only -- safe for CPU inference,
no calibration dataset required) via onnxruntime.quantization, and
benchmark both the FP32 and INT8 ONNX graphs against the native PyTorch
model to show the CPU speed-up. This is what makes the detector genuinely
"deployable on a laptop": no CUDA, no PyTorch runtime needed at inference
time, and a much smaller memory / disk footprint.

Usage:
    python export_onnx.py --checkpoint ../11_Weights/segformer_b0_tonal.pt \
                           --out ../11_Weights/segformer_b0_tonal.onnx
"""
from __future__ import annotations
import argparse
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "07_Deployment_Module"))
sys.path.append(os.path.join(ROOT, "13_Scripts"))
from inference_pipeline import load_model  # noqa: E402
from config_loader import load_config, get as cfg_get  # noqa: E402


def export(checkpoint: str, out_path: str, n_freq: int = 513, n_frames: int = 247):
    model, cfg = load_model(checkpoint, device="cpu")
    model.eval()
    dummy = torch.randn(1, 1, n_freq, n_frames)

    torch.onnx.export(
        model, dummy, out_path,
        input_names=["spectrogram"], output_names=["logits"],
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
    print(f"[export] wrote FP32 ONNX model to {out_path} "
          f"({os.path.getsize(out_path) / 1e6:.2f} MB)  [static shape: 1x1x{n_freq}x{n_frames}]")
    print("         Re-run with --n-freq/--n-frames to export a different fixed chunk size "
          "(the pad-to-multiple-of-32 logic is baked in at trace time).")
    return out_path


def quantize(fp32_path: str, int8_path: str):
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QInt8)
    print(f"[quantize] wrote INT8 ONNX model to {int8_path} "
          f"({os.path.getsize(int8_path) / 1e6:.2f} MB)")
    return int8_path


def bench_onnx(path: str, n_freq=513, n_frames=247, n_runs=10):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
    x = np.random.randn(1, 1, n_freq, n_frames).astype(np.float32)
    name = sess.get_inputs()[0].name
    for _ in range(2):
        sess.run(None, {name: x})
    t0 = time.time()
    for _ in range(n_runs):
        sess.run(None, {name: x})
    dt = (time.time() - t0) / n_runs
    return dt * 1000.0


def bench_torch(checkpoint: str, n_freq=513, n_frames=247, n_runs=10):
    model, cfg = load_model(checkpoint, device="cpu")
    model.eval()
    x = torch.randn(1, 1, n_freq, n_frames)
    with torch.no_grad():
        for _ in range(2):
            model(x)
        t0 = time.time()
        for _ in range(n_runs):
            model(x)
        dt = (time.time() - t0) / n_runs
    return dt * 1000.0


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.path.join(
        ROOT, cfg_get(cfg, "model.segformer_weights", "11_Weights/active/segformer_b0_tonal.pt")))
    ap.add_argument("--out", default=os.path.join(ROOT, "11_Weights", "segformer_b0_tonal.onnx"))
    ap.add_argument("--n-freq", type=int, default=513)
    ap.add_argument("--n-frames", type=int, default=247)
    args = ap.parse_args()

    fp32_path = export(args.checkpoint, args.out, args.n_freq, args.n_frames)
    int8_path = os.path.splitext(args.out)[0] + "_int8.onnx"
    quantize(fp32_path, int8_path)

    print("\n[benchmark] single-chunk CPU latency (lower is better):")
    t_pt = bench_torch(args.checkpoint, args.n_freq, args.n_frames)
    t_fp32 = bench_onnx(fp32_path, args.n_freq, args.n_frames)
    t_int8 = bench_onnx(int8_path, args.n_freq, args.n_frames)
    print(f"  PyTorch (FP32) : {t_pt:8.1f} ms")
    print(f"  ONNX (FP32)    : {t_fp32:8.1f} ms  ({t_pt/t_fp32:.2f}x vs PyTorch)")
    print(f"  ONNX (INT8)    : {t_int8:8.1f} ms  ({t_pt/t_int8:.2f}x vs PyTorch)")


if __name__ == "__main__":
    main()
