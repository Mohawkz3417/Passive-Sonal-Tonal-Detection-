"""
model_manager.py
==================
Discovers available trained models inside 11_Weights/ (PyTorch .pt
checkpoints, TorchScript archives, and ONNX graphs), reports metadata
about each (architecture, parameter count, file size, validation F1 if
known from a sibling *_history.json), and hands back a ready-to-use
model for the active `TonalDetectorPipeline`.

Used by:
    - main.py            ("9. Settings" / model selection before Live Detection)
    - 03_Live_Detector    (--model-manager interactive picker)
    - deploy.py           (auto-picks the best available deployment artifact)
"""
from __future__ import annotations
import json
import os
import sys
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "13_Scripts"))
sys.path.append(os.path.join(ROOT, "07_Deployment_Module"))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_loader import load_config, get as cfg_get, resolve_path  # noqa: E402

# Preference order when auto-selecting a deployment artifact: TorchScript
# is fastest to load with no training-code dependency, then quantized
# ONNX (smallest/fastest on CPU), then FP32 ONNX, then the raw PyTorch
# training checkpoint (most flexible, works even freshly after training).
ARTIFACT_PRIORITY = ("torchscript", "onnx_int8", "onnx_fp32", "pytorch")


@dataclass
class ModelEntry:
    kind: str                 # "pytorch" | "torchscript" | "onnx"
    architecture: str         # "segformer" | "unet" | "unknown"
    path: str
    size_mb: float
    val_f1: float | None = None
    extra: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return os.path.basename(self.path)


def _history_f1(pt_path: str) -> float | None:
    hist_path = os.path.splitext(pt_path)[0] + "_history.json"
    if not os.path.exists(hist_path):
        return None
    try:
        with open(hist_path) as f:
            hist = json.load(f)
        if hist:
            return float(hist[-1].get("f1", hist[-1].get("val_f1")))
    except Exception:  # noqa: BLE001
        return None
    return None


def _architecture_of_pt(path: str) -> str:
    try:
        import torch
        ckpt = torch.load(path, map_location="cpu")
        if isinstance(ckpt, dict):
            return ckpt.get("model_name", "unknown")
    except Exception:  # noqa: BLE001
        pass
    return "unknown"


def discover_models(weights_dir: str | None = None) -> list[ModelEntry]:
    """Scan the weights directory and return every usable model artifact."""
    cfg = load_config()
    weights_dir = resolve_path(weights_dir or cfg_get(cfg, "model.weights_dir", "11_Weights"))
    if not os.path.isdir(weights_dir):
        return []

    entries: list[ModelEntry] = []
    for fn in sorted(os.listdir(weights_dir)):
        path = os.path.join(weights_dir, fn)
        if not os.path.isfile(path):
            continue
        size_mb = os.path.getsize(path) / 1e6
        lower = fn.lower()

        if lower.endswith(".pt") or lower.endswith(".pth"):
            if "torchscript" in lower:
                entries.append(ModelEntry("torchscript", "segformer" if "segformer" in lower else
                                           ("unet" if "unet" in lower else "unknown"),
                                           path, size_mb))
            else:
                arch = _architecture_of_pt(path)
                entries.append(ModelEntry("pytorch", arch, path, size_mb, val_f1=_history_f1(path)))
        elif lower.endswith(".ts"):
            entries.append(ModelEntry("torchscript",
                                       "segformer" if "segformer" in lower else
                                       ("unet" if "unet" in lower else "unknown"),
                                       path, size_mb))
        elif lower.endswith(".onnx"):
            kind = "onnx_int8" if "int8" in lower else "onnx"
            entries.append(ModelEntry(kind, "segformer" if "segformer" in lower else
                                       ("unet" if "unet" in lower else "unknown"),
                                       path, size_mb))
    return entries


def list_models_pretty(weights_dir: str | None = None) -> str:
    entries = discover_models(weights_dir)
    if not entries:
        return "  (no models found in 11_Weights/ -- train one first, option 1)"
    lines = []
    for i, e in enumerate(entries, 1):
        f1 = f"F1={e.val_f1:.3f}" if e.val_f1 is not None else "F1=n/a"
        lines.append(f"  [{i}] {e.name:<40s} {e.kind:<11s} {e.architecture:<10s} "
                      f"{e.size_mb:6.2f} MB  {f1}")
    return "\n".join(lines)


def pick_model_interactive(weights_dir: str | None = None) -> "ModelEntry | None":
    """CLI prompt: list discovered models and let the user pick one by index."""
    entries = discover_models(weights_dir)
    if not entries:
        print("No models found in 11_Weights/. Train one first (main.py option 1).")
        return None
    print("\nAvailable models:")
    print(list_models_pretty(weights_dir))
    choice = input(f"\nSelect model [1-{len(entries)}] (Enter = 1): ").strip()
    if not choice:
        return entries[0]
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(entries):
            return entries[idx]
    except ValueError:
        pass
    print("Invalid selection, using [1].")
    return entries[0]


def best_deployment_artifact(weights_dir: str | None = None) -> "ModelEntry | None":
    """Pick the best available artifact for deploy.py, preferring TorchScript
    > quantized ONNX > FP32 ONNX > raw PyTorch checkpoint (see ARTIFACT_PRIORITY)."""
    entries = discover_models(weights_dir)
    if not entries:
        return None
    by_kind: dict[str, list[ModelEntry]] = {}
    for e in entries:
        by_kind.setdefault(e.kind, []).append(e)
    for kind in ARTIFACT_PRIORITY:
        key = kind
        if key not in by_kind and kind == "pytorch":
            key = "pytorch"
        if key in by_kind:
            return by_kind[key][0]
    return entries[0]


def load_selected_model(entry: "ModelEntry", device: str = "cpu"):
    """Load a ModelEntry into a ready-to-run model object for TonalDetectorPipeline."""
    if entry.kind in ("pytorch", "torchscript"):
        from inference_pipeline import load_model
        return load_model(entry.path, device=device)
    elif entry.kind in ("onnx", "onnx_int8"):
        raise ValueError(
            "ONNX artifacts run through onnxruntime, not the PyTorch TonalDetectorPipeline. "
            "Use 07_Deployment_Module/export_onnx.py's bench_onnx()-style session directly, "
            "or select a .pt / TorchScript entry for the Live Detector / deploy.py."
        )
    raise ValueError(f"unknown model kind: {entry.kind}")


if __name__ == "__main__":
    print("NPOL Tonal Detection Suite -- discovered models:\n")
    print(list_models_pretty())
    best = best_deployment_artifact()
    if best:
        print(f"\nAuto-selected deployment artifact: {best.name} ({best.kind})")
