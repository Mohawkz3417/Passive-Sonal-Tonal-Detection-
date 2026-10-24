# NPOL Tonal Detection Suite

**Robust Low-SNR Passive Sonar Tonal Detection using Transformer-Based
Spectrogram Segmentation (SegFormer-B0)**

Final-year project / NPOL internship deliverable. Target: detect weak
tonal (narrowband) components in hydrophone data down to **-7 dB to
-10 dB SNR**, in real time, on a standard laptop (CPU only, no CUDA
required for deployment).

This repository is a complete, runnable implementation of the pipeline
described in `Project_proposal.docx`, organized into the exact folder
structure requested there. Every module below is implemented and has
passing unit tests (`02_Testing_Module/test_pipeline.py`, 34/34 passing --
20 original + 14 added in the software-engineering upgrade pass covering
config.yaml, the model manager, the expanded signal generator, the
dataset registry, and the TorchScript deployment path)
and has been exercised end-to-end on synthetic data (see `RESULTS.md`).

---

## 1. Why this architecture

Ambient ocean noise and machinery tonals overlap heavily in energy once
SNR drops below 0 dB. A 3x3/5x5 CNN kernel (U-Net) only ever sees a
small local neighbourhood of the spectrogram, so it cannot tell a
persistent, continuous tonal line apart from an isolated noise spike --
both just look like "a bright pixel" locally. This is why plain CNN
segmentation tends to hallucinate extra frequencies at low SNR.

**SegFormer-B0** solves this with self-attention: every patch in the
spectrogram can attend to every other patch from the first encoder stage
onward, so the network can directly learn "this line of bright pixels is
continuous across time -> tonal" vs. "this bright pixel is spatially
isolated -> noise". Its lightweight, MLP-only decode head then keeps the
whole model at **~3.7M parameters**, small enough for real-time CPU
inference on a laptop.

We do **not** rely on the network alone, however -- see the proposal's
own critique of "CNN/segmentation-only" detectors. Detection-grade
performance at -7..-10 dB needs a full DSP + classical-detection +
tracking chain around the network, which is what this repo implements.

## 2. Full pipeline (as built)

```
Hydrophone (WAV / synthetic / live Ethernet UDP)
        |
        v
Signal Acquisition -------------------------- 08_Ethernet_Receiver/receiver.py
        |
        v
DC Removal -> Band-pass Filter -> Normalize -- 13_Scripts/signal_processing.py
        |
        v
Noise Power Estimation
        |
        v
STFT (16 kHz, n_fft=1024, hop=256, Hann) ----- 13_Scripts/signal_processing.py
        |
        v
PCEN (Per-Channel Energy Normalization) ------ 13_Scripts/signal_processing.py
        |
        v
SegFormer-B0 (MixVisionTransformer + --------- 09_Models/segformer_tonal.py
   All-MLP decode head) -> probability mask     09_Models/unet_tonal.py (baseline)
        |
        v
CFAR verification (Cell-Averaging, 1-D) ------ 07_Deployment_Module/postprocess.py
        |
        v
Morphological Opening/Closing + blob/area ---- 07_Deployment_Module/postprocess.py
   filtering (removes isolated noise, keeps
   elongated / continuous structures)
        |
        v
Connected Component Analysis (8-connectivity)- 07_Deployment_Module/postprocess.py
        |
        v
Frequency/Bandwidth/Duration Estimation ------ 07_Deployment_Module/freq_estimation.py
   + Harmonic grouping
        |
        v
Multi-target Kalman Tracking (birth/confirm/-- 07_Deployment_Module/kalman_tracker.py
   death, nearest-neighbour gated association)
        |
        v
Confidence Score (fuses prob, blob size, SNR,- 07_Deployment_Module/freq_estimation.py
   track stability, harmonics)
        |
        v
Live GUI (waveform / spectrogram / mask / ---- 03_Live_Detector/live_detector.py
   detections / confidence / FPS) + CSV log
```

`07_Deployment_Module/inference_pipeline.py` wires every stage above
into a single `TonalDetectorPipeline` class used by both the Live GUI
and the evaluation harness, so there is exactly one code path between
"research" and "deployment".

## 3. Folder structure

| Folder / file | Contents |
|---|---|
| `main.py` | **Single entry point** -- numbered menu launching every module below |
| `deploy.py` | Standalone deployment runner -- no dependency on training code |
| `config.yaml` | **Central configuration** -- every module reads its defaults from here |
| `00_Documentation/` | This file, `SETUP.md`, `RESULTS.md`, `REAL_DATA_GUIDE.md` |
| `01_Training_Module/` | `train.py`, `losses.py` (Dice + BCE) |
| `02_Testing_Module/` | `test_pipeline.py` -- pytest unit tests for every module |
| `03_Live_Detector/` | `live_detector.py` -- Tkinter/Matplotlib real-time GUI (Start/Stop/Pause/Save, CPU/RAM, model manager) |
| `04_Signal_Generator/` | `synthetic_signal_generator.py` (tones/harmonics/ambient noise), `waveforms.py` (chirp/FM/white/pink noise), `generator_cli.py` (CLI + WAV/PNG/mask/metadata export) |
| `05_Dataset_Generator/` | `dataset_builder.py`, `real_data_loader.py`, `pseudo_label_real_tonals.py`, `dataset_registry.py` (named DeepShip/ShipsEar/DS3500/NPOL lookup) |
| `06_Model_Evaluation/` | `evaluate.py` (Pd/Pfa/F1/IoU vs SNR), `report_extras.py` (confusion matrix, ROC/PR, false-alarm & missed-detection); `results/` (legacy) + top-level `reports/` output |
| `07_Deployment_Module/` | `postprocess.py`, `freq_estimation.py`, `kalman_tracker.py`, `inference_pipeline.py`, `export_onnx.py`, `export_torchscript.py` |
| `08_Ethernet_Receiver/` | `receiver.py` (UDP/TCP listener) + `ethernet_sender_sim.py` (test tool) |
| `09_Models/` | `segformer_tonal.py` (SegFormer-B0, from scratch), `unet_tonal.py` (baseline), `model_manager.py` (discovers/selects checkpoints) |
| `10_Datasets/` | Materialized `.npz` datasets, generated signal bundles, detection snapshots |
| `11_Weights/` | Trained checkpoints (`.pt`), TorchScript (`.torchscript.pt`), exported ONNX (`.onnx`) |
| `12_Requirements/` | `requirements.txt` |
| `13_Scripts/` | `signal_processing.py` (DSP primitives), `config_loader.py` (config.yaml), `log_utils.py` (CSV+TXT logging) |
| `logs/` | `detections_log.csv` / `.txt` -- every confirmed detection, any module |
| `reports/` | Confusion matrix, ROC/PR curves, false-alarm/missed-detection, training curves |

## 4. Quick start

**Easiest: the single entry-point launcher**
```bash
pip install -r 12_Requirements/requirements.txt
python main.py
```
gives you a numbered menu (Train / Test / Live Detection / Signal
Generator / Dataset Generator / Model Evaluation / Deployment / Ethernet
Receiver / Settings) that dispatches to every module below and reads
its defaults from `config.yaml`.

**Equivalent, module-by-module (for scripting / CI):**
```bash
cd 12_Requirements && pip install -r requirements.txt

# 1. Train (synthetic data generated on the fly, no download needed)
cd ../01_Training_Module
python train.py --model segformer --epochs 50 --steps-per-epoch 200 --batch-size 8 \
                 --out ../11_Weights/segformer_b0_tonal.pt

# 2. Evaluate across the full SNR range, including the -7..-10 dB target band
#    (also writes confusion matrix / ROC / PR / false-alarm & missed-detection to reports/)
cd ../06_Model_Evaluation
python evaluate.py --checkpoint ../11_Weights/segformer_b0_tonal.pt --n-per-snr 20

# 3. Export for deployment: ONNX + INT8 quantization, and/or TorchScript
cd ../07_Deployment_Module
python export_onnx.py --checkpoint ../11_Weights/segformer_b0_tonal.pt
python export_torchscript.py --checkpoint ../11_Weights/segformer_b0_tonal.pt

# 4. Run the live GUI (synthetic source needs no hardware/files)
cd ../03_Live_Detector
python live_detector.py --source synthetic --snr-db -8

# 4b. ...or run the standalone deployment entry point (no training-code import)
cd ..
python deploy.py                      # auto-picks the best artifact in 11_Weights/
python deploy.py --headless           # console-only, no GUI

# 5. Run unit tests
cd 02_Testing_Module
pytest -v
```

See `SETUP.md` for Windows-specific notes and `RESULTS.md` for what has
already been run and measured in this delivery, and what remains to be
done to reach production-grade accuracy.

## 5. Plugging in real data (ShipsEar / DeepShip / DS3500 / etc.)

Two ready-to-use paths, no manual annotation required for the first one:

1. **Real-noise injection** (recommended default) -- `TonalSegDataset`
   accepts `real_noise_dir`; a fraction of training samples then use a
   real recording as the noise background with a synthetic tonal mixed
   on top at controlled SNR, giving real noise statistics with exact,
   free ground truth. Wired into `train.py` via `--real-data-dir` /
   `--real-noise-prob`.
2. **Pseudo-labeling real tonals** -- `05_Dataset_Generator/pseudo_label_real_tonals.py`
   runs a classical persistent-CFAR detector over your recordings to find
   candidate real tonal lines, with a review PNG for each candidate so
   you can visually confirm before trusting it as ground truth.

Full walkthrough, including which of your four datasets to use for what,
and how to fine-tune on the pseudo-labeled set: **`00_Documentation/REAL_DATA_GUIDE.md`**.

## 6. Models implemented

- **Primary: SegFormer-B0** (`09_Models/segformer_tonal.py`) -- built
  from scratch in PyTorch (no internet/HuggingFace download required):
  4-stage MixVisionTransformer encoder (overlap patch embeddings,
  efficient self-attention with spatial-reduction, Mix-FFN) + an
  all-MLP decode head. ~3.7M parameters.
- **Baseline: U-Net** (`09_Models/unet_tonal.py`) -- a small 4-level
  U-Net (~4.4M parameters) for direct experimental comparison, as
  called for in the proposal.

Both share the same input/output contract (`(B,1,F,T) -> (B,1,F,T)`
logits) so `train.py --model unet` and `evaluate.py` work unchanged.

## 7. Configuration (`config.yaml`) and the Model Manager

Every tunable parameter -- sampling rate, FFT size, hop length, band-pass
limits, window type, active model + weights path, confidence threshold,
morphology parameters, CFAR parameters, Kalman tracker gates, logging/GUI
settings, training hyperparameters, dataset paths -- lives in one file,
`config.yaml`, read via `13_Scripts/config_loader.py`. Every module's
argparse defaults are pulled from it, so editing `config.yaml` (or using
`python main.py` -> "9. Settings") changes the suite's behaviour without
touching code; explicit CLI flags still override it for a single run.

`09_Models/model_manager.py` scans `11_Weights/` and reports every
discovered checkpoint (PyTorch `.pt`, TorchScript `.torchscript.pt`, ONNX
`.onnx`/`_int8.onnx`) with its architecture, size, and last known
validation F1 (from a sibling `*_history.json`). Used by:
- `python main.py` -> Live Detection / Model Evaluation / Deployment (interactive picker)
- `03_Live_Detector/live_detector.py --model-manager`
- `deploy.py` (auto-picks the best deployment artifact: TorchScript >
  quantized ONNX > FP32 ONNX > raw PyTorch checkpoint)

## 8. CFAR, in a nutshell

Cell-Averaging CFAR (`07_Deployment_Module/postprocess.py::ca_cfar_1d`)
estimates the local noise power around each frequency bin from a ring of
*training cells* (skipping a few *guard cells* immediately next to the
bin under test, so the tonal's own energy doesn't bias the noise
estimate), then flags the bin only if its power exceeds
`alpha * noise_estimate`, where `alpha = N*(Pfa^(-1/N) - 1)` is derived
directly from the desired Probability of False Alarm (`pfa` in
`config.yaml`) -- **never a fixed threshold**: the same absolute signal
level that would be flagged in a quiet band is correctly ignored in a
noisier one, and vice versa. Run independently on every frame across the
band-pass filter's passband, then multiplied element-wise with the
SegFormer probability mask (thresholded), CFAR acts as an independent
classical check that a candidate pixel isn't just something the network
imagined -- this "ensemble" combination is what drives the false-alarm
rate down at low SNR compared to using the network alone.

## 9. Kalman tracking, in a nutshell

A per-tonal constant-velocity Kalman filter (`07_Deployment_Module/kalman_tracker.py`)
tracks `[frequency, frequency_rate]` frame to frame. New detections are
associated to existing tracks by nearest-neighbour gating (`gate_hz`); a
track is only reported to the operator once it has accumulated
`confirm_hits` associated detections, and is dropped after `max_misses`
consecutive misses. This is what separates a real, frequency-continuous
tonal (which a Kalman track will confidently confirm and smooth) from a
one-off noise spike that happens to pass CFAR + the network in a single
frame but doesn't persist -- a major source of false alarms if CFAR/the
network output were reported directly, frame by frame, with no tracking.

## 10. PCEN, in a nutshell

Per-Channel Energy Normalization (`13_Scripts/signal_processing.py::pcen`)
replaces the usual log-magnitude spectrogram with
`(E / (eps + M)^alpha + delta)^r - delta^r`, where `M` is a per-bin
exponential moving average of the energy `E` (i.e. an adaptive estimate
of the *recent local* noise floor in that frequency bin). Dividing by
`M^alpha` before the compressive `r`-power automatically suppresses
slowly-varying background energy per-bin -- exactly what a spectrogram of
ambient ocean noise plus a weak tonal needs, since ambient noise power
varies a lot with frequency but a genuine tonal is a *sudden* deviation
from its own bin's recent history, which PCEN accentuates far more than
a plain log-spectrogram does.

## 11. Deployment formats: PyTorch / TorchScript / ONNX

Three interchangeable deployment artifacts, all producible from the same
trained checkpoint:

| Format | Script | When to use |
|---|---|---|
| PyTorch `.pt` (training checkpoint) | `01_Training_Module/train.py` (written automatically) | Most flexible; needs `torch` + `09_Models/*.py` importable |
| TorchScript `.torchscript.pt` | `07_Deployment_Module/export_torchscript.py` | Self-contained graph; needs only `torch`, not the model source files -- `deploy.py`'s default pick when available |
| ONNX (FP32 / INT8) | `07_Deployment_Module/export_onnx.py` | Smallest footprint, usually fastest on CPU via `onnxruntime`; no `torch` needed on the deployment machine at all |

`deploy.py` runs any of the three standalone (no import of
`01_Training_Module` or `losses.py`), with `--headless` for a
console-only run or the full Tkinter GUI by default.

## 12. Reports and logs

- `logs/detections_log.csv` + `logs/detections_log.txt` -- every
  confirmed detection from the Live GUI or `deploy.py`, with timestamp,
  frequency, confidence, SNR, inference time, and CPU usage
  (`13_Scripts/log_utils.py::DetectionLogger`).
- `reports/` -- written by `06_Model_Evaluation/evaluate.py` +
  `report_extras.py`: pixel metrics vs SNR, Pd/Pfa vs SNR, a pixel-level
  confusion matrix, ROC curve, Precision-Recall curve, an explicit
  False Alarm Rate / Missed Detection Rate table, frequency error, a
  latency benchmark, and training curves (from the checkpoint's sibling
  `*_history.json`, if present). The original `06_Model_Evaluation/results/`
  location is still written too, for anything that depended on it.
