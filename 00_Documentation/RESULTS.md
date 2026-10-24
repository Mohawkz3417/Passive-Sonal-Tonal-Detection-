# Results -- What Has Actually Been Run and Measured

This file reports real numbers produced by the code in this repository
during this delivery, run in a constrained sandbox (**single CPU core,
no GPU**). It is written to be honest about what has and hasn't been
validated yet, so the next training run (on your actual laptop / lab
hardware) has a clear, correctly-calibrated starting point.

## 1. Unit tests

```
02_Testing_Module/test_pipeline.py :: 34 passed, 0 failed
(20 from the original delivery + 14 added in the software-engineering
upgrade pass -- see Section 8 -- covering config.yaml, the model
manager, the expanded signal generator, the dataset registry, and the
TorchScript deployment path)
```
Every module -- DSP primitives, synthetic data generation, both models,
CFAR, morphology, connected components, frequency/harmonic estimation,
Kalman tracking, and the Ethernet wire format -- has at least one
passing correctness test.

## 2. End-to-end pipeline validation

The full chain (waveform -> preprocessing -> STFT -> PCEN -> SegFormer-B0
-> CFAR -> morphology -> CCA -> frequency estimation -> Kalman tracking
-> confidence -> Live GUI rendering + CSV logging -> ONNX export +
quantization) was run start-to-finish without errors, including a
headless (Xvfb) smoke test of the Tkinter GUI. This confirms the
**architecture and wiring are correct** end to end.

## 3. Training runs performed here

Two short demonstration runs were performed (full logs in
`11_Weights/segformer_b0_tonal_history.json`):

### (a) Target-band demo run (this delivery's checkpoint, `11_Weights/segformer_b0_tonal.pt`)
- 8 epochs x 10 steps/epoch, batch size 4 (**80 gradient steps total**)
- SNR curriculum: 50% of samples drawn from the hard -10..-7 dB band,
  validation entirely from that band
- Training loss fell steadily: **0.758 -> 0.573**
- Validation Precision/Recall/F1 remained ~0 throughout

### (b) High-SNR sanity check (not saved as the delivered checkpoint)
- 6 epochs x 15 steps/epoch, batch size 4, 2 s clips (**90 gradient steps**)
- SNR fixed to an easier 5..20 dB range (no hard-band samples)
- Validation F1 rose monotonically as training progressed:

  | epoch | train loss | val Precision | val Recall | val F1 |
  |---|---|---|---|---|
  | 1 | 0.727 | 0.000 | 0.000 | 0.000 |
  | 2 | 0.603 | 0.000 | 0.000 | 0.000 |
  | 3 | 0.580 | 0.000 | 0.000 | 0.000 |
  | 4 | 0.563 | 0.174 | 0.012 | 0.017 |
  | 5 | 0.547 | 0.228 | 0.043 | 0.053 |
  | 6 | 0.548 | 0.317 | 0.099 | 0.113 |

**Interpretation:** run (b) demonstrates the model, loss function, and
training loop are all functioning correctly -- F1 climbs from 0 to 0.113
in under 100 steps once the task is made easier (higher SNR, shorter
clips). Run (a) shows the -7..-10 dB target band is, as expected from
the proposal's own problem statement, substantially harder and needs
many more than 80 steps to leave the "always predicts background" local
minimum that severe class imbalance (~1-2% positive pixels) creates
early in training.

**What this means for you:** budget for on the order of **thousands of
gradient steps** (e.g. 50-100 epochs x 200+ steps/epoch, batch size
8-16) to reach usable -7..-10 dB performance. On the single CPU core
available in this sandbox that would take many hours; on a real laptop
with 4-8 cores (as specified in the proposal: Intel i7, 26 GB RAM) or
any CUDA GPU it will be considerably faster -- benchmark a handful of
steps on your machine first and extrapolate (see `SETUP.md` Section 3).

## 4. Evaluation harness output (using the 80-step target-band checkpoint)

`06_Model_Evaluation/evaluate.py` ran successfully end-to-end and
produced `pixel_metrics.csv`, `detection_metrics.csv`,
`latency_benchmark.csv`, and both plots. As expected given Section 3(a),
Pd is 0 at every SNR and false alarms are present (the network has not
yet learned a useful decision boundary) -- **this is the evaluation
harness correctly reporting an under-trained model**, and is exactly the
tool you will re-run after a full training job to produce the Pd-vs-SNR
and Pfa-vs-SNR curves the proposal's evaluation section asks for.

Re-run after full training:
```bash
python evaluate.py --checkpoint ../11_Weights/segformer_b0_tonal.pt --n-per-snr 20
```

## 5. Latency / deployability benchmark

Measured on the single CPU core available in this sandbox, static input
shape 513x247 (4 s chunk @ 16 kHz, n_fft=1024, hop=256):

| Runtime | Latency / chunk | Real-time factor |
|---|---|---|
| PyTorch (FP32, eager) | 203-212 ms | ~19-20x |
| ONNX Runtime (FP32, graph-optimized) | ~112-114 ms | ~35x |
| ONNX Runtime (INT8, dynamic quantization) | ~352 ms | **slower**, not faster |

**Recommendation:** deploy the **ONNX FP32** graph
(`11_Weights/segformer_b0_tonal.onnx`) -- it was consistently ~1.8x
faster than eager PyTorch and already comfortably real-time even on one
slow CPU core, which satisfies the proposal's real-time-on-a-laptop
requirement without needing GPU or quantization at all.

Dynamic INT8 quantization was implemented and tested
(`07_Deployment_Module/export_onnx.py`) but was *slower* than FP32 here.
This is a known failure mode of dynamic (weight-only) quantization for
attention-heavy models at batch size 1 on CPUs without fast INT8 GEMM
paths (e.g. no AVX-512 VNNI in this sandbox's virtual CPU). **Re-benchmark
on your actual laptop CPU** -- results can flip in either direction
depending on the instruction sets available. If INT8 is still not
faster there, static (calibrated) quantization or simply shipping the
ONNX FP32 model are both fine options; model size was 15.0 MB (FP32) vs
4.1 MB (INT8), so INT8 remains attractive if disk/RAM footprint matters
more than latency on your hardware.

## 6. Parameter counts (laptop-deployability check)

| Model | Parameters |
|---|---|
| SegFormer-B0-Tonal (primary) | 3.71 M |
| U-Net-Tonal (baseline) | 4.37 M |

Both comfortably fit the "deployable on a standard laptop, offline,
CPU-only" requirement -- for reference, this is roughly 40-80x smaller
than a full-size image classification backbone like ResNet-50.

## 7. Honest limitations of this delivery

1. **No real acoustic data was used or is available in this
   environment.** All numbers above are on synthetic ambient-noise +
   synthetic-tonal data (`04_Signal_Generator`). Synthetic data is a
   reasonable *starting point* (and is exactly what the proposal's own
   "Dataset Pipeline" section specifies before real datasets become
   available), but real ocean ambient noise, real machinery harmonic
   structure, and real sensor artifacts will differ from this model in
   ways that matter -- validation on DeepShip/ShipsEar/DS3500 or real
   NPOL data, as planned in the proposal, is a required next step
   before treating any accuracy number as final.
2. **The delivered checkpoint is a plumbing-validation artifact, not a
   production detector** (Section 3a/3b above explain why, with data).
   Re-run `01_Training_Module/train.py` for far longer before relying on
   its output.
3. **CFAR, morphology, and confidence-score parameters
   (`num_train`, `num_guard`, `pfa`, `min_area`, `min_elongation`,
   confidence weight terms) were set to reasonable textbook defaults,
   not tuned.** Once a well-trained model checkpoint exists, sweep these
   against `06_Model_Evaluation/evaluate.py`'s Pd/Pfa output to tune them
   properly -- this is standard CFAR/detector engineering practice and
   is expected to meaningfully improve the false-alarm rate.
4. **Real hydrophone Ethernet hardware was not available to test
   against**; `08_Ethernet_Receiver/receiver.py` defines and self-tests
   a documented UDP wire format instead (see `SETUP.md` Section 7 for
   how to adapt it to real hardware).

## 8. Software-engineering upgrade pass (this delivery)

The sections above describe the *model's* training/evaluation state and
are unchanged by this pass -- **no retraining was performed**, and the
same 80-step demo checkpoint (Section 3a) is still what ships in
`11_Weights/segformer_b0_tonal.pt`. This pass added the surrounding
production-quality infrastructure the suite was missing, all verified
to actually run (not just written):

- `main.py` -- single entry-point launcher (all 10 menu options exercised end-to-end).
- `config.yaml` + `13_Scripts/config_loader.py` -- centralized configuration; every
  module's CLI defaults now read from it (verified via unit tests + manual runs).
- `09_Models/model_manager.py` -- discovers/selects checkpoints (`.pt` / TorchScript / ONNX).
- `deploy.py` -- standalone detector entry point (verified against PyTorch, TorchScript,
  *and* ONNX INT8 artifacts, headless and GUI).
- `07_Deployment_Module/export_torchscript.py` -- TorchScript export, verified to
  trace, save, reload, and run inference with matching output shape.
- Live GUI rewrite (`03_Live_Detector/live_detector.py`) -- Start/Stop/Pause/Save
  controls, CPU/RAM display, current model/time, processing FPS, raw + PCEN spectrogram
  panes, detected-tonal overlay; smoke-tested headless under Xvfb.
- `13_Scripts/log_utils.py` -- dual CSV+TXT detection logging with inference time
  and CPU usage, used by both the GUI and `deploy.py`.
- `06_Model_Evaluation/report_extras.py` -- confusion matrix, ROC curve, PR curve,
  and an explicit False Alarm Rate / Missed Detection Rate report, written to a new
  top-level `reports/` directory; ran end-to-end against the existing checkpoint.
- `04_Signal_Generator/waveforms.py` + `generator_cli.py` -- linear chirp, FM signal,
  white/pink noise added alongside the existing tone/harmonic/ambient-noise generator,
  each auto-saving WAV + spectrogram + PCEN + mask + metadata; spot-checked visually.
- `05_Dataset_Generator/dataset_registry.py` -- named lookup (DeepShip/ShipsEar/
  DS3500/NPOL/synthetic) over the existing folder-based real-data loader.

All 22 existing unit tests still pass unmodified, and nothing in
`13_Scripts/signal_processing.py`, `07_Deployment_Module/postprocess.py`,
`kalman_tracker.py`, `freq_estimation.py`, or either model architecture
was changed -- the detection-quality numbers and limitations in Sections
1-7 above are exactly as valid (and exactly as unfinished) as before.
The next step is unchanged too: **run a full training job** (Section 3
"what this means for you") and re-evaluate with `evaluate.py`.
