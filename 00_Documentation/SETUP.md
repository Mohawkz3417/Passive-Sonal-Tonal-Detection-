# Setup Guide

## Requirements

- Python 3.10 - 3.12
- ~500 MB disk (mostly PyTorch); no GPU required
- Windows, Linux, or macOS

## 1. Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

pip install -r 12_Requirements/requirements.txt
```

**Windows note:** Tkinter (needed for `03_Live_Detector`) ships with the
official python.org Windows installer by default -- no extra step needed.
On Ubuntu/Debian Linux you must additionally run:
```bash
sudo apt install python3-tk
```

## 2. Verify the install

```bash
cd 02_Testing_Module
pytest -v
```
You should see all tests pass (signal processing, synthetic data
generation, both models, CFAR/morphology/CCA, frequency estimation,
Kalman tracking, and the Ethernet wire format).

## 2b. Fastest path: the launcher

Once the install is verified, `python main.py` from the project root
gives you a numbered menu covering every step below (train, test, live
detection, signal generator, dataset generator, evaluation, deployment,
Ethernet, settings) -- it's the easiest way to drive the suite without
memorizing each module's flags. Everything below also works standalone,
module by module, which is what the launcher calls under the hood.

## 3. Train a model

Training data is generated on the fly (no download needed) by
`04_Signal_Generator` + `05_Dataset_Generator`, sampling SNR uniformly
over `[-10, +20] dB` with extra emphasis on the `-10..-7 dB` target band.

```bash
cd 01_Training_Module
python train.py --model segformer --epochs 50 --steps-per-epoch 200 \
                 --batch-size 8 --out ../11_Weights/segformer_b0_tonal.pt
```

**Training on your own datasets (ShipsEar / DeepShip / DS3500 / etc.):**
add `--real-data-dir` to mix real recordings in as the noise background
(exact, free ground truth -- see `REAL_DATA_GUIDE.md` for the full story,
including how to also pseudo-label real tonals for a small fine-tuning set):
```bash
python train.py --model segformer --epochs 50 --steps-per-epoch 200 --batch-size 8 \
                 --real-data-dir /path/to/ShipsEar --real-noise-prob 0.6 \
                 --out ../11_Weights/segformer_b0_tonal.pt
```

**Expected wall-clock time** (see `RESULTS.md` for the measured
per-step cost in this delivery's sandbox, which had only a single CPU
core available):
- ~2.2 s/step at batch size 4 on **one** CPU core.
- A typical laptop CPU (4-8 cores, e.g. the i7 specified in the
  proposal) with `torch.set_num_threads()` left at its default will be
  substantially faster than this single-core figure -- benchmark a few
  steps on your machine first (`--epochs 1 --steps-per-epoch 5`) and
  extrapolate before committing to a long run.
- For a *good* detector at -7..-10 dB SNR, plan for on the order of
  **50-100 epochs x 200+ steps/epoch** (i.e. thousands of gradient
  steps) -- the 80-step demo run included in this delivery is a
  pipeline/plumbing validation, not a converged model (see `RESULTS.md`).
- If available, training on a CUDA GPU (even a modest laptop GPU) will
  be 10-50x faster; install the CUDA build of torch from
  https://pytorch.org/get-started/locally/ instead of the CPU wheel.

## 4. Evaluate

```bash
cd 06_Model_Evaluation
python evaluate.py --checkpoint ../11_Weights/segformer_b0_tonal.pt --n-per-snr 20
```
Writes `pixel_metrics.csv`, `detection_metrics.csv`,
`latency_benchmark.csv`, and two PNG plots (`pixel_metrics_vs_snr.png`,
`pd_pfa_vs_snr.png`) to `06_Model_Evaluation/results/`.

## 5. Export for deployment

```bash
cd 07_Deployment_Module
python export_onnx.py --checkpoint ../11_Weights/segformer_b0_tonal.pt
```
Produces `segformer_b0_tonal.onnx` (FP32) and `segformer_b0_tonal_int8.onnx`
(dynamically quantized) in `11_Weights/`, and prints a latency
comparison (PyTorch vs ONNX FP32 vs ONNX INT8). **Benchmark both on
your target laptop** -- see `RESULTS.md` for why INT8 was not
automatically faster in this delivery's environment.

Alternatively (or in addition), export TorchScript -- a self-contained
graph that only needs `torch` at inference time, no dependency on
`09_Models/segformer_tonal.py` being importable:
```bash
python export_torchscript.py --checkpoint ../11_Weights/segformer_b0_tonal.pt
```
Writes `segformer_b0_tonal.torchscript.pt`, verifies it loads and runs,
and prints its CPU latency. `deploy.py` (see step 8 below) automatically
prefers a TorchScript artifact over a raw `.pt` checkpoint when one is
available, since it's what a "no training code" deployment machine needs.

## 6. Run the Live GUI

```bash
cd 03_Live_Detector

# no hardware/files needed -- built-in synthetic generator:
python live_detector.py --source synthetic --snr-db -8

# replay a WAV file in real time:
python live_detector.py --source wav --wav-path path\to\file.wav

# live Ethernet/UDP hydrophone stream (see REAL_DATA_GUIDE.md for TCP
# option and the UDP-listen vs TCP-connect-out --eth-host semantics):
python live_detector.py --source ethernet --eth-protocol udp --eth-host 0.0.0.0 --eth-port 50000
python live_detector.py --source ethernet --eth-protocol tcp --eth-host 192.168.1.50 --eth-port 50000
```

To test the Ethernet path without real hardware, run the simulator in a
second terminal:
```bash
cd 08_Ethernet_Receiver
python ethernet_sender_sim.py --mode synthetic --snr-db -8 --host 127.0.0.1 --port 50000
```

Detections are logged to `logs/detections_log.csv` and `logs/detections_log.txt`
(both CSV and human-readable TXT, including inference time and CPU usage --
see `13_Scripts/log_utils.py`).

## 7. Connecting real hydrophone hardware

`08_Ethernet_Receiver/receiver.py` defines a simple, documented UDP
packet format (see the module docstring), plus a TCP-client alternative
(`EthernetReceiverTCP`) for hardware that runs as a TCP server, or for
lossy links where you need guaranteed delivery -- see
`REAL_DATA_GUIDE.md` for when to use which. Real NPOL/hydrophone
acquisition hardware protocols are proprietary and were not available
while building this suite, so you will need to either:
- add a small adapter that repackages the real acquisition system's
  Ethernet output into `receiver.py`'s packet format, or
- edit `receiver.py`'s `unpack_packet()` to parse the real system's
  native frame format directly.

Both are isolated to that one file -- nothing downstream needs to change.

## 8. Standalone deployment (no training code needed)

```bash
cd ..   # project root
python deploy.py                 # auto-picks the best artifact in 11_Weights/, opens the GUI
python deploy.py --list-models   # see what's discoverable
python deploy.py --headless --n-chunks 20 --source synthetic --snr-db -8
```
`deploy.py` only imports `07_Deployment_Module`, `09_Models`, and
`13_Scripts` -- never `01_Training_Module` or `losses.py` -- so it's
what you'd hand to someone who just needs to *run* the detector.

## 9. Signal Generator (expanded)

```bash
cd 04_Signal_Generator
python generator_cli.py --type linear_chirp --snr-db 5 --duration 3
python generator_cli.py --type pink_noise --duration 10
```
Supported `--type` values: `pure_tone`, `dual_tone`, `multi_tone`,
`harmonics`, `linear_chirp`, `fm_signal`, `white_noise`, `pink_noise`,
`ambient_ocean`. `--snr-db` supports the full **+20 dB to -15 dB** range.
Every run automatically saves a WAV, raw-spectrogram PNG, PCEN-spectrogram
PNG, ground-truth mask PNG/`.npz` (where applicable), and a metadata JSON
to `10_Datasets/generated/` (or `--out-dir`).
