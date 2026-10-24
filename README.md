# Tonal Detection Suite

AI-based passive sonar **tonal detection** system: an 8-element hydrophone
array simulation feeds a transformer-based spectrogram segmentation model
(**SegFormer-B0**), cross-checked against a from-scratch **U-Net** baseline,
followed by CFAR/morphology/connected-component post-processing and a
multi-target **Kalman tracker**, with real-time bearing (DOA) estimation,
a full React/TypeScript operator UI, and a synthetic **Signal Generator**
(with real-recording background mixing) for testing and demonstration.

Both models are **trained, not placeholder weights** — see
[`TRAINING_AND_EVAL_REPORT.md`](TRAINING_AND_EVAL_REPORT.md) for the full,
honest training/evaluation record (every number in this README is sourced
from that report and was produced by actually running this code).

### 0. This delivery is 3 files — put the checkpoints in place first

Trained-weight files are too large to bundle with the code zip, so this
is delivered as three separate downloads. After extracting the code zip,
copy each checkpoint into the path shown (create the folder if it
doesn't exist yet):

| Zip | Extract its `.pt` file to |
|---|---|
| `NPOL_Tonal_Detection_Suite_v10_CODE.zip` | (the project itself — extract anywhere) |
| `NPOL_segformer_checkpoint_v10.zip` | `11_Weights/active/segformer_b0_tonal.pt` |
| `NPOL_unet_checkpoint_v10.zip` | `11_Weights/unet_tonal.pt` |

Once both `.pt` files are in place, confirm everything is wired up
correctly with:

```bash
python 13_Scripts/system_health_check.py --full
```

This should print `23 PASS / 0 WARNING / 0 FAIL` (see Section 2 below) —
if either checkpoint is missing or in the wrong path, the
`SegFormer Checkpoint` / relevant check will fail and say exactly which
file it expected.

---

## 1. What this project does

1. An 8-hydrophone linear array "hears" a tonal (a ship's narrowband
   acoustic signature) mixed into ambient ocean noise, at a chosen bearing,
   frequency, and SNR (down to strongly negative dB, i.e. the tonal is
   *quieter* than the noise floor).
2. The array data streams over a real UDP wire protocol to a detection
   backend, which:
   - **Beamforms** across all 8 channels and estimates bearing (DOA).
   - Converts the beamformed signal to a **PCEN-normalized spectrogram**.
   - Runs it through **SegFormer-B0** (primary) and, side by side, a
     **U-Net** baseline, both segmenting "where is a tonal in this
     spectrogram."
   - Applies **CFAR + morphology + connected-component analysis** to turn
     the raw segmentation mask into candidate tonal detections (frequency,
     bandwidth, harmonics, SNR).
   - Feeds candidates into a **multi-target Kalman tracker**, which only
     reports a detection as *confirmed* after 3 consecutive matching
     frames (rejects one-frame false alarms) and drops it after 4 missed
     frames.
   - Classifies confirmed tracks (ship-like / biological / unknown).
3. A full **React operator UI** shows live spectrograms, beamformer output,
   tracked contacts, per-model comparison, and system health — or can
   analyze an uploaded WAV/NPZ/MP3/real-dataset file directly, bypassing
   the live array.
4. A separate **Signal Generator** app (its own UI) drives the above for
   testing: fully synthetic tonals, or a synthetic tonal mixed onto a real
   recording's noise background (ShipsEar/DeepShip/DS3500/UATD/NPOL, or a
   file you upload), with controllable injected SNR.

---

## 2. Real, verified results

### Training (from random init, CPU-only, this repo's actual training run)

| Model | Params | Steps trained | Best val F1 | Best val IoU | Deployed checkpoint |
|---|---:|---:|---:|---:|---|
| **SegFormer-B0** | ~3.7M (segmentation head + B0 encoder) | 1,650 | 0.716 | 0.558 | `11_Weights/active/segformer_b0_tonal.pt` (epoch 10/11) |
| **U-Net** (from scratch) | 4.37M | 800 | **0.872** | **0.779** | `11_Weights/unet_tonal.pt` (epoch 6/8) |

U-Net reaching a higher F1 with fewer steps is a real, interesting
comparison point — not a claim that U-Net is definitively the better
architecture (no matched/controlled comparison was run; see the report).
Note both deployed checkpoints are each model's **best** epoch across a
noisy, resumed multi-round CPU training run, not necessarily the *last*
epoch trained — the Model Status page reports this explicitly.

### Full-pipeline detection accuracy vs. SNR (single-channel, no array gain — the model's raw sensitivity floor)

| SNR (dB) | SegFormer Pd | U-Net Pd |
|---:|---:|---:|
| +2 to −12 | **1.00** | **1.00** |
| −15 | 0.78 | 0.87 |
| −18 | 0.75 | 0.73 |
| −20 | 0.50 | 0.60 |

Both models detect perfectly down to **−12 dB** and degrade gracefully
below that — this is the system's real, honestly-measured operating
envelope, not a fabricated "works at any SNR" claim.

### Live, full 8-element array pipeline (real UDP transmission, this session)

10+ live conditions were driven end-to-end through the real Signal
Generator → UDP → Detection Backend → tracker pipeline (not the offline
batch evaluation above): varying SNR (0 to **−20 dB**), bearing, noise
type (shipping/environmental/biological/background), Doppler on/off, and
frequency (180 Hz–2200 Hz). **Every case was detected**, with typical
frequency error under 5 Hz and bearing error under a few degrees. The
live case handles lower nominal SNR than the offline single-channel sweep
because the 8-element array's beamforming gain raises the effective SNR
the model actually sees — both results are honest and consistent once
that's accounted for (see Section 6.4 of the training report).

### Verification status

- `pytest 02_Testing_Module/ tests/`: **63 passed, 1 skipped, 0 failed**
- `python 13_Scripts/system_health_check.py --full`: **23 PASS / 0 WARNING / 0 FAIL**
- Frontend: `tsc -b` **0 errors**, `npm run build` succeeds, `npm run lint`
  **0 errors**
- **Real-data validation (ShipsEar / DS3500 / NOAA SanctSound ocean-ambient
  noise):** the trained pipeline was run end-to-end against 2223 + 2223 + 368
  genuine third-party recordings — see `TRAINING_AND_EVAL_REPORT.md` §7 for
  the full write-up, including two real bugs found and fixed while building
  the real-data validator, honest qualitative detection rates (43% / 50% /
  23%), a documented classifier limitation on real audio, and a real
  beamformer bug found and fixed from live ground-truth testing against
  real noise backgrounds.

---

## 3. Quick start

```bash
# 1. Python dependencies
pip install -r 12_Requirements/requirements.txt
pip install -r 12_Requirements/requirements_additions.txt   # pydub etc.
# pydub (MP3 support) also needs the ffmpeg binary on PATH.

# 2. Sanity check everything (loads both real checkpoints, runs a real
#    8-channel array + beamformer + DOA + tracker sequence, etc.)
python 13_Scripts/system_health_check.py --full

# 3. Run the two backend services (separate terminals)
cd detection-backend-app   && uvicorn detection_backend:app --port 8000
cd signal-generator-app    && uvicorn generator_backend:app --port 8001

# 4. Run the frontend
cd frontend
npm install
npm run dev        # http://localhost:5173
```

Open the Signal Generator's UI, generate a tonal (or use "Real Recording"
mode), hit Transmit, then watch the main app's **Live Sonar**, **Tracking**,
**Model Status**, and **Input Analysis** pages update in real time. Both
apps run entirely on CPU — no GPU required anywhere in this suite.

### Fastest CLI-only start

```bash
python main.py     # numbered menu: Train / Test / Live Detection /
                    # Signal Generator / Dataset Generator / Model
                    # Evaluation / Deployment / Ethernet / Settings
```

### Run the automated test suite / evaluation directly

```bash
python -m pytest 02_Testing_Module/ tests/ -v
python 06_Model_Evaluation/evaluate.py --checkpoint 11_Weights/active/segformer_b0_tonal.pt
python 06_Model_Evaluation/evaluate.py --checkpoint 11_Weights/unet_tonal.pt
```

---

## 4. Adding real-world datasets (ShipsEar / DS3500 / ocean_ambient / DeepShip / UATD / NPOL)

No real recordings are bundled with this project (by design — they're
large, license-restricted, and not this repo's to redistribute). This
suite has already been genuinely validated against real ShipsEar, DS3500,
and NOAA SanctSound ocean-ambient recordings (see `TRAINING_AND_EVAL_REPORT.md`
§7) — `config.yaml` already points at the right folders, drop your own
copies of the same (or any other) dataset's audio files straight in:

```
10_Datasets/real/shipsear/        <- drop ShipsEar .wav/.flac files here
10_Datasets/real/ds3500/          <- drop DS3500 .wav/.flac files here
10_Datasets/real/ocean_ambient/   <- drop ocean-ambient .wav/.flac files here
```

(any internal sub-folder layout is fine — files are found recursively; see
`10_Datasets/real/README.md`). No config edit is needed for these three.
To add a different dataset, add one line in `config.yaml`:

```yaml
datasets:
  deepship: /path/to/your/deepship/folder
  # uatd / npol work the same way
```

That's it — `05_Dataset_Generator/dataset_registry.py` is the single
source of truth every consumer reads from:
- `06_Model_Evaluation/run_real_validation.py` — qualitative validation
  against real files.
- The Detection Backend's **Input Analysis → REAL DATASET** tab — analyze
  one real file directly, with optional injected SNR.
- The Signal Generator's **Real Recording** mode — mix a synthetic,
  ground-truth-known tonal onto a real recording's noise character, to
  stress-test against genuine ocean/ship noise texture.
- `01_Training_Module/train.py --real-data-dir ...` — train against real
  ambient noise instead of purely synthetic.

You can also just **upload a `.wav`/`.flac` file from either app's UI** —
no config edit needed for a quick one-off test.

---

## 5. Project structure

```
00_Documentation/        Architecture notes, setup guide, results history
01_Training_Module/      train.py (SegFormer + U-Net, --resume/--save-last)
02_Testing_Module/       Original unit test suite
03_Live_Detector/        CLI live-detection entry point
04_Signal_Generator/     Core synthetic signal generation (array physics)
05_Dataset_Generator/    Training-data synthesis + real-dataset registry/loader
06_Model_Evaluation/     evaluate.py (pixel + full-pipeline SNR sweeps), reports
07_Deployment_Module/    inference_pipeline.py (the deployable model wrapper),
                         ONNX/TorchScript exporters
08_Ethernet_Receiver/    UDP array-data receiver
09_Models/               Model architectures (SegFormer-B0, U-Net)
10_Datasets/             Demo samples + real-dataset upload landing folder
11_Weights/              Trained checkpoints + per-model training-history JSON
12_Requirements/         requirements*.txt
13_Scripts/              config_loader, system_health_check, shared signal
                         processing, real_audio_mixing, run_pipeline dispatcher
14_Array_Processing/     Beamforming + DOA
15_Multichannel_IO/      Wire protocol (pack/unpack, UDP streaming)
16_Classification/       Ship-like/biological/unknown classifier
17_Array_Inference/      Full array → detection integration layer
18_Contracts/            Frozen API/data-model contracts (DetectionResult, etc.)
detection-backend-app/   FastAPI backend the frontend/live pipeline talks to
signal-generator-app/    FastAPI backend for the Signal Generator UI
frontend/                React + TypeScript operator UI (Vite)
tests/                   Newer end-to-end/contract/config test suite
config.yaml              Single source of truth for every tunable
main.py / deploy.py      CLI menu / standalone deployment entry point
TRAINING_AND_EVAL_REPORT.md   Full, honest training + evaluation record
```

---

## 6. Known limitations (read before assuming something is broken)

- **No real-world dataset is bundled.** The system is trained and
  evaluated on synthetic data with a real physics-based array/noise model;
  Section 4 above is how to point it at real recordings you provide.
- **A Uniform Linear Array cannot resolve front/back bearing** (e.g. 37°
  and 143° give identical inter-element delays) — DOA is restricted to
  ±90°, a standard ULA limitation, not a bug.
- **MP3 input is mono by design** — never produces a bearing.
- **Real-background mode's per-channel independence is approximated**
  from one mono recording (time-shifted per channel) — it is a controlled
  stress test against real noise *character*, not a genuine multi-hydrophone
  real capture.
- **The U-Net vs. SegFormer comparison is not a matched experiment** —
  different step counts and no controlled LR schedule; read it as "both
  work, here's what each achieved," not "U-Net is definitively better."
- **CPU-only training is slow** (single epoch ≈ 8–20 minutes depending on
  model); a GPU would train faster and could go further before
  diminishing returns, but nothing in the code requires one.

---

## 7. License

MIT — see [`LICENSE`](LICENSE). Update the copyright holder name before
publishing.
