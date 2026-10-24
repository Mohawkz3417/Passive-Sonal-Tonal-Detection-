# HOW TO RUN

> **Note:** This file predates full model training and the frontend/U-Net/
> real-dataset integration pass. For current, accurate status (both models
> ARE trained, the frontend IS built and wired up, real-dataset support
> exists AND has been validated against genuine ShipsEar/DS3500/NOAA
> SanctSound recordings) see [`README.md`](README.md) and
> [`TRAINING_AND_EVAL_REPORT.md`](TRAINING_AND_EVAL_REPORT.md) first. This
> file is kept for its still-accurate lower-level API/module reference
> (sections 1–7, 10); sections 8–9's "not yet trained" framing and
> section 11's "requires training" bullet are superseded — see the two
> documents above.

Covers the multi-channel array extension and the subsequent architecture
freeze. Environment: Python 3.12 (3.10+ should work), CPU-only is fully
supported (no CUDA required anywhere in this suite).

## 0. Setup

```
pip install -r 12_Requirements/requirements.txt
pip install -r 12_Requirements/requirements_additions.txt
```
`pydub` (MP3 support) also needs the `ffmpeg` binary on PATH (`apt install
ffmpeg` / `brew install ffmpeg` / ffmpeg.org for Windows).

## 1. First thing to run: system health check

```
python 13_Scripts/system_health_check.py            # --quick (default), ~50s (mostly one-time torch import)
python 13_Scripts/system_health_check.py --full      # adds a real Ethernet loopback + full synthetic smoke test
```
Every line actually executes the relevant code (generates a real 8-channel
signal, runs a real beamformer/DOA scan, loads and forward-passes the real
checkpoint, runs a real 5-frame Kalman sequence, etc.) -- nothing is a
hardcoded green light. As delivered: **19 PASS / 1 WARNING / 0 FAIL**
(`--quick`), **21 PASS / 1 WARNING / 0 FAIL** (`--full`). The one warning
is "0 of 6 registered datasets have a configured, existing local folder" --
see section 7.

## 2. The master dispatcher

```
python 13_Scripts/run_pipeline.py --health [--full]
python 13_Scripts/run_pipeline.py --check-training-data [--n-samples N]
python 13_Scripts/run_pipeline.py --generate-data [--n-samples N]
python 13_Scripts/run_pipeline.py --train-smoke        # tiny wiring-only run, ~12s, does NOT touch your active checkpoint
python 13_Scripts/run_pipeline.py --train              # PRINTS the exact training command (resolved from config.yaml) -- does not run it
python 13_Scripts/run_pipeline.py --evaluate
python 13_Scripts/run_pipeline.py --real-validation
python 13_Scripts/run_pipeline.py --full-test          # tests/ + 02_Testing_Module/ + --health --full
```
This is a thin subprocess dispatcher, not a reimplementation -- every flag
calls a script that already exists and is independently runnable.

## 3. Signal Generator app

```
cd signal-generator-app
uvicorn generator_backend:app --port 8001 --host 127.0.0.1
```
Open http://localhost:8001/. Array geometry, sample rate, and the target
Ethernet port all default from `config.yaml` (`array:`, `signal:`,
`array_ethernet:`).

## 4. Detection Backend app

```
cd detection-backend-app
uvicorn detection_backend:app --port 8000 --host 127.0.0.1
```
Open http://localhost:8000/. Listens for Ethernet array data on
`array_ethernet.port` (default UDP 50001) automatically at startup.

Train the classifier once (optional but recommended -- the pipeline runs
fine without it, just skips classification):
```
cd 16_Classification && python acoustic_classifier.py --train --n-per-class 150
```

## 5. Complete end-to-end demonstration

1. Start the Detection Backend (section 4).
2. Start the Signal Generator (section 3).
3. `POST /api/generate` with `bearing_deg: 37` (or use the UI).
4. `POST /api/transmit` -> `{"host":"127.0.0.1","port":50001}`.
5. Poll the Detection Backend: `GET /api/status`, `GET /api/latest_detection`,
   `GET /api/beam_spectrum`, `GET /api/spectrogram` -- or watch the UI,
   which polls these every 1.5s.
6. Direct file test (bypasses Ethernet): `POST /api/analyze_wav`,
   `/api/analyze_npz`, `/api/analyze_mp3` (multipart file upload).

Verified in this delivery: 8/8 sensors ONLINE, bearing estimated within the
DOA scan's grid resolution of the true 37 deg, all pipeline stages READY/
RUNNING, WAV/NPZ/MP3 all produce well-formed `DetectionResult`s, and an
intentionally-wrong-channel-count WAV returns a clean
`{"error":{"code":"INCOMPATIBLE_CHANNEL_COUNT",...}}` (HTTP 400) instead of
crashing.

## 6. REST API / WebSocket contract (frozen)

```
GET  /api/health              liveness
GET  /api/status              full SystemStatus (system/input/array/network/sensors/model/pipeline/performance/warnings)
GET  /api/config              read-only view of array/doa/signal/array_ethernet/api config
GET  /api/model/metrics       training history if available, else {"available": false}
GET  /api/latest_detection    last DetectionResult
GET  /api/beam_spectrum       last BeamSpectrumData
GET  /api/spectrogram         last SpectrogramData
POST /api/analyze_wav         multipart file -> DetectionResult
POST /api/analyze_npz         multipart file -> DetectionResult
POST /api/analyze_mp3         multipart file -> DetectionResult (bearing_available always false)
WS   /ws/live                 {"type": "status"|"detection", "timestamp": ..., "data": {...}}
```
Errors are always `{"error": {"code", "message", "details"}}` with an
appropriate HTTP status -- never a raw Python traceback to the client (the
traceback is logged server-side).

Signal Generator app: `POST /api/generate`, `GET /api/status`,
`GET /api/last`, `GET /api/preview`, `POST /api/transmit`,
`POST /api/save_wav`, `POST /api/save_array`, `GET /api/download/{name}`.

## 7. Real-data validation

```
python 06_Model_Evaluation/run_real_validation.py --smoke
python 06_Model_Evaluation/run_real_validation.py --dataset shipsear --files-per-dataset 30
python 06_Model_Evaluation/run_real_validation.py --input /path/to/file.wav
```
**This suite has been genuinely validated this way against real ShipsEar,
DS3500, and NOAA SanctSound ocean-ambient recordings** -- see
`TRAINING_AND_EVAL_REPORT.md` §7 for the full write-up and numbers.
`config.yaml`'s `datasets:` section already points `shipsear`/`ds3500`/
`ocean_ambient` at `10_Datasets/real/<name>/` -- those folders ship empty
(the recordings themselves are not redistributed); drop your own copies
in and re-run the commands above with no config edit needed. `deepship`/
`uatd`/`npol` are still `null` -- add a `datasets.<name>: <path>` line the
same way to wire up another dataset. The script reuses the existing
`05_Dataset_Generator/dataset_registry.py` (already wired to that same
config section) rather than a second dataset-finding mechanism.

Public underwater-acoustic datasets generally give file-level or no
labels -- not per-frequency tonal ground truth, and these are mono
recordings with no array geometry. So this validation is honestly
qualitative (did the file load, did the pipeline run, what came out),
written to `06_Model_Evaluation/real_validation/validation_results.csv` +
`validation_summary.json`. It does **not** compute segmentation
precision/recall/F1/IoU against these files (no ground truth exists to
compute them against), and `bearing_deg` is `null` for every row (mono
input). See `00_Documentation/REAL_DATA_GUIDE.md` for the existing
(unmodified) real-noise-injection training method, which is the
higher-value way these datasets already feed into this suite.

## 8. Training

**Not run as part of this delivery** (see project rules). Check data
coverage first, then use the printed command:
```
python 13_Scripts/run_pipeline.py --check-training-data
python 13_Scripts/run_pipeline.py --train      # prints the exact command; run it yourself
```
Current config: SNR range -12 to +2 dB (hard/difficult band -12 to -7 dB,
sampled 70% of the time), frequency range 150-3500 Hz across 5 bands,
50 epochs x 200 steps x batch 8. All configurable in `config.yaml`'s
`training:` section -- no source changes needed to adjust epochs/SNR
range/etc.

Coverage checker result at delivery time (300 samples): all 8 SNR bins hit,
all 5 frequency bands hit, harmonics/drift/intermittent/zero-tonal
negatives all present, Doppler-shifted tonals present (see section 11) --
**COVERAGE CHECK: PASS**.

## 9. Checkpoint organization

```
11_Weights/
    active/segformer_b0_tonal.pt      <- the backend loads THIS path (config.yaml: model.segformer_weights)
    archive/                           <- old checkpoints go here if you keep them
    metrics/segformer_b0_tonal_history.json
```
Replacing `active/segformer_b0_tonal.pt` with a newly-trained, same-architecture
checkpoint updates the runtime model automatically -- no code changes.

**Known limitation, unchanged from before**: the checkpoint shipped in
`active/` is the original ~80-gradient-step wiring-test run (see
`00_Documentation/RESULTS.md`) -- confirmed again in this pass (flat ~0.3
probability-mask output regardless of SNR, 0 raw detections). Every stage
around it (array sim, beamforming, DOA, contracts, input adapters,
classifier) is independently verified and real; only the trained detection
*sensitivity* requires the training run in section 8.

## 10. Testing

```
cd tests && python -m pytest -v .                    # 24 passed, 1 informational skip
cd 02_Testing_Module && python -m pytest -v .         # 34 passed (original suite, untouched logic)
```
New test files added this pass: `tests/test_config.py` (config
centralization/consistency), `tests/test_contracts.py` (frozen data
models, input adapters, JSON-safety, bearing-fabrication guards).
`tests/test_end_to_end.py` (from the array-extension pass) was updated in
3 places to match the frozen `DetectionResult`/`SystemStatus` shapes
(checkpoint path now resolved via config instead of hardcoded, `/api/status`
assertions updated to the nested contract, spectrogram payload key names
updated) -- not removed, not weakened.

Given the scope of the original spec (a 9-file test-suite reorganization
was requested), this delivery consolidates into 3 focused files rather
than 9 -- `test_config.py`, `test_contracts.py`, `test_end_to_end.py` --
covering the same ground (config, contracts/input-adapters/API, and full
pipeline integration) without the overhead of near-empty files. Tracking,
classification, and array-processing mechanics are covered inside
`test_end_to_end.py` and `system_health_check.py` rather than separate
`test_tracking.py`/`test_classification.py`/`test_array_processing.py`
files, per the spec's own "do not unnecessarily reorganize a working test
structure" instruction.

## 11. Known limitations (read before assuming something is broken)

- **SegFormer detection sensitivity requires training** -- see section 9.
- **A Uniform Linear Array cannot resolve front/back bearing** (37 deg and
  143 deg give identical delays) -- DOA scan is restricted to +/-90 deg.
- **Live DOA uses a 2-degree scan grid** (`doa.scan_step_deg` in config,
  tunable) for speed -- estimates land on the nearest grid point.
  Standalone verification (`14_Array_Processing/verify_array_doa.py`) uses
  a 1-degree grid and gets exact results.
- **MP3 is single-channel by design** -- never produces a bearing (enforced
  by an assertion in `detection_backend.py`'s MP3 endpoint, not just
  documented).
- **`training.snr_eval_grid_db` now drives `evaluate.py`'s SNR sweep**
  (was a hardcoded list) -- if you compare against old evaluation runs,
  note the grid changed from `[-10..+20]` to `[-12..+2]` to match the new
  training range.
- **Core training generator (`generate_clip`) did not vary noise *profile*
  per sample** (shipping/biological/environmental/background) -- it uses
  one default ambient-noise model; the profile-selectable noise added for
  the Signal Generator app is not yet wired into training data generation.
  Flagged, not silently left uncovered -- see
  `01_Training_Module/check_training_data.py`'s output.
- **Real-data validation is qualitative, not statistical** -- see section 7.
- **No PostgreSQL, Redis, Kafka, Celery, Docker, GraphQL, or auth** --
  explicitly out of scope for this pass.
- **CORS origins default to `http://localhost:5173`** (Vite's default) --
  add your actual frontend origin(s) to `config.yaml`'s `api.cors_origins`
  before building against this from a browser.
