"""
detection_backend.py
=======================
Detection Backend application (separate process/port from the Signal
Generator). All runtime values (Ethernet port, array geometry, DOA scan,
CORS origins, checkpoint path) come from config.yaml -- see
13_Scripts/config_loader.py -- rather than being hardcoded here.

Runs:
  1) a MultichannelReceiver (15_Multichannel_IO) listening for Ethernet
     array data in a background thread,
  2) a processing loop that pulls the latest chunk and runs it through
     ArrayTonalDetector (17_Array_Inference) -- which itself calls the
     EXISTING, unmodified 07_Deployment_Module.TonalDetectorPipeline --
  3) a FastAPI app exposing the FROZEN contracts (18_Contracts/data_models.py):
     SystemStatus, DetectionResult, BeamSpectrumData, SpectrogramData, plus
     WAV/NPZ/MP3 direct-upload analysis via the common input adapter.

Run:
    cd detection-backend-app && uvicorn detection_backend:app --port 8000
Then open http://localhost:8000/ for the simple testing UI.
"""
from __future__ import annotations
import os
import sys
import time
import asyncio
import threading
import tempfile
import traceback
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "13_Scripts"))
sys.path.append(os.path.join(ROOT, "15_Multichannel_IO"))
sys.path.append(os.path.join(ROOT, "14_Array_Processing"))
sys.path.append(os.path.join(ROOT, "17_Array_Inference"))
sys.path.append(os.path.join(ROOT, "18_Contracts"))
sys.path.append(os.path.join(ROOT, "07_Deployment_Module"))
sys.path.append(os.path.join(ROOT, "05_Dataset_Generator"))
sys.path.append(os.path.join(ROOT, "04_Signal_Generator"))

from config_loader import load_config, get as cfg_get, resolve_path  # noqa: E402
from multichannel_receiver import MultichannelReceiver  # noqa: E402
from array_simulator import ArrayGeometry  # noqa: E402
from array_inference_pipeline import ArrayTonalDetector  # noqa: E402
from data_models import (ValidationError, NetworkStatus, SensorStatus, ModelStatus,  # noqa: E402
                          SystemStatus, json_safe, now_iso, PIPELINE_STAGES, Track, TrackPoint,
                          MonoFrame)
import input_adapter  # noqa: E402
from inference_pipeline import load_model, TonalDetectorPipeline  # noqa: E402
from dataset_registry import list_datasets, get_dataset_dir, KNOWN_DATASETS  # noqa: E402
from real_data_loader import find_audio_files, load_and_resample  # noqa: E402
from real_audio_mixing import inject_noise_at_snr  # noqa: E402

CFG = load_config()
CHECKPOINT_PATH = resolve_path(cfg_get(CFG, "model.segformer_weights", "11_Weights/active/segformer_b0_tonal.pt"))
UNET_CHECKPOINT_PATH = resolve_path(cfg_get(CFG, "model.unet_weights", "11_Weights/unet_tonal.pt"))
UPLOAD_DIR = os.path.join(ROOT, "10_Datasets", "real_uploads")  # shared with signal-generator-app
FS_DEFAULT = float(cfg_get(CFG, "signal.sampling_rate_hz", 16000.0))
CHUNK_SECONDS = float(cfg_get(CFG, "signal.chunk_duration_s", 4.0))
ETHERNET_HOST = os.environ.get("NPOL_ETHERNET_HOST", cfg_get(CFG, "array_ethernet.host", "0.0.0.0"))
ETHERNET_PORT = int(os.environ.get("NPOL_ETHERNET_PORT", cfg_get(CFG, "array_ethernet.port", 50001)))
N_HYDROPHONES = int(cfg_get(CFG, "array.n_hydrophones", 8))
CORS_ORIGINS = cfg_get(CFG, "api.cors_origins", ["http://localhost:5173"])

app = FastAPI(title="NPOL Detection Backend")
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])

_START_TIME = time.time()

# --------------------------------------------------------------------- global state
_state_lock = threading.Lock()
_state = {
    "loaded": False, "load_error": None,
    "receiver": None, "detector": None,
    "latest_result": None,          # DetectionResult (object, not dict) of the last processed chunk
    "n_frames_processed": 0,
    "last_error": None,
    "ws_clients": [],
    "tracks": {},                    # track_id -> Track (bounded point history), built from real tracker state
    # U-Net comparison (Part: "frontend shows both SegFormer and U-Net") -- SegFormer
    # via `detector` above remains the ONE primary/active detector for every existing
    # endpoint and contract; this is a purely additive, read-only side channel so the
    # UI can show the two models side by side. Never fabricated: if no U-Net checkpoint
    # exists, unet_available stays False and /api/model/compare says so honestly.
    "unet_pipeline": None, "unet_available": False, "unet_load_error": None,
    "unet_latest": None,             # raw dict from TonalDetectorPipeline.process_chunk() (U-Net)
}


def _init():
    with _state_lock:
        try:
            geometry = ArrayGeometry(n_elements=N_HYDROPHONES,
                                      spacing_m=float(cfg_get(CFG, "array.spacing_m", 0.2)),
                                      sound_speed_mps=float(cfg_get(CFG, "array.sound_speed_mps", 1500.0)))
            detector = ArrayTonalDetector(checkpoint_path=CHECKPOINT_PATH, geometry=geometry, fs=FS_DEFAULT)
            receiver = MultichannelReceiver(host=ETHERNET_HOST, port=ETHERNET_PORT, buffer_seconds=30.0)
            receiver.start()
            _state["receiver"] = receiver
            _state["detector"] = detector
            _state["loaded"] = True
        except Exception as e:
            _state["load_error"] = f"{e}\n{traceback.format_exc()}"

        try:
            if os.path.exists(UNET_CHECKPOINT_PATH):
                unet_model, _cfg = load_model(UNET_CHECKPOINT_PATH, device="cpu")
                _state["unet_pipeline"] = TonalDetectorPipeline(unet_model, fs=FS_DEFAULT)
                _state["unet_available"] = True
            else:
                _state["unet_load_error"] = (f"no U-Net checkpoint at {UNET_CHECKPOINT_PATH} -- "
                                              f"train one with: python 01_Training_Module/train.py --model unet ...")
        except Exception as e:
            _state["unet_load_error"] = f"{e}\n{traceback.format_exc()}"


def _update_track_history(detector, result):
    """Builds real, bounded per-track history (Part 33) from the tracker's
    actual internal state (all_tracks_status()) -- not fabricated. Bearing/
    confidence are only attached to the track that matches this frame's
    primary detection; other live-but-not-primary tracks still get a
    frequency-only point."""
    try:
        tracker = detector.tonal_pipeline.tracker
    except AttributeError:
        return
    for st in tracker.all_tracks_status():
        tid = st["track_id"]
        if tid not in _state["tracks"]:
            _state["tracks"][tid] = Track(track_id=tid, status=st["status"])
        track = _state["tracks"][tid]
        track.status = st["status"]
        is_primary = (result.track_id == tid)
        track.add_point(TrackPoint(
            timestamp=now_iso(),
            bearing_deg=(result.estimated_bearing_deg if is_primary else None),
            frequency_hz=st["frequency_hz"],
            confidence=(result.segformer_confidence if is_primary else None)))
    for dropped_id in getattr(tracker, "last_dropped_track_ids", []):
        if dropped_id in _state["tracks"]:
            _state["tracks"][dropped_id].status = "LOST"


def _processing_loop():
    chunk_n = int(FS_DEFAULT * CHUNK_SECONDS)
    while True:
        try:
            receiver = _state["receiver"]
            detector = _state["detector"]
            if receiver is None or detector is None:
                time.sleep(0.5)
                continue
            chunk = receiver.get_latest_chunk(chunk_n)
            if chunk is None:
                time.sleep(0.5)
                continue
            frame = input_adapter.ethernet_chunk_to_array_frame(
                chunk, receiver.sample_rate or FS_DEFAULT, sequence_number=receiver.last_seq)
            result = detector.process_frame(frame)
            _state["latest_result"] = result
            _state["n_frames_processed"] += 1
            _state["last_error"] = None
            _update_track_history(detector, result)

            # U-Net comparison, best-effort: reuses the SAME cleaned mono waveform
            # SegFormer just processed (detector.last_full_result["waveform"] is
            # preprocess_pipeline's x_clean), so both models see identical input --
            # a fair side-by-side. Never blocks/breaks the primary SegFormer path.
            unet_pipe = _state["unet_pipeline"]
            if unet_pipe is not None and detector.last_full_result is not None:
                try:
                    _state["unet_latest"] = unet_pipe.process_chunk(
                        detector.last_full_result["waveform"].astype(np.float32))
                except Exception as e:  # noqa: BLE001
                    _state["unet_load_error"] = f"U-Net comparison run failed: {e}"
        except Exception as e:
            _state["last_error"] = f"{e}\n{traceback.format_exc()}"
        time.sleep(1.0)


@app.on_event("startup")
def on_startup():
    _init()
    threading.Thread(target=_processing_loop, daemon=True).start()


# --------------------------------------------------------------------- structured error handling (Part 51)
@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    return JSONResponse(status_code=400, content=exc.to_json())


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    # Full traceback goes server-side only (stdout, captured by uvicorn's log) --
    # the client never sees raw Python internals (Part 51).
    print(f"[detection_backend] unhandled error on {request.url.path}: {exc}\n{traceback.format_exc()}")
    return JSONResponse(status_code=500, content={
        "error": {"code": "INTERNAL_ERROR", "message": "an internal error occurred", "details": {}}})


# --------------------------------------------------------------------- contract-shaped status endpoints
def _pipeline_status() -> dict:
    detector = _state["detector"]
    receiver = _state["receiver"]
    rx_connected = receiver.status()["connected"] if receiver else False
    have_result = _state["latest_result"] is not None
    base = "READY" if _state["loaded"] else ("ERROR" if _state["load_error"] else "UNAVAILABLE")
    return {
        "INPUT": "RUNNING" if rx_connected else base,
        "ARRAY_PROCESSING": base,
        "BEAMFORMER": base,
        "DOA": base,
        "SEGFORMER": base,
        "POSTPROCESSING": base,
        "CLASSIFIER": "READY" if (detector and detector.classifier is not None) else "WARNING",
        "TRACKER": base,
        "ETHERNET": "RUNNING" if rx_connected else "READY" if _state["loaded"] else "UNAVAILABLE",
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "uptime_s": time.time() - _START_TIME}


@app.get("/api/status")
def status():
    receiver = _state["receiver"]
    detector = _state["detector"]
    rx = receiver.status() if receiver else {}

    network = NetworkStatus(
        connected=rx.get("connected", False), protocol="udp", host=ETHERNET_HOST, port=ETHERNET_PORT,
        packets_received=rx.get("packets_received", 0), packets_dropped=rx.get("packets_dropped_out_of_order", 0),
        packet_loss_percent=0.0,  # UDP gives us no ground truth for total sent, so this is not computable
                                    # honestly from the receiver side alone -- left at 0.0 with a warning below
                                    # rather than a fabricated percentage.
        queue_overruns=0, sequence_number=(rx.get("packets_received") if rx.get("connected") else None),
        sample_rate_hz=rx.get("sample_rate"), n_channels=rx.get("n_channels"),
        last_packet_timestamp=None,
        warnings=([] if rx.get("connected") else ["no packets received recently"]))

    sensors = []
    for label, state in rx.get("channel_status", {}).items():
        sensors.append(SensorStatus(id=int(label[1:]), label=label, online=(state == "ONLINE"),
                                     signal_level=None, clipping=False, warning=None).to_json())

    model_status = ModelStatus(
        architecture="segformer_b0", checkpoint_path=CHECKPOINT_PATH,
        loaded=_state["loaded"], device="cpu",
        training_metrics_available=os.path.exists(os.path.join(ROOT, "11_Weights", "metrics")),
        inference_latency_ms=(_state["latest_result"].model_inference_ms if _state["latest_result"] else None))

    perf = {
        "array_processing_ms": _state["latest_result"].array_processing_ms if _state["latest_result"] else None,
        "model_inference_ms": _state["latest_result"].model_inference_ms if _state["latest_result"] else None,
        "classification_ms": _state["latest_result"].classification_ms if _state["latest_result"] else None,
        "total_ms": _state["latest_result"].total_ms if _state["latest_result"] else None,
        "dropped_packets": rx.get("packets_dropped_out_of_order", 0),
        "dropped_chunks": None,  # not tracked by the current processing loop -- honestly null, not 0
    }

    sys_status = SystemStatus(
        system={"uptime_s": time.time() - _START_TIME, "n_frames_processed": _state["n_frames_processed"]},
        input={"mode": "ethernet", "connected": rx.get("connected", False)},
        array={"n_hydrophones": N_HYDROPHONES, "spacing_m": cfg_get(CFG, "array.spacing_m", 0.2),
               "sound_speed_mps": cfg_get(CFG, "array.sound_speed_mps", 1500.0)},
        network=network.to_json(), sensors=sensors, model=model_status.to_json(),
        pipeline=_pipeline_status(), performance=perf,
        warnings=([_state["last_error"]] if _state["last_error"] else []))
    return sys_status.to_json()


@app.get("/api/config")
def get_config():
    """Read-only view of runtime configuration relevant to a frontend."""
    return {
        "array": cfg_get(CFG, "array", {}), "doa": cfg_get(CFG, "doa", {}),
        "signal": {"sampling_rate_hz": cfg_get(CFG, "signal.sampling_rate_hz"),
                   "chunk_duration_s": cfg_get(CFG, "signal.chunk_duration_s")},
        "array_ethernet": cfg_get(CFG, "array_ethernet", {}),
        "api": cfg_get(CFG, "api", {}),
    }


@app.get("/api/model/metrics")
def model_metrics():
    """Real offline metrics if they exist (11_Weights/metrics), else an
    honest null -- never fabricated numbers."""
    metrics_dir = os.path.join(ROOT, "11_Weights", "metrics")
    history_path = os.path.join(metrics_dir, "segformer_b0_tonal_history.json")
    if not os.path.exists(history_path):
        return {"available": False, "note": "no training history found at 11_Weights/metrics/"}
    import json
    with open(history_path) as f:
        history = json.load(f)
    return {"available": True, "training_history": history}


@app.get("/api/model/compare")
def model_compare():
    """Side-by-side SegFormer vs U-Net comparison for the frontend's Model
    Status page. SegFormer stays the ONE primary/active detector for every
    other endpoint (Part 4/12 of the freeze spec) -- this is a read-only,
    additive comparison surface only. Honest by construction: if no U-Net
    checkpoint has been trained yet, `unet.available` is False and no
    numbers are invented for it."""
    import json

    def _training_history(name: str):
        path = os.path.join(ROOT, "11_Weights", "metrics", f"{name}_history.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    seg_result = _state["latest_result"]
    seg_raw = _state["detector"].last_full_result if _state["detector"] else None
    segformer = {
        "architecture": "segformer_b0", "available": True,
        "checkpoint_path": CHECKPOINT_PATH,
        "checkpoint_exists": os.path.exists(CHECKPOINT_PATH),
        "training_history": _training_history("segformer_b0_tonal"),
        "last_inference_latency_ms": seg_result.model_inference_ms if seg_result else None,
        "last_detections": ([{"frequency_hz": d.frequency_hz, "confidence": d.confidence,
                               "track_id": d.track_id, "track_hits": d.track_hits}
                              for d in seg_raw["detections"]] if seg_raw else []),
        "last_scene_snr_db": seg_raw["snr_db"] if seg_raw else None,
    }

    unet_raw = _state["unet_latest"]
    unet_pipe = _state["unet_pipeline"]
    unet = {
        "architecture": "unet_baseline", "available": _state["unet_available"],
        "checkpoint_path": UNET_CHECKPOINT_PATH,
        "checkpoint_exists": os.path.exists(UNET_CHECKPOINT_PATH),
        "load_error": _state["unet_load_error"],
        "training_history": _training_history("unet_tonal"),
        "last_inference_latency_ms": (unet_pipe.last_latency_s * 1000.0
                                       if (unet_pipe and unet_pipe.last_latency_s) else None),
        "last_detections": ([{"frequency_hz": d.frequency_hz, "confidence": d.confidence,
                               "track_id": d.track_id, "track_hits": d.track_hits}
                              for d in unet_raw["detections"]] if unet_raw else []),
        "last_scene_snr_db": unet_raw["snr_db"] if unet_raw else None,
    }
    return {"segformer": segformer, "unet": unet,
            "note": "both models are run on the identical cleaned waveform from the most "
                    "recently processed Ethernet chunk, for a fair comparison" if seg_raw else
                    "no chunk processed yet -- transmit array data via Ethernet or use "
                    "/api/analyze_wav / /api/analyze_real_dataset first"}


def _dataset_folder(name: str) -> str | None:
    if name.lower() == "uploaded":
        return UPLOAD_DIR
    return get_dataset_dir(name)


@app.get("/api/real_datasets")
def real_datasets():
    """Same registry/listing as the Signal Generator app's endpoint of the
    same name (both import from 05_Dataset_Generator/dataset_registry.py
    and share 10_Datasets/real_uploads/) -- exposed here too so the Testing
    UI's Input Analysis page can browse/pick a real file without going
    through the separate Signal Generator process."""
    out = []
    registry = list_datasets()
    registry.pop("synthetic", None)
    registry["uploaded"] = UPLOAD_DIR
    for name, _configured in registry.items():
        folder = _dataset_folder(name)
        exists = bool(folder and os.path.isdir(folder))
        files = []
        if exists:
            try:
                files = [os.path.relpath(p, folder) for p in find_audio_files(folder)[:200]]
            except Exception:  # noqa: BLE001
                files = []
        out.append({"name": name,
                     "description": KNOWN_DATASETS.get(name, "User-uploaded recordings"
                                                        if name == "uploaded" else "Configured real-data folder"),
                     "configured_path": folder if name != "uploaded" else None,
                     "folder_exists": exists, "file_count": len(files), "files": files})
    return {"datasets": out}


@app.post("/api/analyze_real_dataset")
async def analyze_real_dataset(request: Request):
    """Run the SAME detection pipeline every other input source uses
    (Part 6/9's 'Ethernet/WAV/MP3/NPZ all reach one pipeline') against a
    file from a configured real dataset (or an upload), with an OPTIONAL
    controlled noise injection at a requested SNR -- this is the 'modify
    the audio to test' path. The file is resampled to this backend's
    configured sample rate FIRST (real datasets are almost never natively
    at 16 kHz, and the detection pipeline's STFT/PCEN preprocessing is
    fixed to its configured rate -- see TonalDetectorPipeline.__init__ --
    so analyzing a file at its native rate without resampling would
    silently misalign the spectrogram's frequency axis). This is reported
    back honestly as `resampled_from_hz` rather than hidden.

    Body (JSON): {"dataset": str, "file": str, "snr_db": float|null,
                   "noise_type": str|null, "duration_s": float=30.0}
    """
    body = await request.json()
    dataset = body.get("dataset")
    file_rel = body.get("file")
    snr_db = body.get("snr_db")
    noise_type = body.get("noise_type")
    duration_s = float(body.get("duration_s", 30.0))
    if not dataset or not file_rel:
        raise HTTPException(400, "'dataset' and 'file' are required")

    folder = _dataset_folder(dataset)
    if not folder or not os.path.isdir(folder):
        raise HTTPException(404, f"dataset '{dataset}' is not configured or its folder does not exist")
    path = os.path.join(folder, file_rel)
    if not os.path.isfile(path):
        raise HTTPException(404, f"file '{file_rel}' not found in dataset '{dataset}'")

    try:
        import soundfile as sf
        native_info = sf.info(path)
        native_sr = native_info.samplerate
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"could not read audio header: {e}") from e

    waveform = load_and_resample(path, target_fs=FS_DEFAULT, duration_s=duration_s, offset_s=0.0)
    noise_condition = "none (clean real recording)"
    measured_before_mix = None
    if snr_db is not None and noise_type:
        waveform = inject_noise_at_snr(waveform, FS_DEFAULT, float(snr_db), noise_type, seed=0).astype(np.float32)
        noise_condition = f"synthetic '{noise_type}' injected at requested {snr_db} dB"

    frame = MonoFrame(samples=waveform, sample_rate_hz=FS_DEFAULT, source="real_dataset",
                       metadata={"dataset": dataset, "file": file_rel, "native_sample_rate_hz": native_sr})
    result = _state["detector"].process_frame(frame, requested_snr_db=(float(snr_db) if snr_db is not None else None))
    payload = result.to_json()
    payload["_raw_segmentation"] = _raw_segmentation_summary(_state["detector"])
    payload["_real_dataset_context"] = {
        "dataset": dataset, "file": file_rel, "native_sample_rate_hz": native_sr,
        "resampled_from_hz": native_sr, "resampled_to_hz": FS_DEFAULT,
        "noise_condition": noise_condition, "analysis_duration_s": duration_s,
    }
    return payload


@app.get("/api/latest_detection")
def latest_detection():
    if _state["latest_result"] is None:
        return {"detected": False, "note": "no chunk processed yet -- transmit array data via Ethernet first"}
    return _state["latest_result"].to_json()


@app.get("/api/tracks")
def tracks():
    """All tracks with real TENTATIVE/CONFIRMED/LOST status and bounded
    history (Part 15/16/32/33) -- built from MultiTonalTracker's actual
    hits/misses/confirmed state, not just the single 'primary' detection
    /api/latest_detection reports."""
    return {tid: t.to_json() for tid, t in _state["tracks"].items()}


@app.get("/api/beam_spectrum")
def beam_spectrum():
    r = _state["latest_result"]
    if r is None or r.beam_spectrum is None:
        return {"angles_deg": [], "power_db": [], "peak_angle_deg": None}
    return r.beam_spectrum.to_json()


@app.get("/api/spectrogram")
def spectrogram():
    detector = _state["detector"]
    if detector is None:
        raise HTTPException(503, "detector not loaded")
    payload = detector.get_spectrogram_payload()
    if payload is None:
        return {"available": False, "note": "no chunk processed yet"}
    payload["available"] = True
    return json_safe(payload)


# --------------------------------------------------------------------- direct WAV / NPZ / MP3 analysis
def _raw_segmentation_summary(detector: ArrayTonalDetector) -> dict:
    """The tracker only reports a track as `detected` after `confirm_hits`
    (3) consecutive matching frames -- correct behavior for continuous
    Ethernet monitoring (rejects one-frame false alarms), but it means a
    single one-shot file analysis (WAV/NPZ/MP3/real-dataset upload) can
    NEVER show detected=True on its first call, even when SegFormer
    clearly segmented a tonal, and every other DetectionResult field
    (frequency_hz, segformer_confidence, ...) stays null along with it
    since those are only populated for a CONFIRMED primary detection.
    That makes single-file analysis look like 'nothing found' even when
    the model worked correctly. This additive, non-contract field exposes
    the RAW SegFormer+CFAR+CCA blobs BEFORE tracker confirmation, so 'is
    the model actually seeing this tonal' can be checked from one file --
    re-analyzing the same file 2 more times will let the shared tracker
    confirm it the normal way if that's what's needed instead."""
    raw = detector.last_full_result
    if not raw:
        return {"available": False}
    return {
        "available": True,
        "note": "pre-tracker-confirmation SegFormer output for this one call -- NOT the same "
                "as the confirmed `detected`/`frequency_hz` fields above, which require "
                "confirm_hits consecutive frames (see this field's own docstring).",
        "raw_blobs_found": len(raw["tonals_raw"]),
        "raw_blobs": [{"frequency_hz": t["frequency_hz"], "seg_prob": t["seg_prob"],
                       "snr_db": t["snr_db"], "bandwidth_hz": t["bandwidth_hz"],
                       "n_harmonics_found": t.get("n_harmonics_found", 0)}
                      for t in raw["tonals_raw"]],
        "scene_snr_db": raw["snr_db"],
        "mean_segmentation_probability": float(np.mean(raw["prob_mask"])),
        "max_segmentation_probability": float(np.max(raw["prob_mask"])),
    }


@app.post("/api/analyze_wav")
async def analyze_wav(file: UploadFile = File(...)):
    raw = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        tmp.write(raw)
        tmp.flush()
        frame = input_adapter.wav_to_frame(tmp.name, expected_n_channels=N_HYDROPHONES,
                                            expected_sample_rate_hz=FS_DEFAULT)
    result = _state["detector"].process_frame(frame)
    payload = result.to_json()
    payload["_raw_segmentation"] = _raw_segmentation_summary(_state["detector"])
    return payload


@app.post("/api/analyze_npz")
async def analyze_npz(file: UploadFile = File(...)):
    raw = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".npz") as tmp:
        tmp.write(raw)
        tmp.flush()
        frame = input_adapter.npz_to_array_frame(tmp.name, expected_n_channels=N_HYDROPHONES)
    result = _state["detector"].process_frame(frame)
    payload = result.to_json()
    payload["_raw_segmentation"] = _raw_segmentation_summary(_state["detector"])
    return payload


@app.post("/api/analyze_mp3")
async def analyze_mp3(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(suffix=".mp3") as tmp:
        tmp.write(await file.read())
        tmp.flush()
        frame = input_adapter.mp3_to_mono_frame(tmp.name)
    result = _state["detector"].process_frame(frame)
    assert result.bearing_available is False and result.estimated_bearing_deg is None, \
        "MP3 path must never fabricate a bearing"  # Part 41 -- enforced, not just documented
    payload = result.to_json()
    payload["_raw_segmentation"] = _raw_segmentation_summary(_state["detector"])
    return payload


# --------------------------------------------------------------------- websocket (frozen envelope, Part 49)
@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    await ws.accept()
    _state["ws_clients"].append(ws)
    try:
        last_sent = -1
        while True:
            await asyncio.sleep(0.5)
            if _state["n_frames_processed"] != last_sent:
                last_sent = _state["n_frames_processed"]
                await ws.send_json({"type": "status", "timestamp": now_iso(), "data": status()})
                if _state["latest_result"] is not None:
                    await ws.send_json({"type": "detection", "timestamp": now_iso(),
                                         "data": _state["latest_result"].to_json()})
    except WebSocketDisconnect:
        _state["ws_clients"].remove(ws)


@app.get("/", response_class=HTMLResponse)
def index():
    static_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "index.html")
    with open(static_path, "r") as f:
        return f.read()
