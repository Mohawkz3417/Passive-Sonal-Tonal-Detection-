"""
generator_backend.py
=======================
Standalone Signal Generator application backend (separate process/port
from the Detection Backend). Array geometry defaults, the array-Ethernet
port, and CORS origins all come from config.yaml (13_Scripts/config_loader.py)
-- previously ArrayGeometry(8, 0.2) and port 50001 were hardcoded here
independently of detection_backend.py; both now read the same `array:`
and `array_ethernet:` sections.

Reuses, unmodified except for the two additive fields documented in
04_Signal_Generator/synthetic_signal_generator.py:
    - TonalSpec / generate_tonal_waveform / generate_ambient_noise
Wraps them with 14_Array_Processing/array_simulator.py to produce real
multi-channel array data, and 15_Multichannel_IO to transmit it.

Run:
    cd signal-generator-app && uvicorn generator_backend:app --port 8001
Then open http://localhost:8001/ for the simple generator UI.
"""
from __future__ import annotations
import os
import shutil
import sys
import time
import uuid
import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "04_Signal_Generator"))
sys.path.append(os.path.join(ROOT, "14_Array_Processing"))
sys.path.append(os.path.join(ROOT, "15_Multichannel_IO"))
sys.path.append(os.path.join(ROOT, "13_Scripts"))
sys.path.append(os.path.join(ROOT, "05_Dataset_Generator"))

from synthetic_signal_generator import TonalSpec, generate_tonal_waveform, generate_ambient_noise  # noqa: E402
from array_simulator import ArrayGeometry, simulate_array_reception  # noqa: E402
from multichannel_sender import stream_array  # noqa: E402
from config_loader import load_config, get as cfg_get, resolve_path  # noqa: E402
from dataset_registry import list_datasets, get_dataset_dir, KNOWN_DATASETS  # noqa: E402
from real_data_loader import find_audio_files, load_and_resample  # noqa: E402
from real_audio_mixing import mix_real_background_per_channel  # noqa: E402

CFG = load_config()
DEFAULT_N_HYDROPHONES = int(cfg_get(CFG, "array.n_hydrophones", 8))
DEFAULT_SPACING_M = float(cfg_get(CFG, "array.spacing_m", 0.2))
DEFAULT_SOUND_SPEED = float(cfg_get(CFG, "array.sound_speed_mps", 1500.0))
DEFAULT_ARRAY_PORT = int(cfg_get(CFG, "array_ethernet.port", 50001))
DEFAULT_SAMPLE_RATE = float(cfg_get(CFG, "signal.sampling_rate_hz", 16000.0))
CORS_ORIGINS = cfg_get(CFG, "api.cors_origins", ["http://localhost:5173"])

app = FastAPI(title="NPOL Signal Generator")
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Files the user uploads directly through the Signal Generator UI (rather
# than pointing config.yaml at a folder they already have on this machine)
# land here, and are exposed as a special "uploaded" pseudo-dataset -- so
# "add a real recording" always works out of the box with zero config
# editing, on top of the normal config.yaml `datasets:` route for a whole
# ShipsEar/DeepShip/DS3500/UATD folder the user already has locally.
UPLOAD_DIR = os.path.join(ROOT, "10_Datasets", "real_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
_AUDIO_EXTS = (".wav", ".WAV", ".flac", ".FLAC")

# in-memory "last generated signal" store (single-user testing tool, not a database)
_LAST = {"array": None, "fs": None, "info": None}


class GenerateRequest(BaseModel):
    signal_type: str = "Ship-like"          # Ship-like | Biological | Unknown
    frequency_hz: float = 443.0
    snr_db: float = -7.0                    # this is the REQUESTED SNR -- see measured_snr_db in the response
    noise_type: str = "shipping"            # shipping | biological | environmental | background
    doppler: bool = False
    harmonics: bool = True
    drift_hz: float = 0.3
    bearing_deg: float = 37.0
    n_hydrophones: int = DEFAULT_N_HYDROPHONES
    sampling_rate: float = DEFAULT_SAMPLE_RATE
    duration_s: float = 30.0
    array_spacing_m: float = DEFAULT_SPACING_M
    seed: int | None = None


class TransmitRequest(BaseModel):
    host: str = "127.0.0.1"
    port: int = DEFAULT_ARRAY_PORT
    realtime: bool = True


class RealBackgroundRequest(BaseModel):
    """Generate array data the same way /api/generate does (a synthetic,
    exactly-known tonal at a chosen bearing/frequency), but with a REAL
    recording standing in for the ambient noise background instead of the
    synthetic noise profiles -- lets the user test the detector against
    genuine ocean/ship recordings (ShipsEar/DeepShip/DS3500/UATD/whatever
    is configured, or a file they just uploaded) while keeping perfect
    ground truth on the tonal itself (see mix_real_background_per_channel's
    docstring for why the array's per-channel independence is approximated,
    not a real multi-hydrophone capture)."""
    dataset: str                              # e.g. "shipsear", "ds3500", "uploaded"
    file: str | None = None                   # relative path within the dataset folder; None = random
    signal_type: str = "Ship-like"
    frequency_hz: float = 443.0
    snr_db: float = -7.0                      # tonal-vs-real-background scene SNR (requested)
    doppler: bool = False
    harmonics: bool = True
    drift_hz: float = 0.3
    bearing_deg: float = 37.0
    n_hydrophones: int = DEFAULT_N_HYDROPHONES
    sampling_rate: float = DEFAULT_SAMPLE_RATE
    duration_s: float = 30.0
    array_spacing_m: float = DEFAULT_SPACING_M
    seed: int | None = None


def _build_tonal_spec(req: GenerateRequest) -> TonalSpec:
    n_harm = 0
    am_depth, am_rate = 0.1, 2.0
    duty, onset = 1.0, 0.0
    if req.signal_type.lower().startswith("ship"):
        n_harm, am_depth, am_rate, duty = (2 if req.harmonics else 0), 0.15, 2.0, 1.0
    elif req.signal_type.lower().startswith("bio"):
        n_harm, am_depth, am_rate, duty, onset = (1 if req.harmonics else 0), 0.40, 8.0, 0.5, 0.2
    else:  # Unknown / Other
        n_harm, am_depth, am_rate, duty = (1 if req.harmonics else 0), 0.20, 4.0, 0.8

    doppler_ramp = 0.0
    if req.doppler:
        # +/- ~1.5% of f0 over the clip duration -- a first-order closing/opening approximation
        doppler_ramp = (0.015 * req.frequency_hz) / max(req.duration_s, 1e-6)
        doppler_ramp *= np.random.default_rng(req.seed).choice([-1.0, 1.0])

    return TonalSpec(f0=req.frequency_hz, n_harmonics=n_harm, drift_hz=req.drift_hz,
                      am_depth=am_depth, am_rate_hz=am_rate, duty=duty, onset_frac=onset,
                      doppler_ramp_hz_per_s=doppler_ramp)


def _measure_snr_db(beamformed: np.ndarray, fs: float) -> float:
    """Real, measured SNR of the GENERATED array data's beamformed output
    (not the requested value fed to the generator) -- uses the SAME
    noise-floor estimator the live detection pipeline itself uses
    (13_Scripts/signal_processing.py), so this number means the same
    thing here as it does in a live DetectionResult. requested_snr_db and
    measured_snr_db are never the same field (Part 4)."""
    from signal_processing import estimate_noise_floor_spectrum, estimate_snr_db, compute_stft
    _freqs, _times, S_mag = compute_stft(beamformed, fs)
    noise_floor = estimate_noise_floor_spectrum(S_mag, percentile=50.0)
    return float(estimate_snr_db(float(np.mean(S_mag ** 2)), float(np.mean(noise_floor ** 2))))


@app.post("/api/generate")
def generate(req: GenerateRequest):
    if req.n_hydrophones < 1 or req.n_hydrophones > 64:
        raise HTTPException(400, "n_hydrophones must be between 1 and 64")

    geometry = ArrayGeometry(n_elements=req.n_hydrophones, spacing_m=req.array_spacing_m,
                              sound_speed_mps=DEFAULT_SOUND_SPEED)
    alias_limit = geometry.max_alias_free_freq_hz()
    warnings = []
    if req.frequency_hz * (3 if req.harmonics else 1) > alias_limit:
        warnings.append(
            f"Highest expected component (~{req.frequency_hz * (3 if req.harmonics else 1):.0f} Hz) "
            f"exceeds this geometry's alias-free limit ({alias_limit:.0f} Hz) -- DOA may show grating-lobe "
            f"ambiguity. Reduce spacing or frequency, or ignore for a controlled demo.")

    n_samples = int(req.sampling_rate * req.duration_s)
    tone = _build_tonal_spec(req)
    seed = req.seed if req.seed is not None else int(time.time()) % (2**31 - 1)
    source_wave, freq_track = generate_tonal_waveform(n_samples, req.sampling_rate, tone, seed=seed)

    def noise_fn(n, fs, sd):
        return generate_ambient_noise(n, fs, seed=sd, profile=req.noise_type)

    channels = simulate_array_reception(
        source_wave, req.sampling_rate, geometry, req.bearing_deg, req.snr_db, noise_fn, seed=seed)

    try:
        sys.path.append(os.path.join(ROOT, "14_Array_Processing"))
        from beamforming import steer_and_sum
        beamformed = (steer_and_sum(channels, req.sampling_rate, geometry, req.bearing_deg)
                      if channels.shape[0] > 1 else channels[0])
        measured_snr_db = _measure_snr_db(beamformed, req.sampling_rate)
    except Exception as e:
        measured_snr_db = None
        warnings.append(f"measured_snr_db unavailable: {e}")

    gen_id = str(uuid.uuid4())[:8]
    info = {
        "id": gen_id,
        "source_type": req.signal_type, "frequency_hz": req.frequency_hz,
        "requested_snr_db": req.snr_db, "measured_snr_db": measured_snr_db,   # NEVER the same field (Part 4)
        "noise_type": req.noise_type,
        "doppler_enabled": req.doppler, "harmonics_enabled": req.harmonics,
        "frequency_drift_hz_per_s": req.drift_hz,
        "bearing_deg": req.bearing_deg, "n_hydrophones": req.n_hydrophones,
        "spacing_m": req.array_spacing_m, "sound_speed_mps": DEFAULT_SOUND_SPEED,
        "sampling_rate_hz": req.sampling_rate, "duration_s": req.duration_s,
        "n_samples_per_channel": n_samples, "shape": [channels.shape[0], channels.shape[1]],
        "channels_identical": bool(np.allclose(channels[0], channels[1])) if channels.shape[0] > 1 else None,
        "per_channel_rms": [float(np.sqrt(np.mean(c ** 2))) for c in channels],
        "alias_free_limit_hz": alias_limit,
        "seed": seed, "warnings": warnings,
    }
    _LAST["array"] = channels
    _LAST["fs"] = req.sampling_rate
    _LAST["info"] = info
    return info


def _dataset_folder(name: str) -> str | None:
    """Resolve a dataset name to a folder path, including the special
    'uploaded' pseudo-dataset that isn't in config.yaml at all."""
    if name.lower() == "uploaded":
        return UPLOAD_DIR
    return get_dataset_dir(name)


@app.get("/api/real_datasets")
def real_datasets():
    """List every dataset name known to the project (config.yaml's
    `datasets:` section, via the SAME registry run_real_validation.py and
    training use) plus the always-available 'uploaded' pseudo-dataset, and
    report an HONEST file count for each -- 0 if the folder isn't
    configured, doesn't exist, or is empty. This never fabricates a
    dataset's presence; it only reports what's actually on disk right now."""
    out = []
    registry = list_datasets()
    registry.pop("synthetic", None)  # not a real dataset -- it's the built-in generator
    registry["uploaded"] = UPLOAD_DIR  # always present as a folder, even if empty
    for name, configured_path in registry.items():
        folder = _dataset_folder(name)
        exists = bool(folder and os.path.isdir(folder))
        files = []
        if exists:
            try:
                all_files = find_audio_files(folder)
                files = [os.path.relpath(p, folder) for p in all_files[:200]]
            except Exception:  # noqa: BLE001 -- a bad folder shouldn't 500 the whole listing
                files = []
        out.append({
            "name": name,
            "description": KNOWN_DATASETS.get(name, "User-uploaded recordings (via this Signal Generator UI)"
                                                if name == "uploaded" else "Configured real-data folder"),
            "configured_path": folder if name != "uploaded" else None,
            "folder_exists": exists,
            "file_count": len(files),
            "files": files,  # capped at 200 for the dropdown; real folders can have more
        })
    return {"datasets": out}


@app.post("/api/real_datasets/upload")
async def upload_real_dataset_file(file: UploadFile = File(...)):
    """Accept a real recording (WAV/FLAC) uploaded straight from the
    browser, no server filesystem access needed on the user's part. Saved
    into 10_Datasets/real_uploads/ and immediately visible under the
    'uploaded' dataset in GET /api/real_datasets."""
    ext = os.path.splitext(file.filename or "")[1]
    if ext not in _AUDIO_EXTS:
        raise HTTPException(400, f"unsupported file type {ext!r} -- expected one of {_AUDIO_EXTS}")
    safe_name = f"{uuid.uuid4().hex[:8]}_{os.path.basename(file.filename)}"
    dest = os.path.join(UPLOAD_DIR, safe_name)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        info = sf.info(dest)
        duration_s = round(info.frames / info.samplerate, 2)
        sample_rate = info.samplerate
    except Exception as e:  # noqa: BLE001
        os.remove(dest)
        raise HTTPException(400, f"uploaded file could not be read as audio: {e}") from e
    return {"filename": safe_name, "dataset": "uploaded", "duration_s": duration_s,
            "sample_rate_hz": sample_rate}


@app.post("/api/real/generate")
def generate_with_real_background(req: RealBackgroundRequest):
    """Same array-data pipeline as /api/generate, except the ambient noise
    background comes from a REAL recording (a configured dataset folder or
    an upload) instead of the synthetic noise model. The tonal itself
    stays fully synthetic and ground-truth-known (frequency/bearing/
    harmonics/Doppler), so this is a controlled test of 'does detection
    still work against genuine ocean/ship background noise', not a claim
    that the array data itself is a real multi-hydrophone recording."""
    if req.n_hydrophones < 1 or req.n_hydrophones > 64:
        raise HTTPException(400, "n_hydrophones must be between 1 and 64")

    folder = _dataset_folder(req.dataset)
    if not folder or not os.path.isdir(folder):
        raise HTTPException(404, f"dataset '{req.dataset}' is not configured or its folder does not exist -- "
                                  f"see GET /api/real_datasets, or upload a file via /api/real_datasets/upload")
    all_files = find_audio_files(folder)
    if not all_files:
        raise HTTPException(404, f"no audio files found under dataset '{req.dataset}' ({folder})")

    seed = req.seed if req.seed is not None else int(time.time()) % (2**31 - 1)
    if req.file:
        candidate = os.path.join(folder, req.file)
        if candidate not in all_files:
            raise HTTPException(404, f"file '{req.file}' not found in dataset '{req.dataset}'")
        chosen_path = candidate
    else:
        chosen_path = all_files[np.random.default_rng(seed).integers(0, len(all_files))]

    # Load a LONG buffer (not a single duration_s-length crop) so
    # mix_real_background_per_channel can give each array channel an
    # independently time-offset slice of the same recording -- see that
    # function's docstring for why identical-content channels would
    # silently corrupt DOA estimation. Bounded to 5 minutes so a very long
    # dataset file doesn't blow up memory on one request.
    try:
        import soundfile as _sf
        _native_duration_s = _sf.info(chosen_path).frames / _sf.info(chosen_path).samplerate
    except Exception:  # noqa: BLE001
        _native_duration_s = req.duration_s
    # Bug found via real-dataset validation (short-clip datasets, e.g. the
    # 5-second ShipsEar/DS3500/ocean_ambient clips added for real-data
    # testing): when the native file is no longer than the requested
    # duration_s, `max(_native_duration_s, req.duration_s)` collapses to
    # exactly req.duration_s, leaving ZERO spare length for
    # mix_real_background_per_channel() to draw distinct per-channel
    # offsets from (its `max_start = len(buf) - n_target` becomes 0, so
    # every channel silently gets the identical start=0 crop). That
    # reintroduces the exact "identical-content-per-channel biases DOA to
    # broadside" failure mode this whole buffer/offset design was built to
    # avoid. Fix: always request several times duration_s worth of buffer
    # (load_and_resample tiles short native files to fill it), so there is
    # real headroom for independent per-channel time-shifts regardless of
    # how short the source clip is.
    buffer_duration_s = min(max(_native_duration_s, req.duration_s * 4.0), 300.0)
    real_buffer = load_and_resample(chosen_path, target_fs=req.sampling_rate,
                                     duration_s=buffer_duration_s, offset_s=0.0, seed=seed)

    geometry = ArrayGeometry(n_elements=req.n_hydrophones, spacing_m=req.array_spacing_m,
                              sound_speed_mps=DEFAULT_SOUND_SPEED)
    alias_limit = geometry.max_alias_free_freq_hz()
    warnings = [f"Real-background mode: per-channel independence is APPROXIMATED from one mono "
                f"recording ({os.path.relpath(chosen_path, folder)}) -- see docs, not a real "
                f"multi-hydrophone capture."]
    if req.frequency_hz * (3 if req.harmonics else 1) > alias_limit:
        warnings.append(
            f"Highest expected component (~{req.frequency_hz * (3 if req.harmonics else 1):.0f} Hz) "
            f"exceeds this geometry's alias-free limit ({alias_limit:.0f} Hz).")

    n_samples = int(req.sampling_rate * req.duration_s)
    tone = _build_tonal_spec(GenerateRequest(signal_type=req.signal_type, frequency_hz=req.frequency_hz,
                                              snr_db=req.snr_db, doppler=req.doppler, harmonics=req.harmonics,
                                              drift_hz=req.drift_hz, seed=seed, duration_s=req.duration_s))
    source_wave, freq_track = generate_tonal_waveform(n_samples, req.sampling_rate, tone, seed=seed)

    per_channel_real = mix_real_background_per_channel(real_buffer, req.n_hydrophones, req.sampling_rate,
                                                        duration_s=req.duration_s, seed=seed)

    def noise_fn(n, fs, sd):
        # index by call order: MultiTonalTracker-free array simulator calls this
        # once per channel in order, so a simple counter closure is safe here.
        idx = noise_fn.counter
        noise_fn.counter += 1
        base = per_channel_real[idx % len(per_channel_real)]
        return base[:n] if len(base) >= n else np.tile(base, int(np.ceil(n / len(base))))[:n]
    noise_fn.counter = 0

    channels = simulate_array_reception(
        source_wave, req.sampling_rate, geometry, req.bearing_deg, req.snr_db, noise_fn, seed=seed)

    try:
        from beamforming import steer_and_sum
        beamformed = (steer_and_sum(channels, req.sampling_rate, geometry, req.bearing_deg)
                      if channels.shape[0] > 1 else channels[0])
        measured_snr_db = _measure_snr_db(beamformed, req.sampling_rate)
    except Exception as e:  # noqa: BLE001
        measured_snr_db = None
        warnings.append(f"measured_snr_db unavailable: {e}")

    gen_id = str(uuid.uuid4())[:8]
    info = {
        "id": gen_id,
        "source_type": f"Real-Background ({req.dataset})", "frequency_hz": req.frequency_hz,
        "requested_snr_db": req.snr_db, "measured_snr_db": measured_snr_db,
        "noise_type": f"real:{req.dataset}/{os.path.relpath(chosen_path, folder)}",
        "real_background_dataset": req.dataset,
        "real_background_file": os.path.relpath(chosen_path, folder),
        "doppler_enabled": req.doppler, "harmonics_enabled": req.harmonics,
        "frequency_drift_hz_per_s": req.drift_hz,
        "bearing_deg": req.bearing_deg, "n_hydrophones": req.n_hydrophones,
        "spacing_m": req.array_spacing_m, "sound_speed_mps": DEFAULT_SOUND_SPEED,
        "sampling_rate_hz": req.sampling_rate, "duration_s": req.duration_s,
        "n_samples_per_channel": n_samples, "shape": [channels.shape[0], channels.shape[1]],
        "channels_identical": bool(np.allclose(channels[0], channels[1])) if channels.shape[0] > 1 else None,
        "per_channel_rms": [float(np.sqrt(np.mean(c ** 2))) for c in channels],
        "alias_free_limit_hz": alias_limit,
        "seed": seed, "warnings": warnings,
    }
    _LAST["array"] = channels
    _LAST["fs"] = req.sampling_rate
    _LAST["info"] = info
    return info


@app.get("/api/status")
def status():
    return {"has_generated_signal": _LAST["info"] is not None,
            "last": _LAST["info"],
            "array_ethernet_default_port": DEFAULT_ARRAY_PORT}


@app.get("/api/last")
def get_last():
    if _LAST["info"] is None:
        raise HTTPException(404, "no signal generated yet")
    return _LAST["info"]


@app.get("/api/preview")
def preview(max_points: int = 2000):
    """Downsampled channel-0 waveform for a quick plot in the UI (not for analysis)."""
    if _LAST["array"] is None:
        raise HTTPException(404, "no signal generated yet")
    ch0 = _LAST["array"][0]
    step = max(1, len(ch0) // max_points)
    return {"fs_effective_points_per_s": _LAST["fs"] / step, "values": ch0[::step].tolist()}


@app.post("/api/transmit")
def transmit(req: TransmitRequest):
    if _LAST["array"] is None:
        raise HTTPException(404, "no signal generated yet -- call /api/generate first")
    n_sent = stream_array(_LAST["array"], int(_LAST["fs"]), req.host, req.port, realtime=req.realtime)
    return {"packets_sent": n_sent, "host": req.host, "port": req.port,
            "n_channels": _LAST["array"].shape[0], "n_samples": _LAST["array"].shape[1]}


@app.post("/api/save_wav")
def save_wav():
    if _LAST["array"] is None:
        raise HTTPException(404, "no signal generated yet")
    fname = f"array_signal_{_LAST['info']['id']}.wav"
    path = os.path.join(OUTPUT_DIR, fname)
    # soundfile defaults to 16-bit PCM for .wav, full-scale +/-1.0 -- our simulated
    # amplitudes are not bounded to that range. Normalize with ONE global scale
    # factor across all channels (not per-channel) to avoid silently clipping
    # while preserving the inter-channel amplitude/phase relationships that
    # beamforming depends on.
    peak = float(np.max(np.abs(_LAST["array"]))) + 1e-12
    scaled = (_LAST["array"] * (0.9 / peak)).T  # soundfile wants (n_samples, n_channels)
    sf.write(path, scaled, int(_LAST["fs"]))
    return {"path": path, "filename": fname, "normalization_scale_applied": 0.9 / peak}


@app.post("/api/save_array")
def save_array():
    if _LAST["array"] is None:
        raise HTTPException(404, "no signal generated yet")
    fname = f"array_signal_{_LAST['info']['id']}.npz"
    path = os.path.join(OUTPUT_DIR, fname)
    np.savez(path, channels=_LAST["array"], fs=_LAST["fs"], info=_LAST["info"])
    return {"path": path, "filename": fname}


@app.get("/api/download/{filename}")
def download(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(404, "file not found")
    return FileResponse(path, filename=filename)


@app.get("/", response_class=HTMLResponse)
def index():
    static_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "index.html")
    with open(static_path, "r") as f:
        return f.read()
