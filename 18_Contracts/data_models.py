"""
18_Contracts/data_models.py
==============================
Frozen internal data contracts. This is the ONLY place these shapes are
defined -- both backend apps (signal-generator-app, detection-backend-app)
and the test suite import from here rather than each inventing their own
dict shape, which is what "freezing the architecture" actually means in
code: one definition, everything else consumes it.

Design rules followed throughout this file (do not relax these when
extending it):
  - Every field that might not be available is Optional and defaults to
    None / False -- NEVER a fabricated placeholder value.
  - `to_json()` on every class returns ONLY JSON-safe types (no numpy
    scalars/arrays, no torch tensors, no Path objects) -- see
    `json_safe()` below, which every FastAPI response in this project
    should be passed through.
  - "requested" and "measured" quantities (e.g. SNR) are always separate
    fields -- never the same field reused for both.
"""
from __future__ import annotations
import time
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional, Any


# --------------------------------------------------------------------- JSON safety
def json_safe(obj: Any) -> Any:
    """Recursively convert numpy/torch/dataclass values into plain JSON-safe
    Python types. Use this to wrap ANY value before it goes into a FastAPI
    response, not just the contract classes below."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if hasattr(obj, "to_json"):
        return obj.to_json()
    if hasattr(obj, "__dataclass_fields__"):
        return json_safe(asdict(obj))
    # torch tensors (avoid importing torch here just for an isinstance check)
    if type(obj).__module__.startswith("torch") and hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)  # last-resort fallback (e.g. Path) -- never crash serialization


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# --------------------------------------------------------------------- frames (input contract)
@dataclass
class ArrayFrame:
    """Multi-channel input, standardized as [n_channels, n_samples] --
    NEVER [n_samples, n_channels]. Every array-processing module in this
    project (array_simulator, beamforming, array_inference_pipeline) is
    written against this shape; input adapters are responsible for
    transposing whatever they read into this convention before it goes
    anywhere else."""
    samples: np.ndarray             # shape (n_channels, n_samples)
    sample_rate_hz: float
    n_channels: int
    source: str                     # "ethernet" | "wav" | "npz" | "synthetic"
    timestamp: str = field(default_factory=now_iso)
    sequence_number: Optional[int] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.samples.ndim != 2:
            raise ValueError(f"ArrayFrame.samples must be 2D [n_channels, n_samples], got shape {self.samples.shape}")
        if self.samples.shape[0] != self.n_channels:
            raise ValueError(f"ArrayFrame.n_channels={self.n_channels} does not match "
                              f"samples.shape[0]={self.samples.shape[0]}")

    def to_json(self) -> dict:
        return {"sample_rate_hz": self.sample_rate_hz, "n_channels": self.n_channels,
                "n_samples": int(self.samples.shape[1]), "source": self.source,
                "timestamp": self.timestamp, "sequence_number": self.sequence_number,
                "metadata": json_safe(self.metadata)}  # samples deliberately excluded -- too large for a status payload


@dataclass
class MonoFrame:
    """Single-channel input (MP3 path, or a mono WAV/dataset file). Never
    carries a bearing -- see DetectionResult.bearing_available."""
    samples: np.ndarray              # shape (n_samples,)
    sample_rate_hz: float
    source: str                      # "mp3" | "wav_mono" | "dataset"
    timestamp: str = field(default_factory=now_iso)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.samples.ndim != 1:
            raise ValueError(f"MonoFrame.samples must be 1D, got shape {self.samples.shape}")

    def to_array_frame(self) -> ArrayFrame:
        """A MonoFrame is always a valid 1-channel ArrayFrame (but not vice
        versa) -- lets MonoFrame reuse ArrayTonalDetector directly, which
        already skips beamforming/DOA for n_channels == 1."""
        return ArrayFrame(samples=self.samples[np.newaxis, :], sample_rate_hz=self.sample_rate_hz,
                           n_channels=1, source=self.source, timestamp=self.timestamp,
                           metadata=self.metadata)

    def to_json(self) -> dict:
        return {"sample_rate_hz": self.sample_rate_hz, "n_samples": int(len(self.samples)),
                "source": self.source, "timestamp": self.timestamp, "metadata": json_safe(self.metadata)}


# --------------------------------------------------------------------- validation errors (Part 22/51)
class ValidationError(Exception):
    """Raised by input adapters on incompatible input. `code` is one of the
    fixed strings below -- callers (FastAPI handlers) turn this into the
    frozen {"error": {"code", "message", "details"}} envelope, never a raw
    stack trace."""
    CODES = ("INCOMPATIBLE_CHANNEL_COUNT", "UNSUPPORTED_SAMPLE_RATE",
             "INSUFFICIENT_DURATION", "UNSUPPORTED_FORMAT", "DECODE_ERROR")

    def __init__(self, code: str, message: str, details: Optional[dict] = None):
        assert code in self.CODES, f"unknown ValidationError code {code!r}"
        self.code = code
        self.details = details or {}
        super().__init__(message)

    def to_json(self) -> dict:
        return {"error": {"code": self.code, "message": str(self), "details": json_safe(self.details)}}


# --------------------------------------------------------------------- detection / classification output
@dataclass
class DetectionResult:
    detected: bool
    timestamp: str
    source: str                                  # "ethernet" | "wav" | "npz" | "mp3"
    n_channels: int

    bearing_available: bool
    estimated_bearing_deg: Optional[float]

    frequency_hz: Optional[float]
    requested_snr_db: Optional[float]             # only set when the input carries a KNOWN ground-truth SNR (synthetic/generator)
    measured_snr_db: Optional[float]              # always set when a detection/scene SNR estimate is available

    segformer_confidence: Optional[float]

    classification: Optional[str]
    classification_confidence: Optional[float]

    track_id: Optional[int]
    track_status: str                             # "TENTATIVE" | "CONFIRMED" | "LOST" | "CLOSED" | "NONE"

    bandwidth_hz: Optional[float]
    n_harmonics: Optional[int]

    array_processing_ms: float
    model_inference_ms: float
    classification_ms: float
    total_ms: float

    beam_spectrum: Optional["BeamSpectrumData"] = None
    all_detections: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def to_json(self) -> dict:
        d = asdict(self)
        if self.beam_spectrum is not None:
            d["beam_spectrum"] = self.beam_spectrum.to_json()
        return json_safe(d)


@dataclass
class DetectionEvidence:
    """Rule-based, computed-only "why was this detected" fields (Part 48).
    Every field is None unless actually available -- no LLM, no invented values."""
    segformer_confidence: Optional[float] = None
    frequency_hz: Optional[float] = None
    snr_db: Optional[float] = None
    bandwidth_hz: Optional[float] = None
    harmonic_count: Optional[int] = None
    bearing_deg: Optional[float] = None
    bearing_stability: Optional[float] = None
    beam_peak_strength_db: Optional[float] = None
    classification: Optional[str] = None
    classification_confidence: Optional[float] = None
    track_confirmed: bool = False

    def to_json(self) -> dict:
        return json_safe(asdict(self))


# --------------------------------------------------------------------- contact / track (Part 15/16)
TRACK_STATES = ("TENTATIVE", "CONFIRMED", "LOST", "CLOSED")


@dataclass
class TrackPoint:
    timestamp: str
    bearing_deg: Optional[float]
    frequency_hz: Optional[float]
    confidence: Optional[float]


@dataclass
class Track:
    track_id: int
    status: str                       # one of TRACK_STATES -- never invented states
    points: list = field(default_factory=list)   # list[TrackPoint], bounded (see MAX_TRACK_HISTORY)
    MAX_TRACK_HISTORY = 200

    def add_point(self, pt: TrackPoint):
        self.points.append(pt)
        if len(self.points) > self.MAX_TRACK_HISTORY:
            self.points.pop(0)

    def to_json(self) -> dict:
        return {"track_id": self.track_id, "status": self.status,
                "points": [json_safe(asdict(p)) for p in self.points]}


@dataclass
class Contact:
    detection_id: str
    detection: DetectionResult
    track_id: Optional[int]

    def to_json(self) -> dict:
        return {"detection_id": self.detection_id, "detection": self.detection.to_json(),
                "track_id": self.track_id}


# --------------------------------------------------------------------- status contracts (Part 9/10/44-47)
@dataclass
class NetworkStatus:
    connected: bool
    protocol: str
    host: str
    port: int
    packets_received: int
    packets_dropped: int
    packet_loss_percent: float
    queue_overruns: int
    sequence_number: Optional[int]
    sample_rate_hz: Optional[float]
    n_channels: Optional[int]
    last_packet_timestamp: Optional[str]
    warnings: list = field(default_factory=list)

    def to_json(self) -> dict:
        return json_safe(asdict(self))


@dataclass
class SensorStatus:
    id: int
    label: str
    online: bool
    signal_level: Optional[float] = None   # None unless actually measured -- never invented
    clipping: bool = False
    warning: Optional[str] = None

    def to_json(self) -> dict:
        return json_safe(asdict(self))


@dataclass
class ModelStatus:
    architecture: str
    checkpoint_path: str
    loaded: bool
    device: str
    training_metrics_available: bool
    inference_latency_ms: Optional[float] = None

    def to_json(self) -> dict:
        return json_safe(asdict(self))


PIPELINE_STAGES = ("INPUT", "ARRAY_PROCESSING", "BEAMFORMER", "DOA", "SEGFORMER",
                   "POSTPROCESSING", "CLASSIFIER", "TRACKER", "ETHERNET")
PIPELINE_STATES = ("READY", "RUNNING", "WARNING", "ERROR", "UNAVAILABLE")


@dataclass
class SystemStatus:
    system: dict
    input: dict
    array: dict
    network: dict
    sensors: list
    model: dict
    pipeline: dict
    performance: dict
    warnings: list = field(default_factory=list)

    def to_json(self) -> dict:
        return json_safe(asdict(self))


# --------------------------------------------------------------------- spectrogram / beam spectrum (Part 12/13)
@dataclass
class BeamSpectrumData:
    timestamp: str
    angles_deg: list
    power_db: list
    peak_angle_deg: Optional[float]

    def to_json(self) -> dict:
        return json_safe(asdict(self))


@dataclass
class SpectrogramData:
    timestamp: str
    sample_rate_hz: float
    frequencies_hz: list
    times_s: list
    magnitude: list
    detection_mask: list

    def to_json(self) -> dict:
        return json_safe(asdict(self))
