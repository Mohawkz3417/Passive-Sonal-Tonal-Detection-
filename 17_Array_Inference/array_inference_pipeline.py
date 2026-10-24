"""
array_inference_pipeline.py
==============================
The integration point requested in Part 2/4 of the original brief, now
producing the FROZEN DetectionResult contract (18_Contracts/data_models.py)
instead of an ad-hoc dataclass:

    ArrayFrame/MonoFrame -> preprocessing -> Delay-and-Sum beamforming
        -> DOA -> spectrogram -> EXISTING SegFormer-B0 -> tonal detection
        -> feature extraction -> classification -> EXISTING Kalman tracker
        -> DetectionResult

Deliberate design choice, unchanged from before: this module does NOT
reimplement any of the spectrogram/SegFormer/CFAR/morphology/CCA/
frequency-estimation/Kalman chain. It imports
07_Deployment_Module.inference_pipeline.TonalDetectorPipeline UNCHANGED
and calls its existing `process_chunk(waveform)` on the beamformer's
OUTPUT waveform. Array processing sits in front of the existing pipeline,
not inside it.

If only 1 channel is supplied, beamforming/DOA is skipped entirely --
`bearing_available=False`, `estimated_bearing_deg=None` (Part 11: never
fabricate a bearing).

Array geometry and DOA scan parameters default to config.yaml's `array:`
and `doa:` sections (previously hardcoded ArrayGeometry(8, 0.2) /
step_deg=1.0 defaults here and in detection_backend.py independently --
now one source of truth, see config_loader).
"""
from __future__ import annotations
import os
import sys
import time
import numpy as np
from dataclasses import asdict
from typing import Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "13_Scripts"))
sys.path.append(os.path.join(ROOT, "09_Models"))
sys.path.append(os.path.join(ROOT, "07_Deployment_Module"))
sys.path.append(os.path.join(ROOT, "14_Array_Processing"))
sys.path.append(os.path.join(ROOT, "16_Classification"))
sys.path.append(os.path.join(ROOT, "18_Contracts"))

from config_loader import load_config, get as cfg_get  # noqa: E402
from inference_pipeline import load_model, TonalDetectorPipeline  # noqa: E402 -- EXISTING, unmodified
from array_simulator import ArrayGeometry  # noqa: E402
from beamforming import estimate_bearing_delay_and_sum, steer_and_sum  # noqa: E402
from data_models import (ArrayFrame, MonoFrame, DetectionResult,  # noqa: E402
                          BeamSpectrumData, now_iso)

try:
    from acoustic_classifier import AcousticClassifier  # noqa: E402
    _CLASSIFIER_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover - classifier is optional per Part 4
    AcousticClassifier = None
    _CLASSIFIER_IMPORT_ERROR = str(e)


def _default_geometry(cfg: dict) -> ArrayGeometry:
    return ArrayGeometry(
        n_elements=int(cfg_get(cfg, "array.n_hydrophones", 8)),
        spacing_m=float(cfg_get(cfg, "array.spacing_m", 0.2)),
        sound_speed_mps=float(cfg_get(cfg, "array.sound_speed_mps", 1500.0)))


class ArrayTonalDetector:
    def __init__(self, checkpoint_path: str, geometry: Optional[ArrayGeometry] = None,
                 fs: float = 16000.0, classifier_path: Optional[str] = None,
                 doa_scan_step_deg: Optional[float] = None, doa_scan_range_deg: Optional[tuple] = None):
        cfg = load_config()
        self.model, _cfg = load_model(checkpoint_path)
        self.checkpoint_path = checkpoint_path
        self.tonal_pipeline = TonalDetectorPipeline(self.model, fs=fs)  # EXISTING pipeline, reused as-is
        self.geometry = geometry or _default_geometry(cfg)
        self.fs = fs
        self.doa_scan_step_deg = doa_scan_step_deg if doa_scan_step_deg is not None \
            else float(cfg_get(cfg, "doa.scan_step_deg", 2.0))
        self.doa_scan_range_deg = doa_scan_range_deg or (
            float(cfg_get(cfg, "doa.scan_min_deg", -90.0)), float(cfg_get(cfg, "doa.scan_max_deg", 90.0)))

        self.classifier = None
        self.classifier_error = _CLASSIFIER_IMPORT_ERROR
        if AcousticClassifier is not None:
            try:
                self.classifier = AcousticClassifier(classifier_path) if classifier_path else AcousticClassifier()
            except FileNotFoundError as e:
                self.classifier_error = str(e)  # classifier is optional; pipeline still runs without it

        self.last_full_result = None  # raw dict from TonalDetectorPipeline.process_chunk (has S_pcen/prob_mask)

    def get_spectrogram_payload(self, max_time_points: int = 250) -> Optional[dict]:
        """Downsampled spectrogram + segmentation mask for UI display -- built
        from the same array the model actually saw, not a separate recompute."""
        r = self.last_full_result
        if r is None:
            return None
        s_pcen = r["S_pcen"]
        prob = r["prob_mask"]
        n_freq, n_time = s_pcen.shape
        step = max(1, n_time // max_time_points)
        return {
            "timestamp": now_iso(), "sample_rate_hz": self.fs,
            "frequencies_hz": r["freqs"].tolist(),
            "times_s": r["times"][::step].tolist(),
            "magnitude": s_pcen[:, ::step].tolist(),
            "detection_mask": (prob[:, ::step] > self.tonal_pipeline.seg_thresh).astype(int).tolist(),
            "seg_threshold": self.tonal_pipeline.seg_thresh,
        }

    def process_frame(self, frame, requested_snr_db: Optional[float] = None) -> DetectionResult:
        """Preferred entry point: takes an ArrayFrame or MonoFrame (see
        18_Contracts/data_models.py / input_adapter.py) so every input
        source (Ethernet/WAV/NPZ/MP3) reaches this pipeline the same way."""
        if isinstance(frame, MonoFrame):
            frame = frame.to_array_frame()
        result = self.process_array_chunk(frame.samples, fs=frame.sample_rate_hz,
                                           requested_snr_db=requested_snr_db)
        result.source = frame.source
        return result

    def process_array_chunk(self, channels: np.ndarray, fs: Optional[float] = None,
                             requested_snr_db: Optional[float] = None) -> DetectionResult:
        fs = fs or self.fs
        warnings = []
        n_channels = channels.shape[0]

        t0 = time.time()
        bearing = None
        beam_spectrum = None
        bearing_available = n_channels > 1
        if bearing_available:
            bearing, angles, powers = estimate_bearing_delay_and_sum(
                channels, fs, self.geometry, scan_range_deg=self.doa_scan_range_deg,
                step_deg=self.doa_scan_step_deg)
            beamformed = steer_and_sum(channels, fs, self.geometry, bearing)
            powers_arr = np.asarray(powers)
            power_db = (10 * np.log10(powers_arr / (np.max(powers_arr) + 1e-30) + 1e-30)).tolist()
            beam_spectrum = BeamSpectrumData(timestamp=now_iso(), angles_deg=list(np.asarray(angles).tolist()),
                                              power_db=power_db, peak_angle_deg=bearing)
        else:
            beamformed = channels[0]
            warnings.append("single-channel input: beamforming/DOA skipped, bearing unavailable")
        array_processing_ms = (time.time() - t0) * 1000.0

        t1 = time.time()
        result = self.tonal_pipeline.process_chunk(beamformed.astype(np.float32))  # EXISTING pipeline call
        model_inference_ms = (time.time() - t1) * 1000.0
        self.last_full_result = result

        detections = result["detections"]
        scene_snr_db = result["snr_db"]

        primary = max(detections, key=lambda d: d.confidence) if detections else None

        classification = classification_confidence = None
        t2 = time.time()
        if primary is not None and self.classifier is not None:
            try:
                classification, classification_confidence, _probs = self.classifier.classify(
                    beamformed, fs, primary.frequency_hz,
                    duration_s=primary.track_hits * (len(beamformed) / fs))
            except Exception as e:
                warnings.append(f"classifier error: {e}")
        elif primary is not None and self.classifier is None:
            warnings.append(f"classifier unavailable: {self.classifier_error}")
        classification_ms = (time.time() - t2) * 1000.0

        # Track status: the EXISTING MultiTonalTracker.step() only ever returns
        # tracks that have already passed confirm_hits consecutive detections
        # (see 07_Deployment_Module/kalman_tracker.py) -- it does not expose a
        # separate "tentative" state through its public API. So the only
        # states this integration can HONESTLY report are CONFIRMED (a track
        # was returned) or NONE (no track this frame) -- reporting TENTATIVE
        # or LOST here would be inventing tracker behavior the current
        # implementation doesn't actually surface.
        track_status = "CONFIRMED" if primary is not None else "NONE"

        return DetectionResult(
            detected=primary is not None,
            timestamp=now_iso(),
            source="array",  # overwritten by process_frame() from frame.source when available
            n_channels=n_channels,
            bearing_available=bearing_available,
            estimated_bearing_deg=bearing,
            frequency_hz=(primary.frequency_hz if primary else None),
            requested_snr_db=requested_snr_db,
            measured_snr_db=(primary.snr_db if primary else scene_snr_db),
            segformer_confidence=(primary.confidence if primary else None),
            classification=classification,
            classification_confidence=classification_confidence,
            track_id=(primary.track_id if primary else None),
            track_status=track_status,
            bandwidth_hz=(primary.bandwidth_hz if primary else None),
            n_harmonics=(primary.n_harmonics if primary else None),
            array_processing_ms=array_processing_ms,
            model_inference_ms=model_inference_ms,
            classification_ms=classification_ms,
            total_ms=array_processing_ms + model_inference_ms + classification_ms,
            beam_spectrum=beam_spectrum,
            all_detections=[asdict(d) for d in detections],
            warnings=warnings,
        )
