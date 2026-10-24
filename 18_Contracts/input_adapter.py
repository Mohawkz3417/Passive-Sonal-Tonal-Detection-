"""
18_Contracts/input_adapter.py
================================
Ethernet -> ArrayFrame
WAV      -> ArrayFrame (multi-channel) or MonoFrame (mono file)
NPZ      -> ArrayFrame (expects the same {channels, fs, info} shape
            signal-generator-app's /api/save_array already writes)
MP3      -> MonoFrame (always -- see module docstring in data_models.py
            and the project's HOW_TO_RUN.md for why MP3 is mono-only by
            design, not by oversight)

Every adapter raises ValidationError (data_models.py) with one of the
fixed codes on incompatible input -- it never silently invents/resamples
a missing channel, and never guesses a bearing for mono input.

This does NOT duplicate the AI pipeline: adapters only produce frames.
Everything downstream (beamforming, DOA, SegFormer, classification,
Kalman) is the SAME array_inference_pipeline.ArrayTonalDetector for every
input source -- see Part 7 of the freeze spec.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import soundfile as sf

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_models import ArrayFrame, MonoFrame, ValidationError  # noqa: E402

MIN_DURATION_S = 0.5  # below this, most of the pipeline's framing (STFT, tracker warm-up) is meaningless


def _check_duration(n_samples: int, sample_rate_hz: float, source: str):
    duration_s = n_samples / sample_rate_hz
    if duration_s < MIN_DURATION_S:
        raise ValidationError(
            "INSUFFICIENT_DURATION",
            f"{source} clip is {duration_s:.2f}s, shorter than the minimum {MIN_DURATION_S}s",
            {"duration_s": duration_s, "minimum_s": MIN_DURATION_S})


def ethernet_chunk_to_array_frame(channels: np.ndarray, sample_rate_hz: float,
                                   sequence_number: int | None = None) -> ArrayFrame:
    """channels is already [n_channels, n_samples] -- this is what
    MultichannelReceiver.get_latest_chunk() returns. Thin wrapper so every
    input source produces the frame contract the same way."""
    if channels.ndim != 2:
        raise ValidationError("UNSUPPORTED_FORMAT",
                               f"expected 2D [n_channels, n_samples] from the Ethernet receiver, got shape {channels.shape}")
    _check_duration(channels.shape[1], sample_rate_hz, "Ethernet")
    return ArrayFrame(samples=channels, sample_rate_hz=sample_rate_hz, n_channels=channels.shape[0],
                       source="ethernet", sequence_number=sequence_number)


def wav_to_frame(path: str, expected_n_channels: int | None = None,
                  expected_sample_rate_hz: float | None = None,
                  sample_rate_tolerance_hz: float = 1.0) -> ArrayFrame | MonoFrame:
    """Reads any WAV. Multi-channel -> ArrayFrame; mono -> MonoFrame.
    If expected_n_channels/expected_sample_rate_hz are given (the live
    detector's configured array geometry/sample rate) and the file
    doesn't match, raises ValidationError rather than silently resampling
    or duplicating/dropping channels."""
    try:
        data, fs = sf.read(path, always_2d=True)  # (n_samples, n_channels)
    except Exception as e:
        raise ValidationError("DECODE_ERROR", f"could not read WAV file: {e}", {"path": path})

    n_channels = data.shape[1]
    if expected_sample_rate_hz is not None and abs(fs - expected_sample_rate_hz) > sample_rate_tolerance_hz:
        raise ValidationError(
            "UNSUPPORTED_SAMPLE_RATE",
            f"file is {fs} Hz, expected {expected_sample_rate_hz} Hz. This project does not "
            f"silently resample real recordings into the configured pipeline rate.",
            {"file_sample_rate_hz": fs, "expected_sample_rate_hz": expected_sample_rate_hz})
    if expected_n_channels is not None and n_channels != 1 and n_channels != expected_n_channels:
        raise ValidationError(
            "INCOMPATIBLE_CHANNEL_COUNT",
            f"file has {n_channels} channels, expected {expected_n_channels} (or 1 for mono)",
            {"file_channels": n_channels, "expected_channels": expected_n_channels})

    channels = data.T.astype(np.float64)  # -> (n_channels, n_samples)
    _check_duration(channels.shape[1], fs, "WAV")

    if n_channels == 1:
        return MonoFrame(samples=channels[0], sample_rate_hz=float(fs), source="wav_mono",
                          metadata={"path": path})
    return ArrayFrame(samples=channels, sample_rate_hz=float(fs), n_channels=n_channels,
                       source="wav", metadata={"path": path})


def npz_to_array_frame(path: str, expected_n_channels: int | None = None) -> ArrayFrame:
    """Expects the {channels: (n_ch, n_samples), fs: scalar, info: dict}
    layout written by signal-generator-app's /api/save_array -- this is
    the project's own NPZ convention, not a generic audio-NPZ reader."""
    try:
        npz = np.load(path, allow_pickle=True)
    except Exception as e:
        raise ValidationError("DECODE_ERROR", f"could not read NPZ file: {e}", {"path": path})

    if "channels" not in npz or "fs" not in npz:
        raise ValidationError("UNSUPPORTED_FORMAT",
                               "NPZ file missing required 'channels' and/or 'fs' arrays "
                               "(expected the layout written by /api/save_array)", {"path": path, "keys": list(npz.keys())})

    channels = np.asarray(npz["channels"], dtype=np.float64)
    fs = float(npz["fs"])
    if channels.ndim != 2:
        raise ValidationError("UNSUPPORTED_FORMAT", f"'channels' must be 2D, got shape {channels.shape}")
    if expected_n_channels is not None and channels.shape[0] != expected_n_channels:
        raise ValidationError(
            "INCOMPATIBLE_CHANNEL_COUNT",
            f"file has {channels.shape[0]} channels, expected {expected_n_channels}",
            {"file_channels": channels.shape[0], "expected_channels": expected_n_channels})
    _check_duration(channels.shape[1], fs, "NPZ")

    info = npz["info"].item() if "info" in npz and npz["info"].shape == () else None
    return ArrayFrame(samples=channels, sample_rate_hz=fs, n_channels=channels.shape[0],
                       source="npz", metadata={"path": path, "generator_info": info})


def mp3_to_mono_frame(path: str) -> MonoFrame:
    """Always mono -- see module docstring. Requires the `pydub` package
    and the `ffmpeg` binary on PATH."""
    try:
        from pydub import AudioSegment
    except ImportError:
        raise ValidationError("DECODE_ERROR", "pydub not installed -- MP3 support unavailable")
    try:
        audio = AudioSegment.from_file(path, format="mp3")
    except Exception as e:
        raise ValidationError("DECODE_ERROR", f"could not decode MP3 (is ffmpeg installed?): {e}", {"path": path})

    samples = np.array(audio.get_array_of_samples()).astype(np.float64)
    if audio.channels > 1:
        samples = samples.reshape(-1, audio.channels).mean(axis=1)  # deliberate mono-sum, see docstring
    samples /= (2 ** (8 * audio.sample_width - 1))
    _check_duration(len(samples), audio.frame_rate, "MP3")
    return MonoFrame(samples=samples, sample_rate_hz=float(audio.frame_rate), source="mp3",
                      metadata={"path": path, "original_channels": audio.channels})
