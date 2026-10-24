"""
multichannel_protocol.py
===========================
Multi-channel extension of 08_Ethernet_Receiver's wire format. The
original single-channel format (magic "NPOL", header <IIII>) is left
completely untouched in receiver.py -- this is a NEW, separate packet
type with its own magic number, so existing single-channel tools
(ethernet_sender_sim.py, receiver.py, live_detector.py) keep working
exactly as before and are not modified.

Packet format (little-endian):
    uint32  magic            = 0x4E504F32  ("NPO2")
    uint32  seq               sequence number
    uint32  sample_rate       Hz
    uint32  n_channels        number of hydrophone channels
    uint32  n_samples         samples PER CHANNEL following
    int16[n_channels * n_samples]   interleaved PCM:
        ch0[0], ch1[0], ..., ch{N-1}[0], ch0[1], ch1[1], ...
    (interleaved-per-sample is the conventional multi-channel PCM layout
    used by real multi-element acquisition hardware.)
"""
from __future__ import annotations
import struct
import numpy as np

MAGIC_V2 = 0x4E504F32           # "NPO2"
HEADER_FMT_V2 = "<IIIII"        # magic, seq, sample_rate, n_channels, n_samples
HEADER_SIZE_V2 = struct.calcsize(HEADER_FMT_V2)

MAX_UDP_PAYLOAD_SAMPLES_PER_CHANNEL = 256  # keeps packets well under typical MTU
                                             # even at 8 channels x int16 x 256 = 4096B + header


def pack_multichannel_packet(seq: int, sample_rate: int, channel_data: np.ndarray) -> bytes:
    """channel_data: (n_channels, n_samples) float array in [-1, 1]."""
    n_channels, n_samples = channel_data.shape
    interleaved = channel_data.T.reshape(-1)  # (n_samples, n_channels) -> flat, per-sample interleaved
    samples_i16 = np.clip(interleaved * 32767, -32768, 32767).astype("<i2")
    header = struct.pack(HEADER_FMT_V2, MAGIC_V2, seq, sample_rate, n_channels, n_samples)
    return header + samples_i16.tobytes()


def unpack_multichannel_packet(data: bytes):
    if len(data) < HEADER_SIZE_V2:
        return None
    magic, seq, sample_rate, n_channels, n_samples = struct.unpack(HEADER_FMT_V2, data[:HEADER_SIZE_V2])
    if magic != MAGIC_V2:
        return None
    expected = HEADER_SIZE_V2 + 2 * n_channels * n_samples
    if len(data) < expected:
        return None
    flat = np.frombuffer(data[HEADER_SIZE_V2:expected], dtype="<i2").astype(np.float32) / 32768.0
    channels = flat.reshape(n_samples, n_channels).T  # -> (n_channels, n_samples)
    return {"seq": seq, "sample_rate": sample_rate, "n_channels": n_channels,
            "n_samples": n_samples, "channels": channels}


def iter_packets(channel_data: np.ndarray, sample_rate: int,
                  packet_samples: int = MAX_UDP_PAYLOAD_SAMPLES_PER_CHANNEL,
                  headroom: float = 0.9):
    """
    Yield (seq, bytes) packets covering the whole (n_channels, n_samples)
    array, encoded as 16-bit PCM.

    IMPORTANT: the wire format's full-scale range is +/-1.0 (like a
    calibrated ADC). This project's simulated array signals are NOT
    guaranteed to stay within that range (a sinusoid + harmonics + noise
    can peak well above 1.0), so this function normalizes the WHOLE array
    by ONE global scale factor (not per-channel, and not per-packet)
    before packetizing. Using a single global factor is essential: scaling
    each channel or each packet independently would corrupt the
    inter-channel amplitude/phase relationships that beamforming and DOA
    estimation depend on. Without this, values above full scale get
    silently clipped by pack_multichannel_packet's np.clip, which distorts
    the waveform non-linearly and would corrupt every downstream stage.
    """
    peak = float(np.max(np.abs(channel_data))) + 1e-12
    scale = headroom / peak
    scaled = channel_data * scale

    n_channels, n_total = scaled.shape
    seq = 0
    for start in range(0, n_total, packet_samples):
        chunk = scaled[:, start:start + packet_samples]
        yield seq, pack_multichannel_packet(seq, sample_rate, chunk)
        seq += 1
