"""
ethernet_sender_sim.py
========================
Standalone test tool: streams either a synthetic hydrophone clip or a WAV
file over UDP using the same wire format as receiver.py, so the Ethernet
input path can be exercised end-to-end without real NPOL hardware.

Usage:
    python ethernet_sender_sim.py --mode synthetic --snr-db -8 --host 127.0.0.1 --port 50000
    python ethernet_sender_sim.py --mode wav --wav-path my_recording.wav
"""
from __future__ import annotations
import argparse
import os
import socket
import sys
import time

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "04_Signal_Generator"))
from synthetic_signal_generator import generate_clip  # noqa: E402
from receiver import pack_packet                        # noqa: E402


def stream(waveform: np.ndarray, sample_rate: int, host: str, port: int,
           packet_samples: int = 512, realtime: bool = True):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    seq = 0
    n = len(waveform)
    t_per_packet = packet_samples / sample_rate
    print(f"[sender] streaming {n/sample_rate:.1f}s of audio to {host}:{port} "
          f"({packet_samples} samples/packet)")
    for start in range(0, n, packet_samples):
        chunk = waveform[start:start + packet_samples]
        pkt = pack_packet(seq, sample_rate, chunk)
        sock.sendto(pkt, (host, port))
        seq += 1
        if realtime:
            time.sleep(t_per_packet)
    print(f"[sender] done, sent {seq} packets")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["synthetic", "wav"], default="synthetic")
    ap.add_argument("--wav-path", type=str, default=None)
    ap.add_argument("--snr-db", type=float, default=-8.0)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--fs", type=int, default=16000)
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=50000)
    ap.add_argument("--no-realtime", action="store_true")
    args = ap.parse_args()

    if args.mode == "wav":
        import scipy.io.wavfile as wavfile
        if not args.wav_path:
            raise SystemExit("--wav-path is required for --mode wav")
        fs, data = wavfile.read(args.wav_path)
        if data.dtype != np.float32:
            data = data.astype(np.float32) / np.iinfo(data.dtype).max if np.issubdtype(data.dtype, np.integer) else data.astype(np.float32)
        if data.ndim > 1:
            data = data.mean(axis=1)
        waveform, sample_rate = data, fs
    else:
        clip = generate_clip(duration_s=args.duration, fs=args.fs, snr_db=args.snr_db, n_tonals=1)
        waveform, sample_rate = clip.waveform.astype(np.float32), args.fs

    stream(waveform, sample_rate, args.host, args.port, realtime=not args.no_realtime)


if __name__ == "__main__":
    main()
