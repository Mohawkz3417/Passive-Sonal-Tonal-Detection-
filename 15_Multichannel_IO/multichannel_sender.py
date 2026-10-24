"""
multichannel_sender.py
=========================
Streams (n_channels, n_samples) array data to a MultichannelReceiver over
UDP using multichannel_protocol.py. Used by the Signal Generator app's
"transmit" action, and independently runnable/testable from the CLI
(mirrors 08_Ethernet_Receiver/ethernet_sender_sim.py's usage pattern).

Usage:
    python multichannel_sender.py --host 127.0.0.1 --port 50001 \
        --bearing 37 --snr-db -7 --n-hydrophones 8 --duration 30
"""
from __future__ import annotations
import argparse
import os
import socket
import sys
import time

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "04_Signal_Generator"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "14_Array_Processing"))
sys.path.append(os.path.dirname(__file__))

from synthetic_signal_generator import TonalSpec, generate_tonal_waveform, generate_ambient_noise  # noqa: E402
from array_simulator import ArrayGeometry, simulate_array_reception  # noqa: E402
from multichannel_protocol import iter_packets, MAX_UDP_PAYLOAD_SAMPLES_PER_CHANNEL  # noqa: E402


def stream_array(channel_data: np.ndarray, sample_rate: int, host: str, port: int,
                  realtime: bool = True) -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sent = 0
    t_per_packet = MAX_UDP_PAYLOAD_SAMPLES_PER_CHANNEL / sample_rate
    for seq, pkt in iter_packets(channel_data, sample_rate):
        sock.sendto(pkt, (host, port))
        sent += 1
        if realtime:
            time.sleep(t_per_packet)
    sock.close()
    return sent


def build_demo_array(bearing_deg: float, snr_db: float, n_hydrophones: int,
                      fs: float, duration_s: float, f0: float = 443.0,
                      noise_profile: str = "shipping", n_harmonics: int = 2,
                      seed: int = 0) -> np.ndarray:
    n_samples = int(fs * duration_s)
    tone = TonalSpec(f0=f0, n_harmonics=n_harmonics, drift_hz=0.3)
    source_wave, _ = generate_tonal_waveform(n_samples, fs, tone, seed=seed)
    geometry = ArrayGeometry(n_elements=n_hydrophones, spacing_m=0.2)

    def noise_fn(n, fsx, sd):
        return generate_ambient_noise(n, fsx, seed=sd, profile=noise_profile)

    return simulate_array_reception(source_wave, fs, geometry, bearing_deg, snr_db, noise_fn, seed=seed)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=50001)
    ap.add_argument("--bearing", type=float, default=37.0)
    ap.add_argument("--snr-db", type=float, default=-7.0)
    ap.add_argument("--n-hydrophones", type=int, default=8)
    ap.add_argument("--fs", type=float, default=16000.0)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--freq", type=float, default=443.0)
    ap.add_argument("--no-realtime", action="store_true")
    args = ap.parse_args()

    data = build_demo_array(args.bearing, args.snr_db, args.n_hydrophones,
                             args.fs, args.duration, f0=args.freq)
    print(f"[sender] built {data.shape[0]}-channel, {data.shape[1]} samples "
          f"({args.duration}s @ {args.fs} Hz), bearing={args.bearing} deg")
    n_sent = stream_array(data, int(args.fs), args.host, args.port,
                           realtime=not args.no_realtime)
    print(f"[sender] done, sent {n_sent} packets to {args.host}:{args.port}")
