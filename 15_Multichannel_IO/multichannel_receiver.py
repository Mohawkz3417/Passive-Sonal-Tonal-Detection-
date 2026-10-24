"""
multichannel_receiver.py
===========================
Background UDP listener for the multi-channel wire format
(multichannel_protocol.py). Mirrors the structure of
08_Ethernet_Receiver/receiver.py's EthernetReceiver, extended to
N channels with real per-channel "last packet seen" bookkeeping, so
the testing UI's "H1 ONLINE / H2 ONLINE / ..." display reflects actual
received data rather than a hardcoded list.
"""
from __future__ import annotations
import socket
import threading
import time
import numpy as np

from multichannel_protocol import unpack_multichannel_packet


class MultichannelReceiver:
    def __init__(self, host: str = "0.0.0.0", port: int = 50001,
                 buffer_seconds: float = 30.0, online_timeout_s: float = 3.0):
        self.host = host
        self.port = port
        self.online_timeout_s = online_timeout_s

        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._running = False

        self.sample_rate: int | None = None
        self.n_channels: int | None = None
        self._buffer_seconds = buffer_seconds
        self._ring: np.ndarray | None = None   # (n_channels, ring_len)
        self._write_pos = 0
        self._total_samples_written = 0
        self._lock = threading.Lock()

        self.packets_received = 0
        self.last_packet_time: float | None = None
        self.last_seq = -1
        self.packets_dropped_out_of_order = 0
        self._last_channel_seen_time: dict[int, float] = {}

    # ------------------------------------------------------------------ lifecycle
    def start(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Real hardware acquisition streams are paced near real-time, but an
        # unpaced burst (e.g. a "fast" test transmit) can still outrun the
        # OS default UDP receive buffer -- widen it defensively. This does
        # NOT make UDP reliable (no flow control exists at this layer); it
        # just moves the realistic loss point further out.
        try:
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        except OSError:
            pass
        self._sock.bind((self.host, self.port))
        self._sock.settimeout(0.5)
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._sock is not None:
            self._sock.close()

    # ------------------------------------------------------------------ receive loop
    def _listen_loop(self):
        while self._running:
            try:
                data, _addr = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            pkt = unpack_multichannel_packet(data)
            if pkt is None:
                continue
            self._handle_packet(pkt)

    def _handle_packet(self, pkt: dict):
        with self._lock:
            if self.sample_rate is None:
                self.sample_rate = pkt["sample_rate"]
                self.n_channels = pkt["n_channels"]
                ring_len = int(self._buffer_seconds * self.sample_rate)
                self._ring = np.zeros((self.n_channels, ring_len), dtype=np.float32)

            if pkt["n_channels"] != self.n_channels:
                return  # ignore mismatched streams rather than corrupt the ring buffer

            if pkt["seq"] < self.last_seq:
                self.packets_dropped_out_of_order += 1
                # still write it (best-effort), but don't rewind bookkeeping
            self.last_seq = max(self.last_seq, pkt["seq"])

            chans = pkt["channels"]
            n = chans.shape[1]
            ring_len = self._ring.shape[1]
            start = self._write_pos
            end = start + n
            if end <= ring_len:
                self._ring[:, start:end] = chans
            else:
                first = ring_len - start
                self._ring[:, start:] = chans[:, :first]
                self._ring[:, :end - ring_len] = chans[:, first:]
            self._write_pos = end % ring_len
            self._total_samples_written += n

            now = time.time()
            self.packets_received += 1
            self.last_packet_time = now
            for i in range(self.n_channels):
                self._last_channel_seen_time[i] = now

    # ------------------------------------------------------------------ consumer API
    def get_latest_chunk(self, n_samples: int) -> np.ndarray | None:
        """Return the most recent (n_channels, n_samples) block, or None if not
        enough data has arrived yet."""
        with self._lock:
            if self._ring is None or self._total_samples_written < n_samples:
                return None
            ring_len = self._ring.shape[1]
            end = self._write_pos
            start = (end - n_samples) % ring_len
            if start < end:
                return self._ring[:, start:end].copy()
            return np.concatenate([self._ring[:, start:], self._ring[:, :end]], axis=1)

    def status(self) -> dict:
        now = time.time()
        with self._lock:
            channel_status = {}
            if self.n_channels:
                for i in range(self.n_channels):
                    last_seen = self._last_channel_seen_time.get(i)
                    online = last_seen is not None and (now - last_seen) < self.online_timeout_s
                    channel_status[f"H{i + 1}"] = "ONLINE" if online else "OFFLINE"
            connected = (self.last_packet_time is not None and
                         (now - self.last_packet_time) < self.online_timeout_s)
            return {
                "connected": connected,
                "sample_rate": self.sample_rate,
                "n_channels": self.n_channels,
                "packets_received": self.packets_received,
                "total_samples_written": self._total_samples_written,
                "packets_dropped_out_of_order": self.packets_dropped_out_of_order,
                "channel_status": channel_status,
                "host": self.host, "port": self.port,
            }
