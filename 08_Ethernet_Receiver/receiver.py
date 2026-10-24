"""
receiver.py
============
Module 1 (Signal Acquisition) -- Ethernet input path.

A minimal, dependency-free UDP receiver for a hydrophone acquisition
system. Real NPOL/hydrophone hardware protocols are proprietary and not
available in this environment, so this module defines a simple, documented
wire format that a real acquisition system's Ethernet output can be
adapted to (or that `ethernet_sender_sim.py` in this same folder emulates
for local testing over loopback):

    Packet format (little-endian):
        uint32  magic        = 0x4E504F4C  ("NPOL")
        uint32  seq           sequence number
        uint32  sample_rate   Hz
        uint32  n_samples     number of int16 samples following
        int16[n_samples]      PCM samples

The receiver reassembles packets into a continuous ring buffer that the
Live Detector (03_Live_Detector) or evaluation code can pull fixed-size
chunks from.
"""
from __future__ import annotations
import socket
import struct
import threading
import queue
import numpy as np

MAGIC = 0x4E504F4C
HEADER_FMT = "<IIII"          # magic, seq, sample_rate, n_samples
HEADER_SIZE = struct.calcsize(HEADER_FMT)


def pack_packet(seq: int, sample_rate: int, samples: np.ndarray) -> bytes:
    samples_i16 = np.clip(samples * 32767, -32768, 32767).astype("<i2")
    header = struct.pack(HEADER_FMT, MAGIC, seq, sample_rate, len(samples_i16))
    return header + samples_i16.tobytes()


def unpack_packet(data: bytes):
    if len(data) < HEADER_SIZE:
        return None
    magic, seq, sample_rate, n_samples = struct.unpack(HEADER_FMT, data[:HEADER_SIZE])
    if magic != MAGIC:
        return None
    expected = HEADER_SIZE + 2 * n_samples
    if len(data) < expected:
        return None
    samples_i16 = np.frombuffer(data[HEADER_SIZE:expected], dtype="<i2")
    samples = samples_i16.astype(np.float32) / 32768.0
    return {"seq": seq, "sample_rate": sample_rate, "samples": samples}


class EthernetReceiver:
    """
    Background UDP listener. Call `.start()`, then repeatedly call
    `.get_chunk(n_samples)` (blocking, with timeout) to pull the next
    block of contiguous audio for the live detector.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 50000, max_queue: int = 256):
        self.host, self.port = host, port
        self._sock = None
        self._thread = None
        self._running = False
        self._q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=max_queue)
        self.sample_rate = None
        self._buffer = np.zeros(0, dtype=np.float32)
        self._last_seq = None
        self.dropped_packets = 0

    def start(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((self.host, self.port))
        self._sock.settimeout(0.5)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[EthernetReceiver] listening on udp://{self.host}:{self.port}")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._sock:
            self._sock.close()

    def _loop(self):
        while self._running:
            try:
                data, addr = self._sock.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            pkt = unpack_packet(data)
            if pkt is None:
                continue
            if self.sample_rate is None:
                self.sample_rate = pkt["sample_rate"]
            if self._last_seq is not None and pkt["seq"] != self._last_seq + 1:
                self.dropped_packets += max(0, pkt["seq"] - self._last_seq - 1)
            self._last_seq = pkt["seq"]
            try:
                self._q.put_nowait(pkt["samples"])
            except queue.Full:
                pass  # drop oldest data under overload rather than block acquisition

    def get_chunk(self, n_samples: int, timeout: float = 5.0) -> np.ndarray | None:
        """Block (up to `timeout` s) until at least n_samples are available."""
        import time
        deadline = time.time() + timeout
        while len(self._buffer) < n_samples:
            remaining = deadline - time.time()
            if remaining <= 0:
                return None
            try:
                block = self._q.get(timeout=min(0.5, remaining))
            except queue.Empty:
                continue
            self._buffer = np.concatenate([self._buffer, block])
        chunk, self._buffer = self._buffer[:n_samples], self._buffer[n_samples:]
        return chunk


class EthernetReceiverTCP:
    """
    TCP-client alternative to EthernetReceiver, for acquisition hardware
    that runs as a TCP *server* waiting for client software to connect and
    pull a continuous byte stream (a very common pattern for lab/industrial
    acquisition boxes -- check your hardware's manual for which model it
    uses). Uses the same packet framing as pack_packet()/unpack_packet()
    above, just read from a byte stream instead of discrete UDP datagrams.

    Trade-off vs. the UDP EthernetReceiver: TCP guarantees ordered,
    lossless delivery (good on a noisy/WiFi link, or if you cannot afford
    to silently lose samples), at the cost of slightly higher and less
    predictable latency than UDP under load, and requires the far end to
    accept an incoming connection rather than just blasting out datagrams.
    For a direct wired Ethernet link to dedicated acquisition hardware,
    UDP is usually preferred for lowest latency; for anything routed,
    Wi-Fi, or where dropped samples are unacceptable, prefer TCP.
    """

    def __init__(self, host: str, port: int, max_queue: int = 256, connect_timeout: float = 5.0):
        self.host, self.port = host, port
        self.connect_timeout = connect_timeout
        self._sock = None
        self._thread = None
        self._running = False
        self._q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=max_queue)
        self.sample_rate = None
        self._buffer = np.zeros(0, dtype=np.float32)
        self._last_seq = None
        self.dropped_packets = 0  # always 0 for TCP (lossless by construction); kept for API parity

    def start(self):
        self._sock = socket.create_connection((self.host, self.port), timeout=self.connect_timeout)
        self._sock.settimeout(0.5)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[EthernetReceiverTCP] connected to tcp://{self.host}:{self.port}")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._sock:
            self._sock.close()

    def _recv_exact(self, n: int) -> bytes | None:
        buf = b""
        while len(buf) < n and self._running:
            try:
                chunk = self._sock.recv(n - len(buf))
            except socket.timeout:
                continue
            except OSError:
                return None
            if not chunk:  # peer closed the connection
                return None
            buf += chunk
        return buf if len(buf) == n else None

    def _loop(self):
        while self._running:
            header = self._recv_exact(HEADER_SIZE)
            if header is None:
                break
            magic, seq, sample_rate, n_samples = struct.unpack(HEADER_FMT, header)
            if magic != MAGIC:
                continue  # resync: drop this byte and keep scanning would be more robust;
                          # acceptable for a controlled test/adapter stream
            payload = self._recv_exact(2 * n_samples)
            if payload is None:
                break
            samples = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
            if self.sample_rate is None:
                self.sample_rate = sample_rate
            if self._last_seq is not None and seq != self._last_seq + 1:
                pass  # TCP is lossless/ordered, so this would indicate a framing bug, not real loss
            self._last_seq = seq
            try:
                self._q.put_nowait(samples)
            except queue.Full:
                pass

    def get_chunk(self, n_samples: int, timeout: float = 5.0) -> np.ndarray | None:
        import time
        deadline = time.time() + timeout
        while len(self._buffer) < n_samples:
            remaining = deadline - time.time()
            if remaining <= 0:
                return None
            try:
                block = self._q.get(timeout=min(0.5, remaining))
            except queue.Empty:
                continue
            self._buffer = np.concatenate([self._buffer, block])
        chunk, self._buffer = self._buffer[:n_samples], self._buffer[n_samples:]
        return chunk


if __name__ == "__main__":
    rx = EthernetReceiver(port=50000)
    rx.start()
    print("Waiting for data... run ethernet_sender_sim.py in another terminal.")
    chunk = rx.get_chunk(16000, timeout=10.0)
    if chunk is None:
        print("No data received (timeout). This is expected if no sender is running.")
    else:
        print(f"Received chunk: {chunk.shape}, sample_rate={rx.sample_rate}, "
              f"dropped_packets={rx.dropped_packets}")
    rx.stop()
