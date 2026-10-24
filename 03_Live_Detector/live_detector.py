"""
live_detector.py
==================
Module 10: Live GUI.

A Tkinter + Matplotlib desktop application (Windows/Linux/Mac compatible,
offline, no web server needed) that runs the full detection pipeline in
real time and displays:

    - Live waveform
    - Live (raw) spectrogram
    - PCEN spectrogram
    - Segmentation mask + detected-tonal overlay
    - Detected frequencies, confidence, estimated SNR
    - Inference time / processing FPS / CPU / RAM usage
    - Current model, current time
    - Detection history (scrolling table + log)
    - Start / Stop / Pause / Save controls
    - CSV + TXT logging of every confirmed detection (13_Scripts/log_utils.py)

Input sources (Module 1):
    --source synthetic   : built-in synthetic generator (no files needed)
    --source wav         : replay a WAV file in real time
    --source ethernet     : live UDP/TCP stream via 08_Ethernet_Receiver

Run:
    python live_detector.py --source synthetic --snr-db -8
    python live_detector.py --model-manager     # pick a model interactively from 11_Weights/
"""
from __future__ import annotations
import argparse
import os
import sys
import threading
import queue
from datetime import datetime

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "07_Deployment_Module"))
sys.path.append(os.path.join(ROOT, "04_Signal_Generator"))
sys.path.append(os.path.join(ROOT, "08_Ethernet_Receiver"))
sys.path.append(os.path.join(ROOT, "09_Models"))
sys.path.append(os.path.join(ROOT, "13_Scripts"))

from inference_pipeline import load_model, TonalDetectorPipeline   # noqa: E402
from synthetic_signal_generator import generate_clip                # noqa: E402
from config_loader import load_config, get as cfg_get               # noqa: E402
from log_utils import DetectionLogger                                # noqa: E402

try:
    import psutil
    _HAVE_PSUTIL = True
except ImportError:
    _HAVE_PSUTIL = False


class ChunkSource:
    """Uniform interface over {synthetic, wav, ethernet} input sources.
    (Unchanged from the original implementation -- this class already worked.)"""

    def __init__(self, mode: str, fs: float, chunk_s: float, snr_db: float = -8.0,
                 wav_path: str | None = None, eth_host="0.0.0.0", eth_port=50000,
                 eth_protocol: str = "udp"):
        self.mode = mode
        self.fs = fs
        self.chunk_n = int(chunk_s * fs)
        self.snr_db = snr_db
        if mode == "wav":
            import scipy.io.wavfile as wavfile
            sr, data = wavfile.read(wav_path)
            if np.issubdtype(data.dtype, np.integer):
                data = data.astype(np.float32) / np.iinfo(data.dtype).max
            if data.ndim > 1:
                data = data.mean(axis=1)
            self._wav = data.astype(np.float32)
            self._wav_pos = 0
            self.fs = sr
            self.chunk_n = int(chunk_s * sr)
        elif mode == "ethernet":
            if eth_protocol == "tcp":
                from receiver import EthernetReceiverTCP
                self._rx = EthernetReceiverTCP(host=eth_host, port=eth_port)
            else:
                from receiver import EthernetReceiver
                self._rx = EthernetReceiver(host=eth_host, port=eth_port)
            self._rx.start()

    def next_chunk(self) -> np.ndarray | None:
        if self.mode == "synthetic":
            n_tonals = np.random.choice([0, 1, 1, 2])
            clip = generate_clip(duration_s=self.chunk_n / self.fs, fs=self.fs,
                                  snr_db=self.snr_db, n_tonals=int(n_tonals))
            return clip.waveform
        elif self.mode == "wav":
            if self._wav_pos + self.chunk_n > len(self._wav):
                self._wav_pos = 0  # loop
            chunk = self._wav[self._wav_pos:self._wav_pos + self.chunk_n]
            self._wav_pos += self.chunk_n
            return chunk
        elif self.mode == "ethernet":
            return self._rx.get_chunk(self.chunk_n, timeout=5.0)
        raise ValueError(self.mode)


class LiveDetectorApp:
    def __init__(self, root: tk.Tk, pipeline: TonalDetectorPipeline, source: ChunkSource,
                 log_path: str, chunk_s: float, model_name: str = "segformer_b0_tonal"):
        self.root = root
        self.pipeline = pipeline
        self.source = source
        self.chunk_s = chunk_s
        self.model_name = model_name
        self.root.title("NPOL Tonal Detection Suite -- Live Detector")

        self._build_layout()

        self.logger = DetectionLogger(csv_path=log_path)

        self._work_q: "queue.Queue" = queue.Queue(maxsize=2)
        self._stop = False
        self._paused = False
        self._running = True
        self._last_result: dict | None = None
        self._frame_times: list[float] = []   # for processing FPS
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        self.root.after(cfg_get(load_config(), "gui.refresh_ms", 150), self._poll_results)
        self.root.after(cfg_get(load_config(), "gui.clock_refresh_ms", 1000), self._tick_clock)
        self.root.after(cfg_get(load_config(), "gui.resource_refresh_ms", 2000), self._tick_resources)

    # ------------------------------------------------------------------ UI
    def _build_layout(self):
        main = ttk.Frame(self.root, padding=6)
        main.pack(fill=tk.BOTH, expand=True)

        # ---- toolbar (Start/Stop/Pause/Save) -----------------------------
        toolbar = ttk.Frame(main)
        toolbar.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))
        self.btn_start = ttk.Button(toolbar, text="\u25b6 Start", command=self._on_start)
        self.btn_start.pack(side=tk.LEFT, padx=2)
        self.btn_pause = ttk.Button(toolbar, text="\u23f8 Pause", command=self._on_pause)
        self.btn_pause.pack(side=tk.LEFT, padx=2)
        self.btn_stop = ttk.Button(toolbar, text="\u23f9 Stop", command=self._on_stop)
        self.btn_stop.pack(side=tk.LEFT, padx=2)
        self.btn_save = ttk.Button(toolbar, text="\U0001f4be Save Snapshot", command=self._on_save)
        self.btn_save.pack(side=tk.LEFT, padx=2)

        self.lbl_status = ttk.Label(toolbar, text="Status: running", foreground="green")
        self.lbl_status.pack(side=tk.LEFT, padx=16)
        self.lbl_clock = ttk.Label(toolbar, text="--:--:--")
        self.lbl_clock.pack(side=tk.RIGHT, padx=4)
        ttk.Label(toolbar, text="Time:").pack(side=tk.RIGHT)

        body = ttk.Frame(main)
        body.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(body)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right = ttk.Frame(body, width=360)
        right.pack(side=tk.RIGHT, fill=tk.Y)

        figsize = tuple(cfg_get(load_config(), "gui.figsize", [9, 7]))
        self.fig = Figure(figsize=figsize, dpi=100)
        self.ax_wave = self.fig.add_subplot(411)
        self.ax_spec = self.fig.add_subplot(412)
        self.ax_pcen = self.fig.add_subplot(413)
        self.ax_mask = self.fig.add_subplot(414)
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ---- model / status panel ----------------------------------------
        model_frame = ttk.LabelFrame(right, text="Model", padding=6)
        model_frame.pack(fill=tk.X, pady=(0, 6))
        self.lbl_model = ttk.Label(model_frame, text=f"Active: {self.model_name}")
        self.lbl_model.pack(anchor="w")

        ttk.Label(right, text="Detected Tonals", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        cols = ("freq", "conf", "snr", "harm")
        self.tree = ttk.Treeview(right, columns=cols, show="headings", height=8)
        for c, w in zip(cols, (90, 70, 70, 60)):
            self.tree.heading(c, text=c.upper())
            self.tree.column(c, width=w, anchor="center")
        self.tree.pack(fill=tk.X, pady=4)

        stats = ttk.LabelFrame(right, text="Performance", padding=6)
        stats.pack(fill=tk.X, pady=8)
        self.lbl_latency = ttk.Label(stats, text="Inference time: -- ms")
        self.lbl_latency.pack(anchor="w")
        self.lbl_fps = ttk.Label(stats, text="Processing FPS: --")
        self.lbl_fps.pack(anchor="w")
        self.lbl_rtf = ttk.Label(stats, text="Throughput: -- x real-time")
        self.lbl_rtf.pack(anchor="w")
        self.lbl_snr = ttk.Label(stats, text="Estimated SNR: -- dB")
        self.lbl_snr.pack(anchor="w")
        self.lbl_cpu = ttk.Label(stats, text="CPU: -- %")
        self.lbl_cpu.pack(anchor="w")
        self.lbl_ram = ttk.Label(stats, text="RAM: -- MB")
        self.lbl_ram.pack(anchor="w")
        if not _HAVE_PSUTIL:
            self.lbl_cpu.config(text="CPU: n/a (pip install psutil)")
            self.lbl_ram.config(text="RAM: n/a (pip install psutil)")

        hist = ttk.LabelFrame(right, text="Detection History (log)", padding=6)
        hist.pack(fill=tk.BOTH, expand=True, pady=8)
        self.history_box = tk.Text(hist, width=40, height=14, state="disabled")
        self.history_box.pack(fill=tk.BOTH, expand=True)

        self.lbl_footer_status = ttk.Label(right, text="Ready.", foreground="gray")
        self.lbl_footer_status.pack(anchor="w", pady=(4, 0))

    # ------------------------------------------------------------- controls
    def _on_start(self):
        self._paused = False
        self._running = True
        self.lbl_status.config(text="Status: running", foreground="green")
        self._set_footer("Detection resumed.")

    def _on_pause(self):
        self._paused = not self._paused
        if self._paused:
            self.lbl_status.config(text="Status: paused", foreground="orange")
            self._set_footer("Detection paused -- worker idle, no new chunks processed.")
        else:
            self.lbl_status.config(text="Status: running", foreground="green")
            self._set_footer("Detection resumed.")

    def _on_stop(self):
        if not messagebox.askyesno("Stop Detection",
                                    "Stop the live detector and close the window?"):
            return
        self.shutdown()
        self.root.destroy()

    def _on_save(self):
        if self._last_result is None:
            self._set_footer("Nothing to save yet -- no chunk processed.")
            return
        from datetime import datetime as _dt
        out_dir = os.path.join(ROOT, "10_Datasets", "snapshots")
        os.makedirs(out_dir, exist_ok=True)
        tag = _dt.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(out_dir, f"snapshot_{tag}.png")
        self.fig.savefig(out_path, dpi=140)
        self._set_footer(f"Saved snapshot -> {out_path}")

    def _set_footer(self, text: str):
        self.lbl_footer_status.config(text=text)

    # -------------------------------------------------------------- worker
    def _worker_loop(self):
        while not self._stop:
            if self._paused or not self._running:
                import time as _time
                _time.sleep(0.1)
                continue
            chunk = self.source.next_chunk()
            if chunk is None:
                continue
            result = self.pipeline.process_chunk(chunk)
            try:
                self._work_q.put(result, timeout=1.0)
            except queue.Full:
                pass

    # ------------------------------------------------------------- polling
    def _poll_results(self):
        try:
            result = self._work_q.get_nowait()
            self._render(result)
        except queue.Empty:
            pass
        if not self._stop:
            self.root.after(cfg_get(load_config(), "gui.refresh_ms", 150), self._poll_results)

    def _tick_clock(self):
        self.lbl_clock.config(text=datetime.now().strftime("%H:%M:%S"))
        if not self._stop:
            self.root.after(cfg_get(load_config(), "gui.clock_refresh_ms", 1000), self._tick_clock)

    def _tick_resources(self):
        if _HAVE_PSUTIL:
            try:
                cpu = psutil.cpu_percent(interval=None)
                ram_mb = psutil.Process(os.getpid()).memory_info().rss / 1e6
                self.lbl_cpu.config(text=f"CPU: {cpu:.0f} %")
                self.lbl_ram.config(text=f"RAM: {ram_mb:.0f} MB")
            except Exception:  # noqa: BLE001
                pass
        if not self._stop:
            self.root.after(cfg_get(load_config(), "gui.resource_refresh_ms", 2000), self._tick_resources)

    def _render(self, result: dict):
        self._last_result = result
        import time as _time
        self._frame_times.append(_time.time())
        self._frame_times = [t for t in self._frame_times if _time.time() - t < 5.0]
        fps = len(self._frame_times) / 5.0

        self.ax_wave.clear()
        t = np.arange(len(result["waveform"])) / self.pipeline.fs
        self.ax_wave.plot(t, result["waveform"], linewidth=0.5, color="tab:blue")
        self.ax_wave.set_title("Waveform")
        self.ax_wave.set_xlim(0, t[-1] if len(t) else 1)

        extent = [0, result["times"][-1] if len(result["times"]) else 1,
                  result["freqs"][0], result["freqs"][-1]]

        self.ax_spec.clear()
        self.ax_spec.imshow(20 * np.log10(result["S_mag"] + 1e-8), aspect="auto", origin="lower",
                             extent=extent, cmap="viridis")
        self.ax_spec.set_title("Raw Spectrogram (dB)")
        self.ax_spec.set_ylabel("Hz")

        self.ax_pcen.clear()
        self.ax_pcen.imshow(result["S_pcen"], aspect="auto", origin="lower",
                             extent=extent, cmap="magma")
        self.ax_pcen.set_title("PCEN Spectrogram")
        self.ax_pcen.set_ylabel("Hz")

        self.ax_mask.clear()
        self.ax_mask.imshow(result["cleaned_mask"], aspect="auto", origin="lower",
                             extent=extent, cmap="gray")
        # Detected-tonal overlay: mark confirmed detections' frequency as a
        # horizontal guide line across the mask panel (post-CFAR + morphology).
        for d in result["detections"]:
            self.ax_mask.axhline(d.frequency_hz, color="cyan", linewidth=1.0, alpha=0.8)
            self.ax_mask.text(extent[1] * 0.99, d.frequency_hz, f" {d.frequency_hz:.0f} Hz",
                               color="cyan", fontsize=7, va="center", ha="right")
        self.ax_mask.set_title("Segmentation Mask + Detected Tonal Overlay")
        self.ax_mask.set_xlabel("Time (s)")
        self.ax_mask.set_ylabel("Hz")

        self.fig.tight_layout()
        self.canvas.draw_idle()

        for row in self.tree.get_children():
            self.tree.delete(row)
        cpu_pct = psutil.cpu_percent(interval=None) if _HAVE_PSUTIL else None
        for d in result["detections"]:
            self.tree.insert("", "end", values=(f"{d.frequency_hz:.1f}", f"{d.confidence:.0f}%",
                                                  f"{d.snr_db:.1f}", d.n_harmonics))
            self.logger.log_detection(
                frequency_hz=d.frequency_hz, confidence_pct=d.confidence, snr_db=d.snr_db,
                bandwidth_hz=d.bandwidth_hz, n_harmonics=d.n_harmonics, track_id=d.track_id,
                inference_time_ms=result["latency_s"] * 1000, cpu_percent=cpu_pct)
            self._append_history(
                f"[{datetime.now().strftime('%H:%M:%S')}] f={d.frequency_hz:7.1f} Hz  "
                f"conf={d.confidence:5.1f}%  SNR={d.snr_db:6.1f} dB  track#{d.track_id} "
                f"(hits={d.track_hits})")

        latency_ms = result["latency_s"] * 1000
        rtf = self.chunk_s / max(result["latency_s"], 1e-6)
        self.lbl_latency.config(text=f"Inference time: {latency_ms:.1f} ms")
        self.lbl_fps.config(text=f"Processing FPS: {fps:.2f}")
        self.lbl_rtf.config(text=f"Throughput: {rtf:.1f}x real-time")
        self.lbl_snr.config(text=f"Estimated SNR: {result['snr_db']:.1f} dB")
        self._set_footer(f"Last update {datetime.now().strftime('%H:%M:%S')} -- "
                          f"{len(result['detections'])} confirmed tonal(s).")

    def _append_history(self, line: str):
        self.history_box.config(state="normal")
        self.history_box.insert("end", line + "\n")
        self.history_box.see("end")
        self.history_box.config(state="disabled")

    def shutdown(self):
        self._stop = True
        self._worker.join(timeout=2.0)
        self.logger.close()
        try:
            if hasattr(self.source, "_rx"):
                self.source._rx.stop()
        except Exception:  # noqa: BLE001
            pass


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                     help="model checkpoint path. If omitted, uses config.yaml's active model "
                          "(or pass --model-manager to pick interactively from 11_Weights/)")
    ap.add_argument("--model-manager", action="store_true",
                     help="interactively pick a model from 11_Weights/ before starting")
    ap.add_argument("--source", choices=["synthetic", "wav", "ethernet"], default="synthetic")
    ap.add_argument("--wav-path", type=str, default=None)
    ap.add_argument("--snr-db", type=float, default=cfg_get(cfg, "signal_generator.snr_db", -8.0))
    ap.add_argument("--fs", type=float, default=cfg_get(cfg, "signal.sampling_rate_hz", 16000.0))
    ap.add_argument("--chunk-s", type=float, default=cfg_get(cfg, "gui.chunk_s", 4.0))
    ap.add_argument("--eth-host", type=str, default=cfg_get(cfg, "ethernet.host", "0.0.0.0"),
                     help="UDP mode: local address to listen on (0.0.0.0 = all interfaces). "
                          "TCP mode: the acquisition system's IP address to connect out to.")
    ap.add_argument("--eth-port", type=int, default=cfg_get(cfg, "ethernet.port", 50000))
    ap.add_argument("--eth-protocol", choices=["udp", "tcp"], default=cfg_get(cfg, "ethernet.protocol", "udp"),
                     help="udp: listen for pushed datagrams (lowest latency, can silently drop "
                          "packets on lossy links). tcp: connect out to the acquisition system as "
                          "a client (ordered, lossless, slightly higher latency).")
    ap.add_argument("--log", type=str, default=os.path.join(ROOT, cfg_get(cfg, "logging.csv_log", "logs/detections_log.csv")))
    args = ap.parse_args()

    checkpoint = args.checkpoint
    model_display_name = "segformer_b0_tonal"
    if args.model_manager:
        from model_manager import pick_model_interactive
        entry = pick_model_interactive()
        if entry is None:
            raise SystemExit(1)
        checkpoint = entry.path
        model_display_name = entry.name
    elif checkpoint is None:
        checkpoint = os.path.join(ROOT, cfg_get(cfg, "model.segformer_weights",
                                                  "11_Weights/segformer_b0_tonal.pt"))
        model_display_name = os.path.basename(checkpoint)
    else:
        model_display_name = os.path.basename(checkpoint)

    model, mcfg = load_model(checkpoint, device="cpu")
    pipeline = TonalDetectorPipeline(model, fs=args.fs)
    source = ChunkSource(args.source, args.fs, args.chunk_s, snr_db=args.snr_db,
                          wav_path=args.wav_path, eth_host=args.eth_host, eth_port=args.eth_port,
                          eth_protocol=args.eth_protocol)

    root = tk.Tk()
    app = LiveDetectorApp(root, pipeline, source, args.log, args.chunk_s,
                           model_name=model_display_name)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.shutdown(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
