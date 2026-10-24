"""
log_utils.py
=============
Centralized detection + run logging for the suite.

Creates `logs/` (project root, path from config.yaml -> logging.*) the
first time it's used, and writes every confirmed detection to BOTH:
    - a CSV  (machine-readable, one row per detection)
    - a TXT  (human-readable running log, one line per detection, plus
      session start/stop banners) -- handy to `tail -f` during a demo.

Also wraps Python's `logging` module with a consistent suite-wide
formatter/handler setup (`setup_logger`) so every module that wants
proper log levels (INFO/WARNING/ERROR) instead of bare `print()` can
opt in with one line, without each module reinventing handler setup.
"""
from __future__ import annotations
import csv
import logging
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config_loader import load_config, get as cfg_get, resolve_path  # noqa: E402

_CSV_FIELDS = ["timestamp", "track_id", "frequency_hz", "confidence_pct", "snr_db",
               "bandwidth_hz", "n_harmonics", "inference_time_ms", "cpu_percent"]


def setup_logger(name: str, level: str | None = None) -> logging.Logger:
    """Consistent logging.Logger for any module: `log = setup_logger(__name__)`."""
    cfg = load_config()
    level_name = level or cfg_get(cfg, "logging.level", "INFO")
    logger = logging.getLogger(name)
    if not logger.handlers:  # avoid duplicate handlers if called more than once
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level_name.upper(), logging.INFO))
    logger.propagate = False
    return logger


class DetectionLogger:
    """
    Dual CSV + TXT detection logger, used by the Live GUI and deploy.py so
    every confirmed tonal detection (whichever module produced it) ends up
    in the same place with the same schema.
    """

    def __init__(self, csv_path: str | None = None, txt_path: str | None = None):
        cfg = load_config()
        self.csv_path = resolve_path(csv_path or cfg_get(cfg, "logging.csv_log", "logs/detections_log.csv"))
        if txt_path is not None:
            self.txt_path = resolve_path(txt_path)
        elif csv_path is not None:
            # a caller passing a custom csv_path (e.g. --log) gets a matching .txt beside it
            self.txt_path = os.path.splitext(self.csv_path)[0] + ".txt"
        else:
            self.txt_path = resolve_path(cfg_get(cfg, "logging.txt_log", "logs/detections_log.txt"))
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.txt_path), exist_ok=True)

        new_csv = not os.path.exists(self.csv_path)
        self._csv_f = open(self.csv_path, "a", newline="")
        self._csv_w = csv.writer(self._csv_f)
        if new_csv:
            self._csv_w.writerow(_CSV_FIELDS)

        self._txt_f = open(self.txt_path, "a")
        self._txt_f.write(f"\n=== session started {datetime.now().isoformat(timespec='seconds')} ===\n")
        self._txt_f.flush()

    def log_detection(self, frequency_hz: float, confidence_pct: float, snr_db: float,
                       bandwidth_hz: float = 0.0, n_harmonics: int = 0, track_id: int = -1,
                       inference_time_ms: float | None = None, cpu_percent: float | None = None,
                       timestamp: str | None = None):
        ts = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [ts, track_id, f"{frequency_hz:.2f}", f"{confidence_pct:.1f}", f"{snr_db:.2f}",
               f"{bandwidth_hz:.2f}", n_harmonics,
               f"{inference_time_ms:.1f}" if inference_time_ms is not None else "",
               f"{cpu_percent:.1f}" if cpu_percent is not None else ""]
        self._csv_w.writerow(row)
        self._csv_f.flush()

        line = (f"[{ts}] f={frequency_hz:8.1f} Hz  conf={confidence_pct:5.1f}%  "
                f"SNR={snr_db:6.1f} dB  bw={bandwidth_hz:.1f} Hz  harmonics={n_harmonics}  "
                f"track#{track_id}")
        if inference_time_ms is not None:
            line += f"  infer={inference_time_ms:.1f} ms"
        if cpu_percent is not None:
            line += f"  cpu={cpu_percent:.0f}%"
        self._txt_f.write(line + "\n")
        self._txt_f.flush()

    def close(self):
        try:
            self._txt_f.write(f"=== session ended {datetime.now().isoformat(timespec='seconds')} ===\n")
            self._txt_f.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            self._csv_f.close()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    log = DetectionLogger()
    log.log_detection(frequency_hz=442.3, confidence_pct=87.5, snr_db=-8.2, bandwidth_hz=6.0,
                       n_harmonics=1, track_id=0, inference_time_ms=41.2, cpu_percent=23.5)
    log.close()
    print(f"wrote {log.csv_path} and {log.txt_path}")
