"""
kalman_tracker.py
===================
Module 9: multi-frame frequency tracking with a constant-velocity Kalman
filter per tonal track. Rejects transient/random false alarms that don't
persist across frames, and smooths the frequency estimate of real tonals.

State per track: x = [frequency (Hz), frequency_rate (Hz/frame)]
"""
from __future__ import annotations
import numpy as np


class KalmanTrack1D:
    """Constant-velocity Kalman filter tracking a single tonal's frequency."""

    def __init__(self, track_id: int, f0: float, process_var: float = 0.5,
                 meas_var: float = 4.0):
        self.id = track_id
        self.x = np.array([[f0], [0.0]])          # state: [freq, freq_rate]
        self.P = np.eye(2) * 10.0                  # state covariance
        self.A = np.array([[1.0, 1.0], [0.0, 1.0]])  # constant-velocity transition
        self.H = np.array([[1.0, 0.0]])             # we only observe frequency
        self.Q = np.eye(2) * process_var            # process noise
        self.R = np.array([[meas_var]])              # measurement noise

        self.age = 1
        self.hits = 1
        self.misses = 0
        self.freq_history = [f0]
        self.confirmed = False

    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        self.age += 1
        return float(self.x[0, 0])

    def update(self, freq_meas: float):
        z = np.array([[freq_meas]])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P
        self.hits += 1
        self.misses = 0
        self.freq_history.append(float(self.x[0, 0]))
        if len(self.freq_history) > 30:
            self.freq_history.pop(0)

    def mark_missed(self):
        self.misses += 1

    @property
    def frequency(self) -> float:
        return float(self.x[0, 0])

    @property
    def variance(self) -> float:
        return float(np.var(self.freq_history)) if len(self.freq_history) > 1 else float("inf")


class MultiTonalTracker:
    """
    Manages a bank of KalmanTrack1D trackers with nearest-neighbour, gated
    association. A track becomes 'confirmed' (i.e. reported to the operator)
    only after `confirm_hits` consecutive/near-consecutive associated
    detections, and is dropped after `max_misses` consecutive missed frames.
    """

    def __init__(self, gate_hz: float = 15.0, confirm_hits: int = 3, max_misses: int = 4,
                 process_var: float = 0.5, meas_var: float = 4.0):
        self.gate_hz = gate_hz
        self.confirm_hits = confirm_hits
        self.max_misses = max_misses
        self.process_var = process_var
        self.meas_var = meas_var
        self.tracks: list[KalmanTrack1D] = []
        self._next_id = 0
        self.last_dropped_track_ids: list[int] = []

    def step(self, detections_hz: list[float]) -> list[KalmanTrack1D]:
        """
        Advance every track one frame and associate the given list of
        detected frequencies (Hz) for this frame. Returns the list of
        currently CONFIRMED tracks (i.e. what the operator should see).
        """
        for tr in self.tracks:
            tr.predict()

        unmatched_dets = list(range(len(detections_hz)))
        for tr in self.tracks:
            # BUG FIX: this used to be `if not unmatched_dets: break`, which
            # exited the loop entirely as soon as detections ran out --
            # silently skipping mark_missed() for every remaining track that
            # frame (and skipping it for ALL tracks whenever a frame had zero
            # detections at all, since unmatched_dets would already be empty
            # before the loop even started). That meant a track whose source
            # stopped producing detections never aged out via max_misses.
            # Removing the early exit is correctness-preserving for the
            # matching logic (the inner loop below is simply a no-op scan
            # over an empty list) and is the minimal fix -- no change to the
            # Kalman filter math itself.
            best_j, best_dist = None, self.gate_hz
            for j in unmatched_dets:
                dist = abs(detections_hz[j] - tr.frequency)
                if dist < best_dist:
                    best_dist, best_j = dist, j
            if best_j is not None:
                tr.update(detections_hz[best_j])
                unmatched_dets.remove(best_j)
            else:
                tr.mark_missed()

        # spawn new tentative tracks for unmatched detections
        for j in unmatched_dets:
            self.tracks.append(KalmanTrack1D(self._next_id, detections_hz[j],
                                               self.process_var, self.meas_var))
            self._next_id += 1

        # confirm / prune
        alive = []
        for tr in self.tracks:
            if tr.hits >= self.confirm_hits:
                tr.confirmed = True
            if tr.misses <= self.max_misses:
                alive.append(tr)
        self.last_dropped_track_ids = [t.id for t in self.tracks if t not in alive]  # NEW: additive side-channel,
                                                                                        # does not change this method's
                                                                                        # existing return type/contract
        self.tracks = alive

        return [t for t in self.tracks if t.confirmed]

    def all_tracks_status(self) -> list[dict]:
        """NEW, additive method (does not touch step()'s algorithm or return
        contract): a real status snapshot of EVERY currently-alive track,
        not just the confirmed ones step() returns. Status is computed
        directly from each track's actual hits/misses/confirmed fields --
        nothing here is invented. 'LOST' tracks (misses just exceeded
        max_misses) are reported via last_dropped_track_ids for one frame
        by the caller, since a pruned track no longer exists to query."""
        out = []
        for tr in self.tracks:
            if not tr.confirmed:
                status = "TENTATIVE"
            else:
                status = "CONFIRMED"  # includes tracks currently coasting (tr.misses > 0 but still alive)
            out.append({"track_id": tr.id, "status": status, "frequency_hz": tr.frequency,
                        "hits": tr.hits, "misses": tr.misses, "age": tr.age})
        return out
