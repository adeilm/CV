"""Per-student rolling EAR/PERCLOS tracker.

PERCLOS = "percentage of eye closure" over a rolling time window. The
classic drowsiness signal. Each frame contributes one EAR sample
(eye-aperture ratio). Samples older than `PERCLOS_WINDOW_SEC` evict
automatically. One `StudentTracker` per `student_id`, kept in a global
registry; `get_tracker()` is the only public access point.
"""
from collections import deque
from threading import Lock
from typing import Optional

import numpy as np

from .config import EAR_THRESHOLD, EAR_NORMALIZED_MAX, PERCLOS_WINDOW_SEC


class StudentTracker:
    def __init__(self):
        self.ear_samples: deque = deque()  # (t, ear, closed)
        self.lock = Lock()

    def add_ear(self, t: float, ear: Optional[float]):
        with self.lock:
            if ear is not None:
                self.ear_samples.append((t, ear, ear < EAR_THRESHOLD))
            cutoff = t - PERCLOS_WINDOW_SEC
            while self.ear_samples and self.ear_samples[0][0] < cutoff:
                self.ear_samples.popleft()

    def perclos(self) -> float:
        with self.lock:
            if not self.ear_samples:
                return 0.0
            closed = sum(1 for _, _, c in self.ear_samples if c)
            return closed / len(self.ear_samples)

    def ear_normalized(self) -> float:
        with self.lock:
            if not self.ear_samples:
                return 1.0
            mean_ear = float(np.mean([e for _, e, _ in self.ear_samples]))
            return float(np.clip(mean_ear / EAR_NORMALIZED_MAX, 0.0, 1.0))


_trackers: dict[str, StudentTracker] = {}
_lock = Lock()


def get_tracker(student_id: str) -> StudentTracker:
    with _lock:
        if student_id not in _trackers:
            _trackers[student_id] = StudentTracker()
        return _trackers[student_id]
