"""
Lightweight per-track keypoint smoothing.
"""
from __future__ import annotations

import numpy as np


class KeypointSmoother:
    def __init__(self, enabled: bool = True, alpha: float = 0.65):
        self.enabled = enabled
        self.alpha = float(alpha)
        self._prev: dict[int, np.ndarray] = {}

    def smooth(self, track_id: int, kpts: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return kpts
        cur = np.asarray(kpts, dtype=float)
        prev = self._prev.get(track_id)
        if prev is None or prev.shape != cur.shape:
            self._prev[track_id] = cur.copy()
            return cur

        out = cur.copy()
        valid = cur[:, 2] > 0
        out[valid, :2] = self.alpha * cur[valid, :2] + (1.0 - self.alpha) * prev[valid, :2]
        self._prev[track_id] = out.copy()
        return out

    def clear(self, track_id: int) -> None:
        self._prev.pop(track_id, None)

