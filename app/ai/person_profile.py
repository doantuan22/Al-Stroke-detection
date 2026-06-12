"""
Per-track person profiles for temporal behavior analysis.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class PersonProfile:
    track_id: int
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    lost_at: Optional[float] = None
    positions: deque = field(default_factory=lambda: deque(maxlen=30))
    velocity_px_s: float = 0.0
    direction: tuple[float, float] = (0.0, 0.0)
    zone_name: Optional[str] = None
    near_baggage: bool = False
    near_weapon: bool = False
    risk_score: float = 0.0

    def update(self, bbox: list, now: float | None = None) -> None:
        now = now or time.time()
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2

        if self.positions:
            prev_t, px, py = self.positions[-1]
            dt = max(now - prev_t, 1e-6)
            dx, dy = cx - px, cy - py
            self.velocity_px_s = float(np.hypot(dx, dy) / dt)
            mag = float(np.hypot(dx, dy))
            self.direction = (dx / mag, dy / mag) if mag > 1e-6 else (0.0, 0.0)

        self.positions.append((now, cx, cy))
        self.last_seen = now
        self.lost_at = None


class PersonProfileStore:
    def __init__(self, max_positions: int = 30):
        self.max_positions = max_positions
        self._profiles: dict[int, PersonProfile] = {}

    def update(self, persons: list[dict], now: float | None = None) -> dict[int, PersonProfile]:
        now = now or time.time()
        active = set()
        for person in persons:
            tid = person.get("track_id")
            bbox = person.get("bbox") or []
            if tid is None or len(bbox) < 4:
                continue
            active.add(tid)
            profile = self._profiles.get(tid)
            if profile is None:
                profile = PersonProfile(track_id=tid)
                profile.positions = deque(maxlen=self.max_positions)
                self._profiles[tid] = profile
            profile.update(bbox, now=now)

        for tid, profile in self._profiles.items():
            if tid not in active and profile.lost_at is None:
                profile.lost_at = now

        return self._profiles

    def mark_proximity(
        self,
        baggage_ids: set[int] | None = None,
        weapon_bearers: set[int] | None = None,
    ) -> None:
        baggage_ids = baggage_ids or set()
        weapon_bearers = weapon_bearers or set()
        for tid, profile in self._profiles.items():
            profile.near_baggage = tid in baggage_ids
            profile.near_weapon = tid in weapon_bearers

    def get(self, track_id: int) -> PersonProfile | None:
        return self._profiles.get(track_id)

    def all(self) -> dict[int, PersonProfile]:
        return self._profiles

