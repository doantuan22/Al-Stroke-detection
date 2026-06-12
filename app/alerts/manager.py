"""
Shared alert state, cooldown, dedup, and evidence buffering.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any


@dataclass
class AlertDecision:
    should_emit: bool
    reason: str
    alert_key: str
    severity: str
    count: int
    evidence_frames: list[Any]


class AlertManager:
    def __init__(self, config: dict | None = None):
        cfg = config or {}
        alert_cfg = cfg.get("alert", {})
        self.dedup_window = float(alert_cfg.get("dedup_window_sec", 10.0))
        self.evidence_before = int(alert_cfg.get("evidence_frames_before", 5))
        self.evidence_after = int(alert_cfg.get("evidence_frames_after", 5))
        self.max_stroke_uploads = int(alert_cfg.get("max_stroke_uploads_per_track", 3))
        self.stroke_upload_every_nth = int(alert_cfg.get("stroke_upload_every_nth_alert", 3))
        self.stroke_track_cooldown = float(alert_cfg.get("stroke_track_cooldown_sec", 5.0))

        self._last_emit: dict[str, float] = {}
        self._counts: defaultdict[str, int] = defaultdict(int)
        self._states: dict[str, str] = {}
        self._evidence: defaultdict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.evidence_before + self.evidence_after + 1)
        )

    def record_frame(self, frame: Any, key: str = "global") -> None:
        if frame is not None:
            self._evidence[key].append(frame)

    def decide(
        self,
        event_type: str,
        track_id: int | str | None,
        severity: str,
        cooldown_sec: float | None = None,
        now: float | None = None,
        frame: Any = None,
        dedup_extra: str = "",
    ) -> AlertDecision:
        now = now or time.time()
        cooldown = self.dedup_window if cooldown_sec is None else cooldown_sec
        entity = "none" if track_id is None else str(track_id)
        key = f"{event_type}:{entity}:{dedup_extra}"
        evidence_key = f"{event_type}:{entity}"

        if frame is not None:
            self.record_frame(frame, evidence_key)

        elapsed = now - self._last_emit.get(key, 0.0)
        if elapsed < cooldown:
            return AlertDecision(
                False, "cooldown", key, severity, self._counts[key],
                list(self._evidence[evidence_key])
            )

        self._last_emit[key] = now
        self._counts[key] += 1
        self._states[key] = severity
        return AlertDecision(
            True, "emit", key, severity, self._counts[key],
            list(self._evidence[evidence_key])
        )

    def decide_stroke(
        self,
        track_id: int,
        severity: str,
        now: float | None = None,
        frame: Any = None,
    ) -> AlertDecision:
        decision = self.decide(
            "stroke_fall_suspected",
            track_id,
            severity,
            cooldown_sec=self.stroke_track_cooldown,
            now=now,
            frame=frame,
        )
        if not decision.should_emit:
            return decision
        if decision.count > self.max_stroke_uploads:
            return AlertDecision(
                False, "max_count", decision.alert_key, severity,
                decision.count, decision.evidence_frames
            )
        return decision

    def should_upload_stroke(self, decision_or_count) -> bool:
        if isinstance(decision_or_count, AlertDecision):
            return (
                decision_or_count.should_emit
                and decision_or_count.count == self.stroke_upload_every_nth
            )
        return int(decision_or_count) == self.stroke_upload_every_nth

    def get_state(self, event_type: str, track_id: int | str | None) -> str | None:
        key = f"{event_type}:{'none' if track_id is None else track_id}:"
        return self._states.get(key)
