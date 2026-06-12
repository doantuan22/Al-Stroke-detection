"""
Temporal classifier interface for future stroke/fall models.

The current fallback is intentionally conservative and lets the existing
rule-based recognizer remain the source of truth until a trained model exists.
"""
from __future__ import annotations

from typing import Any


class TemporalStrokeClassifier:
    def predict(self, keypoint_sequence: list[Any]) -> dict:
        raise NotImplementedError


class RuleFallbackTemporalClassifier(TemporalStrokeClassifier):
    def predict(self, keypoint_sequence: list[Any]) -> dict:
        if not keypoint_sequence:
            return {"detected": False, "confidence": 0.0, "label": "Normal"}
        return {
            "detected": False,
            "confidence": 0.0,
            "label": "RuleFallbackOnly",
        }

