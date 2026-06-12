"""
Configuration loader for the safety system.

The app prefers config/safety_config.yaml, but keeps a full in-code fallback so
missing files or missing PyYAML never stop the GUI from starting.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "safety_config.yaml"


DEFAULT_CONFIG: dict[str, Any] = {
    "camera": {
        "target_width": 1920,
        "target_height": 1080,
        "target_fps": 30,
        "buffer_size": 1,
    },
    "inference": {
        "pose_input_size": 768,
        "object_input_size": 960,
        "pose_skip_default": 2,
        "pose_skip_max": 4,
        "object_skip_default": 2,
        "object_skip_max": 6,
        "adaptive_fps_low": 20,
        "adaptive_fps_high": 28,
        "db_sync_every_frames": 90,
    },
    "tracking": {
        "tracker_type": "bytetrack",
        "tracker_config": "bytetrack_stroke.yaml",
        "enable_reid": False,
        "max_history": 30,
        "grace_frames": 15,
    },
    "alert": {
        "dedup_window_sec": 10.0,
        "evidence_frames_before": 5,
        "evidence_frames_after": 5,
        "max_stroke_uploads_per_track": 3,
        "stroke_upload_every_nth_alert": 3,
        "stroke_track_cooldown_sec": 5.0,
        "severity": {
            "observing": 0.4,
            "warning": 0.65,
            "critical": 0.85,
        },
    },
    "stroke": {
        "cooldown_sec": 15.0,
        "suspected_duration_sec": 1.0,
        "confirmed_duration_sec": 2.0,
        "kpts_conf_min": 0.25,
        "min_valid_kpts": 5,
        "sudden_vel_ratio": 0.07,
        "vel_window": 5,
        "aspect_ratio_min": 1.2,
        "bbox_h_max_ratio": 0.45,
        "head_hip_margin": 0.15,
        "sustained_posture": 6,
        "slump_aspect_min": 0.8,
        "slump_vel_ratio": 0.025,
        "slump_window": 12,
        "slump_sustained": 5,
        "smoothing": {
            "enabled": True,
            "method": "ema",
            "alpha": 0.65,
        },
    },
    "baggage": {
        "class_ids": [24, 26, 28],
        "confidence_threshold": 0.22,
        "owner_radius_px": 160.0,
        "proximity_expand_px": 80.0,
        "abandoned_timeout_sec": 60.0,
        "cooldown_sec": 120.0,
        "owner_presence_grace_sec": 2.0,
        "grace_period_sec": 3.0,
        "stationary_velocity_threshold": 2.0,
        "stationary_duration_sec": 3.0,
        "owner_leave_distance_threshold": 120.0,
        "risk_alert_threshold": 0.75,
        "risk": {
            "stationary_weight": 0.20,
            "owner_left_weight": 0.30,
            "no_nearby_person_weight": 0.20,
            "timeout_weight": 0.30,
            "leave_distance_weight": 0.15,
        },
    },
    "weapon": {
        "model_path": "models/weapon_yolo.pt",
        "fallback_to_coco": True,
        "confidence_threshold": 0.15,
        "bearer_radius_px": 200.0,
        "wrist_distance_threshold": 80.0,
        "pose_confidence_boost": 0.15,
        "cooldown_sec": 5.0,
        "overlay_persist_sec": 2.0,
        "coco_class_ids": [43, 76],
    },
    "zones": {
        "default": {
            "baggage_timeout_sec": 60.0,
            "weapon_severity_boost": "high",
        },
        "security_checkpoint": {
            "baggage_timeout_sec": 20.0,
            "weapon_severity_boost": "critical",
        },
        "waiting_area": {
            "baggage_timeout_sec": 60.0,
            "weapon_severity_boost": "high",
        },
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class ConfigLoader:
    def __init__(self, path: str | Path = DEFAULT_CONFIG_PATH):
        self.path = Path(path)
        self.data = self.load()

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return copy.deepcopy(DEFAULT_CONFIG)

        try:
            import yaml
        except Exception:
            print("[Config] PyYAML missing; using built-in defaults.")
            return copy.deepcopy(DEFAULT_CONFIG)

        try:
            loaded = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
            if not isinstance(loaded, dict):
                print(f"[Config] Invalid config root in {self.path}; using defaults.")
                loaded = {}
            return _deep_merge(DEFAULT_CONFIG, loaded)
        except Exception as exc:
            print(f"[Config] Failed to load {self.path}: {exc}; using defaults.")
            return copy.deepcopy(DEFAULT_CONFIG)

    def get(self, dotted_path: str, default: Any = None) -> Any:
        cur: Any = self.data
        for part in dotted_path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    return ConfigLoader(path).data

