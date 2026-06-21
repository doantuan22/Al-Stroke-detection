"""
SafetyEngine coordinates all AI detectors and temporal state.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from app.ai.detector import PoseDetector
from app.ai.tracker import Tracker
from app.ai.recognizer_v2 import StrokeRecognizerV2, StrokeConfig
from app.ai.object_detector import ObjectDetector
from app.ai.baggage_tracker import AbandonedBaggageTracker
from app.ai.weapon_detector import WeaponDetector
from app.ai.keypoint_smoothing import KeypointSmoother
from app.ai.person_profile import PersonProfileStore
from app.ai.temporal_stroke import RuleFallbackTemporalClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_project_path(path: str) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str(PROJECT_ROOT / p)


class SafetyEngine:
    def __init__(self, config: dict[str, Any], camera_id: str = "CAM_00"):
        self.config = config
        inf = config.get("inference", {})
        tracking = config.get("tracking", {})
        person = config.get("person", {})
        stroke = config.get("stroke", {})
        baggage = config.get("baggage", {})
        weapon = config.get("weapon", {})

        tracker_cfg = _resolve_project_path(tracking.get("tracker_config", "bytetrack_stroke.yaml"))
        self.detector = PoseDetector(
            input_size=int(inf.get("pose_input_size", 768)),
            tracker_cfg=tracker_cfg,
        )
        self.person_min_conf = float(person.get("min_confidence", 0.55))
        self.person_min_valid_kpts = int(person.get("min_valid_keypoints", 6))
        self.person_min_bbox_height = float(person.get("min_bbox_height_px", 48))
        self.tracker = Tracker(max_history=int(tracking.get("max_history", 30)))
        self.recognizer = StrokeRecognizerV2(
            config=StrokeConfig(
                kpts_conf_min=float(stroke.get("kpts_conf_min", 0.25)),
                min_valid_kpts=int(stroke.get("min_valid_kpts", 5)),
                sudden_vel_ratio=float(stroke.get("sudden_vel_ratio", 0.07)),
                vel_window=int(stroke.get("vel_window", 5)),
                aspect_ratio_min=float(stroke.get("aspect_ratio_min", 1.2)),
                bbox_h_max_ratio=float(stroke.get("bbox_h_max_ratio", 0.45)),
                head_hip_margin=float(stroke.get("head_hip_margin", 0.15)),
                sustained_posture=int(stroke.get("sustained_posture", 6)),
                slump_aspect_min=float(stroke.get("slump_aspect_min", 0.8)),
                slump_vel_ratio=float(stroke.get("slump_vel_ratio", 0.025)),
                slump_window=int(stroke.get("slump_window", 12)),
                slump_sustained=int(stroke.get("slump_sustained", 5)),
            ),
            debug=False,
        )

        smoothing_cfg = stroke.get("smoothing", {})
        self.smoother = KeypointSmoother(
            enabled=bool(smoothing_cfg.get("enabled", True)),
            alpha=float(smoothing_cfg.get("alpha", 0.65)),
        )
        self.temporal_classifier = RuleFallbackTemporalClassifier()
        self.person_profiles = PersonProfileStore(
            max_positions=int(tracking.get("max_history", 30))
        )

        self.obj_detector = ObjectDetector(
            model_path="yolov8n.pt",
            input_size=int(inf.get("object_input_size", 960)),
            object_skip=int(inf.get("object_skip_default", 2)),
        )
        self.baggage_tracker = AbandonedBaggageTracker(
            owner_radius=float(baggage.get("owner_radius_px", 160)),
            timeout=float(baggage.get("abandoned_timeout_sec", 60)),
            cooldown=float(baggage.get("cooldown_sec", 120)),
            camera_id=camera_id,
            grace_period=float(baggage.get("grace_period_sec", 3)),
            owner_presence_grace_period=float(baggage.get("owner_presence_grace_sec", 2)),
            config=config,
        )

        weapon_model_path = weapon.get("model_path", "models/weapon_yolo.pt")
        weapon_model_abs = _resolve_project_path(weapon_model_path)
        
        if Path(weapon_model_abs).exists():
            print(f"[SafetyEngine] Loading dedicated weapon model: {weapon_model_abs}")
            weapon_od = ObjectDetector(
                model_path=weapon_model_abs,
                input_size=int(weapon.get("weapon_input_size", 640)), # Fix: dùng 640 để tránh méo ảnh và tăng tốc
                object_skip=1,  # Weapon model chạy độc lập không nên skip
            )
            use_finetune = True
            self.weapon_od_independent = True
        else:
            weapon_od = self.obj_detector
            use_finetune = False
            self.weapon_od_independent = False

        self.weapon_detector = WeaponDetector(
            weapon_od,
            use_finetune=use_finetune,
            conf=float(weapon.get("confidence_threshold", 0.45)),
            bearer_radius=float(weapon.get("bearer_radius_px", 200)),
            cooldown=float(weapon.get("cooldown_sec", 5.0)),
            wrist_distance_threshold=float(weapon.get("wrist_distance_threshold", 100)),
            pose_confidence_boost=float(weapon.get("pose_confidence_boost", 0.15)),
            min_pose_associated_conf=float(weapon.get("min_pose_associated_confidence", 0.55)),
            min_unassociated_conf=float(weapon.get("min_unassociated_confidence", 0.85)),
            require_pose_association=bool(weapon.get("require_pose_association", False)),
            min_persistence_sec=float(weapon.get("min_persistence_sec", 1.0)),
            strict_pose_classes=weapon.get("strict_pose_classes", ["gun", "pistol", "rifle"]),
            class_confidence_thresholds=weapon.get("class_confidence_thresholds", {}),
            class_area_ratio_limits=weapon.get("class_area_ratio_limits", {}),
        )

        self._last_person_results: dict[int, dict] = {}

    def reset_object_skip(self) -> None:
        self.obj_detector.reset_skip_counter()

    def update_camera(self, camera_id: str) -> None:
        self.baggage_tracker.update_camera(camera_id)

    def track_persons(self, frame, run_inference: bool, cached_results: list[dict]) -> list[dict]:
        if run_inference:
            return self._filter_person_results(
                self.detector.track(frame, conf=self.person_min_conf)
            )
        return cached_results

    def _filter_person_results(self, results: list[dict]) -> list[dict]:
        filtered = []
        kpts_conf_min = float(self.config.get("stroke", {}).get("kpts_conf_min", 0.25))
        for res in results:
            if float(res.get("conf", 0.0)) < self.person_min_conf:
                continue

            bbox = res.get("bbox") or []
            if len(bbox) < 4 or (bbox[3] - bbox[1]) < self.person_min_bbox_height:
                continue

            kpts = res.get("kpts")
            if kpts is None:
                continue
            valid = (
                (kpts[:, 2] > kpts_conf_min)
                & (kpts[:, 0] > 0)
                & (kpts[:, 1] > 0)
            )
            if int(valid.sum()) < self.person_min_valid_kpts:
                continue
            filtered.append(res)
        return filtered

    def analyze_persons(self, results: list[dict], frame_shape, run_inference: bool):
        active_ids = []
        analyzed = []
        w, h = frame_shape[1], frame_shape[0]

        self.person_profiles.update(results)
        for res in results:
            track_id = res["track_id"]
            active_ids.append(track_id)

            if run_inference:
                kpts = self.smoother.smooth(track_id, res["kpts"])
                res["kpts"] = kpts
                self.tracker.update_history(track_id, kpts)
                history = self.tracker.get_history(track_id)
                result = self.recognizer.analyze(history, (w, h), track_id=track_id)
                self._last_person_results[track_id] = result
            else:
                result = self._last_person_results.get(
                    track_id, self.recognizer._result(False, 0.0, "Normal", "low")
                )
            analyzed.append((res, result))

        if run_inference:
            self.tracker.clean_old_tracks(active_ids)
            still_in_tracker = set(self.tracker.track_history.keys())
            for lost_tid in list(self._last_person_results.keys()):
                if lost_tid not in still_in_tracker:
                    self.recognizer._clear_state(lost_tid)
                    self.smoother.clear(lost_tid)
                    self._last_person_results.pop(lost_tid, None)

        return analyzed, active_ids

    def track_objects(self, frame, classes: list[int], conf: float):
        return self.obj_detector.track(frame, classes=classes, conf=conf)

    def detect_airport_events(
        self,
        obj_results: list[dict],
        persons: list[dict],
        camera_id: str,
        frame: Any = None,
        zone_name: str | None = None,
    ) -> tuple[list[dict], list[dict]]:
        baggage_alerts = self.baggage_tracker.update(obj_results, persons, zone_name=zone_name)
        
        weapon_input = frame if (self.weapon_od_independent and frame is not None) else obj_results
        
        weapon_alerts = self.weapon_detector.detect_frame(
            weapon_input, persons, coco_objects=obj_results, zone_name=zone_name, camera_id=camera_id
        )
        weapon_bearers = {
            a.get("bearer_id") for a in weapon_alerts
            if a.get("bearer_id") is not None
        }
        self.person_profiles.mark_proximity(weapon_bearers=weapon_bearers)
        return baggage_alerts, weapon_alerts

    def refresh_weapon_overlays(self, obj_results: list[dict], persons: list[dict], camera_id: str, frame: Any = None):
        # KHÔNG CHẠY INFERENCE ở frame skip. Weapon Detector lưu lại trạng thái (persistent)
        # thông qua get_active_overlays() bên ngoài.
        pass

