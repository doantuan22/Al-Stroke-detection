import time
from pathlib import Path

import numpy as np

from app.alerts.manager import AlertManager
from app.config import ConfigLoader
from app.ai.recognizer_v2 import StrokeRecognizerV2
from app.ai.baggage_tracker import AbandonedBaggageTracker
from app.ai.weapon_detector import WeaponDetector


def make_standing_pose(y_offset=120, conf=0.9):
    kpts = np.zeros((17, 3))
    kpts[0] = [320, y_offset, conf]
    kpts[5] = [300, y_offset + 40, conf]
    kpts[6] = [340, y_offset + 40, conf]
    kpts[11] = [310, y_offset + 140, conf]
    kpts[12] = [330, y_offset + 140, conf]
    kpts[13] = [310, y_offset + 200, conf]
    kpts[14] = [330, y_offset + 200, conf]
    kpts[15] = [310, y_offset + 260, conf]
    kpts[16] = [330, y_offset + 260, conf]
    return kpts


def feed_stream(recognizer, frames, track_id=1, img_size=(640, 480)):
    history = []
    last = None
    for frame in frames:
        history.append(frame)
        last = recognizer.analyze(history, img_size, track_id=track_id)
    return last


def make_bag(track_id=1, cx=300, cy=240, class_id=24, conf=0.85):
    return {
        "track_id": track_id,
        "class_id": class_id,
        "class_name": "backpack",
        "bbox": [cx - 40, cy - 30, cx + 40, cy + 30],
        "conf": conf,
    }


def make_person(track_id=5, cx=300, cy=240):
    return {
        "track_id": track_id,
        "bbox": [cx - 30, cy - 80, cx + 30, cy + 80],
        "kpts": np.zeros((17, 3)),
        "conf": 0.9,
    }


def test_config_loader_default_fallback(tmp_path):
    cfg = ConfigLoader(tmp_path / "missing.yaml").data
    assert cfg["camera"]["target_width"] == 1920
    assert cfg["baggage"]["abandoned_timeout_sec"] == 60.0


def test_alert_manager_cooldown_and_dedup():
    mgr = AlertManager({"alert": {"dedup_window_sec": 5, "evidence_frames_before": 2}})
    first = mgr.decide("weapon_detected", 10, "critical", now=100.0)
    second = mgr.decide("weapon_detected", 10, "critical", now=101.0)
    third = mgr.decide("weapon_detected", 10, "critical", now=106.0)

    assert first.should_emit
    assert not second.should_emit
    assert second.reason == "cooldown"
    assert third.should_emit
    assert third.count == 2


def test_stroke_detection_streaming_sudden_fall():
    recognizer = StrokeRecognizerV2(debug=False)
    frames = [make_standing_pose(120) for _ in range(5)]
    frames += [make_standing_pose(280) for _ in range(2)]

    result = feed_stream(recognizer, frames, track_id=42)

    assert result["detected"]
    assert result["symptom"] == "Sudden_Fall"


def test_stroke_detection_streaming_normal_no_false_positive():
    recognizer = StrokeRecognizerV2(debug=False)
    frames = [make_standing_pose(120 + (i % 2)) for i in range(12)]

    result = feed_stream(recognizer, frames, track_id=43)

    assert not result["detected"]
    assert result["risk_level"] == "low"


def test_baggage_stationary_and_owner_association():
    tracker = AbandonedBaggageTracker(
        timeout=0.1,
        cooldown=1.0,
        owner_presence_grace_period=0.0,
        config={
            "baggage": {
                "stationary_duration_sec": 0.05,
                "stationary_velocity_threshold": 2.0,
                "risk_alert_threshold": 0.75,
            }
        },
    )
    bag = make_bag(track_id=11)
    owner = make_person(track_id=7, cx=300, cy=240)
    far_owner = make_person(track_id=7, cx=620, cy=420)

    tracker.update([bag], [owner])
    tracker.update([bag], [far_owner])
    time.sleep(0.12)
    alerts = tracker.update([bag], [far_owner])

    state = tracker.get_all_states()[11]
    assert state.likely_owner_id == 7
    assert state.risk_score >= 0.75
    assert alerts
    assert alerts[0]["stationary"]


def test_weapon_pose_association_boosts_confidence():
    weapon = {
        "track_id": 1,
        "class_id": 43,
        "class_name": "knife",
        "bbox": [300, 230, 320, 260],
        "conf": 0.60,
    }
    person = make_person(track_id=9, cx=310, cy=245)
    person["kpts"][9] = [310, 245, 0.9]

    detector = WeaponDetector(
        object_detector=None,
        conf=0.5,
        cooldown=0.0,
        wrist_distance_threshold=50,
        pose_confidence_boost=0.2,
    )
    alerts = detector.detect_frame([weapon], [person])

    assert alerts
    assert alerts[0]["bearer_id"] == 9
    assert alerts[0]["pose_associated"]
    assert alerts[0]["confidence"] == 0.80

