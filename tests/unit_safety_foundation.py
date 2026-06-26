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
    return {"kpts": kpts, "bbox": [285, y_offset - 5, 355, y_offset + 260]}


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
    assert result["symptom"] == "Sudden_Fall (Suspecting...)"


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


def test_baggage_low_confidence_false_positive_ignored():
    tracker = AbandonedBaggageTracker(
        config={
            "baggage": {
                "confidence_threshold": 0.65,
                "class_confidence_thresholds": {"backpack": 0.68},
            }
        },
    )
    clothing_like_backpack = make_bag(track_id=12, conf=0.50)

    alerts = tracker.update([clothing_like_backpack], [])

    assert alerts == []
    assert tracker.get_all_states() == {}


def test_baggage_moderate_confidence_is_tracked():
    tracker = AbandonedBaggageTracker(
        config={
            "baggage": {
                "confidence_threshold": 0.45,
                "class_confidence_thresholds": {"backpack": 0.50},
                "min_confirmed_frames": 2,
                "min_confirmed_seen_sec": 0.0,
            }
        },
    )
    bag = make_bag(track_id=13, conf=0.55)

    tracker.update([bag], [])
    alerts = tracker.update([bag], [])

    assert 13 in tracker.get_all_states()
    assert alerts == []


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


def test_weapon_rejects_unassociated_false_positive_when_required():
    cup_like_weapon = {
        "track_id": 1,
        "class_id": 43,
        "class_name": "knife",
        "bbox": [300, 230, 320, 260],
        "conf": 0.70,
    }

    detector = WeaponDetector(
        object_detector=None,
        conf=0.45,
        require_pose_association=True,
        min_persistence_sec=0.0,
    )
    alerts = detector.detect_frame([cup_like_weapon], [])

    assert alerts == []


def test_weapon_rejects_person_sized_false_positive_even_with_pose():
    weapon = {
        "track_id": 1,
        "class_id": 0,
        "class_name": "gun",
        "bbox": [260, 160, 380, 360],
        "conf": 0.92,
    }
    person = make_person(track_id=9, cx=320, cy=260)
    person["bbox"] = [250, 120, 390, 380]
    person["kpts"][9] = [330, 255, 0.9]

    detector = WeaponDetector(
        object_detector=None,
        conf=0.45,
        min_persistence_sec=0.0,
        strict_pose_classes=["gun", "pistol", "rifle"],
        class_confidence_thresholds={"gun": 0.78},
        class_area_ratio_limits={"gun": 0.18},
    )
    alerts = detector.detect_frame([weapon], [person])

    assert alerts == []


def test_baggage_shared_ownership():
    tracker = AbandonedBaggageTracker(
        config={
            "baggage": {
                "owner_confirm_time_sec": 0.2,
                "proximity_expand_px": 150.0,
            }
        },
    )
    bag1 = make_bag(track_id=1, cx=250, cy=240)
    bag2 = make_bag(track_id=2, cx=350, cy=240)
    person = make_person(track_id=5, cx=300, cy=240)
    
    tracker.update([bag1, bag2], [person])
    time.sleep(0.1)
    tracker.update([bag1, bag2], [person])
    time.sleep(0.1)
    tracker.update([bag1, bag2], [person])
    time.sleep(0.1)
    tracker.update([bag1, bag2], [person])
    
    states = tracker.get_all_states()
    assert states[1].likely_owner_id == 5
    assert states[2].likely_owner_id == 5
    assert states[1].owner_gone_at is None
    assert states[2].owner_gone_at is None


def test_baggage_ignore_passerby():
    tracker = AbandonedBaggageTracker(
        config={
            "baggage": {
                "owner_confirm_time_sec": 0.5,
                "proximity_expand_px": 150.0,
                "stationary_duration_sec": 0.0,
            }
        },
    )
    bag = make_bag(track_id=1, cx=300, cy=240)
    
    tracker.update([bag], [])
    state = tracker.get_all_states()[1]
    initial_gone_at = state.owner_gone_at
    assert initial_gone_at is not None
    
    time.sleep(0.2)
    passerby = make_person(track_id=9, cx=320, cy=240)
    tracker.update([bag], [passerby])
    time.sleep(0.2)
    tracker.update([bag], [passerby])
    
    time.sleep(0.2)
    tracker.update([bag], [])
    
    state = tracker.get_all_states()[1]
    assert state.owner_gone_at == initial_gone_at
    assert state.owner_seen_since is None
