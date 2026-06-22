"""
Stroke/fall streaming test suite.

Run directly:
    python tests/test_stroke_detection.py

Pytest uses unit_*.py files; this script remains as a human-readable smoke test.
"""
import sys
import time

import numpy as np


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, ".")

from app.ai.recognizer_v2 import StrokeRecognizerV2


GREEN = "\033[92m"
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"

pass_count = 0
fail_count = 0
IMG_SIZE = (640, 480)


def check(name: str, cond: bool, detail: str = ""):
    global pass_count, fail_count
    icon = f"{GREEN}PASS{RESET}" if cond else f"{RED}FAIL{RESET}"
    extra = f" - {detail}" if detail else ""
    print(f"  [{icon}] {name}{extra}")
    if cond:
        pass_count += 1
    else:
        fail_count += 1


def section(title: str):
    print(f"\n{BOLD}{BLUE}{'=' * 60}{RESET}")
    print(f"{BOLD}{BLUE}  {title}{RESET}")
    print(f"{BOLD}{BLUE}{'=' * 60}{RESET}")


def make_standing_pose(y_offset=120, conf=0.9):
    kpts = np.zeros((17, 3))
    kpts[0] = [320, y_offset, conf]
    kpts[1] = [310, y_offset - 5, conf]
    kpts[2] = [330, y_offset - 5, conf]
    kpts[3] = [305, y_offset, conf]
    kpts[4] = [335, y_offset, conf]
    kpts[5] = [300, y_offset + 40, conf]
    kpts[6] = [340, y_offset + 40, conf]
    kpts[7] = [290, y_offset + 80, conf]
    kpts[8] = [350, y_offset + 80, conf]
    kpts[9] = [285, y_offset + 120, conf]
    kpts[10] = [355, y_offset + 120, conf]
    kpts[11] = [310, y_offset + 140, conf]
    kpts[12] = [330, y_offset + 140, conf]
    kpts[13] = [310, y_offset + 200, conf]
    kpts[14] = [330, y_offset + 200, conf]
    kpts[15] = [310, y_offset + 260, conf]
    kpts[16] = [330, y_offset + 260, conf]
    return {"kpts": kpts, "bbox": [285, y_offset - 5, 355, y_offset + 260]}


def make_lying_pose(conf=0.9):
    kpts = np.zeros((17, 3))
    base_y = 350
    points = {
        0: (200, base_y),
        1: (190, base_y - 5),
        2: (210, base_y - 5),
        3: (185, base_y),
        4: (215, base_y),
        5: (250, base_y + 10),
        6: (300, base_y + 10),
        7: (270, base_y + 5),
        8: (330, base_y + 5),
        9: (260, base_y),
        10: (340, base_y),
        11: (350, base_y + 15),
        12: (400, base_y + 15),
        13: (450, base_y + 20),
        14: (500, base_y + 20),
        15: (550, base_y + 25),
        16: (600, base_y + 25),
    }
    for idx, (x, y) in points.items():
        kpts[idx] = [x, y, conf]
    return {"kpts": kpts, "bbox": [185, base_y - 5, 600, base_y + 25]}


def feed_stream(recognizer, frames, track_id=1):
    history = []
    result = None
    for frame in frames:
        history.append(frame)
        result = recognizer.analyze(history, IMG_SIZE, track_id=track_id)
    return result


def test_sudden_fall():
    section("Sudden fall streaming")
    recognizer = StrokeRecognizerV2(debug=False)
    frames = [make_standing_pose(100) for _ in range(5)]
    # Tích lũy đủ fall_distance > 72 (15% của 480)
    frames.append(make_standing_pose(120))
    frames.append(make_standing_pose(140))
    frames.append(make_standing_pose(160))
    frames.append(make_standing_pose(180))
    # Rớt mạnh để vượt max_vel
    frames.append(make_standing_pose(250))
    result = feed_stream(recognizer, frames, track_id=1)
    check("detected", result["detected"], str(result))
    check("symptom", result["symptom"] == "Sudden_Fall (Suspecting...)", result["symptom"])


def test_abnormal_posture():
    section("Abnormal posture streaming")
    recognizer = StrokeRecognizerV2(debug=False)
    history = []
    result = None
    for _ in range(12):
        history.append(make_lying_pose())
        result = recognizer.analyze(history, IMG_SIZE, track_id=2)
        if result["detected"]:
            break
    check("detected", result["detected"], str(result))
    check("symptom", result["symptom"] == "Abnormal_Posture (Suspecting...)", result["symptom"])


def test_normal_activity():
    section("Normal movement")
    recognizer = StrokeRecognizerV2(debug=False)
    frames = [make_standing_pose(120 + (i % 3)) for i in range(15)]
    result = feed_stream(recognizer, frames, track_id=3)
    check("not detected", not result["detected"], str(result))
    check("low risk", result["risk_level"] == "low", result["risk_level"])


def make_sitting_bend(conf=0.9):
    kpts = np.zeros((17, 3))
    base_y = 150
    kpts[0] = [320, base_y + 10, conf] # Nose
    kpts[11] = [300, base_y, conf] # Hips
    kpts[12] = [340, base_y, conf]
    kpts[15] = [300, base_y + 40, conf]
    kpts[16] = [340, base_y + 40, conf]
    return {"kpts": kpts, "bbox": [300, base_y, 340, base_y + 40]}

def test_sitting_bend():
    section("Sitting bend (No false positive)")
    recognizer = StrokeRecognizerV2(debug=False)
    frames = [make_sitting_bend() for _ in range(15)]
    result = feed_stream(recognizer, frames, track_id=5)
    check("not detected", not result["detected"], str(result))


def test_recovery():
    section("Recovery logic")
    recognizer = StrokeRecognizerV2(debug=False)
    
    # 1. Trigger fall
    frames = [make_standing_pose(100) for _ in range(5)]
    frames.append(make_standing_pose(120))
    frames.append(make_standing_pose(140))
    frames.append(make_standing_pose(160))
    frames.append(make_standing_pose(180))
    frames.append(make_standing_pose(250))
    result = feed_stream(recognizer, frames, track_id=4)
    check("detected fall", result["detected"], str(result))
    
    # 2. Recovery frame (đứng lên)
    recovery_frames = [make_standing_pose(100)]
    result = feed_stream(recognizer, recovery_frames, track_id=4)
    check("recovery cancels alert", not result["detected"], str(result))


def test_multiple_tracks():
    section("Multiple tracks independent state")
    recognizer = StrokeRecognizerV2(debug=False)
    fall_frames = [make_standing_pose(100) for _ in range(5)]
    fall_frames.append(make_standing_pose(120))
    fall_frames.append(make_standing_pose(140))
    fall_frames.append(make_standing_pose(160))
    fall_frames.append(make_standing_pose(180))
    fall_frames.append(make_standing_pose(250))
    normal_frames = [make_standing_pose(120) for _ in range(10)]
    fall_result = feed_stream(recognizer, fall_frames, track_id=10)
    normal_result = feed_stream(recognizer, normal_frames, track_id=20)
    check("fall track detected", fall_result["detected"], str(fall_result))
    check("normal track clean", not normal_result["detected"], str(normal_result))


def test_performance():
    section("Performance")
    recognizer = StrokeRecognizerV2(debug=False)
    frames = [make_standing_pose() for _ in range(30)]
    t0 = time.perf_counter()
    for _ in range(1000):
        feed_stream(recognizer, frames, track_id=99)
    avg_ms = (time.perf_counter() - t0) / 1000 * 1000
    check("avg < 2ms", avg_ms < 2.0, f"{avg_ms:.3f}ms")


if __name__ == "__main__":
    for fn in [
        test_sudden_fall,
        test_abnormal_posture,
        test_normal_activity,
        test_sitting_bend,
        test_recovery,
        test_multiple_tracks,
        test_performance,
    ]:
        fn()

    total = pass_count + fail_count
    print(f"\n{BOLD}RESULT: {pass_count} PASS | {fail_count} FAIL | Total: {total}{RESET}")
    sys.exit(1 if fail_count else 0)
