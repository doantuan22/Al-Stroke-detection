"""
Stroke Detection — Test Suite
===============================
Kiểm tra các tình huống phát hiện đột quỵ:
  1. Sudden Fall (Ngã đột ngột)
  2. Abnormal Posture (Tư thế bất thường)
  3. Gradual Collapse (Suy sụp dần)
  4. Normal activities (không phải đột quỵ)
  5. Edge cases

Chạy: python -m tests.test_stroke_detection
"""
import sys
import numpy as np
import logging

sys.path.insert(0, '.')

from app.ai.recognizer_v2 import StrokeRecognizerV2, StrokeConfig

# Setup logging để thấy debug output
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)

# ── Màu terminal ────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

pass_count = 0
fail_count = 0


def check(name: str, cond: bool, detail: str = ""):
    global pass_count, fail_count
    icon  = f"{GREEN}✅{RESET}" if cond else f"{RED}❌{RESET}"
    extra = f" — {detail}" if detail else ""
    print(f"  {icon} {name}{extra}")
    if cond:
        pass_count += 1
    else:
        fail_count += 1


def section(title: str):
    print(f"\n{BOLD}{BLUE}{'═'*60}{RESET}")
    print(f"{BOLD}{BLUE}  {title}{RESET}")
    print(f"{BOLD}{BLUE}{'═'*60}{RESET}")


# ── Helper tạo keypoints giả ────────────────────────────────
def make_standing_pose(y_offset=200, conf=0.9) -> np.ndarray:
    """Tạo pose đứng thẳng (17 keypoints COCO format)."""
    kpts = np.zeros((17, 3))
    # Nose
    kpts[0] = [320, y_offset, conf]
    # Eyes
    kpts[1] = [310, y_offset - 5, conf]
    kpts[2] = [330, y_offset - 5, conf]
    # Ears
    kpts[3] = [305, y_offset, conf]
    kpts[4] = [335, y_offset, conf]
    # Shoulders
    kpts[5] = [300, y_offset + 40, conf]
    kpts[6] = [340, y_offset + 40, conf]
    # Elbows
    kpts[7] = [290, y_offset + 80, conf]
    kpts[8] = [350, y_offset + 80, conf]
    # Wrists
    kpts[9] = [285, y_offset + 120, conf]
    kpts[10] = [355, y_offset + 120, conf]
    # Hips
    kpts[11] = [310, y_offset + 140, conf]
    kpts[12] = [330, y_offset + 140, conf]
    # Knees
    kpts[13] = [310, y_offset + 200, conf]
    kpts[14] = [330, y_offset + 200, conf]
    # Ankles
    kpts[15] = [310, y_offset + 260, conf]
    kpts[16] = [330, y_offset + 260, conf]
    return kpts


def make_falling_pose(y_offset=350, conf=0.9) -> np.ndarray:
    """Tạo pose đang ngã (hip di chuyển xuống nhanh)."""
    kpts = make_standing_pose(y_offset, conf)
    # Hip di chuyển xuống thấp
    kpts[11][1] = y_offset + 80
    kpts[12][1] = y_offset + 80
    return kpts


def make_lying_pose(conf=0.9) -> np.ndarray:
    """Tạo pose nằm ngang (aspect ratio cao)."""
    kpts = np.zeros((17, 3))
    # Tất cả keypoints nằm ngang
    base_y = 350
    # Nose
    kpts[0] = [200, base_y, conf]
    # Shoulders
    kpts[5] = [250, base_y + 10, conf]
    kpts[6] = [300, base_y + 10, conf]
    # Hips
    kpts[11] = [350, base_y + 15, conf]
    kpts[12] = [400, base_y + 15, conf]
    # Knees
    kpts[13] = [450, base_y + 20, conf]
    kpts[14] = [500, base_y + 20, conf]
    # Ankles
    kpts[15] = [550, base_y + 25, conf]
    kpts[16] = [600, base_y + 25, conf]
    # Eyes, ears
    kpts[1] = [190, base_y - 5, conf]
    kpts[2] = [210, base_y - 5, conf]
    kpts[3] = [185, base_y, conf]
    kpts[4] = [215, base_y, conf]
    # Elbows, wrists
    kpts[7] = [270, base_y + 5, conf]
    kpts[8] = [330, base_y + 5, conf]
    kpts[9] = [260, base_y, conf]
    kpts[10] = [340, base_y, conf]
    return kpts


def make_slumping_sequence(frames=15) -> list:
    """Tạo chuỗi frames mô phỏng người từ từ suy sụp."""
    sequence = []
    for i in range(frames):
        progress = i / frames
        # Y offset tăng dần (người từ từ ngã xuống)
        y_offset = 200 + int(progress * 100)
        kpts = make_standing_pose(y_offset, conf=0.9)
        # Aspect ratio tăng dần (người từ đứng thành nằm)
        spread = int(progress * 150)
        kpts[11][0] = 310 - spread // 2
        kpts[12][0] = 330 + spread // 2
        kpts[13][0] = 310 - spread
        kpts[14][0] = 330 + spread
        sequence.append(kpts)
    return sequence


# ══════════════════════════════════════════════════════════════
# TEST 1: Sudden Fall Detection
# ══════════════════════════════════════════════════════════════
section("TEST 1: Sudden Fall — Ngã đột ngột")

recognizer = StrokeRecognizerV2(debug=False)  # Bật debug=True để xem chi tiết
img_size = (640, 480)

# Tạo sequence: đứng → ngã nhanh
history = []
for i in range(3):
    history.append(make_standing_pose(y_offset=200))

# Frame ngã đột ngột (hip di chuyển xuống 100px trong 1 frame)
for i in range(3):
    history.append(make_falling_pose(y_offset=300 + i * 20))

result = recognizer.analyze(history, img_size, track_id=1)
check("Sudden Fall được phát hiện", result['detected'])
check("Symptom = Sudden_Fall", result['symptom'] == 'Sudden_Fall')
check("Confidence cao (>0.85)", result['confidence'] > 0.85,
      f"got {result['confidence']:.2f}")
check("Risk level = high", result['risk_level'] == 'high')


# ══════════════════════════════════════════════════════════════
# TEST 2: Abnormal Posture Detection
# ══════════════════════════════════════════════════════════════
section("TEST 2: Abnormal Posture — Tư thế bất thường")

recognizer2 = StrokeRecognizerV2(debug=False)

# Tạo sequence: người nằm ngang liên tục
# QUAN TRỌNG: Phải gọi analyze() từng frame một để sustained counter hoạt động
history2 = []
result2 = None
for i in range(12):  # Cần sustained >= 6 frames
    history2.append(make_lying_pose(conf=0.9))
    # Gọi analyze với history tích lũy (giống như production)
    if len(history2) >= 5:  # Cần ít nhất 5 frames
        result2 = recognizer2.analyze(history2, img_size, track_id=2)
        if result2['detected']:
            break

check("Abnormal Posture được phát hiện", result2 and result2['detected'])
check("Symptom = Abnormal_Posture", result2 and result2['symptom'] == 'Abnormal_Posture')
check("Confidence hợp lý (>0.80)", result2 and result2['confidence'] > 0.80,
      f"got {result2['confidence'] if result2 else 0:.2f}")


# ══════════════════════════════════════════════════════════════
# TEST 3: Gradual Collapse (Slump)
# ══════════════════════════════════════════════════════════════
section("TEST 3: Gradual Collapse — Suy sụp từ từ")

recognizer3 = StrokeRecognizerV2(debug=False)

# Tạo sequence suy sụp dần và gọi analyze từng frame
history3 = []
result3 = None
slump_frames = make_slumping_sequence(frames=15)
for i, frame in enumerate(slump_frames):
    history3.append(frame)
    if len(history3) >= 5:
        result3 = recognizer3.analyze(history3, img_size, track_id=3)
        if result3['detected']:
            break

check("Gradual Collapse được phát hiện", result3 and result3['detected'])
check("Symptom = Gradual_Collapse", result3 and result3['symptom'] == 'Gradual_Collapse')
check("Confidence hợp lý (>0.70)", result3 and result3['confidence'] > 0.70,
      f"got {result3['confidence'] if result3 else 0:.2f}")


# ══════════════════════════════════════════════════════════════
# TEST 4: Normal Activities — Không phát hiện sai
# ══════════════════════════════════════════════════════════════
section("TEST 4: Normal Activities — Không false positive")

recognizer4 = StrokeRecognizerV2(debug=False)

# Người đứng bình thường
history4 = []
for i in range(10):
    # Thay đổi nhẹ y_offset để mô phỏng chuyển động tự nhiên
    history4.append(make_standing_pose(y_offset=200 + (i % 3) * 2))

result4 = recognizer4.analyze(history4, img_size, track_id=4)
check("Đứng bình thường → không alert", not result4['detected'])
check("Symptom = Normal", result4['symptom'] == 'Normal')
check("Risk level = low", result4['risk_level'] == 'low')


# ══════════════════════════════════════════════════════════════
# TEST 5: Edge Cases
# ══════════════════════════════════════════════════════════════
section("TEST 5: Edge Cases — Dữ liệu bất thường")

recognizer5 = StrokeRecognizerV2(debug=False)

# History quá ngắn
short_history = [make_standing_pose() for _ in range(2)]
result5a = recognizer5.analyze(short_history, img_size, track_id=5)
check("History < 5 frames → không crash", True)
check("History ngắn → không detect", not result5a['detected'])

# Keypoints confidence thấp
low_conf_history = []
for i in range(10):
    kpts = make_standing_pose(conf=0.1)  # Confidence quá thấp
    low_conf_history.append(kpts)

result5b = recognizer5.analyze(low_conf_history, img_size, track_id=6)
check("Low confidence keypoints → không crash", True)
check("Low confidence → không detect", not result5b['detected'])

# Keypoints có y=0 (vị trí mặc định)
zero_y_history = []
for i in range(10):
    kpts = make_standing_pose()
    kpts[0][1] = 0  # Nose y=0
    kpts[5][1] = 0  # Left shoulder y=0
    zero_y_history.append(kpts)

try:
    result5c = recognizer5.analyze(zero_y_history, img_size, track_id=7)
    check("Keypoints y=0 → không crash (đã fix)", True)
except Exception as e:
    check("Keypoints y=0 → crash", False, str(e))


# ══════════════════════════════════════════════════════════════
# TEST 6: Multiple Tracks — Độc lập
# ══════════════════════════════════════════════════════════════
section("TEST 6: Multiple Tracks — State độc lập")

recognizer6 = StrokeRecognizerV2(debug=False)

# Track 1: Sudden fall
history_t1 = []
for i in range(3):
    history_t1.append(make_standing_pose(y_offset=200))
for i in range(3):
    history_t1.append(make_falling_pose(y_offset=300 + i * 20))

# Track 2: Normal
history_t2 = [make_standing_pose(y_offset=200) for _ in range(10)]

result_t1 = recognizer6.analyze(history_t1, img_size, track_id=10)
result_t2 = recognizer6.analyze(history_t2, img_size, track_id=20)

check("Track 1 (fall) được detect", result_t1['detected'])
check("Track 2 (normal) không detect", not result_t2['detected'])
check("Hai tracks độc lập không ảnh hưởng nhau", 
      result_t1['detected'] and not result_t2['detected'])


# ══════════════════════════════════════════════════════════════
# TEST 7: Sustained Counter Reset
# ══════════════════════════════════════════════════════════════
section("TEST 7: Sustained Counter — Reset khi trạng thái thay đổi")

recognizer7 = StrokeRecognizerV2(debug=False)

# Nằm 5 frames (chưa đủ sustained=6)
history7 = []
for i in range(5):
    history7.append(make_lying_pose())
    if len(history7) >= 5:
        result7a = recognizer7.analyze(history7, img_size, track_id=30)

check("5 frames nằm → chưa đủ sustained", not result7a['detected'])

# Đứng lên 2 frames (reset counter)
for i in range(2):
    history7.append(make_standing_pose())
    result7b = recognizer7.analyze(history7, img_size, track_id=30)

check("Đứng lên → counter reset", not result7b['detected'])

# Nằm lại 8 frames (phải đủ sustained mới)
for i in range(8):
    history7.append(make_lying_pose())
    result7c = recognizer7.analyze(history7, img_size, track_id=30)
    if result7c['detected']:
        break

check("Nằm lại 8 frames → detect", result7c['detected'])


# ══════════════════════════════════════════════════════════════
# TEST 8: Performance Benchmark
# ══════════════════════════════════════════════════════════════
section("TEST 8: Performance Benchmark")

import time

recognizer_perf = StrokeRecognizerV2(debug=False)
history_perf = [make_standing_pose() for _ in range(30)]

ITERS = 1000
t0 = time.perf_counter()
for _ in range(ITERS):
    recognizer_perf.analyze(history_perf, img_size, track_id=99)
elapsed = time.perf_counter() - t0

avg_ms = (elapsed / ITERS) * 1000
print(f"\n  StrokeRecognizer.analyze (30 frames history):")
print(f"    {ITERS} iterations → {elapsed*1000:.1f}ms total")
print(f"    Trung bình: {avg_ms:.3f}ms / call")
check(f"StrokeRecognizer < 1ms/call (avg={avg_ms:.3f}ms)", avg_ms < 1.0)


# ══════════════════════════════════════════════════════════════
# KẾT QUẢ
# ══════════════════════════════════════════════════════════════
print(f"\n{BOLD}{'═'*60}")
total = pass_count + fail_count
print(f"  KẾT QUẢ: {GREEN}{pass_count} PASS{RESET} | "
      f"{RED}{fail_count} FAIL{RESET} | Tổng: {total}")
print(f"{'═'*60}{RESET}\n")

if fail_count > 0:
    print(f"{YELLOW}⚠️  Có {fail_count} test thất bại — xem chi tiết bên trên.{RESET}\n")
    sys.exit(1)
else:
    print(f"{GREEN}🎉 Tất cả tests PASS!{RESET}\n")
    sys.exit(0)
