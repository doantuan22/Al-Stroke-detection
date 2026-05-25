"""
Airport AI — Simulation Test Suite
====================================
Giả lập các tình huống thực tế để kiểm tra:
  1. Balo bỏ lại không có chủ (abandoned backpack)
  2. Vali bỏ lại rồi chủ quay lại (false positive prevention)
  3. Dao được phát hiện + xác định bearer
  4. Nhiều người + nhiều túi cùng lúc (crowd scenario)
  5. Edge case: bbox rỗng, track_id trùng, frame rỗng
  6. Performance benchmark: thời gian xử lý mỗi frame

Chạy: python -m tests.test_airport_simulation
      hoặc: python tests/test_airport_simulation.py
"""
import sys
import time
import numpy as np

# Thêm project root vào path
sys.path.insert(0, '.')

from app.ai.baggage_tracker import AbandonedBaggageTracker, BAGGAGE_CLASS_IDS
from app.ai.weapon_detector import WeaponDetector, COCO_WEAPON_CLASSES

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


# ── Helpers tạo dữ liệu giả ────────────────────────────────
def make_bag(track_id: int, cx: float, cy: float,
             class_id: int = 24, conf: float = 0.85) -> dict:
    """Tạo object bag giả (backpack mặc định)."""
    return {
        'track_id' : track_id,
        'class_id' : class_id,
        'class_name': BAGGAGE_CLASS_IDS.get(class_id, 'bag'),
        'bbox'     : [cx - 40, cy - 30, cx + 40, cy + 30],
        'conf'     : conf,
    }


def make_person(track_id: int, cx: float, cy: float) -> dict:
    """Tạo person giả."""
    return {
        'track_id': track_id,
        'bbox'    : [cx - 30, cy - 80, cx + 30, cy + 80],
        'kpts'    : np.zeros((17, 3)),
        'conf'    : 0.90,
    }


def make_weapon(cx: float, cy: float,
                class_id: int = 43, conf: float = 0.75) -> dict:
    """Tạo weapon detection giả (knife mặc định)."""
    return {
        'track_id'  : 0,
        'class_id'  : class_id,
        'class_name': COCO_WEAPON_CLASSES.get(class_id, 'weapon'),
        'bbox'      : [cx - 10, cy - 25, cx + 10, cy + 25],
        'conf'      : conf,
    }


def make_frame(h=480, w=640) -> np.ndarray:
    """Tạo frame trắng giả."""
    return np.zeros((h, w, 3), dtype=np.uint8)


# ══════════════════════════════════════════════════════════════
# TEST 1: Abandoned Baggage — Cơ bản
# ══════════════════════════════════════════════════════════════
section("TEST 1: Abandoned Baggage — Kịch bản cơ bản")

# Dùng timeout=2s để test nhanh thay vì 60s thực tế
tracker = AbandonedBaggageTracker(
    owner_radius=150,
    timeout=2.0,    # 2 giây cho test
    cooldown=5.0,
    camera_id='CAM_TEST',
)

bag     = make_bag(track_id=10, cx=300, cy=240)
person  = make_person(track_id=1, cx=300, cy=240)  # đứng ngay cạnh túi

# Frame 1: có người đứng cạnh → không alert
alerts = tracker.update([bag], [person])
check("Có chủ đứng cạnh → 0 alerts", len(alerts) == 0)

state = tracker.get_all_states().get(10)
check("State được tạo", state is not None)
check("owner_gone_at = None (đang có chủ)", state.owner_gone_at is None)

# Chủ rời đi (person ra xa)
far_person = make_person(track_id=1, cx=600, cy=400)  # cách 350px
alerts = tracker.update([bag], [far_person])           # trigger owner_gone_at
state  = tracker.get_all_states().get(10)
check("Chủ rời đi → owner_gone_at được set", state.owner_gone_at is not None)
check("Chưa timeout → 0 alerts", len(alerts) == 0)

# Đợi timeout (2s timeout, sleep 2.2s để chắc)
time.sleep(2.2)
alerts = tracker.update([bag], [far_person])            # kiểm tra sau timeout
check("Sau 2s → ALERT kích hoạt", len(alerts) == 1)
if alerts:
    a = alerts[0]
    check("event_type đúng", a['event_type'] == 'abandoned_baggage')
    check("object_class đúng", a['object_class'] == 'backpack')
    check("duration_sec >= 2.0", a['duration_sec'] >= 2.0,
          f"got {a['duration_sec']:.1f}s")
    check("risk_level = high", a['risk_level'] == 'high')
    check("camera_id đúng", a['camera_id'] == 'CAM_TEST')

# ══════════════════════════════════════════════════════════════
# TEST 2: Chủ quay lại → Reset (False Positive Prevention)
# ══════════════════════════════════════════════════════════════
section("TEST 2: Chủ quay lại trước timeout → Reset")

tracker2 = AbandonedBaggageTracker(
    owner_radius=150, timeout=5.0, cooldown=10.0, owner_presence_grace_period=0.0)

bag2     = make_bag(track_id=20, cx=200, cy=200)
owner    = make_person(track_id=2, cx=200, cy=200)
stranger = make_person(track_id=2, cx=600, cy=400)

# Có chủ
tracker2.update([bag2], [owner])
# Chủ đi xa
tracker2.update([bag2], [stranger])
state2 = tracker2.get_all_states().get(20)
gone_at = state2.owner_gone_at
check("owner_gone_at được set sau khi chủ đi", gone_at is not None)

# Chủ quay lại ngay (trước timeout)
time.sleep(0.5)
tracker2.update([bag2], [owner])
state2 = tracker2.get_all_states().get(20)
check("Chủ quay lại → owner_gone_at reset về None",
      state2.owner_gone_at is None)
check("alerted = False sau reset", state2.alerted == False)

# Kiểm tra không có alert sau reset
time.sleep(0.2)
alerts2 = tracker2.update([bag2], [owner])
check("Không có alert khi có chủ sau reset", len(alerts2) == 0)


# ══════════════════════════════════════════════════════════════
# TEST 3: Alert Cooldown — Tránh spam
# ══════════════════════════════════════════════════════════════
section("TEST 3: Alert Cooldown — Không spam alert")

tracker3 = AbandonedBaggageTracker(
    owner_radius=150, timeout=1.0, cooldown=3.0)
bag3    = make_bag(track_id=30, cx=300, cy=300)
nobody  = []

# Bước 1: khởi tạo state + trigger owner_gone_at
tracker3.update([bag3], nobody)   # ← gây owner_gone_at = now
# Bước 2: sleep đợi timeout
time.sleep(1.2)
# Bước 3: lấy alert
alerts_1st = tracker3.update([bag3], nobody)
check("Alert đầu tiên được gửi", len(alerts_1st) == 1)

# Ngay sau đó (trong cooldown)
alerts_2nd = tracker3.update([bag3], nobody)
check("Alert thứ 2 bị chặn bởi cooldown", len(alerts_2nd) == 0)

# Sau cooldown
time.sleep(3.2)
alerts_3rd = tracker3.update([bag3], nobody)
check("Sau cooldown → alert lại được gửi", len(alerts_3rd) == 1)


# ══════════════════════════════════════════════════════════════
# TEST 4: Nhiều túi + nhiều người (Crowd Scenario)
# ══════════════════════════════════════════════════════════════
section("TEST 4: Crowd — Nhiều túi và nhiều người")

tracker4 = AbandonedBaggageTracker(
    owner_radius=150, timeout=1.5, cooldown=10.0)

# 3 túi: 1 có chủ, 2 không có chủ
bag_owned    = make_bag(track_id=41, cx=100, cy=100, class_id=28)  # suitcase
bag_abandon1 = make_bag(track_id=42, cx=400, cy=300, class_id=24)  # backpack
bag_abandon2 = make_bag(track_id=43, cx=500, cy=400, class_id=26)  # handbag

# 2 người:
# person_near: đứng ngay cạnh bag_owned (dist ~0px < 150) → bag_owned có chủ
# person_far: đứng góc trái dưới (0,480) → cách tất cả 3 túi > 200px
person_near  = make_person(track_id=101, cx=100, cy=100)   # ngay cạnh bag_owned
person_far   = make_person(track_id=102, cx=0,   cy=480)   # xa tất cả: d(bag_owned)≈430, d(bag1)≈499, d(bag2)≈565


objects = [bag_owned, bag_abandon1, bag_abandon2]
persons = [person_near, person_far]

# Bước 1: khởi tạo states + trigger owner_gone_at cho các túi bỏ lại
tracker4.update(objects, persons)
# Bước 2: đợi timeout (1.5s)
time.sleep(1.6)
# Bước 3: lấy alerts
alerts4 = tracker4.update(objects, persons)

check("3 túi đều được track", len(tracker4.get_all_states()) == 3)
check("Chỉ 2 túi bỏ lại được alert (không phải túi có chủ)",
      len(alerts4) == 2,
      f"got {len(alerts4)} alerts")

abandon_types = {a['object_class'] for a in alerts4}
check("Alert có cả backpack và handbag",
      'backpack' in abandon_types and 'handbag' in abandon_types)


# ══════════════════════════════════════════════════════════════
# TEST 5: Túi biến mất → State bị xóa
# ══════════════════════════════════════════════════════════════
section("TEST 5: Túi biến khỏi frame → State tự xóa")

tracker5 = AbandonedBaggageTracker(timeout=5.0, grace_period=0.0)

bag5     = make_bag(track_id=50, cx=300, cy=300)
tracker5.update([bag5], [])
check("State 50 được tạo", 50 in tracker5.get_all_states())

# Túi biến mất (không còn trong objects)
tracker5.update([], [])
check("State 50 bị xóa khi túi biến khỏi frame",
      50 not in tracker5.get_all_states())


# ══════════════════════════════════════════════════════════════
# TEST 6: Edge Cases — Dữ liệu bất thường
# ══════════════════════════════════════════════════════════════
section("TEST 6: Edge Cases — Dữ liệu bất thường")

tracker6 = AbandonedBaggageTracker(timeout=1.0)

# Empty inputs
try:
    alerts_empty = tracker6.update([], [])
    check("Empty objects + persons → không crash", True)
    check("Empty input → 0 alerts", len(alerts_empty) == 0)
except Exception as e:
    check("Empty inputs → CRASH", False, str(e))

# Bbox thiếu field
bad_person = {'track_id': 99, 'bbox': [], 'kpts': np.zeros((17, 3))}
try:
    tracker6.update([make_bag(60, 300, 300)], [bad_person])
    check("Person với bbox rỗng → không crash", True)
except Exception as e:
    check("Person với bbox rỗng → CRASH", False, str(e))

# Object class không phải hành lý (xe đạp, v.v.)
non_bag = make_bag(track_id=70, cx=300, cy=300, class_id=1)  # bicycle
tracker6.update([non_bag], [])
check("Object class lạ (bicycle) bị bỏ qua",
      70 not in tracker6.get_all_states())

# Object thiếu 'bbox' — đã fix trong baggage_tracker.py bằng lọc trước
bad_obj = {
    'track_id': 80, 'class_id': 24,
    # Thiếu key 'bbox' hoàn toàn
    'conf': 0.8
}
try:
    tracker6.update([bad_obj], [])
    check("Object thiếu 'bbox' → bị skip không crash (sau fix)", True)
except Exception as e:
    check("Object thiếu 'bbox' → vẫn crash sau fix", False, str(e))


# ══════════════════════════════════════════════════════════════
# TEST 7: Weapon Detector — Logic giả lập
# ══════════════════════════════════════════════════════════════
section("TEST 7: Weapon Detector — Giả lập logic")


class MockObjectDetector:
    """Mock ObjectDetector để test WeaponDetector không cần GPU."""
    def __init__(self, return_objects=None):
        self._return = return_objects or []

    def detect(self, frame, classes=None, conf=0.5):
        return self._return


# Scenario: Dao ở gần người
knife_obj = make_weapon(cx=310, cy=250, class_id=43, conf=0.72)
person_w  = make_person(track_id=5, cx=300, cy=240)

mock_od = MockObjectDetector(return_objects=[knife_obj])
wd = WeaponDetector(object_detector=mock_od, conf=0.50, cooldown=2.0)

alerts_w = wd.detect_frame(make_frame(), persons=[person_w])
check("Knife trong frame → 1 alert", len(alerts_w) == 1)
if alerts_w:
    a = alerts_w[0]
    check("event_type = weapon_detected", a['event_type'] == 'weapon_detected')
    check("object_class = knife", a['object_class'] == 'knife')
    check("Bearer được xác định (person 5)", a['bearer_id'] == 5,
          f"got bearer={a['bearer_id']}")
    check("risk_level = high (không có zone)", a['risk_level'] == 'high')
    check("confidence >= 0.5", a['confidence'] >= 0.5,
          f"got {a['confidence']:.2f}")

# Cooldown: alert thứ 2 ngay sau đó bị block
alerts_w2 = wd.detect_frame(make_frame(), persons=[person_w])
check("Weapon alert thứ 2 bị block bởi cooldown", len(alerts_w2) == 0)

# Sau cooldown
time.sleep(2.1)
alerts_w3 = wd.detect_frame(make_frame(), persons=[person_w])
check("Sau weapon cooldown → alert lại", len(alerts_w3) == 1)


# ══════════════════════════════════════════════════════════════
# TEST 8: Weapon — Zone → risk = critical
# ══════════════════════════════════════════════════════════════
section("TEST 8: Weapon trong Zone nhạy cảm → Critical")

wd2     = WeaponDetector(object_detector=mock_od, conf=0.50, cooldown=0.1)
time.sleep(0.15)
alerts_zone = wd2.detect_frame(
    make_frame(), persons=[], zone_name="Security Checkpoint")
check("Weapon trong zone → risk = critical",
      len(alerts_zone) == 1 and alerts_zone[0]['risk_level'] == 'critical')
check("zone_name được ghi đúng",
      alerts_zone and alerts_zone[0].get('zone_name') == "Security Checkpoint")

# Không có zone → high
time.sleep(0.15)
alerts_nozone = wd2.detect_frame(make_frame(), persons=[], zone_name=None)
check("Weapon không có zone → risk = high",
      alerts_nozone and alerts_nozone[0]['risk_level'] == 'high')


# ══════════════════════════════════════════════════════════════
# TEST 9: Weapon — Không có bearer (dao bỏ lên bàn, không người)
# ══════════════════════════════════════════════════════════════
section("TEST 9: Weapon không có bearer (dao trên bàn)")

wd3 = WeaponDetector(
    object_detector=MockObjectDetector([knife_obj]),
    conf=0.50, cooldown=0.1)
time.sleep(0.15)
# Không có người nào trong frame
alerts_nb = wd3.detect_frame(make_frame(), persons=[])
check("Alert vẫn tạo khi không có bearer", len(alerts_nb) == 1)
check("bearer_id = None khi không có người",
      alerts_nb and alerts_nb[0]['bearer_id'] is None)


# ══════════════════════════════════════════════════════════════
# TEST 10: Weapon Detector — Chế độ tối ưu (Merge Detection)
# ══════════════════════════════════════════════════════════════
section("TEST 10: Weapon Detector — Tái sử dụng obj_results (Merge)")

knife_obj_opt = make_weapon(cx=310, cy=250, class_id=43, conf=0.72)
person_w_opt  = make_person(track_id=8, cx=300, cy=240)

wd_opt = WeaponDetector(
    object_detector=MockObjectDetector(return_objects=[]), # Giả lập: detect() rỗng để đảm bảo nó dùng list truyền vào
    conf=0.50, cooldown=0.1
)

obj_results_mock = [
    make_bag(track_id=101, cx=100, cy=100),
    {
        'track_id'  : 0,
        'class_id'  : 43, # knife
        'class_name': 'knife',
        'bbox'      : knife_obj_opt['bbox'],
        'conf'      : 0.72
    }
]

# Truyền trực tiếp list obj_results
alerts_opt = wd_opt.detect_frame(obj_results_mock, persons=[person_w_opt])
check("Truyền trực tiếp obj_results → nhận diện được vũ khí", len(alerts_opt) == 1)
if alerts_opt:
    a = alerts_opt[0]
    check("event_type đúng", a['event_type'] == 'weapon_detected')
    check("object_class đúng (knife)", a['object_class'] == 'knife')
    check("Xác định bearer chuẩn", a['bearer_id'] == 8)


# ══════════════════════════════════════════════════════════════
# TEST 11: Performance Benchmark
# ══════════════════════════════════════════════════════════════
section("TEST 11: Performance Benchmark (không GPU)")

# Tạo scenario lớn: 20 túi, 30 người
many_bags = [
    make_bag(track_id=200 + i, cx=float(i * 30 % 640),
             cy=float(i * 20 % 480),
             class_id=[24, 26, 28][i % 3])
    for i in range(20)
]
many_persons = [
    make_person(track_id=300 + i,
                cx=float(i * 25 % 640),
                cy=float(i * 35 % 480))
    for i in range(30)
]

tracker_perf = AbandonedBaggageTracker(timeout=999)
ITERS = 500

t0 = time.perf_counter()
for _ in range(ITERS):
    tracker_perf.update(many_bags, many_persons)
elapsed = time.perf_counter() - t0

avg_ms = (elapsed / ITERS) * 1000
print(f"\n  BaggageTracker update({len(many_bags)} bags, {len(many_persons)} persons):")
print(f"    {ITERS} iterations → {elapsed*1000:.1f}ms total")
print(f"    Trung bình: {avg_ms:.3f}ms / frame")
check(f"BaggageTracker < 1ms/frame (avg={avg_ms:.3f}ms)", avg_ms < 1.0)

# Weapon mock benchmark
knife_list = [make_weapon(cx=float(i * 50), cy=float(i * 30))
              for i in range(5)]
wd_perf  = WeaponDetector(
    object_detector=MockObjectDetector(knife_list),
    conf=0.0, cooldown=0.0)

t1 = time.perf_counter()
for _ in range(ITERS):
    # Dùng list objects để tối ưu hiệu năng (Merge Mode)
    wd_perf.detect_frame(knife_list, persons=many_persons)
elapsed2 = time.perf_counter() - t1
avg_ms2  = (elapsed2 / ITERS) * 1000
print(f"\n  WeaponDetector.detect_frame (Merge Mode - 5 weapons, 30 persons):")
print(f"    Trung bình: {avg_ms2:.3f}ms / frame (Merge Mode, no GPU)")
check(f"WeaponDetector logic < 2ms/frame (avg={avg_ms2:.3f}ms)", avg_ms2 < 2.0)


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
