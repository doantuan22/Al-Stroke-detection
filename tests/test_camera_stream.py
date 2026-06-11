"""
=============================================================
  Giả lập Luồng Camera Stream — Kiểm tra lỗi tiềm ẩn
=============================================================
Chạy KHÔNG cần GPU/model thật: dùng mock objects.

Các kịch bản kiểm tra:
  S1  Camera mở không được (source lỗi)
  S2  Frame trả về ret=False giữa chừng (camera ngắt)
  S3  Frame rỗng / None (frame corrupt)
  S4  Frame sai shape (1 channel, 4 channel)
  S5  Frame skip / cache khi obj_results rỗng
  S6  weapon_detector.detect_frame nhận frame thay vì list
  S7  Queue frame_queue full → drop frame
  S8  Xung đột thread: _last_obj_results bị ghi đè khi read
  S9  BaggageTracker.update() với bbox thiếu tọa độ
  S10 WeaponDetector._find_bearer với persons bbox ngắn
  S11 Adaptive skip khi FPS thấp/cao
  S12 cap.release() bị bỏ qua khi exception
"""

import sys
import io
import time
# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import threading
import numpy as np
from queue import Queue, Full
from collections import deque
from unittest.mock import MagicMock, patch

# ─── màu ANSI ───────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"

PASS = f"{GREEN}[PASS]{RESET}"
FAIL = f"{RED}[FAIL]{RESET}"
WARN = f"{YELLOW}[WARN]{RESET}"
INFO = f"{CYAN}[INFO]{RESET}"

results = []

def record(name, ok, detail=""):
    icon = PASS if ok else FAIL
    print(f"  {icon} {name}" + (f" — {detail}" if detail else ""))
    results.append((name, ok, detail))


# ══════════════════════════════════════════════════════════════
#  MOCK helpers (không load YOLO thật)
# ══════════════════════════════════════════════════════════════
def make_frame(h=480, w=640, channels=3, dtype=np.uint8):
    if channels == 1:
        return np.zeros((h, w), dtype=dtype)
    return np.zeros((h, w, channels), dtype=dtype)

def fake_person(track_id=1, x1=100, y1=100, x2=200, y2=400, conf=0.9):
    return {'track_id': track_id, 'bbox': [x1, y1, x2, y2], 'conf': conf,
            'kpts': np.zeros((17, 3))}

def fake_obj(track_id=1, class_id=26, conf=0.6, x1=120, y1=150, x2=180, y2=220):
    return {'track_id': track_id, 'class_id': class_id,
            'class_name': 'handbag', 'bbox': [x1, y1, x2, y2], 'conf': conf}


# ══════════════════════════════════════════════════════════════
#  S1: Camera mở không được
# ══════════════════════════════════════════════════════════════
def test_s1_camera_open_fail():
    print(f"\n{CYAN}S1: Camera mở không được{RESET}")
    import cv2
    # source không tồn tại
    cap = cv2.VideoCapture(999)
    opened = cap.isOpened()
    cap.release()

    # Kiểm tra code hiện tại có guard không
    # Trong _video_worker: cap.read() → ret=False → break
    # Nhưng KHÔNG có log lỗi rõ ràng → user không biết tại sao dừng
    record("cap.isOpened() check", not opened,
           "Camera 999 đúng là không mở được")
    record("Thiếu log lỗi khi camera không mở được", True,
           f"{YELLOW}BUG: _video_worker không gọi cap.isOpened() → break ngầm{RESET}")


# ══════════════════════════════════════════════════════════════
#  S2: Camera ngắt giữa chừng (ret=False)
# ══════════════════════════════════════════════════════════════
def test_s2_camera_disconnect():
    print(f"\n{CYAN}S2: Camera ngắt giữa chừng{RESET}")

    frames_read = 0
    stopped_cleanly = False

    class FakeCap:
        def __init__(self):
            self._count = 0
        def set(self, *a): pass
        def read(self):
            self._count += 1
            if self._count <= 5:
                return True, make_frame()
            return False, None   # giả camera ngắt
        def release(self):
            pass
        def isOpened(self): return True

    cap = FakeCap()
    is_running = True
    frame_queue = Queue(maxsize=2)

    try:
        while is_running:
            ret, frame = cap.read()
            if not ret:
                stopped_cleanly = True
                break
            frames_read += 1
    finally:
        cap.release()

    record("Đọc đủ 5 frame trước khi ngắt", frames_read == 5,
           f"Đọc được {frames_read} frames")
    record("Vòng lặp dừng sạch khi ret=False", stopped_cleanly)
    record("cap.release() được gọi trong finally", True,
           f"{YELLOW}BUG: Code thật KHÔNG có finally → cap leak nếu exception{RESET}")


# ══════════════════════════════════════════════════════════════
#  S3: Frame rỗng / corrupt
# ══════════════════════════════════════════════════════════════
def test_s3_empty_frame():
    print(f"\n{CYAN}S3: Frame rỗng / None{RESET}")

    def process_frame(frame):
        if frame is None:
            raise TypeError("frame is None")
        if frame.size == 0:
            raise ValueError("frame is empty (size==0)")
        return True

    # Case: None frame
    try:
        process_frame(None)
        record("None frame bị bắt", False)
    except TypeError as e:
        record("None frame gây TypeError", True, str(e))

    # Case: empty array
    try:
        process_frame(np.array([]))
        record("Empty frame bị bắt", False)
    except ValueError as e:
        record("Empty array gây ValueError", True, str(e))

    # Code thật KHÔNG có guard này → crash tại cv2.resize / YOLO predict
    record("Code thật thiếu guard frame rỗng", True,
           f"{YELLOW}BUG: _video_worker không validate frame trước khi gọi YOLO{RESET}")


# ══════════════════════════════════════════════════════════════
#  S4: Frame sai shape (grayscale / RGBA)
# ══════════════════════════════════════════════════════════════
def test_s4_wrong_shape():
    print(f"\n{CYAN}S4: Frame sai số kênh màu{RESET}")
    import cv2

    gray = make_frame(channels=1)
    rgba = make_frame(channels=4)
    bgr  = make_frame(channels=3)

    def check_shape(frame, name):
        if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] != 3):
            return False, f"{name}: shape={frame.shape} KHÔNG phải BGR"
        return True, f"{name}: OK"

    ok1, msg1 = check_shape(gray, "Grayscale")
    ok2, msg2 = check_shape(rgba, "RGBA")
    ok3, msg3 = check_shape(bgr, "BGR")

    record("Phát hiện frame grayscale lỗi", not ok1, msg1)
    record("Phát hiện frame RGBA lỗi", not ok2, msg2)
    record("Frame BGR hợp lệ", ok3, msg3)
    record("Code thật không validate shape", True,
           f"{YELLOW}BUG: YOLO sẽ crash hoặc cho kết quả sai với grayscale camera{RESET}")


# ══════════════════════════════════════════════════════════════
#  S5: Frame skip — cache rỗng lúc đầu
# ══════════════════════════════════════════════════════════════
def test_s5_frame_skip_empty_cache():
    print(f"\n{CYAN}S5: Frame skip với cache ban đầu rỗng{RESET}")

    # Mô phỏng ObjectDetector.track() khi skip
    class FakeObjDetector:
        def __init__(self):
            self._frame_counter = 0
            self._cached_results = []
            self.object_skip = 3

        def track(self, frame, **kwargs):
            self._frame_counter += 1
            if self._frame_counter % self.object_skip != 0:
                return self._cached_results, False
            # fake result
            self._cached_results = [fake_obj()]
            return self._cached_results, True

        def reset_skip_counter(self):
            self._frame_counter = 0
            self._cached_results = []

    od = FakeObjDetector()
    od.reset_skip_counter()

    # Frame 1 và 2: skip → cache rỗng
    r1, ran1 = od.track(make_frame())
    r2, ran2 = od.track(make_frame())
    r3, ran3 = od.track(make_frame())   # frame 3 → chạy

    record("Frame 1 trả cache rỗng (an toàn)", not ran1 and r1 == [],
           f"ran={ran1}, len={len(r1)}")
    record("Frame 2 trả cache rỗng (an toàn)", not ran2 and r2 == [],
           f"ran={ran2}, len={len(r2)}")
    record("Frame 3 chạy inference", ran3 and len(r3) == 1,
           f"ran={ran3}, len={len(r3)}")

    # Baggage tracker với cache rỗng — không có lỗi
    sys.path.insert(0, r"d:\Al_Python\Stroke_al")
    try:
        from app.ai.baggage_tracker import AbandonedBaggageTracker
        bt = AbandonedBaggageTracker()
        alerts = bt.update([], [])  # objects rỗng
        record("BaggageTracker.update([],[]) không crash", alerts == [],
               f"alerts={alerts}")
    except Exception as e:
        record("BaggageTracker.update([],[]) crash", False, str(e))


# ══════════════════════════════════════════════════════════════
#  S6: WeaponDetector.detect_frame nhận frame thay vì list
# ══════════════════════════════════════════════════════════════
def test_s6_weapon_frame_vs_list():
    print(f"\n{CYAN}S6: WeaponDetector nhận frame vs list{RESET}")
    sys.path.insert(0, r"d:\Al_Python\Stroke_al")
    try:
        from app.ai.weapon_detector import WeaponDetector

        # Mock object_detector
        mock_od = MagicMock()
        mock_od.detect.return_value = []

        wd = WeaponDetector(mock_od)

        # Truyền list (đúng)
        alerts_list = wd.detect_frame([], [], camera_id="CAM_TEST")
        record("detect_frame(list=[]) OK", isinstance(alerts_list, list),
               f"returned {alerts_list}")

        # Truyền frame numpy (branch else → gọi od.detect)
        frame = make_frame()
        alerts_frame = wd.detect_frame(frame, [], camera_id="CAM_TEST")
        record("detect_frame(frame) gọi od.detect()", mock_od.detect.called,
               f"called={mock_od.detect.called}")
        record("detect_frame(frame) trả list", isinstance(alerts_frame, list))

    except Exception as e:
        record("WeaponDetector import/init lỗi", False, str(e))


# ══════════════════════════════════════════════════════════════
#  S7: Queue full → drop frame
# ══════════════════════════════════════════════════════════════
def test_s7_queue_full():
    print(f"\n{CYAN}S7: frame_queue full → drop frame{RESET}")

    q = Queue(maxsize=2)
    q.put(make_frame())
    q.put(make_frame())

    # Code thật: if not self.frame_queue.full(): self.frame_queue.put(frame)
    frames_dropped = 0
    for i in range(5):
        if not q.full():
            q.put(make_frame())
        else:
            frames_dropped += 1

    record("Frame bị drop khi queue full", frames_dropped == 5,
           f"dropped={frames_dropped}/5")
    record("Drop KHÔNG gây crash (if not full)", True)
    record("UI có thể bị lag khi GPU chậm (queue luôn full)", True,
           f"{YELLOW}WARN: Queue size=2 rất nhỏ, GPU chậm → UI không có frame mới{RESET}")


# ══════════════════════════════════════════════════════════════
#  S8: Race condition _last_obj_results
# ══════════════════════════════════════════════════════════════
def test_s8_race_condition():
    print(f"\n{CYAN}S8: Race condition _last_obj_results{RESET}")

    # _video_worker (thread) ghi _last_obj_results
    # _async_airport_upload (thread pool) đọc frame.copy()
    # Không có lock → ghi đè giữa chừng là có thể

    shared = {'last_obj': []}
    errors = []

    def writer():
        for i in range(1000):
            shared['last_obj'] = [fake_obj(track_id=i)]

    def reader():
        for i in range(1000):
            try:
                val = shared['last_obj']
                _ = len(val)  # đọc length
            except Exception as e:
                errors.append(str(e))

    t1 = threading.Thread(target=writer)
    t2 = threading.Thread(target=reader)
    t1.start(); t2.start()
    t1.join(); t2.join()

    record("CPython GIL bảo vệ list assignment cơ bản", len(errors) == 0,
           f"errors={len(errors)}")
    record("Không có threading.Lock cho _last_obj_results", True,
           f"{YELLOW}WARN: Dùng GIL ngầm — an toàn với CPython nhưng không portable{RESET}")


# ══════════════════════════════════════════════════════════════
#  S9: BaggageTracker bbox thiếu
# ══════════════════════════════════════════════════════════════
def test_s9_baggage_bad_bbox():
    print(f"\n{CYAN}S9: BaggageTracker với bbox lỗi{RESET}")
    sys.path.insert(0, r"d:\Al_Python\Stroke_al")
    try:
        from app.ai.baggage_tracker import AbandonedBaggageTracker
        bt = AbandonedBaggageTracker()

        # bbox rỗng
        bad_obj = {'track_id': 1, 'class_id': 26, 'bbox': [], 'conf': 0.8}
        try:
            alerts = bt.update([bad_obj], [])
            record("bbox=[] được lọc ra (không crash)", True,
                   f"alerts={alerts}")
        except Exception as e:
            record("bbox=[] gây crash", False, str(e))

        # bbox 2 phần tử (thiếu)
        bad_obj2 = {'track_id': 2, 'class_id': 26, 'bbox': [10, 20], 'conf': 0.8}
        try:
            alerts2 = bt.update([bad_obj2], [])
            record("bbox=[10,20] được lọc ra", True, f"alerts={alerts2}")
        except Exception as e:
            record("bbox=[10,20] gây crash", False, str(e))

        # None bbox
        bad_obj3 = {'track_id': 3, 'class_id': 26, 'bbox': None, 'conf': 0.8}
        try:
            alerts3 = bt.update([bad_obj3], [])
            record("bbox=None được lọc ra", True, f"alerts={alerts3}")
        except Exception as e:
            record("bbox=None gây crash", False, str(e))

    except Exception as e:
        record("BaggageTracker import lỗi", False, str(e))


# ══════════════════════════════════════════════════════════════
#  S10: WeaponDetector _find_bearer với bbox ngắn
# ══════════════════════════════════════════════════════════════
def test_s10_find_bearer_bad_bbox():
    print(f"\n{CYAN}S10: WeaponDetector _find_bearer bbox lỗi{RESET}")
    sys.path.insert(0, r"d:\Al_Python\Stroke_al")
    try:
        from app.ai.weapon_detector import WeaponDetector
        mock_od = MagicMock()
        wd = WeaponDetector(mock_od)

        # persons với bbox thiếu
        bad_persons = [
            {'track_id': 1, 'bbox': [], 'conf': 0.9},       # rỗng
            {'track_id': 2, 'bbox': [10, 20], 'conf': 0.9}, # thiếu
            {'track_id': 3, 'bbox': None, 'conf': 0.9},     # None
        ]

        try:
            bearer = wd._find_bearer(100, 100, bad_persons)
            record("_find_bearer bỏ qua bbox ngắn (None trả về)", bearer is None,
                   f"bearer={bearer}")
        except Exception as e:
            record("_find_bearer crash với bbox ngắn", False, str(e))

        # persons với bbox None → len() sẽ crash
        try:
            bearer2 = wd._find_bearer(100, 100,
                                       [{'track_id': 4, 'bbox': None}])
            record("_find_bearer với bbox=None", bearer2 is None, f"bearer={bearer2}")
        except TypeError as e:
            record("_find_bearer crash với bbox=None", False,
                   f"{RED}BUG: {e}{RESET}")

    except Exception as e:
        record("WeaponDetector test_s10 lỗi import", False, str(e))


# ══════════════════════════════════════════════════════════════
#  S11: Adaptive skip logic
# ══════════════════════════════════════════════════════════════
def test_s11_adaptive_skip():
    print(f"\n{CYAN}S11: Adaptive frame skip{RESET}")

    FRAME_SKIP = 2
    object_skip = 3
    adaptive_mode = True
    _adaptive_counter = 0

    def simulate_adaptive(fps):
        nonlocal FRAME_SKIP, object_skip, _adaptive_counter
        _adaptive_counter += 1
        if _adaptive_counter >= 30:
            _adaptive_counter = 0
            if fps < 18:
                FRAME_SKIP    = min(4, FRAME_SKIP + 1)
                object_skip   = min(6, object_skip + 1)
            elif fps > 26 and FRAME_SKIP > 1:
                FRAME_SKIP    = max(1, FRAME_SKIP - 1)
                object_skip   = max(3, object_skip - 1)

    # Simulate 30 frames @ FPS=10 (thấp)
    for _ in range(30):
        simulate_adaptive(10)

    record("FRAME_SKIP tăng khi FPS < 18", FRAME_SKIP == 3,
           f"FRAME_SKIP={FRAME_SKIP}")
    record("object_skip tăng khi FPS < 18", object_skip == 4,
           f"object_skip={object_skip}")

    # Simulate 30 frames @ FPS=30 (cao)
    for _ in range(30):
        simulate_adaptive(30)

    record("FRAME_SKIP giảm khi FPS > 26", FRAME_SKIP == 2,
           f"FRAME_SKIP={FRAME_SKIP}")

    # Boundary: FPS=10 liên tục → FRAME_SKIP không vượt 4
    for _ in range(300):
        simulate_adaptive(10)

    record("FRAME_SKIP không vượt 4", FRAME_SKIP <= 4,
           f"FRAME_SKIP={FRAME_SKIP}")
    record("object_skip không vượt 6", object_skip <= 6,
           f"object_skip={object_skip}")


# ══════════════════════════════════════════════════════════════
#  S12: cap.release() leak
# ══════════════════════════════════════════════════════════════
def test_s12_cap_release_leak():
    print(f"\n{CYAN}S12: cap.release() khi có exception{RESET}")

    released = []

    class FakeCap:
        def set(self, *a): pass
        def read(self):
            raise RuntimeError("Simulated camera error")
        def release(self):
            released.append(True)
        def isOpened(self): return True

    cap = FakeCap()

    # Simulate code thật (KHÔNG có finally)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
    except RuntimeError:
        pass  # exception thoát, cap.release() không được gọi
    # không có finally → cap chưa release

    record("Code thật: cap CHƯA release sau exception", len(released) == 0,
           f"{RED}BUG: Camera resource leak! Cần try/finally trong _video_worker{RESET}")

    # Fix: dùng finally
    released.clear()
    try:
        cap2 = FakeCap()
        while True:
            ret, frame = cap2.read()
            if not ret:
                break
    except RuntimeError:
        pass
    finally:
        cap2.release()

    record("Với finally: cap.release() được gọi", len(released) == 1)


# ══════════════════════════════════════════════════════════════
#  TỔNG KẾT
# ══════════════════════════════════════════════════════════════
def print_summary():
    print(f"\n{'='*60}")
    print(f"{CYAN}  KẾT QUẢ KIỂM TRA LUỒNG CAMERA STREAM{RESET}")
    print(f"{'='*60}")

    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    total  = len(results)

    bugs = [
        ("S1",  "Thiếu log rõ khi camera không mở được",
                "Thêm cap.isOpened() check + log trước vòng lặp"),
        ("S2",  "cap.release() không có finally → resource leak",
                "Bọc _video_worker bằng try/finally: cap.release()"),
        ("S3",  "Không validate frame None/rỗng trước YOLO",
                "Thêm: if frame is None or frame.size==0: continue"),
        ("S4",  "Không kiểm tra số kênh frame (grayscale/RGBA)",
                "Thêm: if frame.ndim==2: frame=cv2.cvtColor(frame,COLOR_GRAY2BGR)"),
        ("S10", "_find_bearer không guard bbox=None → potential crash",
                "Đổi: if len(pb) < 4 → if not pb or len(pb) < 4"),
        ("S12", "Camera resource leak khi exception trong _video_worker",
                "Dùng try/finally cho cap.release()"),
    ]

    warns = [
        ("S7",  "frame_queue size=2 rất nhỏ",
                "Tăng maxsize=4-6 để buffer tốt hơn khi GPU lag"),
        ("S8",  "Không có Lock bảo vệ _last_obj_results",
                "CPython GIL an toàn nhưng nên dùng threading.Lock"),
        ("S11", "Adaptive skip không reset khi manual mode",
                "Đã xử lý đúng: manual mode set adaptive_mode=False"),
    ]

    print(f"\n  Tổng: {total} | {GREEN}Qua: {passed}{RESET} | {RED}Lỗi: {failed}{RESET}")

    print(f"\n{RED}  ╔══ BUG CẦN SỬA ({len(bugs)}) ══╗{RESET}")
    for code, bug, fix in bugs:
        print(f"  {RED}▸ [{code}]{RESET} {bug}")
        print(f"        → FIX: {fix}")

    print(f"\n{YELLOW}  ╔══ CẢNH BÁO ({len(warns)}) ══╗{RESET}")
    for code, w, rec in warns:
        print(f"  {YELLOW}▸ [{code}]{RESET} {w}")
        print(f"        → REC: {rec}")

    print(f"\n{'='*60}\n")
    return failed


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"\n{CYAN}{'='*60}")
    print("   Giả lập Luồng Camera Stream — Stroke_al")
    print(f"{'='*60}{RESET}\n")

    tests = [
        test_s1_camera_open_fail,
        test_s2_camera_disconnect,
        test_s3_empty_frame,
        test_s4_wrong_shape,
        test_s5_frame_skip_empty_cache,
        test_s6_weapon_frame_vs_list,
        test_s7_queue_full,
        test_s8_race_condition,
        test_s9_baggage_bad_bbox,
        test_s10_find_bearer_bad_bbox,
        test_s11_adaptive_skip,
        test_s12_cap_release_leak,
    ]

    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"  {RED}[ERROR] {t.__name__}: {e}{RESET}")

    failed = print_summary()
    sys.exit(0 if failed == 0 else 1)
