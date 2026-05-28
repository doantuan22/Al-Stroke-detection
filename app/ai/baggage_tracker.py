"""
Abandoned Baggage Tracker
==========================
Phát hiện hành lý (túi, vali, balo) bị bỏ lại không có chủ.
"""
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

BAGGAGE_CLASS_IDS = {24: 'backpack', 26: 'handbag', 28: 'suitcase'}

OWNER_RADIUS_PX  = 160   # Bán kính center-to-center (px)
PROXIMITY_EXPAND = 80    # Mở rộng bbox của hành lý để kiểm tra chồng lấn
ABANDON_TIMEOUT  = 60.0
ALERT_COOLDOWN   = 120.0


@dataclass
class BaggageState:
    track_id     : int
    object_class : str
    bbox         : list
    camera_id    : str     = 'CAM_00'
    first_seen   : float   = field(default_factory=time.time)
    last_seen    : float   = field(default_factory=time.time)
    owner_gone_at: Optional[float] = None
    owner_seen_at: Optional[float] = None
    last_alert_at: float   = 0.0
    alert_count  : int     = 0
    alerted      : bool    = False
    db_synced    : bool    = False
    # Trạng thái chủ hành lý frame trước (để phát hiện transition)
    _prev_has_owner: bool  = True   # Giả sử ban đầu có chủ để tránh false alarm ngay lúc xuất hiện

    @property
    def abandon_seconds(self) -> float:
        if self.owner_gone_at is None:
            return 0.0
        return time.time() - self.owner_gone_at

    @property
    def is_suspicious(self, timeout: float = ABANDON_TIMEOUT) -> bool:
        return self.abandon_seconds >= ABANDON_TIMEOUT

    def to_db_record(self) -> dict:
        return {
            'track_id'    : self.track_id,
            'camera_id'   : self.camera_id,
            'object_class': self.object_class,
            'has_owner'   : self.owner_gone_at is None,
            'owner_gone_at': (
                None if self.owner_gone_at is None
                else _unix_to_iso(self.owner_gone_at)
            ),
            'alerted'     : self.alerted,
            'bbox'        : {
                'x1': self.bbox[0], 'y1': self.bbox[1],
                'x2': self.bbox[2], 'y2': self.bbox[3],
            },
        }


def _unix_to_iso(ts: float) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _calculate_iou(box1, box2) -> float:
    """Tính Intersection over Union (IoU) giữa 2 bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0
    return intersection / union


class AbandonedBaggageTracker:
    """
    Theo dõi trạng thái từng hành lý theo thời gian.
    Cần gọi update() mỗi lần có kết quả detection mới.
    """

    def __init__(
        self,
        owner_radius  : float = OWNER_RADIUS_PX,
        timeout       : float = ABANDON_TIMEOUT,
        cooldown      : float = ALERT_COOLDOWN,
        camera_id     : str   = 'CAM_00',
        grace_period  : float = 3.0,
        owner_presence_grace_period: float = 2.0,
    ):
        self.owner_radius = owner_radius
        self.timeout      = timeout
        self.cooldown     = cooldown
        self.camera_id    = camera_id
        self.grace_period = grace_period
        self.owner_presence_grace_period = owner_presence_grace_period

        self._states   : dict[int, BaggageState] = {}

        self._dirty_ids: set[int] = set()   # IDs cần sync DB


    # ── PUBLIC ─────────────────────────────────────────────────
    def update(
        self,
        objects : list[dict],
        persons : list[dict],
    ) -> list[dict]:
        """
        Cập nhật tracker với detection mới nhất.

        Args:
            objects : list từ ObjectDetector.track()
                      keys: track_id, class_id, bbox, conf
            persons : list từ PoseDetector.track()
                      keys: track_id, bbox, kpts, conf

        Returns:
            alerts: list[dict] — các alert mới vừa được kích hoạt.
            Mỗi alert có keys:
                event_type, track_id, object_class, bbox,
                duration_sec, confidence, risk_level, camera_id
        """
        now    = time.time()
        alerts = []

        # Lọc chỉ lấy hành lý có class_id hợp lệ, bbox đủ và conf >= 0.22
        bags = [
            o for o in objects
            if o.get('class_id') in BAGGAGE_CLASS_IDS
            and o.get('bbox') and len(o.get('bbox', [])) >= 4
            and o.get('conf', 0) >= 0.22
        ]

        # Áp dụng NMS tránh các box hành lý trùng nhau của YOLO (backpack vs handbag vs suitcase)
        bags = sorted(bags, key=lambda x: x.get('conf', 0), reverse=True)
        keep_bags = []
        for b in bags:
            overlap = False
            for kept in keep_bags:
                if _calculate_iou(b['bbox'], kept['bbox']) > 0.45:
                    overlap = True
                    break
            if not overlap:
                keep_bags.append(b)
        bags = keep_bags

        active_bag_ids = {b['track_id'] for b in bags}

        # 1. Khắc phục chuyển ID (tracker switching): Nếu một state cũ (đang chờ grace period) 
        # bị trùng lấp vị trí với một bag mới hoạt động, ta kế thừa state cũ cho ID mới để giữ nguyên bộ đếm thời gian.
        for tid in list(self._states.keys()):
            if tid not in active_bag_ids:
                old_state = self._states[tid]
                for bag in bags:
                    new_tid = bag['track_id']
                    if _calculate_iou(old_state.bbox, bag['bbox']) > 0.45:
                        if new_tid not in self._states:
                            old_state.track_id = new_tid
                            self._states[new_tid] = old_state
                        del self._states[tid]
                        break

        # 2. Dọn track đã biến mất quá grace_period thực sự
        for tid in list(self._states.keys()):
            if tid not in active_bag_ids:
                state = self._states[tid]
                if now - state.last_seen >= self.grace_period:
                    del self._states[tid]




        # Xử lý từng hành lý
        for bag in bags:
            tid  = bag['track_id']
            bbox = bag.get('bbox')          # BUG FIX: dùng .get() thay vì []
            if not bbox or len(bbox) < 4:   # Skip nếu bbox thiếu hoặc sai
                continue
            cls  = BAGGAGE_CLASS_IDS.get(bag.get('class_id', -1), 'bag')

            # Tạo state mới nếu chưa có
            if tid not in self._states:
                self._states[tid] = BaggageState(
                    track_id=tid,
                    object_class=cls,
                    bbox=bbox,
                    camera_id=self.camera_id,
                )

            state           = self._states[tid]
            state.bbox      = bbox
            state.last_seen = now


            # Tâm của hành lý
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2

            # ── Kiểm tra owner ──────────────────────────────
            has_owner = self._has_nearby_person(cx, cy, bbox, persons)

            # Phat hien transition owner
            if state._prev_has_owner and not has_owner:
                print(f"[BaggageTracker] #{tid} [OWNER LEFT] {cls} -- bat dau dem thoi gian")
            # Phat hien transition: owner quay lai
            elif not state._prev_has_owner and has_owner:
                print(f"[BaggageTracker] #{tid} [OWNER BACK] {cls}")
            state._prev_has_owner = has_owner

            if has_owner:
                # Có người đứng gần túi
                if state.owner_gone_at is not None:
                    # Nếu cấu hình grace period bằng 0, reset ngay lập tức
                    if self.owner_presence_grace_period <= 0.0:
                        state.owner_gone_at = None
                        state.owner_seen_at = None
                        state.alerted       = False
                        self._dirty_ids.add(tid)
                    else:
                        # Cần người đứng gần liên tục >= self.owner_presence_grace_period mới reset
                        if state.owner_seen_at is None:
                            state.owner_seen_at = now
                        elif now - state.owner_seen_at >= self.owner_presence_grace_period:
                            # Xác nhận chủ quay lại thực sự
                            state.owner_gone_at = None
                            state.owner_seen_at = None
                            state.alerted       = False
                            self._dirty_ids.add(tid)
                else:
                    state.owner_seen_at = None

            else:
                # Không có ai ở gần túi
                state.owner_seen_at = None
                if state.owner_gone_at is None:
                    # Bắt đầu tính thời gian bỏ rơi
                    state.owner_gone_at = now
                    self._dirty_ids.add(tid)


            # ── Kiểm tra alert ────────────────────────────────
            # BUG FIX: dùng self.timeout thay vì is_suspicious (vốn dùng constant)
            if state.abandon_seconds >= self.timeout:
                can_alert = (now - state.last_alert_at) >= self.cooldown
                if can_alert:
                    state.last_alert_at = now
                    state.alert_count  += 1
                    state.alerted       = True
                    self._dirty_ids.add(tid)

                    alerts.append({
                        'event_type'  : 'abandoned_baggage',
                        'track_id'    : tid,
                        'object_class': cls,
                        'bbox'        : bbox,
                        'duration_sec': state.abandon_seconds,
                        'confidence'  : float(bag.get('conf', 0.75)),
                        'risk_level'  : 'high',
                        'camera_id'   : self.camera_id,
                        'zone_name'   : None,
                    })

        return alerts

    def get_all_states(self) -> dict[int, BaggageState]:
        """Trả về toàn bộ state hiện tại (dùng để vẽ overlay)."""
        return self._states

    def pop_dirty(self) -> list[BaggageState]:
        """Lấy danh sách states cần sync lên Supabase rồi xóa queue."""
        dirty = [self._states[tid] for tid in self._dirty_ids
                 if tid in self._states]
        self._dirty_ids.clear()
        return dirty

    def update_camera(self, camera_id: str):
        self.camera_id = camera_id
        for s in self._states.values():
            s.camera_id = camera_id

    # ── INTERNAL ───────────────────────────────────────────────
    def _has_nearby_person(
        self, cx: float, cy: float, bag_bbox: list, persons: list[dict]
    ) -> bool:
        """
        Kiểm tra xem có người nào đứng gần hành lý không.
        Dùng 2 phương pháp song song:
          1. Center-to-center distance <= owner_radius
          2. Expanded bag bbox có chồng lấn với person bbox
             (để bắt người cao đứng sát bên, tâm xa nhưng bbox overlap)
        conf người tối thiểu 0.35.
        """
        bx1, by1, bx2, by2 = bag_bbox[:4]
        # Mở rộng bbox hành lý theo 4 hướng
        ex1 = bx1 - PROXIMITY_EXPAND
        ey1 = by1 - PROXIMITY_EXPAND
        ex2 = bx2 + PROXIMITY_EXPAND
        ey2 = by2 + PROXIMITY_EXPAND

        for p in persons:
            if p.get('conf', 0) < 0.35:
                continue
            pb = p.get('bbox', [])
            if len(pb) < 4:
                continue
            px1, py1, px2, py2 = pb[:4]
            pcx = (px1 + px2) / 2
            pcy = (py1 + py2) / 2

            # Phương pháp 1: center-to-center distance
            if np.hypot(cx - pcx, cy - pcy) <= self.owner_radius:
                return True

            # Phương pháp 2: expanded bag bbox overlap với person bbox
            if (ex1 < px2 and ex2 > px1 and ey1 < py2 and ey2 > py1):
                return True

        return False

