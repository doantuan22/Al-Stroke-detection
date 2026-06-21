"""
Abandoned Baggage Tracker
==========================
Phát hiện hành lý (túi, vali, balo) bị bỏ lại không có chủ.
"""
import time
import numpy as np
from collections import defaultdict, deque
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
    likely_owner_id: Optional[int] = None
    stationary_since: Optional[float] = None
    center_velocity: float = 0.0
    risk_score: float = 0.0
    owner_scores: dict[int, float] = field(default_factory=dict)
    center_history: deque = field(default_factory=lambda: deque(maxlen=20))
    seen_count: int = 0
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
        config: dict | None = None,
    ):
        cfg = (config or {}).get('baggage', {})
        self.owner_radius = owner_radius
        self.timeout      = timeout
        self.cooldown     = cooldown
        self.camera_id    = camera_id
        self.grace_period = grace_period
        self.owner_presence_grace_period = owner_presence_grace_period
        self.confidence_threshold = float(cfg.get('confidence_threshold', 0.22))
        self.class_confidence_thresholds = cfg.get('class_confidence_thresholds', {}) or {}
        self.min_confirmed_seen_sec = float(cfg.get('min_confirmed_seen_sec', 0.0))
        self.min_confirmed_frames = int(cfg.get('min_confirmed_frames', 1))
        self.min_box_width = float(cfg.get('min_box_width_px', 0.0))
        self.min_box_height = float(cfg.get('min_box_height_px', 0.0))
        self.min_aspect_ratio = float(cfg.get('min_aspect_ratio', 0.0))
        self.max_aspect_ratio = float(cfg.get('max_aspect_ratio', 999.0))
        self.proximity_expand = float(cfg.get('proximity_expand_px', PROXIMITY_EXPAND))
        self.stationary_velocity_threshold = float(cfg.get('stationary_velocity_threshold', 2.0))
        self.stationary_duration = float(cfg.get('stationary_duration_sec', 3.0))
        self.owner_leave_distance_threshold = float(cfg.get('owner_leave_distance_threshold', 120.0))
        self.risk_alert_threshold = float(cfg.get('risk_alert_threshold', 0.75))
        self.risk_weights = cfg.get('risk', {})
        self.zone_timeouts = (config or {}).get('zones', {})

        self._states   : dict[int, BaggageState] = {}
        self._next_bag_id: int = 1

        self._dirty_ids: set[int] = set()   # IDs cần sync DB


    # ── PUBLIC ─────────────────────────────────────────────────
    def update(
        self,
        objects : list[dict],
        persons : list[dict],
        zone_name: str | None = None,
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
        bags = [o for o in objects if self._is_plausible_baggage(o)]

        # Áp dụng NMS tránh các box hành lý trùng nhau của YOLO (backpack vs handbag vs suitcase)
        bags = sorted(bags, key=lambda x: x.get('conf', 0), reverse=True)
        keep_bags = []
        for b in bags:
            overlap = False
            for kept in keep_bags:
                # Nới lỏng NMS (0.85) để cho phép balo đặt sát nhau
                if _calculate_iou(b['bbox'], kept['bbox']) > 0.85:
                    overlap = True
                    break
            if not overlap:
                keep_bags.append(b)
        bags = keep_bags

        # ── CUSTOM IOU TRACKER ─────────────────────────────────────
        # Gán ID nội bộ bằng IoU matching, phớt lờ hoàn toàn track_id của YOLO
        # Điều này giải quyết dứt điểm lỗi YOLO tự nhảy/đảo ID giữa các túi.
        matched_bag_ids = set()
        for bag in bags:
            best_iou = 0.0
            best_tid = None
            for tid, state in self._states.items():
                if tid in matched_bag_ids:
                    continue
                iou = _calculate_iou(bag['bbox'], state.bbox)
                if iou > 0.45 and iou > best_iou:
                    best_iou = iou
                    best_tid = tid
            
            if best_tid is not None:
                bag['track_id'] = best_tid
                matched_bag_ids.add(best_tid)
            else:
                external_tid = bag.get('track_id')
                if isinstance(external_tid, int) and external_tid not in self._states:
                    bag['track_id'] = external_tid
                    self._next_bag_id = max(self._next_bag_id, external_tid + 1)
                else:
                    bag['track_id'] = self._next_bag_id
                    self._next_bag_id += 1

        active_bag_ids = {b['track_id'] for b in bags}
        person_features = self._prepare_person_features(persons)

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
            state.seen_count += 1


            # Tâm của hành lý
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            self._update_motion_state(state, cx, cy, now)
            self._update_owner_scores(state, cx, cy, person_features, now)

            # ── Kiểm tra owner ──────────────────────────────
            nearby_person_id = self._get_nearby_person_id(cx, cy, bbox, person_features)
            has_owner = (nearby_person_id is not None)

            # Phat hien transition owner
            if state._prev_has_owner and not has_owner:
                print(f"[BaggageTracker] #{tid} [OWNER LEFT] {cls} -- bat dau dem thoi gian")
            # Phat hien transition: owner quay lai
            elif not state._prev_has_owner and has_owner:
                print(f"[BaggageTracker] #{tid} [OWNER BACK] {cls}")
            state._prev_has_owner = has_owner

            if has_owner:
                # Có người đứng gần túi
                is_real_owner = (nearby_person_id == state.likely_owner_id)
                # Người lạ cần đứng gần lâu hơn (5s) mới được tính là owner mới,
                # còn chủ cũ (hoặc chưa xác định) thì dùng grace period mặc định (2s)
                required_grace = self.owner_presence_grace_period if (is_real_owner or state.likely_owner_id is None) else 5.0
                
                if state.owner_gone_at is not None:
                    # Cần người đứng gần liên tục đủ lâu mới reset
                    if required_grace <= 0.0:
                        state.owner_gone_at = None
                        state.owner_seen_at = None
                        state.alerted       = False
                        self._dirty_ids.add(tid)
                    elif state.owner_seen_at is None:
                        state.owner_seen_at = now
                    elif now - state.owner_seen_at >= required_grace:
                        # Xác nhận có chủ quay lại / nhận túi
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
            timeout = self._timeout_for_zone(zone_name)
            state.risk_score = self._calculate_risk(state, has_owner, timeout)
            confirmed_long_enough = (
                state.seen_count >= self.min_confirmed_frames
                and now - state.first_seen >= self.min_confirmed_seen_sec
            )
            if (
                confirmed_long_enough
                and state.abandon_seconds >= timeout
                and state.risk_score >= self.risk_alert_threshold
            ):
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
                        'risk_level'  : self._risk_level(state.risk_score, zone_name),
                        'camera_id'   : self.camera_id,
                        'zone_name'   : zone_name,
                        'risk_score'  : state.risk_score,
                        'likely_owner_id': state.likely_owner_id,
                        'stationary'  : self._is_stationary(state),
                    })

        return alerts

    def _is_plausible_baggage(self, obj: dict) -> bool:
        cid = obj.get('class_id')
        if cid not in BAGGAGE_CLASS_IDS:
            return False
        bbox = obj.get('bbox') or []
        if len(bbox) < 4:
            return False

        cls_name = BAGGAGE_CLASS_IDS.get(cid, 'bag')
        conf = float(obj.get('conf', 0.0))
        required_conf = max(
            self.confidence_threshold,
            float(self.class_confidence_thresholds.get(cls_name, 0.0)),
        )
        if conf < required_conf:
            return False

        width = float(bbox[2] - bbox[0])
        height = float(bbox[3] - bbox[1])
        if width < self.min_box_width or height < self.min_box_height:
            return False
        aspect = width / max(height, 1e-6)
        return self.min_aspect_ratio <= aspect <= self.max_aspect_ratio

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
    def _prepare_person_features(self, persons: list[dict]) -> list[dict]:
        features = []
        for p in persons:
            if p.get('conf', 0) < 0.35:
                continue
            pb = p.get('bbox') or []
            if len(pb) < 4:
                continue
            px1, py1, px2, py2 = pb[:4]
            features.append({
                'track_id': p.get('track_id'),
                'bbox': (px1, py1, px2, py2),
                'cx': (px1 + px2) * 0.5,
                'cy': (py1 + py2) * 0.5,
            })
        return features

    def _get_nearby_person_id(
        self, cx: float, cy: float, bag_bbox: list, persons: list[dict]
    ) -> Optional[int]:
        """
        Trả về track_id của người đứng gần hành lý nhất (nếu có).
        Đo bằng khoảng cách ngắn nhất giữa Bounding Box của người và túi.
        """
        bx1, by1, bx2, by2 = bag_bbox[:4]
        
        closest_id = None
        min_box_dist_sq = float('inf')
        
        allowed_dist_sq = self.proximity_expand * self.proximity_expand

        for p in persons:
            px1, py1, px2, py2 = p['bbox']
            
            # Khoảng cách ngắn nhất giữa 2 hình chữ nhật (0 nếu giao nhau)
            dx = max(0.0, max(bx1 - px2, px1 - bx2))
            dy = max(0.0, max(by1 - py2, py1 - by2))
            box_dist_sq = dx * dx + dy * dy

            if box_dist_sq <= allowed_dist_sq:
                if box_dist_sq < min_box_dist_sq:
                    min_box_dist_sq = box_dist_sq
                    closest_id = p.get('track_id')

        return closest_id

    def _update_motion_state(self, state: BaggageState, cx: float, cy: float, now: float):
        if state.center_history:
            prev_t, px, py = state.center_history[-1]
            dt = max(now - prev_t, 1e-6)
            state.center_velocity = float(np.hypot(cx - px, cy - py) / dt)
        state.center_history.append((now, cx, cy))

        if state.center_velocity <= self.stationary_velocity_threshold:
            if state.stationary_since is None:
                state.stationary_since = now
        else:
            state.stationary_since = None

    def _update_owner_scores(self, state: BaggageState, cx: float, cy: float, persons: list[dict], now: float):
        owner_radius_sq = self.owner_radius * self.owner_radius
        leave_distance_sq = self.owner_leave_distance_threshold * self.owner_leave_distance_threshold
        for p in persons:
            tid = p.get('track_id')
            if tid is None:
                continue
            dx = cx - p['cx']
            dy = cy - p['cy']
            dist_sq = dx * dx + dy * dy
            near_bonus = 1.0 if dist_sq <= owner_radius_sq else 0.0
            leave_penalty = 0.35 if dist_sq > leave_distance_sq else 0.0
            state.owner_scores[tid] = max(
                0.0, state.owner_scores.get(tid, 0.0) + near_bonus - leave_penalty
            )

        if state.owner_scores:
            state.likely_owner_id = max(state.owner_scores, key=state.owner_scores.get)

    def _is_stationary(self, state: BaggageState) -> bool:
        return (
            state.stationary_since is not None
            and time.time() - state.stationary_since >= self.stationary_duration
        )

    def _timeout_for_zone(self, zone_name: str | None) -> float:
        if zone_name and zone_name in self.zone_timeouts:
            return float(self.zone_timeouts[zone_name].get('baggage_timeout_sec', self.timeout))
        default_zone = self.zone_timeouts.get('default', {})
        return float(default_zone.get('baggage_timeout_sec', self.timeout))

    def _calculate_risk(self, state: BaggageState, has_owner: bool, timeout: float) -> float:
        weights = {
            'stationary_weight': 0.20,
            'owner_left_weight': 0.30,
            'no_nearby_person_weight': 0.20,
            'timeout_weight': 0.30,
            'leave_distance_weight': 0.15,
        }
        weights.update(self.risk_weights or {})
        risk = 0.0
        if self._is_stationary(state):
            risk += float(weights['stationary_weight'])
        if state.owner_gone_at is not None:
            risk += float(weights['owner_left_weight'])
            if state.abandon_seconds >= timeout:
                risk += float(weights['timeout_weight'])
        if not has_owner:
            risk += float(weights['no_nearby_person_weight'])
        if state.likely_owner_id is not None and state.owner_gone_at is not None:
            risk += float(weights['leave_distance_weight'])
        return min(1.0, risk)

    @staticmethod
    def _risk_level(score: float, zone_name: str | None = None) -> str:
        if zone_name == 'security_checkpoint' and score >= 0.75:
            return 'critical'
        if score >= 0.75:
            return 'high'
        if score >= 0.5:
            return 'medium'
        return 'low'

