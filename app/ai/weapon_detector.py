"""
Weapon Detector
================
Phát hiện vật thể nguy hiểm (dao, súng, v.v.) trong frame.

Chiến lược 2 giai đoạn:
  Giai đoạn 1 (hiện tại): Dùng yolov8n.pt (COCO) detect knife (class 43)
                           làm baseline — đã hoạt động ngay không cần train.
  Giai đoạn 2 (tương lai): Load fine-tuned weapon model từ Roboflow.
                            Chỉ cần đổi model_path là xong.

Tìm "bearer" — người đang cầm vũ khí — bằng cách tìm người gần nhất.
Alert ngay khi phát hiện (không cần timer như abandoned baggage).
"""
import time
import numpy as np
from typing import Optional

# ── COCO class IDs có thể dùng làm baseline ───────────────────
COCO_WEAPON_CLASSES = {
    43: 'knife',        # dao/kéo
}

# ── Class IDs cho fine-tuned weapon model (Roboflow) ──────────
# Sẽ điều chỉnh theo dataset thực tế khi có model
FINETUNE_WEAPON_CLASSES = {
    0 : 'gun',
    1 : 'knife',
    2 : 'pistol',
    3 : 'rifle',
    4 : 'scissors',
}

WEAPON_CONF_MIN   = 0.15    # Ngưỡng confidence tối thiểu (thấp hơn để bắt dao)
BEARER_RADIUS_PX  = 200     # Pixel tính là "người đang cầm" (rộng hơn)
ALERT_COOLDOWN    = 5.0     # Giây giữa 2 alert upload (ngắn hơn để upload thường xuyên)
OVERLAY_PERSIST   = 2.0     # Giây giữ overlay sau khi mất detection



class WeaponDetector:
    """
    Phát hiện vũ khí bằng YOLO + xác định người đang cầm.

    Dùng chung ObjectDetector đã khởi tạo để tránh load model 2 lần.
    """

    def __init__(
        self,
        object_detector,              # ObjectDetector instance đã khởi tạo
        use_finetune: bool   = False, # True khi có fine-tuned model
        conf: float          = WEAPON_CONF_MIN,
        bearer_radius: float = BEARER_RADIUS_PX,
        cooldown: float      = ALERT_COOLDOWN,
        wrist_distance_threshold: float = 80.0,
        pose_confidence_boost: float = 0.15,
        min_pose_associated_conf: float = 0.40,
        min_unassociated_conf: float = 0.85,
        require_pose_association: bool = False,
        min_persistence_sec: float = 0.0,
        strict_pose_classes: Optional[list[str]] = None,
        class_confidence_thresholds: Optional[dict[str, float]] = None,
        class_area_ratio_limits: Optional[dict[str, float]] = None,
    ):
        """
        Args:
            object_detector : ObjectDetector instance (tái dùng model)
            use_finetune    : True nếu đang dùng fine-tuned weapon model
            conf            : Confidence threshold
            bearer_radius   : Pixel radius để xác định bearer
            cooldown        : Giây giữa 2 alert cùng vị trí
        """
        self.od            = object_detector
        self.use_finetune  = use_finetune
        self.conf          = conf
        self.bearer_radius = bearer_radius
        self.cooldown      = cooldown
        self.wrist_distance_threshold = max(100.0, wrist_distance_threshold)
        self.pose_confidence_boost = pose_confidence_boost
        self.min_pose_associated_conf = min_pose_associated_conf
        self.min_unassociated_conf = min_unassociated_conf
        self.require_pose_association = require_pose_association
        self.min_persistence_sec = min_persistence_sec
        self.strict_pose_classes = {
            str(name).lower() for name in (strict_pose_classes or ["gun", "pistol", "rifle"])
        }
        self.class_confidence_thresholds = {
            str(k).lower(): float(v)
            for k, v in (class_confidence_thresholds or {}).items()
        }
        self.class_area_ratio_limits = {
            str(k).lower(): float(v)
            for k, v in (class_area_ratio_limits or {}).items()
        }

        # Map class IDs theo model đang dùng
        if use_finetune:
            # Lấy trực tiếp từ class names của model chuyên dụng
            self.weapon_classes = {k: str(v).lower() for k, v in self.od.model.names.items()}
            print(f"[WeaponDetector] Using fine-tuned model classes: {self.weapon_classes}")
        else:
            self.weapon_classes = COCO_WEAPON_CLASSES
            
        self.weapon_class_ids = list(self.weapon_classes.keys())

        # Cooldown tracker per location key (cho upload)
        self._last_alert: dict[str, float] = {}

        # Persistent detection state: weapon đang nhìn thấy trong frame hiện tại
        # Key: loc_key, Value: dict {bbox, class_name, conf, last_seen, bearer_id}
        self._active_detections: dict[str, dict] = {}

    # ── PUBLIC ─────────────────────────────────────────────────
    def detect_frame(
        self,
        frame_or_objs,
        persons: list[dict],
        coco_objects: Optional[list[dict]] = None,
        zone_name: Optional[str] = None,
        camera_id: str           = 'CAM_00',
    ) -> list[dict]:
        """
        Detect vũ khí từ obj_results (đã track) hoặc frame và tạo alerts.

        Trả về 2 loại:
        - upload_alerts : alerts mới cần gửi lên cloud (theo cooldown)
        - overlay_detections : TẤT CẢ weapons đang nhìn thấy (dùng cho overlay)

        Args:
            frame_or_objs : list[dict] kết quả track hoặc BGR numpy array
            persons       : Danh sách từ PoseDetector.track()
            zone_name     : Tên zone (nếu có ZoneGuard) → risk = critical
            camera_id     : ID camera

        Returns:
            alerts: list[dict]  — chỉ chứa alerts ĐỦ ĐIỀU KIỆN upload (cooldown OK)
        """
        import numpy as np
        if isinstance(frame_or_objs, list):
            all_objects = [
                obj for obj in frame_or_objs
                if obj.get('class_id') in self.weapon_class_ids
                and obj.get('conf', 0) >= self.conf
            ]
        else:
            all_objects, ran_inf = self.od.track(
                frame_or_objs,
                classes=self.weapon_class_ids,
                conf=self.conf,
            )

        # ── TÍCH HỢP COCO WEAPON (KNIFE = 43) ──
        # Model YOLO tự train nhận diện dao rất kém, trong khi COCO nhận diện dao (43) rất tốt.
        # Chúng ta sẽ hợp nhất kết quả của COCO vào danh sách vũ khí.
        if coco_objects:
            knife_cid = next((k for k, v in self.weapon_classes.items() if v == 'knife'), None)
            if knife_cid is not None:
                for co in coco_objects:
                    if co.get('class_id') == 43 and co.get('conf', 0) >= self.conf:
                        all_objects.append({
                            'class_id': knife_cid,
                            'bbox': co['bbox'],
                            'conf': co['conf']
                        })

        alerts = []
        now    = time.time()
        seen_keys: set[str] = set()

        for obj in all_objects:
            cid  = obj['class_id']
            if cid not in self.weapon_classes:
                continue

            bbox       = obj['bbox']
            class_name = self.weapon_classes[cid]
            conf_val   = obj['conf']

            # ── ENSEMBLE FILTERING (Chống Ảo Giác Model Yếu) ──
            if coco_objects and self.use_finetune:
                is_harmless = False
                # Mở rộng danh sách vật dụng có thể cầm tay hoặc gây nhầm lẫn:
                harmless_classes = {
                    24, 25, 26, 27, 28, 39, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
                    56, 57, 63, 64, 65, 66, 67, 73, 74, 75, 76, 77, 78, 79
                }
                for co in coco_objects:
                    cid_coco = co.get('class_id')
                    if cid_coco in harmless_classes:
                        c_bbox = co['bbox']
                        # Tính Intersection over Minimum Area (IoMin)
                        xA = max(bbox[0], c_bbox[0])
                        yA = max(bbox[1], c_bbox[1])
                        xB = min(bbox[2], c_bbox[2])
                        yB = min(bbox[3], c_bbox[3])
                        interArea = max(0, xB - xA) * max(0, yB - yA)
                        if interArea > 0:
                            boxAArea = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                            boxBArea = (c_bbox[2] - c_bbox[0]) * (c_bbox[3] - c_bbox[1])
                            minArea = min(boxAArea, boxBArea)
                            if minArea > 0:
                                iomin = interArea / float(minArea)
                                # Nếu diện tích giao nhau chiếm > 40% của box nhỏ hơn -> Ảo giác!
                                if iomin > 0.4:
                                    is_harmless = True
                                    break
                if is_harmless:
                    continue

            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2

            # Xác định bearer (người đang cầm) TRƯỚC ĐỂ LÀM KEY KHÓA MỤC TIÊU!
            bearer_id, hand_distance, pose_associated, bearer_bbox = self._find_bearer_details(cx, cy, persons)

            if bearer_id is not None:
                # Khóa mục tiêu theo người đang cầm (Bearer Tracking)
                loc_key = f"weapon_{class_name}_bearer_{bearer_id}"
            else:
                # Nếu không có người cầm, dùng spatial tracking tĩnh
                matched_key = None
                for key, det in self._active_detections.items():
                    if det['class_name'] == class_name and det['bearer_id'] is None:
                        dbbox = det['bbox']
                        dcx = (dbbox[0] + dbbox[2]) / 2
                        dcy = (dbbox[1] + dbbox[3]) / 2
                        if (cx - dcx)**2 + (cy - dcy)**2 < 150**2:
                            matched_key = key
                            break
                if matched_key:
                    loc_key = matched_key
                else:
                    loc_key = f"{class_name}_static_{int(cx)}_{int(cy)}"
                    
            seen_keys.add(loc_key)
            
            # ── BẢO VỆ KÉP (DUAL THRESHOLD) ──
            if class_name in self.strict_pose_classes and not pose_associated:
                continue
            if not pose_associated and self.require_pose_association:
                continue

            req_conf = max(self.conf, self._class_min_conf(class_name, pose_associated))
            if conf_val < req_conf:
                continue

            if not self._weapon_bbox_sane(class_name, bbox, bearer_bbox):
                continue
            
            adjusted_conf = min(
                1.0,
                conf_val + (self.pose_confidence_boost if pose_associated else 0.0)
            )

            # Risk level
            risk = 'critical' if zone_name else 'high'

            # Cập nhật persistent detection
            if loc_key not in self._active_detections:
                # Vũ khí mới xuất hiện
                self._active_detections[loc_key] = {
                    'bbox'        : bbox,
                    'class_name'  : class_name,
                    'conf'        : adjusted_conf,
                    'bearer_id'   : bearer_id,
                    'hand_distance': hand_distance,
                    'risk_level'  : risk,
                    'zone_name'   : zone_name,
                    'camera_id'   : camera_id,
                    'first_seen'  : now,
                    'last_seen'   : now,
                    'alerted'     : False,
                }
            else:
                # Vũ khí đã có, cập nhật trạng thái mới nhất
                det = self._active_detections[loc_key]
                det['bbox'] = bbox
                det['conf'] = adjusted_conf
                det['bearer_id'] = bearer_id
                det['hand_distance'] = hand_distance
                det['risk_level'] = risk
                det['zone_name'] = zone_name
                det['last_seen'] = now

            det = self._active_detections[loc_key]

            # Kiểm tra thời gian xuất hiện liên tục mới được phép gửi alert.
            if now - det['first_seen'] >= self.min_persistence_sec:
                # Đã đủ 5 giây liên tục
                if not det['alerted']:
                    det['alerted'] = True
                    self._last_alert[loc_key] = now
                    alerts.append({
                        'event_type'  : 'weapon_detected',
                        'track_id'    : bearer_id,
                        'object_class': class_name,
                        'bbox'        : bbox,
                        'confidence'  : adjusted_conf,
                        'risk_level'  : risk,
                        'zone_name'   : zone_name,
                        'camera_id'   : camera_id,
                        'duration_sec': round(now - det['first_seen'], 1),
                        'bearer_id'   : bearer_id,
                        'hand_distance': hand_distance,
                        'pose_associated': pose_associated,
                    })
                else:
                    # Đã gửi alert trước đó rồi, tiếp tục gửi theo chu kỳ cooldown
                    if now - self._last_alert.get(loc_key, 0) >= self.cooldown:
                        self._last_alert[loc_key] = now
                        alerts.append({
                            'event_type'  : 'weapon_detected',
                            'track_id'    : bearer_id,
                            'object_class': class_name,
                            'bbox'        : bbox,
                            'confidence'  : adjusted_conf,
                            'risk_level'  : risk,
                            'zone_name'   : zone_name,
                            'camera_id'   : camera_id,
                            'duration_sec': round(now - det['first_seen'], 1),
                            'bearer_id'   : bearer_id,
                            'hand_distance': hand_distance,
                            'pose_associated': pose_associated,
                        })

        # Dọn các detection không còn nhìn thấy lâu hơn OVERLAY_PERSIST giây
        expired = [
            k for k, v in self._active_detections.items()
            if now - v['last_seen'] > OVERLAY_PERSIST and k not in seen_keys
        ]
        for k in expired:
            self._active_detections.pop(k, None)

        return alerts

    def get_active_overlays(self) -> list[dict]:
        """
        Trả về danh sách weapon detections đang hiển thị (kể cả giữa các frame skip).
        Dùng để vẽ overlay liên tục lên video — KHÔNG phụ thuộc vào cooldown upload.
        """
        now = time.time()
        result = []
        for det in self._active_detections.values():
            if now - det['last_seen'] <= OVERLAY_PERSIST:
                result.append({
                    'event_type'  : 'weapon_detected',
                    'object_class': det['class_name'],
                    'bbox'        : det['bbox'],
                    'confidence'  : det['conf'],
                    'bearer_id'   : det['bearer_id'],
                    'risk_level'  : det['risk_level'],
                })
        return result

    # ── INTERNAL ───────────────────────────────────────────────
    def _find_bearer(
        self, cx: float, cy: float, persons: list[dict]
    ) -> Optional[int]:
        """Tìm track_id của người đứng gần vũ khí nhất."""
        best_dist = float('inf')
        best_id   = None
        for p in persons:
            pb = p.get('bbox') or []
            if len(pb) < 4:
                continue
            px = (pb[0] + pb[2]) / 2
            py = (pb[1] + pb[3]) / 2
            d  = np.hypot(cx - px, cy - py)
            if d < self.bearer_radius and d < best_dist:
                best_dist = d
                best_id   = p.get('track_id')
        return best_id

    def _find_bearer_details(
        self, cx: float, cy: float, persons: list[dict]
    ) -> tuple[Optional[int], Optional[float], bool, Optional[list[float]]]:
        best_dist = float('inf')
        best_id = None
        best_hand_distance = None
        best_pose_associated = False
        best_bbox = None

        for p in persons:
            pb = p.get('bbox') or []
            if len(pb) < 4:
                continue
            px = (pb[0] + pb[2]) / 2
            py = (pb[1] + pb[3]) / 2
            d = np.hypot(cx - px, cy - py)
            if d < self.bearer_radius and d < best_dist:
                best_dist = d
                best_id = p.get('track_id')
                best_bbox = pb[:4]
                best_hand_distance, best_pose_associated = self._closest_hand_distance(cx, cy, p)

        return best_id, best_hand_distance, best_pose_associated, best_bbox

    def _closest_hand_distance(self, cx: float, cy: float, person: dict) -> tuple[Optional[float], bool]:
        kpts = person.get('kpts')
        if kpts is None:
            return None, False

        best = float('inf')
        # Bổ sung khớp vai (5,6) và hông (11,12) ngoài khuỷu tay/cổ tay
        for idx in (5, 6, 7, 8, 9, 10, 11, 12):
            if len(kpts) <= idx:
                continue
            kp = kpts[idx]
            if len(kp) < 3 or kp[2] < 0.25:
                continue
            best = min(best, float(np.hypot(cx - kp[0], cy - kp[1])))

        if best == float('inf'):
            return None, False
            
        # Tính ngưỡng linh hoạt dựa trên chiều rộng cơ thể người
        pb = person.get('bbox')
        if pb and len(pb) >= 4:
            pw = float(pb[2] - pb[0])
            # Giới hạn ngưỡng từ 30px đến wrist_distance_threshold
            dynamic_threshold = max(30.0, min(self.wrist_distance_threshold, pw * 0.6))
        else:
            dynamic_threshold = self.wrist_distance_threshold
            
        return best, best <= dynamic_threshold

    def _class_min_conf(self, class_name: str, pose_associated: bool) -> float:
        name = str(class_name).lower()
        if name in self.class_confidence_thresholds:
            return self.class_confidence_thresholds[name]
        if pose_associated:
            return self.min_pose_associated_conf
        return self.min_unassociated_conf

    def _weapon_bbox_sane(
        self,
        class_name: str,
        bbox: list[float],
        person_bbox: Optional[list[float]],
    ) -> bool:
        if len(bbox) < 4:
            return False

        w = float(bbox[2] - bbox[0])
        h = float(bbox[3] - bbox[1])
        if w <= 0 or h <= 0:
            return False

        name = str(class_name).lower()
        ratio_limit = self.class_area_ratio_limits.get(name, 0.3)
        if person_bbox and len(person_bbox) >= 4:
            pw = float(person_bbox[2] - person_bbox[0])
            ph = float(person_bbox[3] - person_bbox[1])
            person_area = max(pw * ph, 1e-6)
            weapon_area = w * h
            if weapon_area > person_area * ratio_limit:
                return False

            if name in {"gun", "pistol", "rifle"}:
                # Súng trường có thể vắt ngang nên chiều rộng có thể gấp đôi người
                if w > pw * 2.5 or h > ph * 0.8:
                    return False

        return True

    def get_class_ids(self) -> list[int]:
        """Trả về danh sách class IDs đang theo dõi."""
        return self.weapon_class_ids
