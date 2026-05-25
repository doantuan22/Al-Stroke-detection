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
    76: 'scissors',     # kéo (có thể nguy hiểm)
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

WEAPON_CONF_MIN   = 0.25    # Ngưỡng confidence tối thiểu
BEARER_RADIUS_PX  = 130     # Pixel tính là "người đang cầm"
ALERT_COOLDOWN    = 25.0    # Giây giữa 2 alert cùng vị trí



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

        # Map class IDs theo model đang dùng
        self.weapon_classes = (
            FINETUNE_WEAPON_CLASSES if use_finetune
            else COCO_WEAPON_CLASSES
        )
        self.weapon_class_ids = list(self.weapon_classes.keys())

        # Cooldown tracker per location key
        self._last_alert: dict[str, float] = {}

    # ── PUBLIC ─────────────────────────────────────────────────
    def detect_frame(
        self,
        frame_or_objs,
        persons: list[dict],
        zone_name: Optional[str] = None,
        camera_id: str           = 'CAM_00',
    ) -> list[dict]:
        """
        Detect vũ khí từ obj_results (đã track) hoặc frame và tạo alerts.

        Args:
            frame_or_objs : list[dict] kết quả track hoặc BGR numpy array
            persons       : Danh sách từ PoseDetector.track()
            zone_name     : Tên zone (nếu có ZoneGuard) → risk = critical
            camera_id     : ID camera

        Returns:
            alerts: list[dict]
        """
        import numpy as np
        if isinstance(frame_or_objs, list):
            # Tái sử dụng kết quả object detection đã chạy
            all_objects = [
                obj for obj in frame_or_objs
                if obj.get('class_id') in self.weapon_class_ids
                and obj.get('conf', 0) >= self.conf
            ]
        else:
            # Fallback chạy inference nếu truyền vào frame numpy
            all_objects = self.od.detect(
                frame_or_objs,
                classes=self.weapon_class_ids,
                conf=self.conf,
            )

        alerts = []
        now    = time.time()

        for obj in all_objects:
            cid  = obj['class_id']
            if cid not in self.weapon_classes:
                continue

            bbox       = obj['bbox']
            class_name = self.weapon_classes[cid]
            conf_val   = obj['conf']

            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2

            # Cooldown check theo cell lưới 50px
            loc_key = f"{class_name}_{int(cx//50)}_{int(cy//50)}"
            if now - self._last_alert.get(loc_key, 0) < self.cooldown:
                continue

            # Xác định bearer (người đang cầm)
            bearer_id = self._find_bearer(cx, cy, persons)

            # Risk level
            risk = 'critical' if zone_name else 'high'

            self._last_alert[loc_key] = now
            alerts.append({
                'event_type'  : 'weapon_detected',
                'track_id'    : bearer_id,
                'object_class': class_name,
                'bbox'        : bbox,
                'confidence'  : conf_val,
                'risk_level'  : risk,
                'zone_name'   : zone_name,
                'camera_id'   : camera_id,
                'duration_sec': 0.0,
                'bearer_id'   : bearer_id,
            })

        return alerts

    # ── INTERNAL ───────────────────────────────────────────────
    def _find_bearer(
        self, cx: float, cy: float, persons: list[dict]
    ) -> Optional[int]:
        """Tìm track_id của người đứng gần vũ khí nhất."""
        best_dist = float('inf')
        best_id   = None
        for p in persons:
            pb = p.get('bbox', [])
            if len(pb) < 4:
                continue
            px = (pb[0] + pb[2]) / 2
            py = (pb[1] + pb[3]) / 2
            d  = np.hypot(cx - px, cy - py)
            if d < self.bearer_radius and d < best_dist:
                best_dist = d
                best_id   = p.get('track_id')
        return best_id

    def get_class_ids(self) -> list[int]:
        """Trả về danh sách class IDs đang theo dõi."""
        return self.weapon_class_ids
