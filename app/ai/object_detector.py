"""
Object Detector — yolov8n.pt (COCO 80 classes)
===============================================
Detect và track general objects (túi, vali, balo, v.v.)
Thiết kế để chạy SONG SONG với PoseDetector.

Tối ưu hiệu năng:
  - Gọi inference mỗi OBJECT_SKIP frames (mặc định 3)
  - Cache kết quả giữa các frames bị skip
  - FP16 trên GPU
"""
import torch
import numpy as np
from ultralytics import YOLO

# COCO class IDs quan trọng cho airport
BAGGAGE_CLASS_IDS = {24: 'backpack', 26: 'handbag', 28: 'suitcase'}

# Class IDs COCO khác có thể dùng sau
PERSON_CLASS_ID   = 0
KNIFE_CLASS_ID    = 43   # dao trong COCO (scissors=76)


class ObjectDetector:
    """
    Wrapper quanh YOLOv8 dùng cho object detection.
    Tái sử dụng yolov8n.pt (file đã có sẵn), không cần model mới.
    """

    def __init__(
        self,
        model_path: str = 'yolov8n.pt',
        device: str = None,
        input_size: int = 640,
        object_skip: int = 3,
    ):
        """
        Args:
            model_path  : Path đến model YOLO (mặc định yolov8n.pt)
            device      : 'cuda' | 'cpu' (auto-detect nếu None)
            input_size  : Resolution YOLO input
            object_skip : Chạy inference mỗi N frames (tiết kiệm GPU)
        """
        import os
        self.device     = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size
        self.object_skip = object_skip

        # Tự động phát hiện model TensorRT (.engine) nếu chạy trên CUDA
        if self.device == 'cuda':
            base_path, _ = os.path.splitext(model_path)
            engine_path = base_path + '.engine'
            if os.path.exists(engine_path):
                print(f"[ObjectDetector] Found TensorRT engine: {engine_path}. Switching to engine for maximum performance!")
                model_path = engine_path
                self.use_half = False  # Engine đã được compile cứng kiểu FP16/INT8, không cần predict(half=True)
            else:
                self.use_half = True
        else:
            self.use_half = False

        self._frame_counter  = 0
        self._cached_results : list[dict] = []   # cache giữa frames skip

        print(f"[ObjectDetector] Loading {model_path} on {self.device}...")
        self.model = YOLO(model_path)
        
        # model.to(device) không khả dụng với file .engine (nó tự chạy trên GPU)
        if not model_path.endswith('.engine'):
            self.model.to(self.device)

        # Warm-up
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.model.predict(
            dummy, imgsz=self.input_size,
            half=self.use_half, verbose=False, device=self.device
        )
        print(f"[ObjectDetector] Ready ✓  skip={object_skip} frames")

    # ──────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────
    def track(
        self,
        frame: np.ndarray,
        classes: list[int] = None,
        conf: float = 0.40,
    ) -> tuple[list[dict], bool]:
        """
        Track objects trong frame.
        Chỉ chạy inference mỗi `object_skip` frames; các frame còn lại
        trả về cache.

        Args:
            frame   : BGR numpy array
            classes : Lọc theo class ID (VD: [24, 26, 28] cho hành lý)
            conf    : Ngưỡng confidence tối thiểu

        Returns:
            (results, ran_inference)
            results       : list[dict] với keys: track_id, class_id,
                            class_name, bbox, conf
            ran_inference : True nếu frame này thực sự chạy AI
        """
        self._frame_counter += 1
        if self._frame_counter % self.object_skip != 0:
            return self._cached_results, False   # dùng cache

        # Thực sự chạy inference
        kwargs: dict = dict(
            imgsz=self.input_size,
            conf=conf,
            half=self.use_half,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
            device=self.device,
        )
        if classes:
            kwargs['classes'] = classes

        results = self.model.track(frame, **kwargs)
        self._cached_results = self._parse(results)
        return self._cached_results, True

    def detect(
        self,
        frame: np.ndarray,
        classes: list[int] = None,
        conf: float = 0.50,
    ) -> list[dict]:
        """
        Single-shot detect (không tracking, không skip).
        Dùng cho weapon detection khi cần detect từng frame độc lập.
        """
        kwargs: dict = dict(
            imgsz=self.input_size,
            conf=conf,
            half=self.use_half,
            verbose=False,
            device=self.device,
        )
        if classes:
            kwargs['classes'] = classes

        results = self.model.predict(frame, **kwargs)
        return self._parse(results, use_track_id=False)

    def reset_skip_counter(self):
        """Reset frame counter (khi restart camera)."""
        self._frame_counter  = 0
        self._cached_results = []

    # ──────────────────────────────────────────────────────────────
    # INTERNAL
    # ──────────────────────────────────────────────────────────────
    def _parse(self, results, use_track_id: bool = True) -> list[dict]:
        out = []
        if not results or results[0].boxes is None:
            return out

        boxes = results[0].boxes.cpu()
        for i in range(len(boxes)):
            cid = int(boxes.cls[i].item())
            tid = i   # fallback
            if use_track_id and boxes.id is not None:
                tid = int(boxes.id[i].item())

            out.append({
                'track_id'  : tid,
                'class_id'  : cid,
                'class_name': self.model.names.get(cid, str(cid)),
                'bbox'      : boxes.xyxy[i].numpy().tolist(),   # [x1, y1, x2, y2]
                'conf'      : float(boxes.conf[i].item()),
            })
        return out
