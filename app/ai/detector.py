"""
Modern Pose & Person Detector using YOLOv8-Pose (GPU Optimized)
v2: FP16 half precision + built-in ByteTrack tracking + frame-resize
"""
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# File tracker tùy chỉnh (cùng thư mục gốc project)
_TRACKER_CFG = str(Path(__file__).parents[2] / 'bytetrack_stroke.yaml')


class PoseDetector:
    def __init__(self, model_path='yolov8n-pose.pt', device=None, input_size=640):
        """
        Initialize YOLOv8-Pose detector with optimized settings.
        Args:
            model_path : Path to pose model
            device     : 'cpu' or 'cuda' (auto-detect if None)
            input_size : YOLO input resolution (smaller = faster)
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.input_size = input_size
        self.use_half   = (self.device == 'cuda')   # FP16 only on GPU

        print(f"[AI Engine] Loading {model_path} on {self.device}...")
        self.model = YOLO(model_path)
        self.model.to(self.device)

        # Warm-up: chạy 1 lần giả để JIT biên dịch tránh lag frame đầu
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.model.predict(dummy, imgsz=self.input_size, half=self.use_half,
                           verbose=False, device=self.device)

        if self.device == 'cuda':
            print(f"🚀 GPU Accelerated: {torch.cuda.get_device_name(0)}"
                  f" | FP16={self.use_half} | imgsz={self.input_size}")

    # ──────────────────────────────────────────────────────────
    # PREDICT mode (không tracking) — dùng khi chỉ cần keypoints
    # ──────────────────────────────────────────────────────────
    def detect(self, frame, conf=0.25):
        """
        Detect persons and poses in a single pass (no tracking).
        conf mặc định 0.25 (thay vì 0.4) để không bỏ sót người
        có contrast thấp với nền (áo xám/sàn xám).
        Returns list of dicts: [{'bbox', 'conf', 'kpts', 'track_id'}, ...]
        """
        results = self.model.predict(
            frame,
            imgsz=self.input_size,
            conf=conf,
            half=self.use_half,
            verbose=False,
            device=self.device,
        )
        return self._parse(results)

    # ──────────────────────────────────────────────────────────
    # TRACK mode — dùng ByteTrack, trả track_id ổn định
    # ──────────────────────────────────────────────────────────
    def track(self, frame, conf=0.25, persist=True):
        """
        Detect + track using built-in ByteTrack.
        conf mặc định 0.25 để bắt người có độ tương phản thấp.
        track_id is now stable across frames.
        Returns list of dicts: [{'bbox', 'conf', 'kpts', 'track_id'}, ...]
        """
        results = self.model.track(
            frame,
            imgsz=self.input_size,
            conf=conf,
            half=self.use_half,
            persist=persist,         # giữ state tracker giữa các frame
            tracker=_TRACKER_CFG,   # dùng config tùy chỉnh track_buffer=60
            verbose=False,
            device=self.device,
        )
        return self._parse(results, use_track_id=True)

    # ──────────────────────────────────────────────────────────
    # Internal parser
    # ──────────────────────────────────────────────────────────
    def _parse(self, results, use_track_id=False):
        processed = []
        if not results:
            return processed

        result = results[0]
        if result.boxes is None or result.keypoints is None:
            return processed

        boxes     = result.boxes.cpu()
        keypoints = result.keypoints.cpu().numpy()

        for i in range(len(boxes)):
            # Track ID từ ByteTrack (nếu có), fallback sang index
            if use_track_id and boxes.id is not None:
                tid = int(boxes.id[i].item())
            else:
                tid = i + 1

            processed.append({
                'bbox'    : boxes.xyxy[i].numpy().tolist(),
                'conf'    : float(boxes.conf[i].item()),
                'kpts'    : keypoints.data[i],   # (17, 3) [x, y, conf]
                'track_id': tid,
            })

        return processed

    def draw(self, frame, results):
        """Placeholder — visualization done in utils/visualization.py"""
        return frame
