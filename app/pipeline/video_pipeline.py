"""
Video/camera helper used by the GUI worker.
"""
from __future__ import annotations

import cv2


class VideoPipeline:
    def __init__(self, config: dict | None = None, log=None):
        cfg = (config or {}).get("camera", {})
        self.target_width = int(cfg.get("target_width", 1920))
        self.target_height = int(cfg.get("target_height", 1080))
        self.target_fps = int(cfg.get("target_fps", 30))
        self.buffer_size = int(cfg.get("buffer_size", 1))
        self.log = log or (lambda msg: None)

    def open(self, source):
        cap = cv2.VideoCapture(source)
        if isinstance(source, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
            cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
        return cap

    def describe(self, cap) -> tuple[int, int, float]:
        return (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            float(cap.get(cv2.CAP_PROP_FPS)),
        )

    @staticmethod
    def preprocess(frame):
        if frame is None or frame.size == 0:
            return None
        if frame.ndim == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.ndim == 3 and frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame

