"""
Stroke and Fall Recognizer v2 - Enhanced Detection
===================================================
Phát hiện đột quỵ và té ngã với 3 detectors:
  1. Sudden Fall - Ngã đột ngột
  2. Abnormal Posture - Tư thế bất thường
  3. Gradual Collapse - Suy sụp từ từ
"""
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional
import logging

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class StrokeConfig:
    kpts_conf_min: float = 0.25
    min_valid_kpts: int = 5
    
    sudden_vel_ratio: float = 0.07
    vel_window: int = 5
    
    aspect_ratio_min: float = 1.2
    bbox_h_max_ratio: float = 0.45
    head_hip_margin: float = 0.15
    sustained_posture: int = 6
    
    slump_aspect_min: float = 0.8
    slump_vel_ratio: float = 0.025
    slump_window: int = 12
    slump_sustained: int = 5


class StrokeRecognizerV2:
    
    def __init__(self, config: Optional[StrokeConfig] = None, debug: bool = False):
        self.config = config or StrokeConfig()
        self.debug = debug
        
        self._sustained_posture: dict[int, int] = {}
        self._sustained_slump: dict[int, int] = {}
        self._vel_history: dict[int, deque] = {}
        self._aspect_history: dict[int, deque] = {}
        
        if self.debug:
            logger.setLevel(logging.DEBUG)
            logger.info(f"[StrokeRecognizerV2] Initialized with config: {self.config}")
    
    
    def analyze(self, history: list, img_size: tuple, track_id: int = 0) -> dict:
        if len(history) < 5:
            self._reset(track_id)
            return self._result(False, 0.0, 'Normal', 'low')
        
        pts = np.array(history)
        w, h = img_size
        
        if track_id not in self._vel_history:
            self._vel_history[track_id] = deque(maxlen=self.config.vel_window)
            self._aspect_history[track_id] = deque(maxlen=self.config.slump_window)
        
        latest = pts[-1]
        valid_mask = (
            (latest[:, 2] > self.config.kpts_conf_min) & 
            (latest[:, 0] > 0) & 
            (latest[:, 1] > 0)
        )
        valid = latest[valid_mask]
        
        if len(valid) < self.config.min_valid_kpts:
            if self.debug:
                logger.debug(f"[Track {track_id}] Insufficient valid keypoints: {len(valid)}")
            self._reset(track_id)
            return self._result(False, 0.0, 'Normal', 'low')
        
        hip_y_series = (pts[:, 11, 1] + pts[:, 12, 1]) / 2
        vel_series = np.diff(hip_y_series)
        
        for vel in vel_series:
            self._vel_history[track_id].append(float(vel))
        
        cur_ar = float(np.ptp(valid[:, 0])) / (float(np.ptp(valid[:, 1])) + 1e-6)
        self._aspect_history[track_id].append(cur_ar)
        
        result = self._detect_sudden_fall(track_id, h)
        if result['detected']:
            return result
        
        result = self._detect_abnormal_posture(track_id, pts, latest, valid, h)
        if result['detected']:
            return result
        
        result = self._detect_gradual_collapse(track_id, h)
        if result['detected']:
            return result
        
        return self._result(False, 0.0, 'Normal', 'low')
    
    
    def _detect_sudden_fall(self, track_id: int, frame_height: float) -> dict:
        vel_buf = self._vel_history[track_id]
        if not vel_buf:
            return self._result(False, 0.0, 'Normal', 'low')
        
        max_vel = float(np.max(vel_buf))
        threshold = self.config.sudden_vel_ratio * frame_height
        
        if self.debug:
            logger.debug(
                f"[Track {track_id}] Sudden Fall Check: "
                f"max_vel={max_vel:.1f}, threshold={threshold:.1f}"
            )
        
        if max_vel > threshold:
            if self.debug:
                logger.info(f"[Track {track_id}] ✅ SUDDEN FALL DETECTED!")
            self._reset(track_id)
            return self._result(True, 0.92, 'Sudden_Fall', 'high')
        
        return self._result(False, 0.0, 'Normal', 'low')
    
    
    def _detect_abnormal_posture(
        self, 
        track_id: int, 
        pts: np.ndarray, 
        latest: np.ndarray, 
        valid: np.ndarray, 
        frame_height: float
    ) -> dict:
        bbox_w = float(np.ptp(valid[:, 0]))
        bbox_h = float(np.ptp(valid[:, 1]))
        aspect = bbox_w / (bbox_h + 1e-6)
        
        cond_horizontal = (
            aspect > self.config.aspect_ratio_min and 
            bbox_h < self.config.bbox_h_max_ratio * frame_height
        )
        
        nose = latest[0]
        l_hip = latest[11]
        r_hip = latest[12]
        l_sho = latest[5]
        r_sho = latest[6]
        
        cond_head_low = False
        hip_y = -1
        
        if l_hip[2] > self.config.kpts_conf_min and r_hip[2] > self.config.kpts_conf_min:
            hip_y = (l_hip[1] + r_hip[1]) / 2
            
            if nose[2] > self.config.kpts_conf_min:
                cond_head_low = nose[1] > (hip_y - self.config.head_hip_margin * frame_height)
            elif l_sho[2] > self.config.kpts_conf_min and r_sho[2] > self.config.kpts_conf_min:
                sho_y = (l_sho[1] + r_sho[1]) / 2
                cond_head_low = sho_y > (hip_y - self.config.head_hip_margin * frame_height)
        
        cond_trend = False
        if len(pts) >= 3:
            ratios = []
            for p in pts[-3:]:
                vm = (p[:, 2] > self.config.kpts_conf_min) & (p[:, 0] > 0) & (p[:, 1] > 0)
                vk = p[vm]
                if len(vk) >= self.config.min_valid_kpts:
                    ratios.append(float(np.ptp(vk[:, 0])) / (float(np.ptp(vk[:, 1])) + 1e-6))
            cond_trend = len(ratios) == 3 and all(r > self.config.aspect_ratio_min for r in ratios)
        
        if self.debug:
            logger.debug(
                f"[Track {track_id}] Abnormal Posture Check:\n"
                f"  - Aspect ratio: {aspect:.2f} (min: {self.config.aspect_ratio_min})\n"
                f"  - BBox height: {bbox_h:.1f} (max: {self.config.bbox_h_max_ratio * frame_height:.1f})\n"
                f"  - Horizontal: {cond_horizontal}\n"
                f"  - Head low: {cond_head_low} (hip_y={hip_y:.1f})\n"
                f"  - Trend: {cond_trend}"
            )
        
        is_posture_bad = cond_horizontal and cond_head_low and cond_trend
        
        if is_posture_bad:
            cnt = self._sustained_posture.get(track_id, 0) + 1
            self._sustained_posture[track_id] = cnt
            
            if self.debug:
                logger.debug(
                    f"[Track {track_id}] Abnormal posture sustained: "
                    f"{cnt}/{self.config.sustained_posture}"
                )
            
            if cnt >= self.config.sustained_posture:
                if self.debug:
                    logger.info(f"[Track {track_id}] ✅ ABNORMAL POSTURE DETECTED!")
                return self._result(True, 0.87, 'Abnormal_Posture', 'high')
            
            return self._result(False, 0.0, 'Observing', 'low')
        else:
            self._sustained_posture[track_id] = 0
        
        return self._result(False, 0.0, 'Normal', 'low')
    
    
    def _detect_gradual_collapse(self, track_id: int, frame_height: float) -> dict:
        ar_buf = list(self._aspect_history[track_id])
        vel_buf = self._vel_history[track_id]
        
        if len(ar_buf) < self.config.slump_window:
            self._sustained_slump[track_id] = 0
            return self._result(False, 0.0, 'Normal', 'low')
        
        half = self.config.slump_window // 2
        ar_early = float(np.mean(ar_buf[:half]))
        ar_late = float(np.mean(ar_buf[half:]))
        ar_trend_up = ar_late > ar_early + 0.15
        
        avg_vel = float(np.mean(list(vel_buf))) if vel_buf else 0.0
        vel_positive = avg_vel > self.config.slump_vel_ratio * frame_height
        
        cur_ar_ok = ar_late > self.config.slump_aspect_min
        
        if self.debug:
            logger.debug(
                f"[Track {track_id}] Gradual Collapse Check:\n"
                f"  - AR early: {ar_early:.2f}, late: {ar_late:.2f}\n"
                f"  - AR trend up: {ar_trend_up}\n"
                f"  - Avg velocity: {avg_vel:.1f} (threshold: {self.config.slump_vel_ratio * frame_height:.1f})\n"
                f"  - Vel positive: {vel_positive}\n"
                f"  - Current AR OK: {cur_ar_ok}"
            )
        
        if ar_trend_up and vel_positive and cur_ar_ok:
            cnt = self._sustained_slump.get(track_id, 0) + 1
            self._sustained_slump[track_id] = cnt
            
            if self.debug:
                logger.debug(
                    f"[Track {track_id}] Gradual collapse sustained: "
                    f"{cnt}/{self.config.slump_sustained}"
                )
            
            if cnt >= self.config.slump_sustained:
                if self.debug:
                    logger.info(f"[Track {track_id}] ✅ GRADUAL COLLAPSE DETECTED!")
                return self._result(True, 0.78, 'Gradual_Collapse', 'high')
            
            return self._result(False, 0.0, 'Observing', 'low')
        else:
            self._sustained_slump[track_id] = 0
        
        return self._result(False, 0.0, 'Normal', 'low')
    
    
    def _reset(self, track_id: int):
        self._sustained_posture[track_id] = 0
        self._sustained_slump[track_id] = 0
        if track_id in self._vel_history:
            self._vel_history[track_id].clear()
        if track_id in self._aspect_history:
            self._aspect_history[track_id].clear()
    
    @staticmethod
    def _result(detected: bool, confidence: float, symptom: str, risk: str) -> dict:
        return {
            'detected': detected,
            'confidence': confidence,
            'symptom': symptom,
            'risk_level': risk
        }
    
    def set_debug(self, enabled: bool):
        self.debug = enabled
        if enabled:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
