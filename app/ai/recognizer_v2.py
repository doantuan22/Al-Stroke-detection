"""
Stroke and Fall Recognizer v2 - Enhanced Detection
===================================================
Phát hiện đột quỵ và té ngã với 3 detectors:
  1. Sudden Fall - Ngã đột ngột
  2. Abnormal Posture - Tư thế bất thường
  3. Gradual Collapse - Suy sụp từ từ
"""
import numpy as np
import time
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
    
    sudden_vel_ratio: float = 0.12  # Tăng lên để bắt buộc rớt nhanh
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
        
        # Trạng thái nghi ngờ và xác nhận cảnh báo
        self._suspect_start_time: dict[int, float] = {}
        self._confirmed: dict[int, bool] = {}
        self._symptom: dict[int, str] = {}
        self.CONFIRM_TIME_SEC = 5.0
        
        if self.debug:
            logger.setLevel(logging.DEBUG)
    
    def analyze(self, history: list, img_size: tuple, track_id: int = 0) -> dict:
        if len(history) < 5:
            self._clear_state(track_id)
            return self._result(False, 0.0, 'Normal', 'low')
        
        w, h = img_size
        
        if track_id not in self._vel_history:
            self._vel_history[track_id] = deque(maxlen=self.config.vel_window)
            self._aspect_history[track_id] = deque(maxlen=self.config.slump_window)
        
        latest_dict = history[-1]
        latest_kpts = latest_dict["kpts"]
        latest_bbox = latest_dict["bbox"]
        
        valid_mask = (
            (latest_kpts[:, 2] > self.config.kpts_conf_min) & 
            (latest_kpts[:, 0] > 0) & 
            (latest_kpts[:, 1] > 0)
        )
        valid = latest_kpts[valid_mask]
        
        is_suspected = track_id in self._suspect_start_time
        
        if len(valid) < self.config.min_valid_kpts:
            if is_suspected:
                elapsed = time.time() - self._suspect_start_time[track_id]
                symptom = self._symptom.get(track_id, 'Unknown')
                if elapsed >= self.CONFIRM_TIME_SEC:
                    self._confirmed[track_id] = True
                    return self._result(True, 0.85, symptom, 'high')
                return self._result(True, 0.50, symptom + ' (Suspecting...)', 'medium')
            self._clear_state(track_id)
            return self._result(False, 0.0, 'Normal', 'low')
        
        prev_dict = history[-2]
        prev_kpts = prev_dict["kpts"]
        prev_bbox = prev_dict["bbox"]
        
        prev_hip_y = (prev_kpts[11, 1] + prev_kpts[12, 1]) * 0.5
        hip_y = (latest_kpts[11, 1] + latest_kpts[12, 1]) * 0.5
        
        bbox_w = max(1.0, latest_bbox[2] - latest_bbox[0])
        bbox_h = max(1.0, latest_bbox[3] - latest_bbox[1])
        cur_ar = float(bbox_w) / float(bbox_h)
        
        # Recovery check: Khóa cứng mục tiêu
        if is_suspected:
            cx_cur = (latest_bbox[0] + latest_bbox[2]) / 2
            cy_cur = (latest_bbox[1] + latest_bbox[3]) / 2
            cx_prev = (prev_bbox[0] + prev_bbox[2]) / 2
            cy_prev = (prev_bbox[1] + prev_bbox[3]) / 2
            dist_moved = float(np.hypot(cx_cur - cx_prev, cy_cur - cy_prev))
            vel_up = prev_hip_y - hip_y
            
            # Chỉ thoát cảnh báo nếu đứng lên (ar < 0.8) VÀ có cử động rõ ràng
            strong_movement = (dist_moved > 0.03 * h) or (vel_up > 0.05 * h)
            is_standing = cur_ar < 0.8
            
            if strong_movement and is_standing:
                if self.debug:
                    logger.info(f"[Track {track_id}] 🔄 Recovery detected (Strong movement). Canceling alert.")
                self._suspect_start_time.pop(track_id, None)
                self._confirmed[track_id] = False
                self._symptom.pop(track_id, None)
            else:
                elapsed = time.time() - self._suspect_start_time[track_id]
                if elapsed >= self.CONFIRM_TIME_SEC:
                    self._confirmed[track_id] = True
                
                symptom = self._symptom.get(track_id, 'Unknown')
                if self._confirmed[track_id]:
                    return self._result(True, 0.85, symptom, 'high')
                else:
                    return self._result(True, 0.50, symptom + ' (Suspecting...)', 'medium')
        
        self._vel_history[track_id].append(float(hip_y - prev_hip_y))
        self._aspect_history[track_id].append(cur_ar)
        
        result = self._detect_sudden_fall(track_id, h, hip_y, cur_ar)
        if result['detected']:
            return self._start_suspected(track_id, result)
        
        result = self._detect_abnormal_posture(track_id, history, latest_kpts, bbox_w, bbox_h, h)
        if result['detected']:
            return self._start_suspected(track_id, result)
        
        result = self._detect_gradual_collapse(track_id, h)
        if result['detected']:
            return self._start_suspected(track_id, result)
        
        return self._result(False, 0.0, 'Normal', 'low')
    
    def _detect_sudden_fall(self, track_id: int, frame_height: float, hip_y: float, cur_ar: float) -> dict:
        vel_buf = self._vel_history[track_id]
        if not vel_buf:
            return self._result(False, 0.0, 'Normal', 'low')
        
        max_vel = float(max(vel_buf))
        fall_distance = sum(v for v in vel_buf if v > 0)
        
        threshold = self.config.sudden_vel_ratio * frame_height
        dist_threshold = 0.15 * frame_height
        
        # Bắt buộc rớt nhanh (chống "nằm từ từ")
        is_fast_drop = max_vel > threshold
        
        # Ngã từ ghế: Aspect ratio > 1.2, hông thấp, và có sự rớt xuống
        is_chair_fall = (cur_ar > 1.2) and (fall_distance > 0.08 * frame_height) and (max_vel > 0.05 * frame_height) and (hip_y > frame_height * 0.5)

        if (is_fast_drop and fall_distance > dist_threshold and hip_y > frame_height * 0.4) or is_chair_fall:
            if self.debug:
                logger.info(f"[Track {track_id}] ✅ SUDDEN FALL DETECTED! (Chair fall: {is_chair_fall})")
            return self._result(True, 0.92, 'Sudden_Fall', 'high')
        
        return self._result(False, 0.0, 'Normal', 'low')
    
    def _detect_abnormal_posture(
        self, 
        track_id: int, 
        history: list,
        latest_kpts: np.ndarray, 
        bbox_w: float,
        bbox_h: float,
        frame_height: float
    ) -> dict:
        aspect = bbox_w / max(1e-6, bbox_h)
        
        cond_horizontal = (
            aspect > max(1.5, self.config.aspect_ratio_min) and 
            bbox_h < self.config.bbox_h_max_ratio * frame_height
        )
        if not cond_horizontal:
            self._sustained_posture[track_id] = 0
            return self._result(False, 0.0, 'Normal', 'low')
        
        nose = latest_kpts[0]
        l_hip = latest_kpts[11]
        r_hip = latest_kpts[12]
        l_sho = latest_kpts[5]
        r_sho = latest_kpts[6]
        
        cond_head_low = False
        hip_y = -1
        
        if l_hip[2] > self.config.kpts_conf_min and r_hip[2] > self.config.kpts_conf_min:
            hip_y = (l_hip[1] + r_hip[1]) / 2
            
            if hip_y > frame_height * 0.4:
                if nose[2] > self.config.kpts_conf_min:
                    cond_head_low = nose[1] > (hip_y - self.config.head_hip_margin * frame_height)
                elif l_sho[2] > self.config.kpts_conf_min and r_sho[2] > self.config.kpts_conf_min:
                    sho_y = (l_sho[1] + r_sho[1]) / 2
                    cond_head_low = sho_y > (hip_y - self.config.head_hip_margin * frame_height)
        
        cond_trend = False
        if len(history) >= 3:
            ratios = []
            for p_dict in history[-3:]:
                pw = max(1.0, p_dict["bbox"][2] - p_dict["bbox"][0])
                ph = max(1.0, p_dict["bbox"][3] - p_dict["bbox"][1])
                ratios.append(float(pw) / float(ph))
            cond_trend = len(ratios) == 3 and all(r > self.config.aspect_ratio_min for r in ratios)
        
        is_posture_bad = cond_horizontal and cond_head_low and cond_trend
        
        if is_posture_bad:
            cnt = self._sustained_posture.get(track_id, 0) + 1
            self._sustained_posture[track_id] = cnt
            
            if cnt >= self.config.sustained_posture:
                return self._result(True, 0.87, 'Abnormal_Posture', 'high')
            
            return self._result(False, 0.0, 'Observing', 'low')
        else:
            self._sustained_posture[track_id] = 0
        
        return self._result(False, 0.0, 'Normal', 'low')
    
    def _detect_gradual_collapse(self, track_id: int, frame_height: float) -> dict:
        ar_buf = self._aspect_history[track_id]
        vel_buf = self._vel_history[track_id]
        
        if len(ar_buf) < self.config.slump_window:
            self._sustained_slump[track_id] = 0
            return self._result(False, 0.0, 'Normal', 'low')
        
        half = self.config.slump_window // 2
        ar_values = tuple(ar_buf)
        ar_early = sum(ar_values[:half]) / half
        ar_late = sum(ar_values[half:]) / (len(ar_values) - half)
        ar_trend_up = ar_late > ar_early + 0.15
        
        avg_vel = (sum(vel_buf) / len(vel_buf)) if vel_buf else 0.0
        vel_positive = avg_vel > self.config.slump_vel_ratio * frame_height
        
        cur_ar_ok = ar_late > self.config.slump_aspect_min
        
        if ar_trend_up and vel_positive and cur_ar_ok:
            cnt = self._sustained_slump.get(track_id, 0) + 1
            self._sustained_slump[track_id] = cnt
            
            if cnt >= self.config.slump_sustained:
                return self._result(True, 0.78, 'Gradual_Collapse', 'high')
            
            return self._result(False, 0.0, 'Observing', 'low')
        else:
            self._sustained_slump[track_id] = 0
        
        return self._result(False, 0.0, 'Normal', 'low')
    
    def _start_suspected(self, track_id: int, result: dict):
        self._suspect_start_time[track_id] = time.time()
        self._confirmed[track_id] = False
        self._symptom[track_id] = result['symptom']
        return self._result(True, 0.50, result['symptom'] + ' (Suspecting...)', 'medium')

    def _clear_state(self, track_id: int):
        self._sustained_posture[track_id] = 0
        self._sustained_slump[track_id] = 0
        self._suspect_start_time.pop(track_id, None)
        self._confirmed[track_id] = False
        self._symptom.pop(track_id, None)
        if track_id in self._vel_history:
            self._vel_history[track_id].clear()
        if track_id in self._aspect_history:
            self._aspect_history[track_id].clear()

    def _reset(self, track_id: int):
        self._clear_state(track_id)
    
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
