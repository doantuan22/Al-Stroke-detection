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


# Số frame giữ trạng thái detected=True sau khi phát hiện (tránh flicker)
_DETECT_COOLDOWN_FRAMES = 150


class StrokeRecognizerV2:
    
    def __init__(self, config: Optional[StrokeConfig] = None, debug: bool = False):
        self.config = config or StrokeConfig()
        self.debug = debug
        
        self._sustained_posture: dict[int, int] = {}
        self._sustained_slump: dict[int, int] = {}
        self._vel_history: dict[int, deque] = {}
        self._aspect_history: dict[int, deque] = {}
        
        # Cooldown per track: số frame còn lại trong trạng thái "đang bị đột quỵ"
        # Khi > 0: analyze() trả về detected=True ngay lập tức mà không cần tính lại
        # Mỗi người (track_id) có cooldown độc lập → không ảnh hưởng lẫn nhau
        self._detect_cooldown: dict[int, int] = {}
        # Cache kết quả cuối để trả về trong thời gian cooldown
        self._last_detect_result: dict[int, dict] = {}
        
        if self.debug:
            logger.setLevel(logging.DEBUG)
            logger.info(f"[StrokeRecognizerV2] Initialized with config: {self.config}")
    
    
    def analyze(self, history: list, img_size: tuple, track_id: int = 0) -> dict:
        if len(history) < 5:
            self._clear_state(track_id)
            return self._result(False, 0.0, 'Normal', 'low')
        
        w, h = img_size
        
        if track_id not in self._vel_history:
            self._vel_history[track_id] = deque(maxlen=self.config.vel_window)
            self._aspect_history[track_id] = deque(maxlen=self.config.slump_window)
        
        latest = history[-1]
        valid_mask = (
            (latest[:, 2] > self.config.kpts_conf_min) & 
            (latest[:, 0] > 0) & 
            (latest[:, 1] > 0)
        )
        valid = latest[valid_mask]
        
        cooldown = self._detect_cooldown.get(track_id, 0)
        
        if len(valid) < self.config.min_valid_kpts:
            if cooldown > 0:
                self._detect_cooldown[track_id] = cooldown - 1
                return self._last_detect_result.get(track_id, self._result(True, 0.85, 'Detected', 'high'))
            if self.debug:
                logger.debug(f"[Track {track_id}] Insufficient valid keypoints: {len(valid)}")
            self._clear_state(track_id)
            return self._result(False, 0.0, 'Normal', 'low')
        
        prev = history[-2]
        prev_hip_y = (prev[11, 1] + prev[12, 1]) * 0.5
        hip_y = (latest[11, 1] + latest[12, 1]) * 0.5
        
        cur_ar = float(np.ptp(valid[:, 0])) / (float(np.ptp(valid[:, 1])) + 1e-6)
        
        # Recovery check: nếu đang báo động mà đứng lên hoặc hông di chuyển lên nhanh -> hủy báo động
        if cooldown > 0:
            vel_up = prev_hip_y - hip_y
            if cur_ar < 1.0 or vel_up > 0.05 * h:
                if self.debug:
                    logger.info(f"[Track {track_id}] 🔄 Recovery detected. Canceling alert.")
                self._detect_cooldown[track_id] = 0
            else:
                self._detect_cooldown[track_id] = cooldown - 1
                return self._last_detect_result.get(track_id, self._result(True, 0.85, 'Detected', 'high'))
        
        # Chỉ thêm giá trị CUỐI vào buffer (tránh thêm trùng các frame cũ)
        self._vel_history[track_id].append(float(hip_y - prev_hip_y))
        self._aspect_history[track_id].append(cur_ar)
        
        result = self._detect_sudden_fall(track_id, h, hip_y)
        if result['detected']:
            self._start_cooldown(track_id, result)
            return result
        
        result = self._detect_abnormal_posture(track_id, history, latest, valid, h)
        if result['detected']:
            self._start_cooldown(track_id, result)
            return result
        
        result = self._detect_gradual_collapse(track_id, h)
        if result['detected']:
            self._start_cooldown(track_id, result)
            return result
        
        return self._result(False, 0.0, 'Normal', 'low')
    
    
    def _detect_sudden_fall(self, track_id: int, frame_height: float, hip_y: float) -> dict:
        vel_buf = self._vel_history[track_id]
        if not vel_buf:
            return self._result(False, 0.0, 'Normal', 'low')
        
        max_vel = float(max(vel_buf))
        fall_distance = sum(v for v in vel_buf if v > 0)
        
        threshold = self.config.sudden_vel_ratio * frame_height
        dist_threshold = 0.15 * frame_height
        
        if self.debug:
            logger.debug(
                f"[Track {track_id}] Sudden Fall Check: "
                f"max_vel={max_vel:.1f}, fall_dist={fall_distance:.1f}, threshold={threshold:.1f}, hip_y={hip_y:.1f}"
            )
        
        if max_vel > threshold and fall_distance > dist_threshold and hip_y > frame_height * 0.4:
            if self.debug:
                logger.info(f"[Track {track_id}] ✅ SUDDEN FALL DETECTED!")
            # KHÔNG gọi _reset() ở đây — cooldown sẽ giữ trạng thái ổn định
            return self._result(True, 0.92, 'Sudden_Fall', 'high')
        
        return self._result(False, 0.0, 'Normal', 'low')
    
    
    def _detect_abnormal_posture(
        self, 
        track_id: int, 
        history: list,
        latest: np.ndarray, 
        valid: np.ndarray, 
        frame_height: float
    ) -> dict:
        bbox_w = float(np.ptp(valid[:, 0]))
        bbox_h = float(np.ptp(valid[:, 1]))
        aspect = bbox_w / (bbox_h + 1e-6)
        
        # Yêu cầu bẹt hơn (> 1.5) thay vì 1.2 để chống báo giả khi ngồi
        cond_horizontal = (
            aspect > max(1.5, self.config.aspect_ratio_min) and 
            bbox_h < self.config.bbox_h_max_ratio * frame_height
        )
        if not cond_horizontal:
            self._sustained_posture[track_id] = 0
            return self._result(False, 0.0, 'Normal', 'low')
        
        nose = latest[0]
        l_hip = latest[11]
        r_hip = latest[12]
        l_sho = latest[5]
        r_sho = latest[6]
        
        cond_head_low = False
        hip_y = -1
        
        if l_hip[2] > self.config.kpts_conf_min and r_hip[2] > self.config.kpts_conf_min:
            hip_y = (l_hip[1] + r_hip[1]) / 2
            
            # Phải nằm thấp dưới màn hình thì mới tính là bất thường
            if hip_y > frame_height * 0.4:
                if nose[2] > self.config.kpts_conf_min:
                    cond_head_low = nose[1] > (hip_y - self.config.head_hip_margin * frame_height)
                elif l_sho[2] > self.config.kpts_conf_min and r_sho[2] > self.config.kpts_conf_min:
                    sho_y = (l_sho[1] + r_sho[1]) / 2
                    cond_head_low = sho_y > (hip_y - self.config.head_hip_margin * frame_height)
        
        cond_trend = False
        if len(history) >= 3:
            ratios = []
            for p in history[-3:]:
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
    
    
    def _start_cooldown(self, track_id: int, result: dict):
        """Bắt đầu cooldown sau khi phát hiện đột quỵ.
        
        Trong thời gian cooldown, analyze() trả về cached result mà không
        tính lại — giúp loại bỏ flicker khi có nhiều người trong khung hình.
        Buffer được GIỮ NGUYÊN (không xóa) để khi cooldown hết, detector
        vẫn còn đủ dữ liệu để tiếp tục nhận diện.
        """
        self._detect_cooldown[track_id] = _DETECT_COOLDOWN_FRAMES
        self._last_detect_result[track_id] = result
        if self.debug:
            logger.info(f"[Track {track_id}] 🔒 Cooldown started ({_DETECT_COOLDOWN_FRAMES} frames)")

    def _clear_state(self, track_id: int):
        """Xóa toàn bộ state khi track biến mất (history < 5 frames).
        
        Khác với _reset() cũ (gọi sau mỗi lần detect), _clear_state() chỉ
        được gọi khi người thực sự rời khỏi khung hình.
        """
        self._sustained_posture[track_id] = 0
        self._sustained_slump[track_id] = 0
        self._detect_cooldown[track_id] = 0
        self._last_detect_result.pop(track_id, None)
        if track_id in self._vel_history:
            self._vel_history[track_id].clear()
        if track_id in self._aspect_history:
            self._aspect_history[track_id].clear()

    def _reset(self, track_id: int):
        """Legacy alias — giữ để tương thích nếu bên ngoài gọi trực tiếp."""
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
