"""
Stroke and Fall Recognizer (COCO Format Optimized)
Logic-based detection for 17-point skeletons — v2 (Reduced False Positives)
"""
import numpy as np

class StrokeRecognizer:
    def __init__(self, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        # COCO Keypoints:
        #   0: Nose, 5: L_Shoulder, 6: R_Shoulder
        #   11: L_Hip, 12: R_Hip, 13: L_Knee, 14: R_Knee

        # Bộ đếm frame liên tiếp để tránh false positive thoáng qua
        self._sustained_count = {}   # track_id -> int
        self.SUSTAINED_THRESHOLD = 8  # Cần >= 8 frame liên tiếp

    def analyze(self, history, img_size, track_id=0):
        """
        Analyze 17-point keypoints history.
        history: List of np.array(17, 3)
        track_id: Dùng để theo dõi sustained frames (mặc định 0 nếu không truyền)
        """
        if len(history) < 5:
            self._reset_sustained(track_id)
            return self._result(False, 0.0, 'Normal', 'low')

        pts = np.array(history)  # (T, 17, 3)
        w, h = img_size

        # ─────────────────────────────────────────────
        # 1. Phát hiện Ngã đột ngột (Sudden Fall)
        #    Dùng trung điểm hông (Index 11, 12)
        # ─────────────────────────────────────────────
        hips_y = (pts[:, 11, 1] + pts[:, 12, 1]) / 2
        y_velocity = np.diff(hips_y)
        max_velocity = np.max(y_velocity) if len(y_velocity) > 0 else 0

        if max_velocity > 0.15 * h:
            # Ngã đột ngột → báo ngay (không cần sustained)
            self._reset_sustained(track_id)
            return self._result(True, 0.9, 'Sudden_Fall', 'high')

        # ─────────────────────────────────────────────
        # 2. Phát hiện Tư thế Nằm Ngang (Abnormal Posture)
        #    Yêu cầu nhiều điều kiện kết hợp để tránh false positive
        # ─────────────────────────────────────────────
        latest_pts = pts[-1]
        valid_kpts = latest_pts[latest_pts[:, 2] > 0.3]

        if len(valid_kpts) < 5:
            self._reset_sustained(track_id)
            return self._result(False, 0.0, 'Normal', 'low')

        x_min, y_min = np.min(valid_kpts[:, 0]), np.min(valid_kpts[:, 1])
        x_max, y_max = np.max(valid_kpts[:, 0]), np.max(valid_kpts[:, 1])

        bbox_w = x_max - x_min
        bbox_h = y_max - y_min
        aspect_ratio = bbox_w / (bbox_h + 1e-6)

        # Điều kiện 1: Bounding box phải nằm ngang rõ ràng (tăng từ 1.3 → 1.6)
        cond_horizontal = aspect_ratio > 1.6 and bbox_h < 0.35 * h

        # Điều kiện 2: Đầu (Nose) phải ở vị trí bất thường so với hông
        #   Khi ngã/nằm, đầu sẽ thấp ngang hoặc thấp hơn hông
        nose = latest_pts[0]
        l_hip = latest_pts[11]
        r_hip = latest_pts[12]
        cond_head_low = False
        if nose[2] > 0.3 and l_hip[2] > 0.3 and r_hip[2] > 0.3:
            hip_y_avg = (l_hip[1] + r_hip[1]) / 2
            # Đầu gần bằng hoặc thấp hơn hông (trong 10% chiều cao frame)
            cond_head_low = nose[1] > (hip_y_avg - 0.10 * h)

        # Điều kiện 3: Kiểm tra xu hướng nhiều frame (không chỉ 1 frame)
        #   Aspect ratio trong 3 frame cuối đều > 1.4
        if len(pts) >= 3:
            recent_ratios = []
            for p in pts[-3:]:
                vk = p[p[:, 2] > 0.3]
                if len(vk) >= 5:
                    rw = np.max(vk[:, 0]) - np.min(vk[:, 0])
                    rh = np.max(vk[:, 1]) - np.min(vk[:, 1])
                    recent_ratios.append(rw / (rh + 1e-6))
            cond_trend = len(recent_ratios) == 3 and all(r > 1.4 for r in recent_ratios)
        else:
            cond_trend = False

        # Kết hợp: Cần TẤT CẢ 3 điều kiện
        is_abnormal = cond_horizontal and cond_head_low and cond_trend

        if is_abnormal:
            # Tăng bộ đếm sustained
            count = self._sustained_count.get(track_id, 0) + 1
            self._sustained_count[track_id] = count

            if count >= self.SUSTAINED_THRESHOLD:
                return self._result(True, 0.85, 'Abnormal_Posture', 'high')
            else:
                # Đang trong quá trình tích luỹ frame — chưa báo
                return self._result(False, 0.0, 'Observing', 'low')
        else:
            # Reset bộ đếm nếu điều kiện không còn thỏa
            self._reset_sustained(track_id)
            return self._result(False, 0.0, 'Normal', 'low')

    def _reset_sustained(self, track_id):
        self._sustained_count[track_id] = 0

    def _result(self, detected, confidence, symptom, risk):
        return {
            'detected': detected,
            'confidence': confidence,
            'symptom': symptom,
            'risk_level': risk
        }
