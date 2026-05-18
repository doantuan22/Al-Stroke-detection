"""
Stroke and Fall Recognizer (COCO Format Optimized)
Logic-based detection for 17-point skeletons
"""
import numpy as np

class StrokeRecognizer:
    def __init__(self, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        # COCO Keypoints: 11: L_Hip, 12: R_Hip, 5: L_Shoulder, 6: R_Shoulder
        
    def analyze(self, history, img_size):
        """
        Analyze 17-point keypoints history
        history: List of np.array(17, 3)
        """
        if len(history) < 5: # Giảm số frame tối thiểu để phản ứng nhanh hơn
            return self._result(False, 0.0, 'Normal', 'low')
            
        pts = np.array(history) # (T, 17, 3)
        w, h = img_size
        
        # 1. Phát hiện Ngã (Sudden Fall)
        # Sử dụng trung điểm của hông (Index 11, 12)
        hips_y = (pts[:, 11, 1] + pts[:, 12, 1]) / 2
        y_velocity = np.diff(hips_y)
        max_velocity = np.max(y_velocity) if len(y_velocity) > 0 else 0
        
        # 2. Phát hiện Tư thế nằm (Horizontal Posture)
        latest_pts = pts[-1]
        valid_kpts = latest_pts[latest_pts[:, 2] > 0.3] # Chỉ lấy điểm có độ tin cậy > 0.3
        
        if len(valid_kpts) < 5:
            return self._result(False, 0.0, 'Normal', 'low')

        x_min, y_min = np.min(valid_kpts[:, 0]), np.min(valid_kpts[:, 1])
        x_max, y_max = np.max(valid_kpts[:, 0]), np.max(valid_kpts[:, 1])
        
        bbox_w = x_max - x_min
        bbox_h = y_max - y_min
        aspect_ratio = bbox_w / (bbox_h + 1e-6)
        
        # Logic tối ưu cho tốc độ phản hồi
        if max_velocity > 0.12 * h: 
            return self._result(True, 0.9, 'Sudden_Fall', 'high')
            
        if aspect_ratio > 1.3 and bbox_h < 0.35 * h: 
            return self._result(True, 0.85, 'Abnormal_Posture', 'high')
            
        return self._result(False, 0.0, 'Normal', 'low')

    def _result(self, detected, confidence, symptom, risk):
        return {
            'detected': detected,
            'confidence': confidence,
            'symptom': symptom,
            'risk_level': risk
        }
