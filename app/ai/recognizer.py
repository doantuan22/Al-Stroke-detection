"""
Stroke and Fall Recognizer  — v5 (Production-Ready)
==========================================================
3 detectors doc lap:

  1. Sudden Fall     : Max hip velocity trong cua so (khong chi last frame)
  2. Abnormal Posture: 4 dieu kien ket hop, nguong mo rong de bat near-wall
  3. Slump/Collapse  : Trend aspect ratio + velocity trung binh (dot quy cham)

Cac fix tu v4:
  - Vel detector dung MAX trong buffer thay vi MEAN (bat duoc sudden drop)
  - Loai bo kpt co y=0 (vi tri mac dinh) khoi tinh bbox
  - near-wall fallback: neu nose_y khong the xac dinh -> dung shoulder
  - SLUMP_SUSTAINED = 6 (giam tu 10 de bat som hon)
"""
import numpy as np
from collections import deque


class StrokeRecognizer:
    # ── Ngưỡng có thể chỉnh ────────────────────────────────────
    KPTS_CONF          = 0.28
    MIN_VALID_KPTS     = 5

    # Detector 1 — Sudden Fall
    SUDDEN_VEL_RATIO   = 0.10   # max hip velocity / H trong cua so
    VEL_WINDOW         = 5      # so frame de giu velocity

    # Detector 2 — Abnormal Posture
    ASPECT_RATIO_MIN   = 1.4    # nguong aspect ratio (giam de bat near-wall)
    BBOX_H_MAX_RATIO   = 0.40   # tang de bat nguoi ngoi guc sap xuong
    HEAD_HIP_MARGIN    = 0.12   # tang de xu ly nga nghieng (head o ben canh)
    TREND_ASPECT_MIN   = 1.2
    KNEE_GAP_MAX_RATIO = 0.16   # 16%H: cui 90 gap=20% (no alert), nga flat gap=1% (alert)
    SUSTAINED_POSTURE  = 8

    # Detector 3 — Slump (dot quy cham)
    SLUMP_ASPECT_MIN   = 1.0
    SLUMP_VEL_RATIO    = 0.03
    SLUMP_WINDOW       = 12
    SLUMP_SUSTAINED    = 6      # giam tu 10 de bat som hon

    def __init__(self):
        self._sustained_posture : dict[int, int]   = {}
        self._sustained_slump   : dict[int, int]   = {}
        self._vel_history       : dict[int, deque] = {}
        self._aspect_history    : dict[int, deque] = {}

    # ─────────────────────────────────────────────────────────
    def analyze(self, history: list, img_size: tuple, track_id: int = 0) -> dict:
        if len(history) < 5:
            self._reset(track_id)
            return self._result(False, 0.0, 'Normal', 'low')

        pts  = np.array(history)
        w, h = img_size

        if track_id not in self._vel_history:
            self._vel_history[track_id]    = deque(maxlen=self.VEL_WINDOW)
            self._aspect_history[track_id] = deque(maxlen=self.SLUMP_WINDOW)

        # ── Lay frame cuoi ──────────────────────────────────
        latest = pts[-1]
        # Chi lay kpt co conf OK VA co vi tri thuc (y > 0 va x > 0)
        valid_mask = (latest[:, 2] > self.KPTS_CONF) & (latest[:, 0] > 0) & (latest[:, 1] > 0)
        valid  = latest[valid_mask]
        if len(valid) < self.MIN_VALID_KPTS:
            self._reset(track_id)
            return self._result(False, 0.0, 'Normal', 'low')

        # ── Cap nhat velocity buffer ────────────────────────
        hip_y_series = (pts[:, 11, 1] + pts[:, 12, 1]) / 2
        vel_series   = np.diff(hip_y_series)
        if len(vel_series) > 0:
            # Them moi PHAN TU CUOI vao buffer (khong them trung binh)
            self._vel_history[track_id].append(float(vel_series[-1]))

        # ── Cap nhat aspect history ─────────────────────────
        cur_ar = float(np.ptp(valid[:, 0])) / (float(np.ptp(valid[:, 1])) + 1e-6)
        self._aspect_history[track_id].append(cur_ar)

        # ══════════════════════════════════════════════════
        # DETECTOR 1: Sudden Fall — MAX velocity trong buffer
        # ══════════════════════════════════════════════════
        vel_buf = self._vel_history[track_id]
        max_vel = float(np.max(vel_buf)) if vel_buf else 0.0

        if max_vel > self.SUDDEN_VEL_RATIO * h:
            self._reset(track_id)
            return self._result(True, 0.92, 'Sudden_Fall', 'high')

        # ══════════════════════════════════════════════════
        # DETECTOR 2: Abnormal Posture
        # ══════════════════════════════════════════════════
        bbox_w = float(np.ptp(valid[:, 0]))
        bbox_h = float(np.ptp(valid[:, 1]))
        aspect = bbox_w / (bbox_h + 1e-6)

        cond_horizontal = (aspect > self.ASPECT_RATIO_MIN
                           and bbox_h < self.BBOX_H_MAX_RATIO * h)

        # Dieu kien 2: Dau (hoac vai) o vi tri thap bat thuong
        nose  = latest[0];  l_hip = latest[11]; r_hip = latest[12]
        l_sho = latest[5];  r_sho = latest[6]
        cond_head_low = False
        hip_y = (l_hip[1] + r_hip[1]) / 2 if (l_hip[2] > self.KPTS_CONF and r_hip[2] > self.KPTS_CONF) else -1

        if hip_y > 0:
            if nose[2] > self.KPTS_CONF:
                # Dau nam trong 12% so voi hong (tang vien de xu ly nga nghieng)
                cond_head_low = nose[1] > (hip_y - self.HEAD_HIP_MARGIN * h)
            elif l_sho[2] > self.KPTS_CONF and r_sho[2] > self.KPTS_CONF:
                # Fallback: dung vai neu khong thay mat (nga sap / bi che mat)
                sho_y = (l_sho[1] + r_sho[1]) / 2
                cond_head_low = sho_y > (hip_y - self.HEAD_HIP_MARGIN * h)

        # Dieu kien 3: Xu huong lien tuc qua 3 frames
        cond_trend = False
        if len(pts) >= 3:
            ratios = []
            for p in pts[-3:]:
                vm = (p[:, 2] > self.KPTS_CONF) & (p[:, 0] > 0) & (p[:, 1] > 0)
                vk = p[vm]
                if len(vk) >= self.MIN_VALID_KPTS:
                    ratios.append(float(np.ptp(vk[:,0])) / (float(np.ptp(vk[:,1])) + 1e-6))
            cond_trend = len(ratios) == 3 and all(r > self.TREND_ASPECT_MIN for r in ratios)

        # Dieu kien 4: Chan bi suy sup / nam ngang
        l_knee = latest[13]; r_knee = latest[14]
        cond_legs = False
        if all(k[2] > self.KPTS_CONF for k in [l_hip, r_hip, l_knee, r_knee]):
            knee_y = (l_knee[1] + r_knee[1]) / 2
            cond_legs = (knee_y - hip_y) < self.KNEE_GAP_MAX_RATIO * h

        is_posture_bad = cond_horizontal and cond_head_low and cond_trend and cond_legs

        if is_posture_bad:
            cnt = self._sustained_posture.get(track_id, 0) + 1
            self._sustained_posture[track_id] = cnt
            if cnt >= self.SUSTAINED_POSTURE:
                return self._result(True, 0.87, 'Abnormal_Posture', 'high')
            return self._result(False, 0.0, 'Observing', 'low')
        else:
            self._sustained_posture[track_id] = 0

        # ══════════════════════════════════════════════════
        # DETECTOR 3: Slump / Gradual Collapse
        # ══════════════════════════════════════════════════
        ar_buf = list(self._aspect_history[track_id])
        if len(ar_buf) >= self.SLUMP_WINDOW:
            half    = self.SLUMP_WINDOW // 2
            ar_early = float(np.mean(ar_buf[:half]))
            ar_late  = float(np.mean(ar_buf[half:]))
            ar_trend_up = ar_late > ar_early + 0.20

            # Velocity trung binh trong buffer phai > nguong slump
            avg_vel = float(np.mean(list(vel_buf))) if vel_buf else 0.0
            vel_positive = avg_vel > self.SLUMP_VEL_RATIO * h

            cur_ar_ok = ar_late > self.SLUMP_ASPECT_MIN

            if ar_trend_up and vel_positive and cur_ar_ok:
                cnt2 = self._sustained_slump.get(track_id, 0) + 1
                self._sustained_slump[track_id] = cnt2
                if cnt2 >= self.SLUMP_SUSTAINED:
                    return self._result(True, 0.78, 'Gradual_Collapse', 'high')
                return self._result(False, 0.0, 'Observing', 'low')
            else:
                self._sustained_slump[track_id] = 0
        else:
            self._sustained_slump[track_id] = 0

        return self._result(False, 0.0, 'Normal', 'low')

    # ─────────────────────────────────────────────────────────
    def _reset(self, track_id: int):
        self._sustained_posture[track_id] = 0
        self._sustained_slump[track_id]   = 0
        if track_id in self._vel_history:
            self._vel_history[track_id].clear()
        if track_id in self._aspect_history:
            self._aspect_history[track_id].clear()

    @staticmethod
    def _result(detected, confidence, symptom, risk):
        return {'detected': detected, 'confidence': confidence,
                'symptom': symptom, 'risk_level': risk}
