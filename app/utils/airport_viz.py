"""
Airport Visualization Helpers
==============================
Các hàm vẽ overlay cho 2 tính năng sân bay:
  - draw_baggage_overlays() : Vẽ bounding box + timer cho hành lý
  - draw_weapon_alerts()    : Vẽ cảnh báo đỏ nổi bật cho vũ khí
  - draw_airport_stats()    : HUD thống kê góc trái dưới
"""
import cv2
import numpy as np
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.ai.baggage_tracker import BaggageState

# ── Màu sắc ───────────────────────────────────────────────────
COLOR_SAFE       = (80, 200, 80)     # Xanh lá — có chủ
COLOR_WARNING    = (0, 180, 255)     # Cam — đang theo dõi (< 60s)
COLOR_DANGER     = (0, 50, 255)      # Đỏ — bỏ lại quá lâu
COLOR_WEAPON     = (0, 0, 255)       # Đỏ thuần — vũ khí
COLOR_WEAPON_BG  = (20, 20, 180)     # Nền label vũ khí
COLOR_TEXT       = (240, 240, 240)
FONT             = cv2.FONT_HERSHEY_SIMPLEX


# ══════════════════════════════════════════════════════════════
#  Abandoned Baggage Overlays
# ══════════════════════════════════════════════════════════════
def draw_baggage_overlays(
    frame: np.ndarray,
    states: dict,            # dict[int, BaggageState]
    abandon_timeout: float = 60.0,
) -> np.ndarray:
    """
    Vẽ bounding box + timer countdown cho từng hành lý đang theo dõi.

    Màu box:
      Xanh lá  → đang có chủ
      Cam      → chủ vừa đi, đang đếm ngược
      Đỏ       → quá timeout → đã alert
    """
    for tid, state in states.items():
        if not state.bbox or len(state.bbox) < 4:
            continue

        x1, y1, x2, y2 = [int(v) for v in state.bbox]
        elapsed = state.abandon_seconds

        # ── Chọn màu theo trạng thái ─────────────────────
        if state.owner_gone_at is None:
            color       = COLOR_SAFE
            status_text = f"{state.object_class} ✓ có chủ"
            thickness   = 2
        elif elapsed < abandon_timeout:
            ratio       = elapsed / abandon_timeout
            color       = COLOR_WARNING
            remaining   = abandon_timeout - elapsed
            status_text = f"{state.object_class} ⚠ {remaining:.0f}s"
            thickness   = 2
            # Vẽ progress bar bên dưới box
            _draw_progress_bar(frame, x1, y2, x2, y2 + 6, ratio, COLOR_WARNING)
        else:
            color       = COLOR_DANGER
            status_text = f"{state.object_class} 🚨 BỎ LẠI {elapsed:.0f}s"
            thickness   = 3
            # Hiệu ứng nhấp nháy khi đã alert
            if int(time.time() * 2) % 2 == 0:
                _draw_glow_rect(frame, x1, y1, x2, y2, COLOR_DANGER)

        # ── Bounding box ──────────────────────────────────
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # ── Label background ──────────────────────────────
        label_h   = 22
        label_y   = max(0, y1 - label_h)
        label_w   = min(len(status_text) * 9 + 8, frame.shape[1] - x1)
        cv2.rectangle(frame,
                      (x1, label_y),
                      (x1 + label_w, y1),
                      color, -1)
        cv2.putText(frame, status_text,
                    (x1 + 4, y1 - 5),
                    FONT, 0.45, (10, 10, 10), 1, cv2.LINE_AA)

        # ── Track ID ──────────────────────────────────────
        cv2.putText(frame, f"#{tid}",
                    (x1 + 4, y2 - 6),
                    FONT, 0.38, color, 1, cv2.LINE_AA)

    return frame


# ══════════════════════════════════════════════════════════════
#  Weapon Alert Overlays
# ══════════════════════════════════════════════════════════════
def draw_weapon_alerts(
    frame: np.ndarray,
    weapon_detections: list[dict],
) -> np.ndarray:
    """
    Vẽ cảnh báo nổi bật màu đỏ cho từng vũ khí phát hiện được.
    Hiệu ứng: box dày + label lớn + glow border.
    """
    for det in weapon_detections:
        bbox = det.get('bbox', [])
        if len(bbox) < 4:
            continue

        x1, y1, x2, y2  = [int(v) for v in bbox]
        class_name       = det.get('object_class', 'weapon')
        conf             = det.get('confidence', 0.0)
        bearer_id        = det.get('bearer_id')
        risk             = det.get('risk_level', 'high')

        # Glow effect
        _draw_glow_rect(frame, x1, y1, x2, y2, COLOR_WEAPON, glow_size=8)

        # Box chính
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_WEAPON, 3)

        # Label trên
        top_text = f"⚠ {class_name.upper()} {conf:.0%}"
        _draw_label_box(frame, top_text,
                        x=x1, y=y1,
                        bg_color=COLOR_WEAPON,
                        text_color=COLOR_TEXT,
                        font_scale=0.55,
                        above=True)

        # Label dưới — bearer nếu có
        if bearer_id is not None:
            bot_text = f"Bearer: #{bearer_id}"
            _draw_label_box(frame, bot_text,
                            x=x1, y=y2,
                            bg_color=(50, 0, 0),
                            text_color=(200, 150, 255),
                            font_scale=0.42,
                            above=False)

        # Risk badge góc phải dưới của box
        risk_color = (0, 0, 220) if risk == 'critical' else (0, 80, 200)
        cv2.rectangle(frame, (x2 - 60, y2 - 20), (x2, y2), risk_color, -1)
        cv2.putText(frame, risk.upper(),
                    (x2 - 57, y2 - 5),
                    FONT, 0.38, COLOR_TEXT, 1, cv2.LINE_AA)

    return frame


# ══════════════════════════════════════════════════════════════
#  Airport Stats HUD
# ══════════════════════════════════════════════════════════════
def draw_airport_stats(
    frame       : np.ndarray,
    n_bags      : int,
    n_abandoned : int,
    n_weapons   : int,
) -> np.ndarray:
    """
    Vẽ HUD thống kê an ninh sân bay ở góc trái dưới frame.
    """
    h, w = frame.shape[:2]
    lines = [
        (f"Hanh ly theo doi: {n_bags}",     (180, 220, 180)),
        (f"Bo lai canh bao: {n_abandoned}",  (0, 150, 255) if n_abandoned > 0 else (150, 150, 150)),
        (f"Vu khi: {n_weapons}",             (0, 50, 255)  if n_weapons  > 0 else (150, 150, 150)),
    ]

    panel_h = 24 * len(lines) + 8
    panel_w = 240
    x0, y0  = 10, h - panel_h - 10

    # Nền semi-transparent
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0 - 4, y0 - 4),
                  (x0 + panel_w, y0 + panel_h), (10, 10, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    for i, (text, color) in enumerate(lines):
        y = y0 + 20 + i * 24
        cv2.putText(frame, text, (x0, y),
                    FONT, 0.45, color, 1, cv2.LINE_AA)

    return frame


# ══════════════════════════════════════════════════════════════
#  Internal helpers
# ══════════════════════════════════════════════════════════════
def _draw_progress_bar(frame, x1, y1, x2, y2, ratio, color):
    """Progress bar ngang."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 40, 40), -1)
    fill_x = int(x1 + (x2 - x1) * min(ratio, 1.0))
    if fill_x > x1:
        cv2.rectangle(frame, (x1, y1), (fill_x, y2), color, -1)


def _draw_glow_rect(frame, x1, y1, x2, y2, color, glow_size=6):
    """Hiệu ứng glow bằng cách vẽ nhiều rectangle mờ dần ra ngoài."""
    alpha_steps = [0.15, 0.10, 0.06]
    for i, alpha in enumerate(alpha_steps):
        g = glow_size - i * 2
        if g <= 0:
            continue
        overlay = frame.copy()
        cv2.rectangle(overlay,
                      (max(0, x1 - g), max(0, y1 - g)),
                      (min(frame.shape[1], x2 + g), min(frame.shape[0], y2 + g)),
                      color, 2)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _draw_label_box(frame, text, x, y, bg_color, text_color,
                    font_scale=0.5, above=True):
    """Vẽ label box với nền màu."""
    (tw, th), _ = cv2.getTextSize(text, FONT, font_scale, 1)
    pad = 4
    if above:
        y_top  = max(0, y - th - pad * 2)
        y_bot  = y
        ty     = y - pad
    else:
        y_top  = y
        y_bot  = y + th + pad * 2
        ty     = y + th + pad

    cv2.rectangle(frame, (x, y_top), (x + tw + pad * 2, y_bot), bg_color, -1)
    cv2.putText(frame, text, (x + pad, ty),
                FONT, font_scale, text_color, 1, cv2.LINE_AA)
