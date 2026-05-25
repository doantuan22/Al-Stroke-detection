"""
Visualization Helpers (Optimized for YOLO-Pose COCO 17-keypoint)
v2: Dynamic alert position per-person (no overlap), optimized loops.
"""
import cv2
import numpy as np

# COCO 17-keypoint skeleton connections
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # head
    (5, 6),                                      # shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),            # arms
    (5, 11), (6, 12), (11, 12),                  # torso
    (11, 13), (13, 15), (12, 14), (14, 16)       # legs
]

# Color per risk level (BGR)
RISK_COLOR = {
    'high'  : (0,   0,   255),   # red
    'medium': (0,   165, 255),   # orange
    'low'   : (0,   255, 0),     # green
}


def draw_skeleton(frame: np.ndarray, kpts: np.ndarray, conf_thresh: float = 0.4) -> np.ndarray:
    """
    Draw 17-point COCO skeleton on frame (in-place).
    Uses pre-filtered valid points to avoid redundant checks in the loop.
    """
    h, w = frame.shape[:2]

    # Pre-compute valid keypoint mask once
    valid_mask = kpts[:, 2] > conf_thresh

    # Draw connections
    for i, j in SKELETON_CONNECTIONS:
        if valid_mask[i] and valid_mask[j]:
            pt1 = (int(kpts[i, 0]), int(kpts[i, 1]))
            pt2 = (int(kpts[j, 0]), int(kpts[j, 1]))
            cv2.line(frame, pt1, pt2, (0, 255, 255), 2, cv2.LINE_AA)

    # Draw keypoint circles
    for idx in range(len(kpts)):
        if valid_mask[idx]:
            cv2.circle(frame, (int(kpts[idx, 0]), int(kpts[idx, 1])), 4,
                       (0, 0, 255), -1, cv2.LINE_AA)

    return frame


def draw_info(frame: np.ndarray, track_id: int, bbox: list,
              result: dict, fps: float = None) -> np.ndarray:
    """
    Draw bounding box, label, and alert text anchored to each person's bbox.
    Alert text is drawn near the person (not fixed at frame corner) so
    multiple people's alerts don't overlap.
    """
    x1, y1, x2, y2 = map(int, bbox)
    color = RISK_COLOR.get(result['risk_level'], RISK_COLOR['low'])

    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    # Label bar above box
    label = f"ID:{track_id} | {result['symptom']}"
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    bar_y1 = max(y1 - 22, 0)
    bar_y2 = y1
    cv2.rectangle(frame, (x1, bar_y1), (x1 + lw + 4, bar_y2), color, -1)
    cv2.putText(frame, label, (x1 + 2, bar_y2 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # Alert text anchored just below the bbox (per-person, no overlap)
    if result['detected']:
        alert_text = f"⚠ ALERT: {result['symptom'].upper()}"
        (aw, ah), _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        ax = max(x1, 0)
        ay = min(y2 + ah + 8, frame.shape[0] - 4)
        cv2.putText(frame, alert_text, (ax, ay),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    # FPS overlay — drawn only once by the last call (caller responsibility)
    # (moved to main_window to draw once per frame, not per person)

    return frame


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Draw FPS counter once per frame at top-right corner."""
    text = f"FPS: {fps:.1f}"
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    x = frame.shape[1] - tw - 10
    cv2.putText(frame, text, (x, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return frame
