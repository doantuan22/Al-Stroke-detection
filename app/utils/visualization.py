"""
Visualization Helpers (Optimized for YOLO-Pose)
"""
import cv2
import numpy as np

# COCO Skeleton connections
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), 
    (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]

def draw_skeleton(frame, kpts, conf_thresh=0.4):
    """Draw 17-point skeleton"""
    for i, j in SKELETON_CONNECTIONS:
        pt1 = kpts[i]
        pt2 = kpts[j]
        
        if pt1[2] > conf_thresh and pt2[2] > conf_thresh:
            cv2.line(frame, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (0, 255, 255), 2)
            
    for pt in kpts:
        if pt[2] > conf_thresh:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)
    return frame

def draw_info(frame, track_id, bbox, result, fps=None):
    """Draw detection info with premium aesthetics"""
    x1, y1, x2, y2 = map(int, bbox)
    
    color = (0, 255, 0)
    if result['risk_level'] == 'high':
        color = (0, 0, 255)
    elif result['risk_level'] == 'medium':
        color = (0, 165, 255)
        
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    label = f"ID: {track_id} | {result['symptom']}"
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (x1, y1 - 25), (x1 + w, y1), color, -1)
    cv2.putText(frame, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    if result['detected']:
        cv2.putText(frame, f"ALERT: {result['symptom'].upper()}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
    if fps is not None:
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 120, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return frame
