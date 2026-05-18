"""
Pose Estimator using MediaPipe
"""
import cv2
import numpy as np
import mediapipe as mp
# Robust import for submodules
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing

class PoseEstimator:
    def __init__(self, static_mode=False, model_complexity=0, smooth_landmarks=True):
        """
        Initialize MediaPipe Pose
        Args:
            static_mode: Whether to treat images as batch or stream
            model_complexity: 0, 1, or 2
            smooth_landmarks: Whether to smooth landmarks across frames
        """
        self.pose = mp_pose.Pose(
            static_image_mode=static_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        
    def estimate(self, frame):
        """
        Estimate pose landmarks in a frame
        Args:
            frame: BGR image
        Returns:
            landmarks: List of landmark objects or None
            processed_frame: Frame with landmarks drawn (optional)
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        return results.pose_landmarks

    def get_keypoints(self, landmarks, img_w, img_h):
        """
        Convert MediaPipe landmarks to numpy keypoints array
        Returns: np.array of shape (33, 3) -> [x, y, visibility]
        """
        if not landmarks:
            return None
            
        keypoints = []
        for lm in landmarks.landmark:
            keypoints.append([lm.x * img_w, lm.y * img_h, lm.visibility])
            
        return np.array(keypoints)

    def draw(self, frame, landmarks):
        """Draw landmarks on frame"""
        if landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )
        return frame
