"""
Modern Pose & Person Detector using YOLOv8-Pose (GPU Optimized)
"""
import torch
import numpy as np
from ultralytics import YOLO

class PoseDetector:
    def __init__(self, model_path='yolov8n-pose.pt', device=None):
        """
        Initialize YOLOv8-Pose detector
        Args:
            model_path: Path to pose model (e.g., yolov8n-pose.pt)
            device: 'cpu' or 'cuda'
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"[AI Engine] Loading {model_path} on {self.device}...")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        if self.device == 'cuda':
            print(f"🚀 GPU Accelerated: {torch.cuda.get_device_name(0)}")
        
    def detect(self, frame, conf=0.4):
        """
        Detect persons and poses in a single pass
        Returns:
            results: List of dicts containing bbox and keypoints
        """
        # Run inference on GPU
        results = self.model.predict(frame, conf=conf, verbose=False, device=self.device)
        
        processed_results = []
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes.cpu().numpy()
            keypoints = result.keypoints.cpu().numpy()
            
            for i in range(len(boxes)):
                processed_results.append({
                    'bbox': boxes[i].xyxy[0].tolist(),
                    'conf': boxes[i].conf[0],
                    'kpts': keypoints.data[i] # Shape (17, 3) -> [x, y, conf]
                })
                
        return processed_results

    def draw(self, frame, results):
        """Draw skeletons using YOLOv8 built-in plotting (optional) or custom"""
        # We'll use custom drawing for a premium look in the GUI
        return frame
