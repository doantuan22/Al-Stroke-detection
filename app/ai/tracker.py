"""
Object Tracker wrapper for YOLOv8 Tracking
"""
import numpy as np

class Tracker:
    def __init__(self):
        """
        YOLOv8 has built-in tracking (ByteTrack/BoT-SORT)
        This class will help manage track history if needed
        """
        self.track_history = {} # track_id -> list of keypoints/states
        
    def update_history(self, track_id, data, max_len=30):
        """Keep a buffer of data for each track"""
        if track_id not in self.track_history:
            self.track_history[track_id] = []
        
        self.track_history[track_id].append(data)
        
        if len(self.track_history[track_id]) > max_len:
            self.track_history[track_id].pop(0)
            
    def get_history(self, track_id):
        return self.track_history.get(track_id, [])

    def clean_old_tracks(self, active_ids):
        """Remove tracks that are no longer active"""
        inactive = [tid for tid in self.track_history if tid not in active_ids]
        for tid in inactive:
            del self.track_history[tid]
