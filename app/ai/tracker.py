"""
Object Tracker — wraps per-person keypoint history.
v2: Uses collections.deque (O(1) append/pop) instead of list.pop(0).
    track_id now comes from YOLOv8 ByteTrack (stable across frames).
"""
from collections import deque


class Tracker:
    def __init__(self, max_history=30):
        """
        Args:
            max_history: number of frames to keep per person
        """
        self.max_history  = max_history
        self.track_history: dict[int, deque] = {}

    def update_history(self, track_id: int, data):
        """Append keypoints/state for a tracked person."""
        if track_id not in self.track_history:
            self.track_history[track_id] = deque(maxlen=self.max_history)
        self.track_history[track_id].append(data)

    def get_history(self, track_id: int) -> list:
        """Return history as a plain list (for numpy compatibility)."""
        return list(self.track_history.get(track_id, []))

    def clean_old_tracks(self, active_ids: list[int]):
        """Remove tracks that are no longer visible."""
        inactive = [tid for tid in self.track_history if tid not in active_ids]
        for tid in inactive:
            del self.track_history[tid]
