"""
Object Tracker — wraps per-person keypoint history.
v3: Grace period trước khi xóa track — giúp nhận diện ổn định
    khi YOLO bỏ sót 1-2 frame (người contrast thấp với nền).

    Vấn đề cũ: clean_old_tracks() xóa ngay khi track không có
    trong frame → lần sau YOLO detect lại được, ByteTrack tạo
    track_id mới → history mất trắng → chập chờn.

    Giải pháp: giữ track thêm GRACE_FRAMES frame trước khi xóa.
    track_buffer trong bytetrack_stroke.yaml đã tăng lên 60 nên
    ByteTrack sẽ giữ cùng track_id, grace period ở đây giúp
    history layer không bị xóa sớm hơn tracker.
"""
from collections import deque


# Số frame chờ trước khi xóa track (tránh xóa khi YOLO tạm bỏ sót)
_GRACE_FRAMES = 15


class Tracker:
    def __init__(self, max_history=30):
        """
        Args:
            max_history: number of frames to keep per person
        """
        self.max_history  = max_history
        self.track_history: dict[int, deque] = {}
        # Đếm số frame vắng mặt liên tiếp cho mỗi track
        self._absent_count: dict[int, int] = {}

    def update_history(self, track_id: int, data):
        """Append keypoints/state for a tracked person."""
        if track_id not in self.track_history:
            self.track_history[track_id] = deque(maxlen=self.max_history)
        self.track_history[track_id].append(data)
        # Reset bộ đếm vắng mặt vì track đang hoạt động
        self._absent_count[track_id] = 0

    def get_history(self, track_id: int) -> list:
        """Return history as a plain list (for numpy compatibility)."""
        return list(self.track_history.get(track_id, []))

    def clean_old_tracks(self, active_ids: list[int]):
        """
        Xóa track sau khi vắng mặt GRACE_FRAMES frame liên tiếp.

        Thay vì xóa ngay khi không thấy trong frame (như v2),
        v3 cho phép YOLO bỏ sót tối đa _GRACE_FRAMES frame mà
        không mất lịch sử — quan trọng với người nằm trên sàn
        cùng màu (contrast thấp → confidence dao động).
        """
        all_ids = list(self.track_history.keys())
        for tid in all_ids:
            if tid in active_ids:
                # Đang hoạt động → reset absent count
                self._absent_count[tid] = 0
            else:
                # Vắng mặt → tăng đếm
                count = self._absent_count.get(tid, 0) + 1
                self._absent_count[tid] = count
                if count >= _GRACE_FRAMES:
                    # Hết grace period → xóa thật sự
                    del self.track_history[tid]
                    del self._absent_count[tid]
