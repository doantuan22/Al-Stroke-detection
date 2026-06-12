"""
Thin cloud synchronization facade.

This keeps upload failures away from the video pipeline. The existing Supabase
clients still do the real storage/database work.
"""
from __future__ import annotations

import traceback
from typing import Any


class CloudSync:
    def __init__(self, stroke_cloud, airport_cloud, baggage_tracker=None, log=None):
        self.stroke_cloud = stroke_cloud
        self.airport_cloud = airport_cloud
        self.baggage_tracker = baggage_tracker
        self.log = log or (lambda msg: None)

    def upload_stroke(self, frame, track_id: int, result: dict, camera_id: str):
        try:
            local_path = self.stroke_cloud.save_local(frame, track_id, result)
            if local_path:
                self.log(f"Saved local evidence: {local_path}")
            return self.stroke_cloud.upload_alert(
                frame, track_id, result, camera_id=camera_id
            )
        except Exception as exc:
            self.log(f"[CloudSync] Stroke upload failed: {exc}")
            traceback.print_exc()
            return None, None

    def upload_airport(self, frame, alert: dict, camera_id: str):
        try:
            url, path = self.airport_cloud.upload_airport_alert(
                frame, alert, camera_id=camera_id
            )
            self.sync_baggage_tracks()
            return url, path
        except Exception as exc:
            self.log(f"[CloudSync] Airport upload failed: {exc}")
            traceback.print_exc()
            return None, None

    def sync_baggage_tracks(self):
        if not self.baggage_tracker:
            return False
        try:
            dirty = self.baggage_tracker.pop_dirty()
            if dirty:
                return self.airport_cloud.upsert_baggage_tracks(dirty)
            return False
        except Exception as exc:
            self.log(f"[CloudSync] Baggage track sync failed: {exc}")
            return False

    def clean_baggage_tracks(self, active_track_ids: list[int]):
        try:
            return self.airport_cloud.clean_baggage_tracks(active_track_ids)
        except Exception as exc:
            self.log(f"[CloudSync] Baggage track cleanup failed: {exc}")
            return None

