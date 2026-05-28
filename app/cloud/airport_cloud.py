"""
Airport Cloud Client
=====================
Upload sự kiện an ninh sân bay lên Supabase.
Kế thừa SupabaseClient để tái dùng kết nối + Storage helper.

Bảng:
  - airport_events  : Lịch sử alert (persistent)
  - baggage_tracks  : Trạng thái realtime hành lý (upsert)

Storage subfolders:
  surveillance-images/
  ├── baggage/    ← ảnh hành lý bỏ lại
  └── weapons/    ← ảnh vũ khí phát hiện
"""
import os
import cv2
import uuid
import traceback
from datetime import datetime, timezone
from app.cloud.supabase import SupabaseClient


class AirportCloudClient(SupabaseClient):
    """
    Mở rộng SupabaseClient với các bảng airport.
    Tái dùng toàn bộ kết nối, storage, và helper của lớp cha.
    """

    AIRPORT_TABLE  = "airport_events"
    BAGGAGE_TABLE  = "baggage_tracks"

    # Sub-folder trong Storage bucket
    _SUBFOLDERS = {
        'abandoned_baggage': 'baggage',
        'weapon_detected'  : 'weapons',
    }

    # ── Upload sự kiện chính ────────────────────────────────────
    def upload_airport_alert(
        self,
        frame,
        alert: dict,
        camera_id: str = 'CAM_00',
    ) -> tuple[str | None, str | None]:
        """
        Upload ảnh snapshot lên Storage rồi insert record vào airport_events.

        Args:
            frame     : BGR numpy array (frame gốc từ camera)
            alert     : dict alert từ BaggageTracker / WeaponDetector
                        keys: event_type, object_class, confidence,
                              risk_level, zone_name, duration_sec,
                              track_id, bbox
            camera_id : ID camera

        Returns:
            (public_url, storage_path) hoặc (None, None) nếu lỗi
        """
        if not self.enabled:
            return None, None

        try:
            ts       = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            etype    = alert.get('event_type', 'event')
            subfolder = self._SUBFOLDERS.get(etype, 'alerts')
            uid      = uuid.uuid4().hex[:8]
            filename = f"{etype}_{ts}_{uid}.jpg"
            path     = f"{subfolder}/{filename}"

            # ── 1. Encode + Upload ảnh ───────────────────────────────
            _, buf = cv2.imencode(
                '.jpg', frame,
                [cv2.IMWRITE_JPEG_QUALITY, 85]
            )
            # upsert=True → tránh lỗi file đã tồn tại (x-upsert header)
            try:
                self.client.storage.from_(self.bucket_name).upload(
                    path=path,
                    file=buf.tobytes(),
                    file_options={
                        "content-type": "image/jpeg",
                        "upsert": "true",
                    },
                )
            except Exception as upload_err:
                print(f"[AirportCloud] Storage upload warning: {upload_err} — retrying as update...")
                self.client.storage.from_(self.bucket_name).update(
                    path=path,
                    file=buf.tobytes(),
                    file_options={"content-type": "image/jpeg"},
                )

            # ── 2. Lấy public URL ───────────────────────────────
            url = self.client.storage.from_(self.bucket_name).get_public_url(path)

            # ── 3. Insert vào airport_events ────────────────
            bbox = alert.get('bbox', [])
            data = {
                "id"          : str(uuid.uuid4()),
                "event_type"  : etype,
                "camera_id"   : camera_id,
                "track_id"    : alert.get('track_id'),
                "object_class": alert.get('object_class'),
                "confidence"  : float(alert.get('confidence', 0.0)),
                "risk_level"  : alert.get('risk_level', 'high'),
                "zone_name"   : alert.get('zone_name'),
                "duration_sec": float(alert.get('duration_sec', 0.0)),
                "image_url"   : url,
                "metadata"    : {
                    "bbox"     : bbox,
                    "bearer_id": alert.get('bearer_id'),
                },
                "resolved"    : False,
                "created_at"  : datetime.now(timezone.utc).isoformat(),
            }
            self.client.table(self.AIRPORT_TABLE).insert(data).execute()
            print(f"[AirportCloud] OK {etype} uploaded -> {path}")
            return url, path

        except Exception as e:
            print(f"[AirportCloud] FAILED Upload failed: {e}")
            traceback.print_exc()
            return None, None

    # ── Realtime baggage_tracks upsert ──────────────────────────
    def upsert_baggage_tracks(self, states: list) -> bool:
        """
        Upsert danh sách BaggageState vào bảng baggage_tracks.
        Dùng upsert (insert hoặc update theo track_id) để tránh duplicate.

        Args:
            states: list[BaggageState] — từ BaggageTracker.pop_dirty()

        Returns:
            True nếu thành công
        """
        if not self.enabled or not states:
            return False
        try:
            records = []
            now_iso = datetime.now(timezone.utc).isoformat()
            for s in states:
                rec = s.to_db_record()
                rec['last_seen_at'] = now_iso
                records.append(rec)

            self.client.table(self.BAGGAGE_TABLE).upsert(
                records,
                on_conflict='track_id'
            ).execute()
            return True
        except Exception as e:
            print(f"[AirportCloud] Baggage upsert failed: {e}")
            return False

    # ── Fetch airport_events ────────────────────────────────────
    def fetch_airport_events(
        self,
        limit: int           = 200,
        event_type: str      = None,
        resolved: bool       = None,
    ) -> list[dict]:
        """
        Lấy danh sách sự kiện sân bay, mới nhất trước.

        Args:
            limit      : Số lượng tối đa
            event_type : Lọc theo loại ('abandoned_baggage' / 'weapon_detected')
            resolved   : Lọc theo trạng thái xử lý

        Returns:
            list[dict]
        """
        if not self.enabled:
            return []
        try:
            q = (
                self.client.table(self.AIRPORT_TABLE)
                .select("*")
                .order("created_at", desc=True)
                .limit(limit)
            )
            if event_type:
                q = q.eq("event_type", event_type)
            if resolved is not None:
                q = q.eq("resolved", resolved)

            return q.execute().data or []
        except Exception as e:
            print(f"[AirportCloud] Fetch failed: {e}")
            return []

    # ── Delete airport_events ───────────────────────────────────
    def delete_airport_event(
        self,
        event_id: str,
        image_url: str = None,
    ) -> bool:
        """Xóa 1 sự kiện và ảnh trên Storage."""
        if not self.enabled:
            return False
        try:
            # Xóa ảnh Storage
            path = self._path_from_url(image_url)
            if path:
                self.client.storage.from_(self.bucket_name).remove([path])

            # Xóa record
            self.client.table(self.AIRPORT_TABLE).delete().eq("id", event_id).execute()
            return True
        except Exception as e:
            print(f"[AirportCloud] Delete failed: {e}")
            return False

    def delete_airport_events_batch(self, events: list[dict]) -> tuple[int, int]:
        """Xóa nhiều sự kiện (ảnh Storage + DB records)."""
        if not self.enabled:
            return 0, len(events)

        success = 0
        fail    = 0

        try:
            # Batch xóa ảnh trên Storage
            paths = [self._path_from_url(e.get("image_url")) for e in events]
            paths = [p for p in paths if p]
            if paths:
                self.client.storage.from_(self.bucket_name).remove(paths)

            # Xóa từng record
            for ev in events:
                try:
                    self.client.table(self.AIRPORT_TABLE)\
                        .delete().eq("id", ev["id"]).execute()
                    success += 1
                except Exception:
                    fail += 1
        except Exception as e:
            print(f"[AirportCloud] Batch delete failed: {e}")
            fail = len(events) - success

        return success, fail

    def mark_resolved(self, event_id: str) -> bool:
        """Đánh dấu sự kiện đã xử lý (resolved=True)."""
        if not self.enabled:
            return False
        try:
            self.client.table(self.AIRPORT_TABLE).update({
                "resolved"   : True,
                "resolved_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", event_id).execute()
            return True
        except Exception as e:
            print(f"[AirportCloud] mark_resolved failed: {e}")
            return False

    # ── Baggage tracks cleanup ───────────────────────────────────
    def clean_baggage_tracks(self, active_track_ids: list[int]) -> None:
        """Xóa các track đã biến mất khỏi bảng baggage_tracks."""
        if not self.enabled:
            return
        try:
            # Lấy tất cả track IDs trong DB
            resp = self.client.table(self.BAGGAGE_TABLE)\
                .select("track_id").execute()
            db_ids = {r['track_id'] for r in (resp.data or [])}
            inactive = db_ids - set(active_track_ids)
            for tid in inactive:
                self.client.table(self.BAGGAGE_TABLE)\
                    .delete().eq("track_id", tid).execute()
        except Exception as e:
            print(f"[AirportCloud] Clean tracks failed: {e}")
