"""
Supabase Cloud Integration - Optimized for stroke_events schema
v3: Extract storage path from image_url — no extra column needed
"""
import os
import cv2
import uuid
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class SupabaseClient:
    def __init__(self):
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_KEY")
        self.bucket_name = os.getenv("SUPABASE_BUCKET", "surveillance-images")
        self.table_name = "stroke_events"

        self.enabled = False
        if self.url and self.key:
            try:
                from supabase import create_client, Client
                self.client: Client = create_client(self.url, self.key)
                self.enabled = True
                print(f"[Cloud] Supabase connected: {self.url}")
            except Exception as e:
                print(f"[Cloud] Connection failed: {e}")
        else:
            print("[Cloud] Credentials missing in .env")

    # ── Internal helper ────────────────────────────────────────────
    def _path_from_url(self, url: str) -> str | None:
        """
        Trích xuất storage path từ Supabase public URL.
        VD: https://xxx.supabase.co/storage/v1/object/public/surveillance-images/alerts/foo.jpg
            → 'alerts/foo.jpg'
        """
        if not url:
            return None
        marker = f"/object/public/{self.bucket_name}/"
        if marker in url:
            return url.split(marker, 1)[-1]
        # Fallback: thử tìm '/alerts/' trong URL
        if "/alerts/" in url:
            idx = url.index("/alerts/")
            return url[idx + 1:]   # bỏ dấu '/' đầu → 'alerts/...'
        return None

    # ─────────────────────────────────────────────────────────────
    # UPLOAD
    # ─────────────────────────────────────────────────────────────
    def upload_alert(self, frame, track_id, result, camera_id="CAM_01"):
        """
        Upload alert image → Storage, rồi insert record vào stroke_events.
        Returns: (public_url, storage_path) hoặc (None, None) nếu lỗi.
        """
        if not self.enabled:
            return None, None

        try:
            timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            event_type = result['symptom'].replace(' ', '_')
            filename = f"{event_type}_{timestamp_str}.jpg"
            path = f"alerts/{filename}"

            _, buffer = cv2.imencode('.jpg', frame)

            # 1. Upload lên Storage
            self.client.storage.from_(self.bucket_name).upload(
                path=path,
                file=buffer.tobytes(),
                file_options={"content-type": "image/jpeg"}
            )

            # 2. Lấy public URL
            public_url = self.client.storage.from_(self.bucket_name).get_public_url(path)

            # 3. Insert record (chỉ dùng các cột đã có trong schema)
            data = {
                "id": str(uuid.uuid4()),
                "image_url": public_url,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "camera_id": str(camera_id),
                "confidence": float(result['confidence']),
                "event_type": result['symptom'],
            }
            self.client.table(self.table_name).insert(data).execute()

            return public_url, path

        except Exception as e:
            print(f"[Cloud] Upload failed: {e}")
            return None, None

    def save_local(self, frame, track_id, result, folder='output/images'):
        """Lưu ảnh alert cục bộ làm backup."""
        os.makedirs(folder, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        symptom = result['symptom'].replace(' ', '_')
        filename = f"alert_{track_id}_{symptom}_{timestamp}.jpg"
        filepath = os.path.join(folder, filename)
        cv2.imwrite(filepath, frame)
        return filepath

    # ─────────────────────────────────────────────────────────────
    # FETCH
    # ─────────────────────────────────────────────────────────────
    def fetch_events(self, limit=200):
        """
        Lấy danh sách sự kiện từ bảng stroke_events, mới nhất trước.
        Returns: list[dict]
        """
        if not self.enabled:
            return []
        try:
            response = (
                self.client.table(self.table_name)
                .select("*")
                .order("timestamp", desc=True)
                .limit(limit)
                .execute()
            )
            return response.data or []
        except Exception as e:
            print(f"[Cloud] Fetch failed: {e}")
            return []

    # ─────────────────────────────────────────────────────────────
    # DELETE
    # ─────────────────────────────────────────────────────────────
    def delete_event(self, event_id: str, image_url: str = None):
        """
        Xóa 1 record khỏi DB và xóa ảnh trên Storage.
        Path storage được trích xuất từ image_url — không cần cột riêng.
        """
        if not self.enabled:
            return False
        try:
            # Xóa ảnh trên Storage
            path = self._path_from_url(image_url)
            if path:
                self.client.storage.from_(self.bucket_name).remove([path])
                print(f"[Cloud] Storage deleted: {path}")

            # Xóa record trong DB
            self.client.table(self.table_name).delete().eq("id", event_id).execute()
            print(f"[Cloud] DB deleted: {event_id}")
            return True
        except Exception as e:
            print(f"[Cloud] Delete failed: {e}")
            return False

    def delete_events_batch(self, events: list[dict]):
        """
        Xóa nhiều sự kiện: ảnh trên Storage (batch) + records trong DB.
        Returns: (success_count, fail_count)
        """
        if not self.enabled:
            return 0, len(events)

        success = 0
        fail = 0

        try:
            # Gom tất cả storage paths để remove batch
            paths = [self._path_from_url(e.get("image_url")) for e in events]
            paths = [p for p in paths if p]   # lọc None
            if paths:
                self.client.storage.from_(self.bucket_name).remove(paths)
                print(f"[Cloud] Storage batch deleted: {len(paths)} files")

            # Xóa từng record (hoặc có thể dùng .in_() nếu SDK hỗ trợ)
            ids = [e["id"] for e in events if e.get("id")]
            for eid in ids:
                try:
                    self.client.table(self.table_name).delete().eq("id", eid).execute()
                    success += 1
                except Exception as ie:
                    print(f"[Cloud] DB delete failed for {eid}: {ie}")
                    fail += 1

        except Exception as e:
            print(f"[Cloud] Batch delete error: {e}")
            fail = len(events) - success

        return success, fail

    def delete_all_events(self):
        """Xóa TOÀN BỘ sự kiện và ảnh tương ứng."""
        events = self.fetch_events(limit=1000)
        if not events:
            return 0, 0
        return self.delete_events_batch(events)
