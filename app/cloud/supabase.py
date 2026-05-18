"""
Supabase Cloud Integration - Optimized for stroke_events schema
"""
import os
import cv2
import time
import uuid
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SupabaseClient:
    def __init__(self):
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_KEY")
        # Đồng bộ với thiết kế của người dùng
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

    def upload_alert(self, frame, track_id, result, camera_id="CAM_01"):
        """
        Upload alert image and log to 'stroke_events' table
        """
        if not self.enabled:
            return None
            
        try:
            # 1. Chuẩn bị file ảnh
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            event_type = result['symptom'].replace(' ', '_')
            filename = f"{event_type}_{timestamp_str}.jpg"
            
            # Encode image to buffer
            _, buffer = cv2.imencode('.jpg', frame)
            
            # 2. Upload lên Storage (Bucket: surveillance-images)
            path = f"alerts/{filename}"
            self.client.storage.from_(self.bucket_name).upload(
                path=path,
                file=buffer.tobytes(),
                file_options={"content-type": "image/jpeg"}
            )
            
            # 3. Lấy Public URL (vì bucket là public)
            public_url = self.client.storage.from_(self.bucket_name).get_public_url(path)
            
            # 4. Insert vào bảng 'stroke_events' theo đúng schema thiết kế
            data = {
                "id": str(uuid.uuid4()), # Tự động tạo ID duy nhất
                "image_url": public_url,
                "timestamp": datetime.now().isoformat(),
                "camera_id": str(camera_id),
                "confidence": float(result['confidence']),
                "event_type": result['symptom']
            }
            
            self.client.table(self.table_name).insert(data).execute()
            
            return public_url
            
        except Exception as e:
            print(f"[Cloud] Upload failed: {e}")
            return None
            
    def save_local(self, frame, track_id, result, folder='output/images'):
        """Save alert image locally as backup"""
        os.makedirs(folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        symptom = result['symptom'].replace(' ', '_')
        filename = f"alert_{track_id}_{symptom}_{timestamp}.jpg"
        filepath = os.path.join(folder, filename)
        
        cv2.imwrite(filepath, frame)
        return filepath
