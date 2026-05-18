"""
Premium Stroke Detection Dashboard - Multi-threaded GPU Version
Optimized for RTX 4000 Series (30-60+ FPS)
"""
import cv2
import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk
import time
import threading
from pathlib import Path
from queue import Queue
from datetime import datetime

# AI Engine
from app.ai.detector import PoseDetector
from app.ai.tracker import Tracker
from app.ai.recognizer import StrokeRecognizer
from app.cloud.supabase import SupabaseClient
from app.utils.visualization import draw_info, draw_skeleton

class StrokeApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("AI Stroke Detection - ULTRA PERFORMANCE (GPU)")
        self.geometry("1280x800")
        ctk.set_appearance_mode("dark")

        # Core Engines
        self.detector = PoseDetector()
        self.tracker = Tracker()
        self.recognizer = StrokeRecognizer()
        self.cloud = SupabaseClient()

        # Threading & Queues
        self.frame_queue = Queue(maxsize=2)
        self.is_running = False
        self.processing_thread = None
        
        # Stats
        self.fps = 0
        self.source = 0 # Default webcam
        self.camera_id = "CAM_00"
        self.alert_count = {}
        self.last_alert_time = {}

        self._create_widgets()
        
    def _create_widgets(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar
        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        ctk.CTkLabel(self.sidebar, text="STROKE AI", font=ctk.CTkFont(size=24, weight="bold")).pack(pady=20)
        
        self.source_option = ctk.CTkOptionMenu(self.sidebar, values=["Webcam 0", "Webcam 1", "Video File"], 
                                             command=self._on_source_change)
        self.source_option.pack(pady=10)

        self.start_btn = ctk.CTkButton(self.sidebar, text="START ENGINE", fg_color="green", command=self.toggle_engine)
        self.start_btn.pack(pady=20, padx=20)

        # Main Panel
        self.main_content = ctk.CTkFrame(self, corner_radius=10)
        self.main_content.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_content.grid_columnconfigure(0, weight=1)
        self.main_content.grid_rowconfigure(0, weight=1)

        self.video_canvas = ctk.CTkLabel(self.main_content, text="")
        self.video_canvas.grid(row=0, column=0, sticky="nsew")

        self.log_panel = ctk.CTkTextbox(self.main_content, height=150)
        self.log_panel.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

    def log_msg(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_panel.insert("0.0", f"[{ts}] {msg}\n")

    def _on_source_change(self, value):
        if value == "Video File":
            self.source = tk.filedialog.askopenfilename()
            if self.source:
                self.camera_id = Path(self.source).name
        else:
            self.source = int(value.split()[-1])
            self.camera_id = f"CAM_{self.source:02d}"
        self.log_msg(f"Source: {self.source} | ID: {self.camera_id}")

    def toggle_engine(self):
        if not self.is_running:
            self.is_running = True
            self.start_btn.configure(text="STOP ENGINE", fg_color="red")
            # Start the AI processing thread
            self.processing_thread = threading.Thread(target=self.video_processing_worker, daemon=True)
            self.processing_thread.start()
            # Start the UI update loop
            self.update_ui_loop()
        else:
            self.is_running = False
            self.start_btn.configure(text="START ENGINE", fg_color="green")

    def video_processing_worker(self):
        """Dedicated thread for Video Capture and AI Inference"""
        cap = cv2.VideoCapture(self.source)
        # Tối ưu buffer camera
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        
        prev_time = time.time()
        
        while self.is_running:
            ret, frame = cap.read()
            if not ret: break

            # 1. AI Inference (GPU)
            results = self.detector.detect(frame)
            
            for i, res in enumerate(results):
                bbox, kpts = res['bbox'], res['kpts']
                track_id = i + 1
                
                # 2. Logic
                self.tracker.update_history(track_id, kpts)
                history = self.tracker.get_history(track_id)
                result = self.recognizer.analyze(history, (frame.shape[1], frame.shape[0]))
                
                # 3. Alert
                if result['detected'] and result['risk_level'] == 'high':
                    now = time.time()
                    count = self.alert_count.get(track_id, 0)
                    if count < 3 and (now - self.last_alert_time.get(track_id, 0)) >= 10:
                        current_count = count + 1
                        self.log_msg(f"ALERT [{current_count}/3]: {result['symptom']}")
                        
                        # 1. Luôn lưu cục bộ vào máy
                        local_path = self.cloud.save_local(frame, track_id, result)
                        if local_path:
                            self.log_msg(f"Backup local: {local_path}")
                            
                        # 2. Chỉ đẩy lên Supabase nếu là ảnh thứ 3 (Ảnh cuối)
                        if current_count == 3:
                            self.log_msg(f"🚀 Pushing FINAL evidence to Supabase...")
                            url = self.cloud.upload_alert(frame, track_id, result, camera_id=self.camera_id)
                            if url:
                                self.log_msg(f"Cloud Sync Success!")
                        
                        self.alert_count[track_id] = current_count
                        self.last_alert_time[track_id] = now

                # 4. Visualization
                frame = draw_skeleton(frame, kpts)
                frame = draw_info(frame, track_id, bbox, result, self.fps)

            # Calc FPS
            curr_time = time.time()
            self.fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            # Push to UI queue (Resize here using OpenCV is much faster than PIL)
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
            
        cap.release()

    def update_ui_loop(self):
        """Main UI Thread: Only handles display"""
        if not self.is_running: return

        if not self.frame_queue.empty():
            frame = self.frame_queue.get()
            
            # Convert to RGB & Resize for UI (Using OpenCV)
            cw, ch = self.video_canvas.winfo_width(), self.video_canvas.winfo_height()
            if cw > 1 and ch > 1:
                frame = cv2.resize(frame, (cw, ch))
            
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_canvas.imgtk = imgtk
            self.video_canvas.configure(image=imgtk)

        self.after(1, self.update_ui_loop) # Chạy gần như liên tục

if __name__ == "__main__":
    app = StrokeApp()
    app.mainloop()
