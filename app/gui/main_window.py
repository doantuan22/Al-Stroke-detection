"""
Premium Stroke Detection Dashboard v2
- Multi-threaded GPU detection
- Database Manager tab with image preview
"""
import cv2
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import customtkinter as ctk
from PIL import Image
import time
import threading
import requests
from io import BytesIO
from pathlib import Path
from queue import Queue
from datetime import datetime

# AI Engine
from app.ai.detector import PoseDetector
from app.ai.tracker import Tracker
from app.ai.recognizer import StrokeRecognizer
from app.cloud.supabase import SupabaseClient
from app.utils.visualization import draw_info, draw_skeleton


# ══════════════════════════════════════════════════════════════
#  Helper: tải ảnh từ URL trả về CTkImage
# ══════════════════════════════════════════════════════════════
def load_image_from_url(url, max_size=(420, 320)):
    try:
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        img.thumbnail(max_size, Image.LANCZOS)
        return ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
    except Exception as e:
        print(f"[Preview] Load failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════
#  Tab 2: Database Manager
# ══════════════════════════════════════════════════════════════
class DatabaseTab(ctk.CTkFrame):
    """Panel quản lý dữ liệu Supabase: xem, preview ảnh, xóa"""

    COL_DEFS = [
        ("timestamp",  "Thời gian",   160),
        ("camera_id",  "Camera",       80),
        ("event_type", "Loại sự kiện", 140),
        ("confidence", "Độ tin cậy",   100),
        ("id",         "ID",           260),
    ]

    def __init__(self, master, cloud: SupabaseClient, **kwargs):
        super().__init__(master, **kwargs)
        self.cloud = cloud
        self._events = []          # cache dữ liệu hiện tại
        self._preview_thread = None

        self._build_ui()

    # ── Build UI ───────────────────────────────────────────────
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(1, weight=1)

        # ── Toolbar ──
        toolbar = ctk.CTkFrame(self, height=50, fg_color="transparent")
        toolbar.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=(10, 4))

        ctk.CTkLabel(toolbar, text="📦 Database Manager",
                     font=ctk.CTkFont(size=16, weight="bold")).pack(side="left", padx=8)

        self.btn_refresh = ctk.CTkButton(
            toolbar, text="🔄 Refresh", width=110,
            fg_color="#2563EB", hover_color="#1D4ED8",
            command=self._refresh)
        self.btn_refresh.pack(side="right", padx=4)

        self.btn_del_sel = ctk.CTkButton(
            toolbar, text="🗑 Xóa đã chọn", width=130,
            fg_color="#DC2626", hover_color="#B91C1C",
            command=self._delete_selected)
        self.btn_del_sel.pack(side="right", padx=4)

        self.btn_del_all = ctk.CTkButton(
            toolbar, text="⚠ Xóa tất cả", width=120,
            fg_color="#7C3AED", hover_color="#6D28D9",
            command=self._delete_all)
        self.btn_del_all.pack(side="right", padx=4)

        self.lbl_count = ctk.CTkLabel(toolbar, text="", text_color="gray")
        self.lbl_count.pack(side="right", padx=10)

        # ── Table (TreeView) ──
        table_frame = ctk.CTkFrame(self)
        table_frame.grid(row=1, column=0, sticky="nsew", padx=(10, 4), pady=4)
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)

        style = ttk.Style()
        style.theme_use("default")
        style.configure("DB.Treeview",
                        background="#1e1e2e",
                        foreground="#cdd6f4",
                        rowheight=28,
                        fieldbackground="#1e1e2e",
                        font=("Consolas", 10))
        style.configure("DB.Treeview.Heading",
                        background="#313244",
                        foreground="#cba6f7",
                        font=("Segoe UI", 10, "bold"))
        style.map("DB.Treeview",
                  background=[("selected", "#45475a")],
                  foreground=[("selected", "#f5c2e7")])

        cols = [c[0] for c in self.COL_DEFS]
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings",
                                 selectmode="extended", style="DB.Treeview")

        for col_id, col_name, col_w in self.COL_DEFS:
            self.tree.heading(col_id, text=col_name, anchor="w")
            self.tree.column(col_id, width=col_w, minwidth=60, anchor="w")

        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")

        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        # ── Preview Panel ──
        preview_frame = ctk.CTkFrame(self, corner_radius=12)
        preview_frame.grid(row=1, column=1, sticky="nsew", padx=(4, 10), pady=4)
        preview_frame.grid_rowconfigure(2, weight=1)
        preview_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(preview_frame, text="🖼 Xem trước ảnh",
                     font=ctk.CTkFont(size=13, weight="bold")).grid(
            row=0, column=0, pady=(12, 4))

        self.lbl_preview = ctk.CTkLabel(
            preview_frame, text="← Chọn một sự kiện\nđể xem ảnh",
            text_color="gray", font=ctk.CTkFont(size=11))
        self.lbl_preview.grid(row=1, column=0, pady=4, sticky="ew")

        self.img_canvas = ctk.CTkLabel(preview_frame, text="")
        self.img_canvas.grid(row=2, column=0, padx=10, pady=6, sticky="nsew")

        # Metadata chi tiết
        self.txt_meta = ctk.CTkTextbox(preview_frame, height=120,
                                       font=ctk.CTkFont(family="Consolas", size=10))
        self.txt_meta.grid(row=3, column=0, padx=10, pady=(4, 12), sticky="ew")
        self.txt_meta.configure(state="disabled")

        # Status bar
        self.lbl_status = ctk.CTkLabel(self, text="Chưa tải dữ liệu",
                                       text_color="gray",
                                       font=ctk.CTkFont(size=10))
        self.lbl_status.grid(row=2, column=0, columnspan=2,
                             sticky="w", padx=14, pady=(0, 6))

    # ── Data operations ────────────────────────────────────────
    def _refresh(self):
        self.btn_refresh.configure(state="disabled", text="⏳ Đang tải...")
        self.lbl_status.configure(text="Đang kết nối Supabase...")

        def _worker():
            events = self.cloud.fetch_events(limit=200)
            self.after(0, lambda: self._populate_table(events))

        threading.Thread(target=_worker, daemon=True).start()

    def _populate_table(self, events):
        self.tree.delete(*self.tree.get_children())
        self._events = events

        for ev in events:
            ts = ev.get("timestamp", "")[:19].replace("T", " ")
            conf = f"{ev.get('confidence', 0):.2f}"
            self.tree.insert("", "end",
                             iid=ev["id"],
                             values=(ts,
                                     ev.get("camera_id", ""),
                                     ev.get("event_type", ""),
                                     conf,
                                     ev["id"]))

        count = len(events)
        self.lbl_count.configure(text=f"{count} sự kiện")
        self.lbl_status.configure(
            text=f"✅ Đã tải {count} sự kiện lúc {datetime.now().strftime('%H:%M:%S')}")
        self.btn_refresh.configure(state="normal", text="🔄 Refresh")

    def _on_select(self, _event=None):
        sel = self.tree.selection()
        if not sel:
            return
        event_id = sel[-1]  # Lấy cái cuối nếu chọn nhiều
        ev = next((e for e in self._events if e["id"] == event_id), None)
        if not ev:
            return

        # Hiện metadata
        self.txt_meta.configure(state="normal")
        self.txt_meta.delete("1.0", "end")
        self.txt_meta.insert("end",
            f"ID        : {ev.get('id','')}\n"
            f"Camera    : {ev.get('camera_id','')}\n"
            f"Loại      : {ev.get('event_type','')}\n"
            f"Tin cậy   : {ev.get('confidence',0):.2f}\n"
            f"Thời gian : {ev.get('timestamp','')[:19]}\n"
            f"URL       : {ev.get('image_url','')}"
        )
        self.txt_meta.configure(state="disabled")

        # Load ảnh trong thread riêng
        url = ev.get("image_url", "")
        if url:
            self.lbl_preview.configure(text="⏳ Đang tải ảnh...")
            self.img_canvas.configure(image=None, text="")

            def _load():
                img = load_image_from_url(url)
                self.after(0, lambda: self._show_preview(img))

            t = threading.Thread(target=_load, daemon=True)
            t.start()

    def _show_preview(self, ctk_img):
        if ctk_img:
            self.lbl_preview.configure(text="")
            self.img_canvas.configure(image=ctk_img, text="")
            self.img_canvas.image = ctk_img  # giữ reference
        else:
            self.lbl_preview.configure(text="❌ Không tải được ảnh")
            self.img_canvas.configure(image=None, text="")

    # ── Delete ─────────────────────────────────────────────────
    def _delete_selected(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Thông báo", "Vui lòng chọn ít nhất 1 sự kiện.")
            return

        n = len(sel)
        if not messagebox.askyesno("Xác nhận xóa",
                                   f"Bạn có chắc muốn xóa {n} sự kiện đã chọn?\n"
                                   f"Ảnh trên Storage cũng sẽ bị xóa vĩnh viễn."):
            return

        to_delete = [e for e in self._events if e["id"] in sel]
        self.btn_del_sel.configure(state="disabled", text="⏳ Đang xóa...")
        self.lbl_status.configure(text="Đang xóa...")

        def _worker():
            ok, fail = self.cloud.delete_events_batch(to_delete)
            self.after(0, lambda: self._after_delete(ok, fail))

        threading.Thread(target=_worker, daemon=True).start()

    def _delete_all(self):
        total = len(self._events)
        if total == 0:
            messagebox.showinfo("Thông báo", "Database đang trống.")
            return

        if not messagebox.askyesno("⚠ XÓA TẤT CẢ",
                                   f"Bạn có CHẮC CHẮN muốn xóa TOÀN BỘ {total} sự kiện?\n"
                                   f"Tất cả ảnh trên Storage cũng sẽ bị xóa.\n\n"
                                   f"Hành động này KHÔNG THỂ HOÀN TÁC!",
                                   icon="warning"):
            return

        self.btn_del_all.configure(state="disabled", text="⏳ Đang xóa...")
        self.lbl_status.configure(text="Đang xóa toàn bộ...")

        def _worker():
            ok, fail = self.cloud.delete_all_events()
            self.after(0, lambda: self._after_delete(ok, fail))

        threading.Thread(target=_worker, daemon=True).start()

    def _after_delete(self, ok, fail):
        self.btn_del_sel.configure(state="normal", text="🗑 Xóa đã chọn")
        self.btn_del_all.configure(state="normal", text="⚠ Xóa tất cả")
        self.lbl_status.configure(
            text=f"✅ Đã xóa {ok} sự kiện" + (f" | ❌ Lỗi {fail}" if fail else ""))
        self._refresh()   # Tải lại bảng


# ══════════════════════════════════════════════════════════════
#  Main Application Window
# ══════════════════════════════════════════════════════════════
class StrokeApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("AI Stroke Detection — v2 (GPU)")
        self.geometry("1400x860")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

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
        self.source = 0
        self.camera_id = "CAM_00"
        self.alert_count = {}
        self.last_alert_time = {}

        self._create_widgets()

    # ── Layout ────────────────────────────────────────────────
    def _create_widgets(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ── Sidebar ──
        sidebar = ctk.CTkFrame(self, width=230, corner_radius=0,
                               fg_color="#13131f")
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)

        ctk.CTkLabel(sidebar, text="🧠 STROKE AI",
                     font=ctk.CTkFont(size=22, weight="bold"),
                     text_color="#a6e3a1").pack(pady=(24, 4))
        ctk.CTkLabel(sidebar, text="Hệ thống phát hiện đột quỵ",
                     font=ctk.CTkFont(size=10),
                     text_color="gray").pack(pady=(0, 20))

        ttk.Separator(sidebar).pack(fill="x", padx=16, pady=4)

        ctk.CTkLabel(sidebar, text="Nguồn video",
                     font=ctk.CTkFont(size=11),
                     text_color="gray").pack(anchor="w", padx=16, pady=(12, 2))

        self.source_option = ctk.CTkOptionMenu(
            sidebar,
            values=["Webcam 0", "Webcam 1", "Video File"],
            command=self._on_source_change,
            fg_color="#1e1e2e", button_color="#2563EB",
            button_hover_color="#1D4ED8")
        self.source_option.pack(pady=4, padx=16, fill="x")

        self.start_btn = ctk.CTkButton(
            sidebar, text="▶  BẮT ĐẦU",
            fg_color="#16a34a", hover_color="#15803d",
            font=ctk.CTkFont(size=13, weight="bold"),
            height=44, corner_radius=10,
            command=self.toggle_engine)
        self.start_btn.pack(pady=16, padx=16, fill="x")

        ttk.Separator(sidebar).pack(fill="x", padx=16, pady=4)

        # Live stats
        ctk.CTkLabel(sidebar, text="Thống kê",
                     font=ctk.CTkFont(size=11),
                     text_color="gray").pack(anchor="w", padx=16, pady=(12, 2))

        self.lbl_fps = ctk.CTkLabel(sidebar, text="FPS: —",
                                    font=ctk.CTkFont(family="Consolas", size=12))
        self.lbl_fps.pack(anchor="w", padx=16, pady=2)

        self.lbl_alerts = ctk.CTkLabel(sidebar, text="Cảnh báo: 0",
                                       font=ctk.CTkFont(family="Consolas", size=12),
                                       text_color="#f38ba8")
        self.lbl_alerts.pack(anchor="w", padx=16, pady=2)

        self.lbl_persons = ctk.CTkLabel(sidebar, text="Người: 0",
                                        font=ctk.CTkFont(family="Consolas", size=12))
        self.lbl_persons.pack(anchor="w", padx=16, pady=2)

        # ── Main content (Tab View) ──
        self.tabs = ctk.CTkTabview(self, corner_radius=10,
                                   fg_color="#1e1e2e",
                                   segmented_button_fg_color="#13131f",
                                   segmented_button_selected_color="#2563EB",
                                   segmented_button_selected_hover_color="#1D4ED8",
                                   command=self._on_tab_change)
        self.tabs.grid(row=0, column=1, padx=16, pady=16, sticky="nsew")

        self.tabs.add("📹  Phát hiện")
        self.tabs.add("📦  Database")
        self.tabs.tab("📹  Phát hiện").grid_columnconfigure(0, weight=1)
        self.tabs.tab("📹  Phát hiện").grid_rowconfigure(0, weight=1)
        self.tabs.tab("📦  Database").grid_columnconfigure(0, weight=1)
        self.tabs.tab("📦  Database").grid_rowconfigure(0, weight=1)

        # ── Detection Tab ──
        det_frame = ctk.CTkFrame(self.tabs.tab("📹  Phát hiện"),
                                 fg_color="transparent")
        det_frame.grid(sticky="nsew")
        det_frame.grid_columnconfigure(0, weight=1)
        det_frame.grid_rowconfigure(0, weight=1)

        self.video_canvas = ctk.CTkLabel(det_frame, text="")
        self.video_canvas.grid(row=0, column=0, sticky="nsew")

        self.log_panel = ctk.CTkTextbox(
            det_frame, height=130,
            font=ctk.CTkFont(family="Consolas", size=10),
            fg_color="#13131f", text_color="#a6e3a1")
        self.log_panel.grid(row=1, column=0, padx=6, pady=(4, 6), sticky="ew")

        # ── Database Tab ──
        self.db_tab = DatabaseTab(
            self.tabs.tab("📦  Database"),
            cloud=self.cloud,
            fg_color="transparent")
        self.db_tab.grid(sticky="nsew")

        self._total_alerts = 0

    # ── Tab switch ─────────────────────────────────────────────
    def _on_tab_change(self):
        if self.tabs.get() == "📦  Database":
            self.db_tab._refresh()

    # ── Logging ────────────────────────────────────────────────
    def log_msg(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_panel.insert("0.0", f"[{ts}] {msg}\n")

    # ── Source control ─────────────────────────────────────────
    def _on_source_change(self, value):
        if value == "Video File":
            self.source = filedialog.askopenfilename(
                filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All", "*.*")])
            if self.source:
                self.camera_id = Path(self.source).name
        else:
            self.source = int(value.split()[-1])
            self.camera_id = f"CAM_{self.source:02d}"
        self.log_msg(f"Nguồn: {self.source} | ID: {self.camera_id}")

    # ── Engine control ─────────────────────────────────────────
    def toggle_engine(self):
        if not self.is_running:
            self.is_running = True
            self.start_btn.configure(text="⏹  DỪNG", fg_color="#DC2626",
                                     hover_color="#B91C1C")
            self.processing_thread = threading.Thread(
                target=self._video_worker, daemon=True)
            self.processing_thread.start()
            self._ui_loop()
        else:
            self.is_running = False
            self.start_btn.configure(text="▶  BẮT ĐẦU",
                                     fg_color="#16a34a", hover_color="#15803d")

    # ── AI worker thread ───────────────────────────────────────
    def _video_worker(self):
        cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        prev_time = time.time()

        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break

            # 1. AI Inference
            results = self.detector.detect(frame)
            active_ids = []

            for i, res in enumerate(results):
                bbox, kpts = res['bbox'], res['kpts']
                track_id = i + 1
                active_ids.append(track_id)

                # 2. Tracker + Recognition
                self.tracker.update_history(track_id, kpts)
                history = self.tracker.get_history(track_id)
                result = self.recognizer.analyze(
                    history, (frame.shape[1], frame.shape[0]),
                    track_id=track_id)   # Truyền track_id cho sustained counter

                # 3. Alert logic
                if result['detected'] and result['risk_level'] == 'high':
                    now = time.time()
                    count = self.alert_count.get(track_id, 0)
                    if count < 3 and (now - self.last_alert_time.get(track_id, 0)) >= 10:
                        current_count = count + 1
                        self._total_alerts += 1
                        self.after(0, lambda c=current_count, s=result['symptom']:
                                   self.log_msg(f"🚨 CẢNH BÁO [{c}/3]: {s}"))
                        self.after(0, lambda:
                                   self.lbl_alerts.configure(
                                       text=f"Cảnh báo: {self._total_alerts}"))

                        # Lưu cục bộ
                        local_path = self.cloud.save_local(frame, track_id, result)
                        if local_path:
                            self.after(0, lambda p=local_path:
                                       self.log_msg(f"💾 Lưu local: {p}"))

                        # Đẩy lên Supabase chỉ ảnh thứ 3
                        if current_count == 3:
                            url, _ = self.cloud.upload_alert(
                                frame, track_id, result,
                                camera_id=self.camera_id)
                            if url:
                                self.after(0, lambda:
                                           self.log_msg("☁ Cloud sync thành công!"))

                        self.alert_count[track_id] = current_count
                        self.last_alert_time[track_id] = now

                # 4. Visualization
                frame = draw_skeleton(frame, kpts)
                frame = draw_info(frame, track_id, bbox, result, self.fps)

            # Clean stale tracks
            self.tracker.clean_old_tracks(active_ids)

            # FPS
            curr_time = time.time()
            self.fps = 1 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time

            # Update sidebar stats
            n_persons = len(results)
            self.after(0, lambda f=self.fps, n=n_persons: (
                self.lbl_fps.configure(text=f"FPS: {f:.1f}"),
                self.lbl_persons.configure(text=f"Người: {n}")
            ))

            if not self.frame_queue.full():
                self.frame_queue.put(frame)

        cap.release()

    # ── UI render loop ─────────────────────────────────────────
    def _ui_loop(self):
        if not self.is_running:
            return

        if not self.frame_queue.empty():
            frame = self.frame_queue.get()
            cw = self.video_canvas.winfo_width()
            ch = self.video_canvas.winfo_height()
            if cw > 1 and ch > 1:
                frame = cv2.resize(frame, (cw, ch))
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img,
                                   size=(cw, ch))
            self.video_canvas._ctk_image = ctk_img  # giữ reference
            self.video_canvas.configure(image=ctk_img)

        self.after(1, self._ui_loop)


if __name__ == "__main__":
    app = StrokeApp()
    app.mainloop()
