"""
AI Safety Dashboard v4 — Stroke + Airport Security
- Multi-threaded GPU detection with ByteTrack
- Async cloud upload (no pipeline blocking)
- ObjectDetector chạy mỗi 3 frames (tiết kiệm GPU)
- Abandoned Baggage + Weapon Detection
"""
import cv2
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import customtkinter as ctk
from PIL import Image, ImageTk
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import requests
from io import BytesIO
from pathlib import Path
from queue import Queue
from datetime import datetime

# AI Engine — Stroke
from app.ai.detector import PoseDetector
from app.ai.tracker import Tracker
from app.ai.recognizer_v2 import StrokeRecognizerV2, StrokeConfig
# AI Engine — Airport
from app.ai.object_detector import ObjectDetector, BAGGAGE_CLASS_IDS
from app.ai.baggage_tracker import AbandonedBaggageTracker
from app.ai.weapon_detector import WeaponDetector
# Cloud
from app.cloud.supabase import SupabaseClient
from app.cloud.airport_cloud import AirportCloudClient
# Visualization
from app.utils.visualization import draw_info, draw_skeleton, draw_fps
from app.utils.airport_viz import draw_baggage_overlays, draw_weapon_alerts, draw_airport_stats


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
#  Tab: Airport Database Manager
# ══════════════════════════════════════════════════════════════
class AirportDatabaseTab(ctk.CTkFrame):
    """Panel xem và quản lý sự kiện airport_events từ Supabase."""

    COL_DEFS = [
        ("created_at",   "Thời gian",    160),
        ("camera_id",    "Camera",        70),
        ("event_type",   "Loại",         160),
        ("object_class", "Đối tượng",     90),
        ("risk_level",   "Mức độ",        70),
        ("confidence",   "Độ tin cậy",    90),
        ("duration_sec", "Thời gian(s)",  90),
        ("resolved",     "Đã xử lý",      70),
        ("id",           "ID",           250),
    ]

    def __init__(self, master, airport_cloud, **kwargs):
        super().__init__(master, **kwargs)
        self.cloud   = airport_cloud
        self._events = []
        self._filter = None   # None = tất cả
        self._build_ui()

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(1, weight=1)

        # Toolbar
        toolbar = ctk.CTkFrame(self, height=50, fg_color="transparent")
        toolbar.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=(10, 4))

        ctk.CTkLabel(toolbar, text="🛡️ Airport Security Events",
                     font=ctk.CTkFont(size=16, weight="bold"),
                     text_color="#fab387").pack(side="left", padx=8)

        self.btn_refresh = ctk.CTkButton(
            toolbar, text="🔄 Refresh", width=110,
            fg_color="#2563EB", hover_color="#1D4ED8",
            command=self._refresh)
        self.btn_refresh.pack(side="right", padx=4)

        self.btn_resolve = ctk.CTkButton(
            toolbar, text="✅ Đã xử lý", width=110,
            fg_color="#16a34a", hover_color="#15803d",
            command=self._mark_resolved)
        self.btn_resolve.pack(side="right", padx=4)

        self.btn_del_sel = ctk.CTkButton(
            toolbar, text="🗑 Xóa chọn", width=110,
            fg_color="#DC2626", hover_color="#B91C1C",
            command=self._delete_selected)
        self.btn_del_sel.pack(side="right", padx=4)

        # Filter buttons
        filter_frame = ctk.CTkFrame(toolbar, fg_color="transparent")
        filter_frame.pack(side="right", padx=8)
        for label, val in [("Tất cả", None), ("Hành lý", "abandoned_baggage"), ("Vũ khí", "weapon_detected")]:
            ctk.CTkButton(
                filter_frame, text=label, width=80,
                fg_color="#374151", hover_color="#4B5563",
                command=lambda v=val: self._set_filter(v)
            ).pack(side="left", padx=2)

        self.lbl_count = ctk.CTkLabel(toolbar, text="", text_color="gray")
        self.lbl_count.pack(side="right", padx=10)

        # Table
        table_frame = ctk.CTkFrame(self)
        table_frame.grid(row=1, column=0, sticky="nsew", padx=(10, 4), pady=4)
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)

        style = ttk.Style()
        style.configure("Airport.Treeview",
                        background="#1a1a2e",
                        foreground="#e2e8f0",
                        rowheight=26,
                        fieldbackground="#1a1a2e",
                        font=("Consolas", 10))
        style.configure("Airport.Treeview.Heading",
                        background="#2d1b69",
                        foreground="#fab387",
                        font=("Segoe UI", 10, "bold"))
        style.map("Airport.Treeview",
                  background=[("selected", "#4a3728")],
                  foreground=[("selected", "#fbbf24")])

        cols = [c[0] for c in self.COL_DEFS]
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings",
                                 selectmode="extended", style="Airport.Treeview")
        for col_id, col_name, col_w in self.COL_DEFS:
            self.tree.heading(col_id, text=col_name, anchor="w")
            self.tree.column(col_id, width=col_w, minwidth=50, anchor="w")

        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        # Row tags (màu theo risk_level)
        self.tree.tag_configure("critical", foreground="#ef4444")
        self.tree.tag_configure("high",     foreground="#f97316")
        self.tree.tag_configure("resolved", foreground="#6b7280")

        # Preview Panel
        preview_frame = ctk.CTkFrame(self, corner_radius=12)
        preview_frame.grid(row=1, column=1, sticky="nsew", padx=(4, 10), pady=4)
        preview_frame.grid_rowconfigure(2, weight=1)
        preview_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(preview_frame, text="🖼 Ảnh sự kiện",
                     font=ctk.CTkFont(size=13, weight="bold")).grid(
            row=0, column=0, pady=(12, 4))

        self.lbl_preview = ctk.CTkLabel(
            preview_frame, text="← Chọn sự kiện để xem",
            text_color="gray", font=ctk.CTkFont(size=11))
        self.lbl_preview.grid(row=1, column=0, pady=4)

        self.img_canvas = ctk.CTkLabel(preview_frame, text="")
        self.img_canvas.grid(row=2, column=0, padx=10, pady=6, sticky="nsew")

        self.txt_meta = ctk.CTkTextbox(preview_frame, height=140,
                                       font=ctk.CTkFont(family="Consolas", size=10))
        self.txt_meta.grid(row=3, column=0, padx=10, pady=(4, 12), sticky="ew")
        self.txt_meta.configure(state="disabled")

        self.lbl_status = ctk.CTkLabel(self, text="Chưa tải", text_color="gray",
                                       font=ctk.CTkFont(size=10))
        self.lbl_status.grid(row=2, column=0, columnspan=2,
                             sticky="w", padx=14, pady=(0, 6))

    def _set_filter(self, val):
        self._filter = val
        self._refresh()

    def _refresh(self):
        self.btn_refresh.configure(state="disabled", text="⏳ Đang tải...")
        self.lbl_status.configure(text="Đang kết nối Supabase...")

        def _worker():
            events = self.cloud.fetch_airport_events(
                limit=200, event_type=self._filter)
            self.after(0, lambda: self._populate(events))

        threading.Thread(target=_worker, daemon=True).start()

    def _populate(self, events):
        self.tree.delete(*self.tree.get_children())
        self._events = events

        for ev in events:
            ts       = ev.get("created_at", "")[:19].replace("T", " ")
            conf     = f"{ev.get('confidence', 0):.2f}"
            dur      = f"{ev.get('duration_sec', 0):.0f}s"
            resolved = "✅" if ev.get("resolved") else "—"
            risk     = ev.get("risk_level", "")
            tag      = "resolved" if ev.get("resolved") else risk

            self.tree.insert("", "end", iid=ev["id"],
                             values=(ts,
                                     ev.get("camera_id", ""),
                                     ev.get("event_type", ""),
                                     ev.get("object_class", ""),
                                     risk,
                                     conf,
                                     dur,
                                     resolved,
                                     ev["id"]),
                             tags=(tag,))

        n = len(events)
        self.lbl_count.configure(text=f"{n} sự kiện")
        self.lbl_status.configure(
            text=f"✅ Tải {n} sự kiện lúc {datetime.now().strftime('%H:%M:%S')}")
        self.btn_refresh.configure(state="normal", text="🔄 Refresh")

    def _on_select(self, _=None):
        sel = self.tree.selection()
        if not sel:
            return
        ev = next((e for e in self._events if e["id"] == sel[-1]), None)
        if not ev:
            return

        meta = ev.get("metadata", {})
        self.txt_meta.configure(state="normal")
        self.txt_meta.delete("1.0", "end")
        self.txt_meta.insert("end",
            f"ID         : {ev.get('id', '')}\n"
            f"Loại       : {ev.get('event_type', '')}\n"
            f"Đối tượng  : {ev.get('object_class', '')}\n"
            f"Risk       : {ev.get('risk_level', '')}\n"
            f"Confidence : {ev.get('confidence', 0):.2f}\n"
            f"Duration   : {ev.get('duration_sec', 0):.1f}s\n"
            f"Zone       : {ev.get('zone_name', '—')}\n"
            f"Camera     : {ev.get('camera_id', '')}\n"
            f"BBox       : {meta.get('bbox', '—')}\n"
            f"Resolved   : {ev.get('resolved', False)}\n"
            f"Thời gian  : {ev.get('created_at', '')[:19]}\n"
            f"URL        : {ev.get('image_url', '')}"
        )
        self.txt_meta.configure(state="disabled")

        url = ev.get("image_url", "")
        if url:
            self.lbl_preview.configure(text="⏳ Đang tải ảnh...")
            self.img_canvas.configure(image=None, text="")

            def _load():
                img = load_image_from_url(url)
                self.after(0, lambda: self._show_img(img))

            threading.Thread(target=_load, daemon=True).start()

    def _show_img(self, ctk_img):
        if ctk_img:
            self.lbl_preview.configure(text="")
            self.img_canvas.configure(image=ctk_img, text="")
            self.img_canvas.image = ctk_img
        else:
            self.lbl_preview.configure(text="❌ Không tải được ảnh")

    def _mark_resolved(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Thông báo", "Vui lòng chọn ít nhất 1 sự kiện.")
            return
        for eid in sel:
            self.cloud.mark_resolved(eid)
        self.lbl_status.configure(text=f"✅ Đã đánh dấu {len(sel)} sự kiện xử lý")
        self._refresh()

    def _delete_selected(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Thông báo", "Vui lòng chọn ít nhất 1 sự kiện.")
            return
        if not messagebox.askyesno("Xác nhận xóa",
                                   f"Xóa {len(sel)} sự kiện và ảnh Storage?"):
            return
        to_del = [e for e in self._events if e["id"] in sel]
        self.btn_del_sel.configure(state="disabled", text="⏳ Đang xóa...")

        def _worker():
            ok, fail = self.cloud.delete_airport_events_batch(to_del)
            self.after(0, lambda: self._after_delete(ok, fail))

        threading.Thread(target=_worker, daemon=True).start()

    def _after_delete(self, ok, fail):
        self.btn_del_sel.configure(state="normal", text="🗑 Xóa chọn")
        self.lbl_status.configure(
            text=f"✅ Đã xóa {ok}" + (f" | ❌ Lỗi {fail}" if fail else ""))
        self._refresh()


# ══════════════════════════════════════════════════════════════
#  Main Application Window
# ══════════════════════════════════════════════════════════════
class StrokeApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Hệ thống nhận diện nguy hiểm bằng AI")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # ── Responsive window: mở rộng tối đa ngay khi khởi động ──
        # Trên màn hình lớn và laptop đều hiển thị đầy đủ không bị khuất
        self.update_idletasks()                      # bắt buộc tk render xong
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        # Kích thước tối thiểu đảm bảo GUI không bị vỡ bố cục
        self.minsize(900, 580)
        # Mở cửa sổ ở trạng thái maximized → tự động fit với mọi độ phân giải
        self.state('zoomed')
        # Lưu thông tin màn hình để các widget dùng
        self._screen_w = sw
        self._screen_h = sh
        # Chiều rộng sidebar: nhỏ hơn trên màn hình nhỏ
        self._sidebar_w = 200 if sw < 1400 else 230
        # Chiều cao log panel: gọn hơn trên màn hình nhỏ
        self._log_h = 90 if sh < 800 else 120

        # ── Stroke Engines ────────────────────────────────────
        self.detector   = PoseDetector(input_size=640)
        self.tracker    = Tracker(max_history=30)
        self.recognizer = StrokeRecognizerV2(debug=False)  # v2 with improved detection

        # ── Airport Engines ───────────────────────────────────
        self.obj_detector    = ObjectDetector(model_path='yolov8n.pt', object_skip=3)
        self.baggage_tracker = AbandonedBaggageTracker(camera_id='CAM_00')
        self.weapon_detector = WeaponDetector(
            self.obj_detector,
            conf=0.15,          # ngưỡng thấp để nhận dao dễ hơn
            bearer_radius=200,  # tăng rá dò để bắt tốt hơn
            cooldown=5.0,       # upload mỗi 5 giây
        )

        # ── Cloud ─────────────────────────────────────────────
        self.cloud         = SupabaseClient()       # stroke_events
        self.airport_cloud = AirportCloudClient()   # airport_events

        # ── Threading & Queues ────────────────────────────────
        self.frame_queue      = Queue(maxsize=2)
        self.is_running       = False
        self.processing_thread = None
        self._upload_pool     = ThreadPoolExecutor(max_workers=3, thread_name_prefix="cloud")

        # ── Frame counters ────────────────────────────────────
        self.FRAME_SKIP      = 2
        self._frame_counter  = 0
        self._last_results   = []
        self._last_obj_results = []   # cache object detection
        # DB sync throttle: upsert baggage_tracks mỗi 90 frames
        self._db_sync_counter = 0
        self.DB_SYNC_EVERY    = 90
        self.adaptive_mode    = True  # Tự động điều chỉnh frame skip để giữ FPS ổn định
        self._adaptive_counter = 0


        # ── Stats ─────────────────────────────────────────────
        self.fps             = 0
        self.source          = 0
        self.camera_id       = "CAM_00"
        self.alert_count     = {}
        self.last_alert_time = {}
        self._total_alerts   = 0
        self._airport_alerts = 0   # counter airport alerts
        # Cache kết quả nhận diện per-track (dùng khi frame skip)
        self._last_person_results: dict[int, dict] = {}

        # Weapon alert overlay: lấy tự weapon_detector.get_active_overlays() mỗi frame
        # Không cần _weapon_flash state nữa — overlay là liên tục

        self._create_widgets()

    # ── Layout ────────────────────────────────────────────────
    def _create_widgets(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ── Sidebar ──
        sidebar = ctk.CTkFrame(self, width=self._sidebar_w, corner_radius=0,
                               fg_color="#13131f")
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)

        ctk.CTkLabel(sidebar, text="🚨 DANGER AI",
                     font=ctk.CTkFont(size=22, weight="bold"),
                     text_color="#f38ba8").pack(pady=(24, 4))
        ctk.CTkLabel(sidebar, text="Hệ thống nhận diện nguy hiểm bằng AI",
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

        self.lbl_airport = ctk.CTkLabel(
            sidebar, text="Hành lý: 0  |  Vật thể: 0",
            font=ctk.CTkFont(family="Consolas", size=11),
            text_color="#fab387")
        self.lbl_airport.pack(anchor="w", padx=16, pady=2)

        ttk.Separator(sidebar).pack(fill="x", padx=16, pady=4)

        # Frame-skip slider
        ctk.CTkLabel(sidebar, text="Frame Skip (1=mọi frame)",
                     font=ctk.CTkFont(size=10), text_color="gray").pack(
            anchor="w", padx=16, pady=(10, 0))

        self.skip_slider = ctk.CTkSlider(
            sidebar, from_=1, to=4, number_of_steps=3,
            command=self._on_skip_change)
        self.skip_slider.set(self.FRAME_SKIP)
        self.skip_slider.pack(padx=16, pady=4, fill="x")

        self.lbl_skip = ctk.CTkLabel(
            sidebar, text=f"Skip={self.FRAME_SKIP}",
            font=ctk.CTkFont(family="Consolas", size=11))
        self.lbl_skip.pack(anchor="w", padx=16)

        # Baggage Timeout Control
        ctk.CTkLabel(sidebar, text="Timeout Hành lý (giây)",
                     font=ctk.CTkFont(size=10), text_color="gray").pack(
            anchor="w", padx=16, pady=(10, 0))

        init_timeout = int(self.baggage_tracker.timeout)
        self.timeout_slider = ctk.CTkSlider(
            sidebar, from_=3, to=120, number_of_steps=117,
            command=self._on_timeout_change)
        self.timeout_slider.set(init_timeout)
        self.timeout_slider.pack(padx=16, pady=4, fill="x")

        self.lbl_timeout = ctk.CTkLabel(
            sidebar, text=f"Timeout={init_timeout}s",
            font=ctk.CTkFont(family="Consolas", size=11))
        self.lbl_timeout.pack(anchor="w", padx=16)

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
        self.tabs.add("🛡️  Airport DB")
        for tab_name in ["📹  Phát hiện", "📦  Database", "🛡️  Airport DB"]:
            self.tabs.tab(tab_name).grid_columnconfigure(0, weight=1)
            self.tabs.tab(tab_name).grid_rowconfigure(0, weight=1)

        # ── Detection Tab ──
        det_frame = ctk.CTkFrame(self.tabs.tab("📹  Phát hiện"),
                                 fg_color="transparent")
        det_frame.grid(sticky="nsew")
        det_frame.grid_columnconfigure(0, weight=1)
        det_frame.grid_rowconfigure(0, weight=1)

        # Dùng tk.Canvas thay vì CTkLabel để hiển thị video
        # Canvas có kích thước CỐ ĐỊNH theo grid → winfo_width/height()
        # luôn trả về kích thước thật của vùng chứa, co giãn theo cửa sổ.
        # CTkLabel cũ tự phình theo ảnh → vòng resize không bao giờ thu nhỏ.
        self.video_canvas = tk.Canvas(
            det_frame,
            bg='black',
            highlightthickness=0,   # không viền xanh của tk.Canvas
            bd=0
        )
        self.video_canvas.grid(row=0, column=0, sticky="nsew")
        self._det_frame = det_frame
        self._video_photo = None   # giữ reference ImageTk.PhotoImage

        self.log_panel = ctk.CTkTextbox(
            det_frame, height=self._log_h,
            font=ctk.CTkFont(family="Consolas", size=10),
            fg_color="#13131f", text_color="#a6e3a1")
        self.log_panel.grid(row=1, column=0, padx=6, pady=(4, 6), sticky="ew")

        # ── Stroke Database Tab ──
        self.db_tab = DatabaseTab(
            self.tabs.tab("📦  Database"),
            cloud=self.cloud,
            fg_color="transparent")
        self.db_tab.grid(sticky="nsew")

        # ── Airport Database Tab ──
        self.airport_db_tab = AirportDatabaseTab(
            self.tabs.tab("🛡️  Airport DB"),
            airport_cloud=self.airport_cloud,
            fg_color="transparent")
        self.airport_db_tab.grid(sticky="nsew")


    # ── Tab switch ─────────────────────────────────────────────
    def _on_tab_change(self):
        current = self.tabs.get()
        if current == "📦  Database":
            self.db_tab._refresh()
        elif current == "🛡️  Airport DB":
            self.airport_db_tab._refresh()

    # ── Frame skip control ─────────────────────────────────────
    def _on_skip_change(self, value):
        self.FRAME_SKIP = max(1, int(value))
        self.adaptive_mode = False
        self.lbl_skip.configure(text=f"Skip={self.FRAME_SKIP} (Manual)")

    # ── Baggage timeout control ────────────────────────────────
    def _on_timeout_change(self, value):
        timeout_val = int(value)
        self.baggage_tracker.timeout = timeout_val
        self.lbl_timeout.configure(text=f"Timeout={timeout_val}s")



    # ── Clean shutdown ─────────────────────────────────────────
    def _on_close(self):
        self.is_running = False
        self._upload_pool.shutdown(wait=False)
        self.destroy()

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
        # Cập nhật camera_id cho airport engines
        self.baggage_tracker.update_camera(self.camera_id)
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

    # ── Async cloud upload (Stroke) ─────────────────────────────
    def _async_upload(self, frame_copy, track_id, result, current_count):
        """Stroke events — chạy trong thread pool, không block pipeline."""
        local_path = self.cloud.save_local(frame_copy, track_id, result)
        if local_path:
            self.after(0, lambda p=local_path: self.log_msg(f"💾 Lưu local: {p}"))
        if current_count == 3:
            url, _ = self.cloud.upload_alert(frame_copy, track_id, result,
                                              camera_id=self.camera_id)
            if url:
                self.after(0, lambda: self.log_msg("☁ Stroke cloud sync!"))

    # ── Async cloud upload (Airport) ────────────────────────────
    def _async_airport_upload(self, frame_copy, alert: dict):
        """Airport alert — upload ảnh + insert airport_events."""
        url, path = self.airport_cloud.upload_airport_alert(
            frame_copy, alert, camera_id=self.camera_id
        )
        if url:
            etype = alert.get('event_type', '')
            emoji = "🗃" if 'baggage' in etype else "⚔️"
            self.after(0, lambda: self.log_msg(
                f"{emoji} Airport alert ↑ Supabase: {alert.get('object_class')}"
            ))
        # Upsert baggage_tracks realtime (dirty states)
        dirty = self.baggage_tracker.pop_dirty()
        if dirty:
            self.airport_cloud.upsert_baggage_tracks(dirty)


    # ── AI worker thread ───────────────────────────────────────
    def _video_worker(self):

        cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        prev_time = time.time()
        self.obj_detector.reset_skip_counter()

        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break

            self._frame_counter += 1
            run_inference = (self._frame_counter % self.FRAME_SKIP == 0)

            # ── POSE inference (Stroke) ──────────────────────
            if run_inference:
                results = self.detector.track(frame)
                self._last_results = results
            else:
                results = self._last_results

            # ── OBJECT inference (Airport, mỗi 3 frames) ────
            # Hạ threshold gốc xuống 0.20 để bắt được các vật thể mờ/nhỏ/che khuất như dao, balo
            obj_results, obj_ran = self.obj_detector.track(
                frame,
                classes=[24, 26, 28, 43, 76],
                conf=0.20,
            )

            if obj_ran:
                self._last_obj_results = obj_results
            else:
                obj_results = self._last_obj_results

            # ── Stroke detection loop ─────────────────────────
            active_ids = []
            for res in results:
                bbox     = res['bbox']
                kpts     = res['kpts']
                track_id = res['track_id']
                active_ids.append(track_id)

                if run_inference:
                    self.tracker.update_history(track_id, kpts)
                    history = self.tracker.get_history(track_id)
                    result  = self.recognizer.analyze(
                        history, (frame.shape[1], frame.shape[0]),
                        track_id=track_id)

                    if result['detected'] and result['risk_level'] == 'high':
                        now   = time.time()
                        count = self.alert_count.get(track_id, 0)
                        # Chup anh moi 5s, toi da 3 lan — lan thu 3 gui len DB
                        if count < 3 and (now - self.last_alert_time.get(track_id, 0)) >= 5:
                            cnt = count + 1
                            self._total_alerts += 1
                            self.alert_count[track_id]     = cnt
                            self.last_alert_time[track_id] = now
                            self.after(0, lambda c=cnt, s=result['symptom']:
                                       self.log_msg(f"🚨 CẢNH BÁO [{c}/3]: {s}"))
                            self.after(0, lambda:
                                       self.lbl_alerts.configure(
                                           text=f"Cảnh báo: {self._total_alerts}"))
                            frame_copy = frame.copy()
                            self._upload_pool.submit(
                                self._async_upload, frame_copy, track_id, result, cnt)
                    # Lưu kết quả vào cache riêng per-track (không lưu trong res dict)
                    self._last_person_results[track_id] = result
                else:
                    # Frame skip: lấy kết quả cache của từng người riêng biệt
                    # → mỗi track_id có kết quả độc lập, không bị ghi đè bởi người khác
                    result = self._last_person_results.get(
                        track_id, self.recognizer._result(False, 0.0, 'Normal', 'low'))

                frame = draw_skeleton(frame, kpts)
                frame = draw_info(frame, track_id, bbox, result)

            if run_inference:
                self.tracker.clean_old_tracks(active_ids)
                # Dọn state recognizer CHỈ KHI Tracker đã thực sự xóa track
                # (sau grace period _GRACE_FRAMES=15 frame vắng mặt).
                # KHÔNG xóa ngay khi YOLO bỏ sót 1-2 frame — như vậy
                # history vẫn được giữ nguyên và nhận diện không bị chập chờn.
                still_in_tracker = set(self.tracker.track_history.keys())
                all_cached = list(self._last_person_results.keys())
                for lost_tid in all_cached:
                    if lost_tid not in still_in_tracker:
                        self.recognizer._clear_state(lost_tid)
                        self._last_person_results.pop(lost_tid, None)

            # ── Airport detection (mỗi lần obj_ran) ──────────
            weapon_alerts = []
            if obj_ran:
                baggage_alerts = self.baggage_tracker.update(obj_results, results)
                for alert in baggage_alerts:
                    self._airport_alerts += 1
                    self.after(0, lambda a=alert:
                               self.log_msg(
                                   f"📄 HÀNH LÝ BỎ LẠI: "
                                   f"{a['object_class']} ({a['duration_sec']:.0f}s)"))
                    self._upload_pool.submit(
                        self._async_airport_upload, frame.copy(), alert)

                weapon_alerts = self.weapon_detector.detect_frame(
                    obj_results, results, camera_id=self.camera_id)

                for alert in weapon_alerts:
                    self._airport_alerts += 1
                    self.after(0, lambda a=alert:
                               self.log_msg(
                                   f"WEAPON: "
                                   f"{a['object_class']} conf={a['confidence']:.0%}"))
                    self._upload_pool.submit(
                        self._async_airport_upload, frame.copy(), alert)

            else:
                # Frame skip: vẫn chạy detect_frame với cache để cập nhật active_detections
                self.weapon_detector.detect_frame(
                    self._last_obj_results, results, camera_id=self.camera_id)

                # Periodic DB sync
                self._db_sync_counter += 1
                if self._db_sync_counter >= self.DB_SYNC_EVERY:
                    self._db_sync_counter = 0
                    dirty = self.baggage_tracker.pop_dirty()
                    if dirty:
                        self._upload_pool.submit(
                            self.airport_cloud.upsert_baggage_tracks, dirty)
                    active_bag_ids = list(self.baggage_tracker.get_all_states().keys())
                    self._upload_pool.submit(
                        self.airport_cloud.clean_baggage_tracks, active_bag_ids)

                states    = self.baggage_tracker.get_all_states()
                n_bags    = len(states)
                n_abandon = sum(1 for s in states.values() if s.alerted)
                n_w       = len(self.weapon_detector.get_active_overlays())
                self.after(0, lambda b=n_bags, a=n_abandon, w=n_w:
                           self.lbl_airport.configure(
                               text=f"Hành lý: {b} (!{a})  Vật thể: {w}"))

            # ── Airport visualization ─────────────────────────
            states = self.baggage_tracker.get_all_states()
            frame  = draw_baggage_overlays(frame, states, abandon_timeout=self.baggage_tracker.timeout)

            # Overlay vũ khí liên tục: lấy từ persistent state, không phụ thuộc cooldown
            active_weapons = self.weapon_detector.get_active_overlays()
            if active_weapons:
                frame = draw_weapon_alerts(frame, active_weapons)

            n_w_hud = len(active_weapons)
            frame = draw_airport_stats(
                frame, len(states),
                sum(1 for s in states.values() if s.alerted),
                n_w_hud)

            frame = draw_fps(frame, self.fps)

            curr_time = time.time()
            self.fps  = 1.0 / (curr_time - prev_time + 1e-9)
            prev_time = curr_time

            # Adaptive Frame Skip logic (mỗi 30 frames)
            if self.adaptive_mode:
                self._adaptive_counter += 1
                if self._adaptive_counter >= 30:
                    self._adaptive_counter = 0
                    if self.fps < 18:
                        # FPS quá thấp → Tăng skip để giảm tải GPU
                        old_skip = self.FRAME_SKIP
                        self.FRAME_SKIP = min(4, self.FRAME_SKIP + 1)
                        self.obj_detector.object_skip = min(6, self.obj_detector.object_skip + 1)
                        if self.FRAME_SKIP != old_skip:
                            self.after(0, lambda s=self.FRAME_SKIP: (
                                self.lbl_skip.configure(text=f"Skip={s} (Auto)"),
                                self.skip_slider.set(s)
                            ))
                    elif self.fps > 26 and self.FRAME_SKIP > 1:
                        # FPS dư dả → Giảm skip để nhận diện mượt hơn
                        old_skip = self.FRAME_SKIP
                        self.FRAME_SKIP = max(1, self.FRAME_SKIP - 1)
                        self.obj_detector.object_skip = max(3, self.obj_detector.object_skip - 1)
                        if self.FRAME_SKIP != old_skip:
                            self.after(0, lambda s=self.FRAME_SKIP: (
                                self.lbl_skip.configure(text=f"Skip={s} (Auto)"),
                                self.skip_slider.set(s)
                            ))

            if run_inference:
                n_persons = len(results)
                # Hiển thị FPS kèm chế độ adaptive
                mode_str = " (Auto)" if self.adaptive_mode else " (Manual)"
                self.after(0, lambda f=self.fps, n=n_persons, m=mode_str: (
                    self.lbl_fps.configure(text=f"FPS: {f:.1f}"),
                    self.lbl_persons.configure(text=f"Người: {n}"),
                    self.lbl_skip.configure(text=f"Skip={self.FRAME_SKIP}{m}")
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

            # tk.Canvas winfo_width/height() = kích thước thực của canvas
            # (không bị ảnh hưởng bởi nội dung vẽ bên trong)
            # → tự động co giãn đúng khi thay đổi kích thước cửa sổ
            cw = self.video_canvas.winfo_width()
            ch = self.video_canvas.winfo_height()

            if cw > 4 and ch > 4:
                fh, fw = frame.shape[:2]
                # Letterbox: vừa khít, giữ tỉ lệ, không bóp không cắt
                ratio = min(cw / fw, ch / fh)
                nw    = max(1, int(fw * ratio))
                nh    = max(1, int(fh * ratio))
                frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
            else:
                nw, nh = frame.shape[1], frame.shape[0]
                cw, ch = nw, nh

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # ImageTk.PhotoImage — phương pháp đúng cho tk.Canvas
            self._video_photo = ImageTk.PhotoImage(image=img)

            # Vẽ ảnh vào giữa canvas (letterbox → hai bên có viền đen tự động)
            self.video_canvas.delete("all")
            self.video_canvas.create_image(
                cw // 2, ch // 2,
                anchor="center",
                image=self._video_photo
            )

        self.after(33, self._ui_loop)   # ~30fps render rate


if __name__ == "__main__":
    app = StrokeApp()
    app.mainloop()
