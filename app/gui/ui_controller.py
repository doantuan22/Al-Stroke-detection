"""
Small UI facade for thread-safe widget updates from worker code.
"""
from __future__ import annotations

from datetime import datetime


class UIController:
    def __init__(self, app):
        self.app = app

    def call(self, func, *args, **kwargs):
        self.app.after(0, lambda: func(*args, **kwargs))

    def log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.app.log_panel.insert("0.0", f"[{ts}] {msg}\n")

    def log_async(self, msg: str):
        self.call(self.log, msg)

    def set_running_button(self, running: bool):
        if running:
            self.app.start_btn.configure(
                text="⏹  DUNG", fg_color="#DC2626", hover_color="#B91C1C"
            )
        else:
            self.app.start_btn.configure(
                text="▶  BAT DAU", fg_color="#16a34a", hover_color="#15803d"
            )

    def update_live_stats(self, fps: float, persons: int, frame_skip: int, mode: str):
        self.app.lbl_fps.configure(text=f"FPS: {fps:.1f}")
        self.app.lbl_persons.configure(text=f"Nguoi: {persons}")
        self.app.lbl_skip.configure(text=f"Skip={frame_skip}{mode}")

    def update_airport_stats(self, bags: int, abandoned: int, weapons: int):
        self.app.lbl_airport.configure(
            text=f"Hanh ly: {bags} (!{abandoned})  Vat the: {weapons}"
        )

    def update_alert_count(self, total: int):
        self.app.lbl_alerts.configure(text=f"Canh bao: {total}")

