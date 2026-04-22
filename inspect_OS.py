import sys
import os
import cv2
import time
import serial
import serial.tools.list_ports
import io
import tempfile
from datetime import datetime
from ultralytics import YOLO
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QTextEdit, QFrame,
    QSlider, QGroupBox, QComboBox, QScrollArea,
    QTabWidget, QSizePolicy, QProgressBar,
    QGraphicsDropShadowEffect, QGridLayout
)
from PySide6.QtCore import QTimer, Qt, QSize, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QImage, QPixmap, QFont, QColor, QPainter, QPen, QBrush, QLinearGradient

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ===============================
# LOAD YOLO MODEL
# ===============================
_base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
model = YOLO(os.path.join(_base_path, "best.pt"))

PILL_ID    = 1
MISSING_ID = 0

INSPECT_DURATION_MS = 5000
ACTION_DURATION_MS  = 10000
MIN_FAIL_VOTES_PCT  = 0.5


def _classify_severity(fail_votes, total_votes, max_missing, avg_confidence):
    """Rule-based severity: CRITICAL / MAJOR / MINOR / WARNING / OK."""
    if total_votes == 0:
        return "OK"
    ratio = fail_votes / total_votes
    if max_missing >= 3 or ratio > 0.85:
        return "CRITICAL"
    if max_missing >= 2 or ratio > 0.7:
        return "MAJOR"
    if max_missing >= 1 and ratio > 0.5:
        return "MINOR"
    if ratio > 0.3 or avg_confidence < 0.6:
        return "WARNING"
    return "OK"


_SEV_COLORS = {
    "CRITICAL": "#ff1744", "MAJOR": "#ff5722",
    "MINOR": "#ff9800", "WARNING": "#ffc107", "OK": "#4caf50",
}


# =========================================================
# DESIGN TOKENS — modern dark theme palette
# =========================================================
class T:
    # Backgrounds
    BG          = "#0a0e1a"   # app background
    BG_SIDE     = "#0d1220"   # side-rail background
    CARD        = "#141a2e"   # card surface
    CARD_HI     = "#1b2240"   # elevated surface (hover / input)
    CARD_LO     = "#0f1423"   # sunken surface (console / pre)
    BORDER      = "#242c48"   # subtle border
    BORDER_HI   = "#3a4775"   # focused border

    # Text
    TEXT        = "#e6ebff"
    TEXT_DIM    = "#8b93b7"
    TEXT_MUTED  = "#5a6487"

    # Brand + semantic
    PRIMARY     = "#4ea1ff"   # brand cyan-blue
    PRIMARY_HI  = "#7ab8ff"
    ACCENT      = "#9b6bff"   # brand purple
    SUCCESS     = "#2ee59d"
    DANGER      = "#ff4d6d"
    WARNING     = "#ffb020"
    INFO        = "#4cc9f0"


class CircularProgress(QWidget):
    """Lightweight ring-progress widget used on the Live tab."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(170, 170)
        self._value = 0.0          # 0.0–1.0
        self._label = "—"
        self._sub   = "IDLE"
        self._color = QColor(T.PRIMARY)

    def set_state(self, value, label, sub, color):
        self._value = max(0.0, min(1.0, float(value)))
        self._label = label
        self._sub   = sub
        if isinstance(color, str):
            color = QColor(color)
        self._color = color
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        side = min(self.width(), self.height())
        pad  = 10
        rect = self.rect().adjusted(pad, pad, -pad, -pad)

        # Track
        pen = QPen(QColor(T.BORDER), 10, Qt.SolidLine, Qt.RoundCap)
        p.setPen(pen)
        p.drawArc(rect, 0, 360 * 16)

        # Progress arc (Qt: 0° at 3-o'clock, CCW positive — use -angle for CW)
        pen.setColor(self._color)
        p.setPen(pen)
        span = int(-360 * 16 * self._value)
        p.drawArc(rect, 90 * 16, span)

        # Center big label
        p.setPen(QColor(T.TEXT))
        f = QFont("Segoe UI", 22, QFont.Bold)
        p.setFont(f)
        p.drawText(rect, Qt.AlignCenter, self._label)

        # Sub label
        p.setPen(QColor(T.TEXT_DIM))
        p.setFont(QFont("Segoe UI", 9, QFont.DemiBold))
        sub_rect = rect.adjusted(0, 46, 0, 0)
        p.drawText(sub_rect, Qt.AlignHCenter | Qt.AlignTop, self._sub)
        p.end()


def _shadow(widget, radius=24, alpha=140, dy=4):
    eff = QGraphicsDropShadowEffect(widget)
    eff.setBlurRadius(radius)
    eff.setXOffset(0)
    eff.setYOffset(dy)
    eff.setColor(QColor(0, 0, 0, alpha))
    widget.setGraphicsEffect(eff)


class InspectOS(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("INSPECT-OS — Blister Pack Inspection")
        self.setMinimumSize(1400, 860)

        # --- Arduino ---
        self.arduino       = None
        self.arduino_ready = False
        self.hw_last_heartbeat = 0

        # --- Camera ---
        self.cap   = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # --- Serial reader ---
        self.serial_timer = QTimer()
        self.serial_timer.timeout.connect(self.read_serial)
        self.serial_timer.start(50)

        # --- Heartbeat watchdog ---
        self.hb_timer = QTimer()
        self.hb_timer.timeout.connect(self.check_heartbeat)
        self.hb_timer.start(3000)

        # --- Clock ---
        self.clock_timer = QTimer()
        self.clock_timer.timeout.connect(self.update_clock)
        self.clock_timer.start(1000)

        # --- State Machine ---
        self.state = "IDLE"

        self.vote_buffer      = []
        self.inspect_timer    = QTimer()
        self.inspect_timer.setSingleShot(True)
        self.inspect_timer.timeout.connect(self.finish_inspection)

        self.action_result    = None
        self.action_timer     = QTimer()
        self.action_timer.setSingleShot(True)
        self.action_timer.timeout.connect(self.finish_action)

        self.countdown_end_ms = 0

        # --- Counters ---
        self.total  = 0
        self.passed = 0
        self.failed = 0

        # --- Motor speed ---
        self.motor_speed = 70

        # --- Detection gating ---
        self.detect_count  = 0
        self.detect_needed = 3
        self.awaiting_clear = False
        self.clear_count   = 0

        # --- Phase 1: Data Infrastructure ---
        self.inspection_log = []           # list of inspection record dicts
        self.session_start  = datetime.now().isoformat()
        self._charts_stale  = False
        # Per-frame accumulators (reset each inspection cycle)
        self._frame_pills       = []
        self._frame_missing     = []
        self._frame_confidences = []
        self._frame_max_missing = 0

        self.build_ui()

    # ===============================
    # UI
    # ===============================
    def build_ui(self):
        self.setStyleSheet(f"""
        /* ----- Global ----- */
        QWidget {{
            background-color: {T.BG};
            color: {T.TEXT};
            font-family: "Segoe UI", "Inter", "SF Pro Text", sans-serif;
            font-size: 13px;
        }}

        /* ----- Cards ----- */
        QFrame#card {{
            background: {T.CARD};
            border: 1px solid {T.BORDER};
            border-radius: 14px;
        }}
        QFrame#sideRail {{
            background: {T.BG_SIDE};
            border: none;
        }}
        QFrame#header {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #111830, stop:0.6 #151c38, stop:1 #1a1f44);
            border-bottom: 1px solid {T.BORDER};
        }}
        QFrame#statCard {{
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 {T.CARD_HI}, stop:1 {T.CARD});
            border: 1px solid {T.BORDER};
            border-radius: 12px;
        }}
        QLabel#cardTitle {{
            color: {T.TEXT_DIM};
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            padding: 2px 0;
        }}
        QLabel#sectionHint {{
            color: {T.TEXT_MUTED};
            font-size: 11px;
        }}
        QLabel#brandTitle {{
            color: {T.TEXT};
            font-size: 22px;
            font-weight: 800;
            letter-spacing: 2px;
        }}
        QLabel#brandSub {{
            color: {T.TEXT_DIM};
            font-size: 11px;
            letter-spacing: 3px;
        }}
        QLabel#clock {{
            color: {T.TEXT};
            font-family: "JetBrains Mono", Consolas, monospace;
            font-size: 18px;
            font-weight: 600;
            letter-spacing: 2px;
        }}
        QLabel#statBig {{
            color: {T.TEXT};
            font-size: 26px;
            font-weight: 800;
        }}
        QLabel#statLabel {{
            color: {T.TEXT_DIM};
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 1.3px;
        }}

        /* ----- Buttons ----- */
        QPushButton {{
            padding: 10px 14px;
            font-size: 13px;
            font-weight: 600;
            border-radius: 10px;
            border: 1px solid {T.BORDER};
            background: {T.CARD_HI};
            color: {T.TEXT};
        }}
        QPushButton:hover   {{ background: #262e52; border-color: {T.BORDER_HI}; }}
        QPushButton:pressed {{ background: #1a2044; }}
        QPushButton:disabled {{
            color: {T.TEXT_MUTED};
            background: {T.CARD};
            border-color: {T.BORDER};
        }}
        QPushButton#start {{
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #25d07b, stop:1 #17a860);
            color: #001a10; border: 1px solid #2ee59d; font-weight: 800;
        }}
        QPushButton#start:hover {{ background: #2ee59d; }}
        QPushButton#stop {{
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #ff5277, stop:1 #d32e52);
            color: white; border: 1px solid #ff6b87; font-weight: 800;
        }}
        QPushButton#stop:hover {{ background: #ff6b87; }}
        QPushButton#primary {{
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 {T.PRIMARY_HI}, stop:1 {T.PRIMARY});
            color: #03142b; border: 1px solid {T.PRIMARY_HI}; font-weight: 800;
        }}
        QPushButton#primary:hover {{ background: {T.PRIMARY_HI}; }}
        QPushButton#accent {{
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #b589ff, stop:1 {T.ACCENT});
            color: white; border: 1px solid #b589ff; font-weight: 800;
        }}
        QPushButton#accent:hover {{ background: #b589ff; }}
        QPushButton#warn {{
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #ffc759, stop:1 {T.WARNING});
            color: #1e1400; border: 1px solid #ffc759; font-weight: 800;
        }}
        QPushButton#warn:hover {{ background: #ffc759; }}
        QPushButton#iconBtn {{
            padding: 6px; font-size: 15px; min-width: 36px; max-width: 36px;
        }}

        /* ----- Inputs ----- */
        QComboBox {{
            background: {T.CARD_HI};
            color: {T.TEXT};
            border: 1px solid {T.BORDER};
            border-radius: 8px;
            padding: 7px 10px;
            font-size: 12px;
        }}
        QComboBox:hover {{ border-color: {T.BORDER_HI}; }}
        QComboBox::drop-down {{ border: none; width: 22px; }}
        QComboBox QAbstractItemView {{
            background: {T.CARD_HI};
            color: {T.TEXT};
            border: 1px solid {T.BORDER_HI};
            selection-background-color: {T.PRIMARY};
            selection-color: #03142b;
            outline: 0;
        }}

        /* ----- Slider ----- */
        QSlider::groove:horizontal   {{ background: {T.CARD_LO}; height: 6px; border-radius: 3px; }}
        QSlider::handle:horizontal   {{
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 {T.PRIMARY_HI}, stop:1 {T.PRIMARY});
            width: 18px; margin: -7px 0; border-radius: 9px;
            border: 2px solid #0a1224;
        }}
        QSlider::sub-page:horizontal {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 {T.ACCENT}, stop:1 {T.PRIMARY});
            border-radius: 3px;
        }}

        /* ----- Tabs ----- */
        QTabWidget::pane {{
            border: 1px solid {T.BORDER};
            background: {T.CARD};
            border-radius: 12px;
            top: -1px;
        }}
        QTabBar {{ qproperty-drawBase: 0; background: transparent; }}
        QTabBar::tab {{
            background: transparent;
            color: {T.TEXT_DIM};
            padding: 9px 18px;
            border: none;
            margin-right: 4px;
            font-weight: 600;
            font-size: 12px;
            letter-spacing: 0.5px;
        }}
        QTabBar::tab:selected {{
            color: {T.TEXT};
            background: {T.CARD_HI};
            border-radius: 10px;
            border: 1px solid {T.BORDER_HI};
        }}
        QTabBar::tab:hover:!selected {{
            color: {T.TEXT};
        }}

        /* ----- Text areas ----- */
        QTextEdit {{
            background: {T.CARD_LO};
            border: 1px solid {T.BORDER};
            border-radius: 10px;
            color: #cde3ff;
            font-family: "JetBrains Mono", "Cascadia Code", Consolas, monospace;
            font-size: 11px;
            padding: 8px;
        }}
        QTextEdit#console {{
            background: #070b16;
            color: #7ae0b5;
            border: 1px solid {T.BORDER};
            border-radius: 12px;
        }}

        /* ----- Scroll ----- */
        QScrollArea {{ border: none; background: transparent; }}
        QScrollBar:vertical {{
            background: transparent; width: 10px; margin: 2px;
        }}
        QScrollBar::handle:vertical {{
            background: {T.BORDER_HI}; border-radius: 5px; min-height: 30px;
        }}
        QScrollBar::handle:vertical:hover {{ background: {T.PRIMARY}; }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
        }}

        /* ----- GroupBox (kept hidden — cards replace it) ----- */
        QGroupBox {{ border: none; margin: 0; padding: 0; }}
        """)

        # =======================================================
        # HEADER — gradient branding bar with status pill & clock
        # =======================================================
        header_frame = QFrame()
        header_frame.setObjectName("header")
        header_frame.setFixedHeight(68)
        header = QHBoxLayout(header_frame)
        header.setContentsMargins(22, 10, 22, 10)
        header.setSpacing(14)

        logo = QLabel("◉")
        logo.setStyleSheet(f"color:{T.PRIMARY}; font-size:28px;")
        logo.setFixedWidth(34)
        header.addWidget(logo)

        brand_col = QVBoxLayout()
        brand_col.setSpacing(0)
        brand_title = QLabel("INSPECT-OS")
        brand_title.setObjectName("brandTitle")
        brand_sub   = QLabel("BLISTER PACK VISION INSPECTION")
        brand_sub.setObjectName("brandSub")
        brand_col.addWidget(brand_title)
        brand_col.addWidget(brand_sub)
        header.addLayout(brand_col)
        header.addStretch()

        # Live status pill (updated in set_state_label)
        self.status_pill = QLabel("● IDLE")
        self.status_pill.setAlignment(Qt.AlignCenter)
        self.status_pill.setFixedHeight(30)
        self.status_pill.setStyleSheet(
            f"padding:4px 14px; border-radius:15px; "
            f"background:{T.CARD_HI}; color:{T.TEXT_DIM}; "
            f"font-weight:700; font-size:11px; letter-spacing:1.3px;"
        )
        header.addWidget(self.status_pill)

        self.clock_label = QLabel("00:00:00")
        self.clock_label.setObjectName("clock")
        header.addWidget(self.clock_label)

        # =======================================================
        # LEFT SIDE-RAIL — cards
        # =======================================================
        left = QVBoxLayout()
        left.setSpacing(12)
        left.setContentsMargins(14, 14, 14, 14)

        # --- Arduino Connection card ---
        conn_card = self._card("Arduino Connection")
        conn_body = conn_card.body

        port_row = QHBoxLayout()
        self.port_combo        = QComboBox()
        self.refresh_ports_btn = QPushButton("⟳")
        self.refresh_ports_btn.setObjectName("iconBtn")
        self.refresh_ports_btn.setToolTip("Refresh ports")
        port_row.addWidget(self.port_combo, 1)
        port_row.addWidget(self.refresh_ports_btn)
        conn_body.addLayout(port_row)

        btn_row = QHBoxLayout()
        self.connect_btn    = QPushButton("CONNECT")
        self.connect_btn.setObjectName("primary")
        self.disconnect_btn = QPushButton("DISCONNECT")
        self.disconnect_btn.setEnabled(False)
        btn_row.addWidget(self.connect_btn)
        btn_row.addWidget(self.disconnect_btn)
        conn_body.addLayout(btn_row)

        self.arduino_lbl = QLabel("● DISCONNECTED")
        self.arduino_lbl.setStyleSheet(
            f"color:{T.DANGER}; font-weight:700; font-size:11px; "
            f"background:rgba(255,77,109,0.08); padding:6px 10px; border-radius:8px;"
        )
        self.arduino_lbl.setAlignment(Qt.AlignCenter)
        conn_body.addWidget(self.arduino_lbl)
        left.addWidget(conn_card)

        # --- System controls card ---
        ctrl_card = self._card("System Controls")
        self.start_btn = QPushButton("▶   START")
        self.start_btn.setObjectName("start")
        self.start_btn.setMinimumHeight(44)
        self.stop_btn  = QPushButton("■   STOP")
        self.stop_btn.setObjectName("stop")
        self.stop_btn.setMinimumHeight(44)
        self.test_servo_btn = QPushButton("⟳   TEST SERVO")
        self.test_servo_btn.setObjectName("warn")
        ctrl_card.body.addWidget(self.start_btn)
        ctrl_card.body.addWidget(self.stop_btn)
        ctrl_card.body.addWidget(self.test_servo_btn)
        left.addWidget(ctrl_card)

        # --- Hardware indicators card ---
        hw_card = self._card("Hardware Status")
        hw_grid = QGridLayout()
        hw_grid.setHorizontalSpacing(8)
        hw_grid.setVerticalSpacing(8)
        self.ind_arduino = self._make_indicator("Arduino")
        self.ind_camera  = self._make_indicator("Camera")
        self.ind_motor   = self._make_indicator("Motor")
        self.ind_servo   = self._make_indicator("Servo")
        pairs = [self.ind_arduino, self.ind_camera, self.ind_motor, self.ind_servo]
        for i, (chip, _dot) in enumerate(pairs):
            hw_grid.addWidget(chip, i // 2, i % 2)
        hw_card.body.addLayout(hw_grid)
        left.addWidget(hw_card)

        # --- Motor speed card ---
        spd_card = self._card("Belt Speed")
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(0, 100)
        self.speed_slider.setValue(self.motor_speed)
        self.speed_slider.setTickInterval(10)
        self.speed_lbl = QLabel(f"{self.motor_speed}%")
        self.speed_lbl.setAlignment(Qt.AlignCenter)
        self.speed_lbl.setStyleSheet(
            f"color:{T.PRIMARY_HI}; font-weight:800; font-size:22px; "
            f"font-family:'JetBrains Mono',Consolas,monospace;"
        )
        spd_card.body.addWidget(self.speed_lbl)
        spd_card.body.addWidget(self.speed_slider)
        scale_row = QHBoxLayout()
        for tick in ("0", "25", "50", "75", "100"):
            t = QLabel(tick)
            t.setStyleSheet(f"color:{T.TEXT_MUTED}; font-size:9px;")
            t.setAlignment(Qt.AlignCenter)
            scale_row.addWidget(t)
        spd_card.body.addLayout(scale_row)
        left.addWidget(spd_card)

        # --- Run statistics: 2x2 grid of stat tiles ---
        stats_card = self._card("Run Statistics")
        stats_grid = QGridLayout()
        stats_grid.setHorizontalSpacing(8)
        stats_grid.setVerticalSpacing(8)
        self.tile_total  = self._stat_tile("TOTAL",  "0", T.PRIMARY)
        self.tile_passed = self._stat_tile("PASSED", "0", T.SUCCESS)
        self.tile_failed = self._stat_tile("FAILED", "0", T.DANGER)
        self.tile_yield  = self._stat_tile("YIELD",  "0.0%", T.ACCENT)
        stats_grid.addWidget(self.tile_total[0],  0, 0)
        stats_grid.addWidget(self.tile_passed[0], 0, 1)
        stats_grid.addWidget(self.tile_failed[0], 1, 0)
        stats_grid.addWidget(self.tile_yield[0],  1, 1)
        stats_card.body.addLayout(stats_grid)
        left.addWidget(stats_card)

        left.addStretch()

        scroll_content = QWidget()
        scroll_content.setLayout(left)
        scroll = QScrollArea()
        scroll.setWidget(scroll_content)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        left_panel = QFrame()
        left_panel.setObjectName("sideRail")
        lp_layout  = QVBoxLayout(left_panel)
        lp_layout.setContentsMargins(0, 0, 0, 0)
        lp_layout.addWidget(scroll)
        left_panel.setFixedWidth(320)

        # =======================================================
        # CAMERA VIEWPORT
        # =======================================================
        cam_wrap = QFrame()
        cam_wrap.setObjectName("card")
        cam_layout = QVBoxLayout(cam_wrap)
        cam_layout.setContentsMargins(12, 12, 12, 12)
        cam_layout.setSpacing(8)

        # Top HUD bar above camera
        hud = QHBoxLayout()
        cam_title = QLabel("LIVE VISION FEED")
        cam_title.setStyleSheet(
            f"color:{T.TEXT_DIM}; font-size:10px; font-weight:800; letter-spacing:2px;"
        )
        self.rec_dot = QLabel("●  OFFLINE")
        self.rec_dot.setStyleSheet(
            f"color:{T.TEXT_MUTED}; font-size:10px; font-weight:700; letter-spacing:1.5px;"
        )
        hud.addWidget(cam_title)
        hud.addStretch()
        hud.addWidget(self.rec_dot)
        cam_layout.addLayout(hud)

        self.camera_view = QLabel("◉  AWAITING CAMERA SIGNAL")
        self.camera_view.setAlignment(Qt.AlignCenter)
        self.camera_view.setMinimumHeight(420)
        self.camera_view.setStyleSheet(
            f"background:#05080f; "
            f"border:1px solid {T.BORDER}; border-radius:10px; "
            f"font-size:16px; color:{T.TEXT_MUTED}; letter-spacing:2px;"
        )
        cam_layout.addWidget(self.camera_view, 1)

        # Progress bar sits beneath the camera feed
        self.inspect_progress = QProgressBar()
        self.inspect_progress.setRange(0, 100)
        self.inspect_progress.setValue(0)
        self.inspect_progress.setTextVisible(False)
        self.inspect_progress.setFixedHeight(6)
        self.inspect_progress.setStyleSheet(f"""
            QProgressBar {{
                background: {T.CARD_LO};
                border: none; border-radius: 3px;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {T.ACCENT}, stop:1 {T.PRIMARY});
                border-radius: 3px;
            }}
        """)
        cam_layout.addWidget(self.inspect_progress)

        # =======================================================
        # RIGHT PANEL — TABS
        # =======================================================
        self.right_tabs = QTabWidget()
        self.right_tabs.setFixedWidth(400)
        self.right_tabs.setDocumentMode(True)

        # --- TAB 1: LIVE ---
        live_tab = QWidget()
        live_layout = QVBoxLayout(live_tab)
        live_layout.setContentsMargins(14, 14, 14, 14)
        live_layout.setSpacing(12)

        # Result card (gradient surface, huge text)
        result_card = QFrame()
        result_card.setObjectName("card")
        rc_layout = QVBoxLayout(result_card)
        rc_layout.setContentsMargins(16, 18, 16, 18)
        rc_layout.setSpacing(6)
        rc_title = QLabel("INSPECTION RESULT")
        rc_title.setObjectName("cardTitle")
        rc_title.setAlignment(Qt.AlignCenter)
        rc_layout.addWidget(rc_title)

        self.result_lbl = QLabel("READY")
        self.result_lbl.setAlignment(Qt.AlignCenter)
        self.result_lbl.setFont(QFont("Segoe UI", 44, QFont.Black))
        self.result_lbl.setStyleSheet(f"color:{T.PRIMARY}; padding:4px 0;")
        rc_layout.addWidget(self.result_lbl)

        self.severity_lbl = QLabel("")
        self.severity_lbl.setAlignment(Qt.AlignCenter)
        self.severity_lbl.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.severity_lbl.setStyleSheet(
            f"color:{T.TEXT_MUTED}; padding:4px 10px; letter-spacing:1.2px;"
        )
        rc_layout.addWidget(self.severity_lbl)
        live_layout.addWidget(result_card)

        # Circular cycle progress + state
        ring_card = QFrame()
        ring_card.setObjectName("card")
        rng_layout = QVBoxLayout(ring_card)
        rng_layout.setContentsMargins(12, 14, 12, 14)
        rng_layout.setSpacing(8)
        rng_title = QLabel("CYCLE STATE")
        rng_title.setObjectName("cardTitle")
        rng_title.setAlignment(Qt.AlignCenter)
        rng_layout.addWidget(rng_title)

        self.cycle_ring = CircularProgress()
        ring_row = QHBoxLayout()
        ring_row.addStretch()
        ring_row.addWidget(self.cycle_ring)
        ring_row.addStretch()
        rng_layout.addLayout(ring_row)

        self.state_lbl = QLabel("IDLE")
        self.state_lbl.setAlignment(Qt.AlignCenter)
        self.state_lbl.setStyleSheet(
            f"font-size:12px; font-weight:800; color:{T.TEXT_DIM}; "
            f"letter-spacing:2px; padding-top:4px;"
        )
        rng_layout.addWidget(self.state_lbl)

        # Kept for compatibility — hidden
        self.countdown_lbl = QLabel("")
        self.countdown_lbl.hide()
        rng_layout.addWidget(self.countdown_lbl)

        live_layout.addWidget(ring_card)

        # Arduino HW feedback card
        hw_card_live = QFrame()
        hw_card_live.setObjectName("card")
        hwl = QVBoxLayout(hw_card_live)
        hwl.setContentsMargins(14, 12, 14, 12)
        hwl.setSpacing(6)
        hwt = QLabel("HARDWARE FEEDBACK")
        hwt.setObjectName("cardTitle")
        hwl.addWidget(hwt)
        self.hw_feedback_lbl = QLabel("Arduino: waiting…")
        self.hw_feedback_lbl.setWordWrap(True)
        self.hw_feedback_lbl.setStyleSheet(
            f"color:{T.TEXT_DIM}; font-family:'JetBrains Mono',Consolas,monospace; "
            f"font-size:11px; background:{T.CARD_LO}; "
            f"border:1px solid {T.BORDER}; border-radius:8px; padding:8px 10px;"
        )
        hwl.addWidget(self.hw_feedback_lbl)
        live_layout.addWidget(hw_card_live)

        live_layout.addStretch()
        self.right_tabs.addTab(live_tab, "Live")

        # --- TAB 2: DASHBOARD ---
        dash_tab = QWidget()
        dash_layout = QVBoxLayout(dash_tab)
        dash_layout.setContentsMargins(10, 10, 10, 10)
        dash_layout.setSpacing(10)

        chart_bg = T.CARD
        self.pie_fig    = Figure(figsize=(3.4, 2.2), dpi=80)
        self.yield_fig  = Figure(figsize=(3.4, 2.2), dpi=80)
        self.sev_fig    = Figure(figsize=(3.4, 2.2), dpi=80)
        self.hourly_fig = Figure(figsize=(3.4, 2.2), dpi=80)
        for fig, attr in [
            (self.pie_fig,    "pie_canvas"),
            (self.yield_fig,  "yield_canvas"),
            (self.sev_fig,    "sev_canvas"),
            (self.hourly_fig, "hourly_canvas"),
        ]:
            fig.patch.set_facecolor(chart_bg)
            canvas = FigureCanvas(fig)
            canvas.setStyleSheet(
                f"background:{chart_bg}; border:1px solid {T.BORDER}; border-radius:10px;"
            )
            setattr(self, attr, canvas)
            dash_layout.addWidget(canvas)

        dash_scroll = QScrollArea()
        dash_scroll.setWidget(dash_tab)
        dash_scroll.setWidgetResizable(True)
        self.right_tabs.addTab(dash_scroll, "Dashboard")
        self._draw_empty_charts()

        # --- TAB 3: REPORTS ---
        report_tab = QWidget()
        report_layout = QVBoxLayout(report_tab)
        report_layout.setContentsMargins(14, 14, 14, 14)
        report_layout.setSpacing(10)

        report_title = QLabel("Inspection Reports")
        report_title.setStyleSheet(
            f"color:{T.TEXT}; font-size:16px; font-weight:800; letter-spacing:0.5px;"
        )
        report_sub = QLabel("Session summary, severity breakdown, and per-sheet log")
        report_sub.setObjectName("sectionHint")
        report_layout.addWidget(report_title)
        report_layout.addWidget(report_sub)

        self.report_preview = QTextEdit()
        self.report_preview.setReadOnly(True)
        self.report_preview.setPlaceholderText(
            "Click 'Preview Report' to see the inspection summary…"
        )
        report_layout.addWidget(self.report_preview, 1)

        report_btn_row = QHBoxLayout()
        self.preview_report_btn = QPushButton("⊙   PREVIEW")
        self.preview_report_btn.setObjectName("primary")
        self.generate_pdf_btn = QPushButton("⇩   SAVE PDF")
        self.generate_pdf_btn.setObjectName("accent")
        report_btn_row.addWidget(self.preview_report_btn)
        report_btn_row.addWidget(self.generate_pdf_btn)
        report_layout.addLayout(report_btn_row)

        self.report_status_lbl = QLabel("")
        self.report_status_lbl.setStyleSheet(f"color:{T.TEXT_MUTED}; font-size:11px;")
        report_layout.addWidget(self.report_status_lbl)

        self.right_tabs.addTab(report_tab, "Reports")

        # =======================================================
        # LOG CONSOLE
        # =======================================================
        log_wrap = QFrame()
        log_wrap.setObjectName("card")
        log_v = QVBoxLayout(log_wrap)
        log_v.setContentsMargins(12, 10, 12, 12)
        log_v.setSpacing(6)
        log_h = QHBoxLayout()
        log_title = QLabel("SYSTEM LOG")
        log_title.setObjectName("cardTitle")
        log_h.addWidget(log_title)
        log_h.addStretch()
        self.log_counter = QLabel("0 events")
        self.log_counter.setStyleSheet(f"color:{T.TEXT_MUTED}; font-size:10px;")
        log_h.addWidget(self.log_counter)
        log_v.addLayout(log_h)

        self.log = QTextEdit()
        self.log.setObjectName("console")
        self.log.setReadOnly(True)
        self.log.setFixedHeight(130)
        self.log.setPlaceholderText("Waiting for system events…")
        log_v.addWidget(self.log)

        # =======================================================
        # ASSEMBLE
        # =======================================================
        top = QHBoxLayout()
        top.setSpacing(14)
        top.setContentsMargins(14, 14, 14, 0)
        top.addWidget(left_panel)
        top.addWidget(cam_wrap, 1)
        top.addWidget(self.right_tabs)

        log_outer = QHBoxLayout()
        log_outer.setContentsMargins(14, 10, 14, 14)
        log_outer.addWidget(log_wrap)

        main = QVBoxLayout(self)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)
        main.addWidget(header_frame)
        main.addLayout(top, 1)
        main.addLayout(log_outer)

        # Signals
        self.start_btn.clicked.connect(self.start_system)
        self.stop_btn.clicked.connect(self.stop_system)
        self.test_servo_btn.clicked.connect(self.test_servo)
        self.speed_slider.valueChanged.connect(self.on_speed_changed)
        self.refresh_ports_btn.clicked.connect(self.refresh_ports)
        self.connect_btn.clicked.connect(self.connect_arduino)
        self.disconnect_btn.clicked.connect(self.disconnect_arduino)
        self.preview_report_btn.clicked.connect(self.preview_report)
        self.generate_pdf_btn.clicked.connect(self.generate_report_pdf)
        self.right_tabs.currentChanged.connect(self._on_tab_changed)

        # Initial visual sync
        self.set_state_label("IDLE — waiting", T.TEXT_DIM)
        self.cycle_ring.set_state(0.0, "—", "IDLE", T.PRIMARY)

        self.refresh_ports()

    # ===============================
    # HELPERS
    # ===============================
    def _card(self, title_text):
        """Create a titled card frame. Returns the frame; `.body` is the content layout."""
        card = QFrame()
        card.setObjectName("card")
        v = QVBoxLayout(card)
        v.setContentsMargins(14, 12, 14, 14)
        v.setSpacing(10)
        title = QLabel(title_text.upper())
        title.setObjectName("cardTitle")
        v.addWidget(title)
        body = QVBoxLayout()
        body.setSpacing(8)
        v.addLayout(body)
        card.body = body
        _shadow(card, radius=18, alpha=110, dy=3)
        return card

    def _stat_tile(self, label, value, accent):
        """A small stat tile used in the run-stats grid. Returns (frame, value_label)."""
        f = QFrame()
        f.setObjectName("statCard")
        v = QVBoxLayout(f)
        v.setContentsMargins(12, 10, 12, 10)
        v.setSpacing(2)
        val = QLabel(value)
        val.setObjectName("statBig")
        val.setStyleSheet(f"color:{accent}; font-size:22px; font-weight:800;")
        lab = QLabel(label)
        lab.setObjectName("statLabel")
        v.addWidget(val)
        v.addWidget(lab)
        return (f, val)

    def _make_indicator(self, name):
        """A pill-shaped indicator with name + status dot."""
        chip = QFrame()
        chip.setStyleSheet(
            f"background:{T.CARD_LO}; border:1px solid {T.BORDER}; border-radius:10px;"
        )
        h = QHBoxLayout(chip)
        h.setContentsMargins(10, 6, 10, 6)
        h.setSpacing(8)
        dot = QLabel("●")
        dot.setStyleSheet(f"color:{T.TEXT_MUTED}; font-size:16px;")
        lbl = QLabel(name)
        lbl.setStyleSheet(f"color:{T.TEXT}; font-size:11px; font-weight:700; letter-spacing:0.5px;")
        h.addWidget(dot)
        h.addWidget(lbl)
        h.addStretch()
        return chip, dot

    def _set_indicator(self, tup, state):
        colors = {
            "on":   T.SUCCESS,
            "warn": T.WARNING,
            "off":  T.TEXT_MUTED,
            "err":  T.DANGER,
        }
        tup[1].setStyleSheet(f"color:{colors.get(state, T.TEXT_MUTED)}; font-size:16px;")

    def update_clock(self):
        self.clock_label.setText(datetime.now().strftime("%H:%M:%S"))

    def _on_tab_changed(self, index):
        """Refresh charts when switching to Dashboard tab if data changed."""
        if index == 1 and self._charts_stale:
            self._charts_stale = False
            QTimer.singleShot(50, self._refresh_charts)

    def log_event(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        lower = msg.lower()
        if "error" in lower or "fail" in lower:
            color = T.DANGER
        elif "warn" in lower or "⚠" in msg:
            color = T.WARNING
        elif "pass" in lower or "ready" in lower or "connected" in lower:
            color = T.SUCCESS
        else:
            color = "#9fd1ff"
        self.log.append(
            f"<span style='color:{T.TEXT_MUTED}'>[{ts}]</span> "
            f"<span style='color:{color}'>{msg}</span>"
        )
        if hasattr(self, "log_counter"):
            doc = self.log.document()
            self.log_counter.setText(f"{doc.blockCount()} events")

    def set_state_label(self, text, color=None):
        color = color or T.TEXT_DIM
        # Compact state label inside the cycle-state card
        self.state_lbl.setText(text.upper())
        self.state_lbl.setStyleSheet(
            f"font-size:12px; font-weight:800; color:{color}; "
            f"letter-spacing:2px; padding-top:4px;"
        )
        # Sync header status pill
        first_word = text.split("—", 1)[0].strip().split()[0].upper() if text else "IDLE"
        bg = QColor(color)
        bg.setAlpha(50)
        bg_css = f"rgba({bg.red()},{bg.green()},{bg.blue()},60)"
        self.status_pill.setText(f"● {first_word}")
        self.status_pill.setStyleSheet(
            f"padding:4px 14px; border-radius:15px; "
            f"background:{bg_css}; color:{color}; "
            f"font-weight:700; font-size:11px; letter-spacing:1.3px; "
            f"border:1px solid {color};"
        )

    def update_hw_feedback(self, text):
        ts = datetime.now().strftime("%H:%M:%S")
        self.hw_feedback_lbl.setText(f"[{ts}]  {text}")

    # ===============================
    # ARDUINO
    # ===============================
    def refresh_ports(self):
        self.port_combo.clear()
        ports = serial.tools.list_ports.comports()
        for p in ports:
            self.port_combo.addItem(f"{p.device} — {p.description}", p.device)
        if not ports:
            self.port_combo.addItem("No ports found", "")

    def connect_arduino(self):
        port = self.port_combo.currentData()
        if not port:
            self.log_event("ERROR: no port selected")
            return
        try:
            self.arduino = serial.Serial(port, 9600, timeout=0.1)
            time.sleep(2)
            self.arduino_ready     = True
            self.hw_last_heartbeat = time.time()
            self.arduino_lbl.setText(f"● CONNECTED · {port}")
            self.arduino_lbl.setStyleSheet(
                f"color:{T.SUCCESS}; font-weight:700; font-size:11px; "
                f"background:rgba(46,229,157,0.10); padding:6px 10px; border-radius:8px;"
            )
            self._set_indicator(self.ind_arduino, "on")
            self.connect_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(True)
            self.port_combo.setEnabled(False)
            self.log_event(f"Arduino connected on {port}")
        except Exception as e:
            self.log_event(f"ERROR: {e}")
            self._set_indicator(self.ind_arduino, "err")

    def disconnect_arduino(self):
        if self.arduino and self.arduino.is_open:
            self.arduino.close()
        self.arduino       = None
        self.arduino_ready = False
        self.arduino_lbl.setText("● DISCONNECTED")
        self.arduino_lbl.setStyleSheet(
            f"color:{T.DANGER}; font-weight:700; font-size:11px; "
            f"background:rgba(255,77,109,0.08); padding:6px 10px; border-radius:8px;"
        )
        self._set_indicator(self.ind_arduino, "off")
        self._set_indicator(self.ind_motor,   "off")
        self._set_indicator(self.ind_servo,   "off")
        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)
        self.port_combo.setEnabled(True)
        self.log_event("Arduino disconnected")

    def send(self, cmd):
        """Send a command string to Arduino."""
        if self.arduino and self.arduino.is_open:
            self.arduino.write(f"{cmd}\n".encode())

    # ===============================
    # SERIAL READER
    # ===============================
    def read_serial(self):
        if not self.arduino or not self.arduino.is_open:
            return
        try:
            while self.arduino.in_waiting > 0:
                line = self.arduino.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                self.hw_last_heartbeat = time.time()
                self.handle_arduino_msg(line)
        except (serial.SerialException, OSError) as e:
            self.log_event(f"Serial error: {e}")
            self.disconnect_arduino()

    def handle_arduino_msg(self, msg):
        if msg == "READY":
            self.log_event("Arduino: READY")
            self._set_indicator(self.ind_arduino, "on")
            self.update_hw_feedback("Arduino ready")
            # Arduino rebooted mid-action — re-send motor command to recover
            if self.state == "ACTING":
                self.send(f"SET_SPEED:{self.motor_speed}")
                self.send("START")
                self.log_event("⚠ Arduino rebooted mid-action — motor re-started")
            else:
                # Spontaneous reboot in IDLE — ensure motor is off
                self.send("STOP")
                self.log_event("⚠ Arduino rebooted in IDLE — STOP sent to kill motor")
        elif msg == "MOTOR_ON":
            self._set_indicator(self.ind_motor, "on")
            self.update_hw_feedback("Motor ON")
        elif msg == "MOTOR_OFF":
            self._set_indicator(self.ind_motor, "off")
            self.update_hw_feedback("Motor OFF")
        elif msg == "FLAP_OPEN":
            self._set_indicator(self.ind_servo, "warn")
            self.update_hw_feedback("Servo OPEN")
        elif msg == "FLAP_CLOSED":
            self._set_indicator(self.ind_servo, "off")
            self.update_hw_feedback("Servo CLOSED")
        elif msg == "HB":
            pass
        else:
            self.update_hw_feedback(msg)

    def check_heartbeat(self):
        if not self.arduino_ready:
            return
        if time.time() - self.hw_last_heartbeat > 10:
            self._set_indicator(self.ind_arduino, "warn")

    # ===============================
    # SYSTEM START / STOP
    # ===============================
    def start_system(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if self.cap.isOpened():
            self._set_indicator(self.ind_camera, "on")
            self.rec_dot.setText("●  LIVE")
            self.rec_dot.setStyleSheet(
                f"color:{T.DANGER}; font-size:10px; font-weight:700; letter-spacing:1.5px;"
            )
            self.log_event("Camera opened — place sheet under camera to begin")
        else:
            self._set_indicator(self.ind_camera, "err")
            self.log_event("ERROR: camera not found")
            return

        # Make sure motor is OFF at start
        self.send("STOP")
        self.send(f"SET_SPEED:{self.motor_speed}")

        self.state = "IDLE"
        self.set_state_label("IDLE — waiting for sheet", T.TEXT_DIM)
        self.cycle_ring.set_state(0.0, "—", "WAITING", T.PRIMARY)
        self.timer.start(33)   # ~30fps

    def stop_system(self):
        # Stop everything immediately
        self.inspect_timer.stop()
        self.action_timer.stop()
        self.send("STOP")
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.state = "IDLE"
        self._set_indicator(self.ind_camera, "off")
        self._set_indicator(self.ind_motor,  "off")
        self._set_indicator(self.ind_servo,  "off")
        self.rec_dot.setText("●  OFFLINE")
        self.rec_dot.setStyleSheet(
            f"color:{T.TEXT_MUTED}; font-size:10px; font-weight:700; letter-spacing:1.5px;"
        )
        self.camera_view.setText("◉  AWAITING CAMERA SIGNAL")
        self.countdown_lbl.setText("")
        self.inspect_progress.setValue(0)
        self.cycle_ring.set_state(0.0, "■", "STOPPED", T.DANGER)
        self.set_state_label("STOPPED", T.DANGER)
        self.result_lbl.setText("STOPPED")
        self.result_lbl.setStyleSheet(f"color:{T.DANGER}; padding:4px 0;")
        self.log_event("System stopped — motor and servo reset")

    def test_servo(self):
        self.send("TEST")
        self.log_event("Servo test triggered")

    def on_speed_changed(self, val):
        self.motor_speed = val
        self.speed_lbl.setText(f"{val}%")
        self.send(f"SET_SPEED:{val}")

    # ===============================
    # INSPECTION LOGIC
    # ===============================
    def start_inspection(self):
        """Called when sheet is first detected in IDLE state."""
        self.state = "INSPECTING"
        self.vote_buffer.clear()
        self._frame_pills.clear()
        self._frame_missing.clear()
        self._frame_confidences.clear()
        self._frame_max_missing = 0
        self._inspect_start = time.time()
        self.countdown_end_ms = int(time.time() * 1000) + INSPECT_DURATION_MS
        self.inspect_timer.start(INSPECT_DURATION_MS)
        self.set_state_label("INSPECTING — 5s", T.INFO)
        self.result_lbl.setText("SCANNING")
        self.result_lbl.setStyleSheet(f"color:{T.INFO}; padding:4px 0;")
        self.severity_lbl.setText("")
        self.cycle_ring.set_state(0.0, "5.0s", "INSPECTING", T.INFO)
        self.log_event("Sheet detected — 5 second inspection started")

    def finish_inspection(self):
        """Called after 5-second inspection timer fires."""
        if not self.vote_buffer:
            self.state = "IDLE"
            self.set_state_label("IDLE — waiting for sheet", T.TEXT_DIM)
            self.result_lbl.setText("READY")
            self.result_lbl.setStyleSheet(f"color:{T.PRIMARY}; padding:4px 0;")
            self.cycle_ring.set_state(0.0, "—", "IDLE", T.PRIMARY)
            self.inspect_progress.setValue(0)
            self.log_event("Inspection ended — no sheet detected")
            return

        fail_votes  = sum(self.vote_buffer)
        total_votes = len(self.vote_buffer)
        fail_ratio  = fail_votes / total_votes

        avg_pill    = sum(self._frame_pills) / len(self._frame_pills) if self._frame_pills else 0
        avg_missing = sum(self._frame_missing) / len(self._frame_missing) if self._frame_missing else 0
        avg_conf    = sum(self._frame_confidences) / len(self._frame_confidences) if self._frame_confidences else 0
        duration_ms = int((time.time() - self._inspect_start) * 1000)
        severity    = _classify_severity(fail_votes, total_votes, self._frame_max_missing, avg_conf)

        if fail_ratio > MIN_FAIL_VOTES_PCT:
            self.action_result = "FAIL"
        else:
            self.action_result = "PASS"

        self.total += 1

        # Build inspection record
        record = {
            "pack_id": self.total,
            "result": self.action_result,
            "severity": severity,
            "fail_votes": fail_votes,
            "total_votes": total_votes,
            "avg_pill_count": round(avg_pill, 1),
            "avg_missing_count": round(avg_missing, 2),
            "max_missing_in_frame": self._frame_max_missing,
            "avg_confidence": round(avg_conf, 3),
            "belt_speed": self.motor_speed,
            "inspection_duration_ms": duration_ms,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        }
        self.inspection_log.append(record)

        self.start_action(severity)

    def start_action(self, severity="OK"):
        """Motor ON for 10s. If FAIL → servo OPEN too."""
        self.state = "ACTING"
        self.countdown_end_ms = int(time.time() * 1000) + ACTION_DURATION_MS

        self.send("START")

        sev_color = _SEV_COLORS.get(severity, T.TEXT_MUTED)
        self.severity_lbl.setText(f"SEVERITY · {severity}")
        self.severity_lbl.setStyleSheet(
            f"color:{sev_color}; font-weight:800; font-size:11px; letter-spacing:2px; "
            f"background:rgba(255,255,255,0.03); border:1px solid {sev_color}; "
            f"padding:6px 14px; border-radius:12px;"
        )

        if self.action_result == "PASS":
            self.passed += 1
            self.result_lbl.setText("PASS")
            self.result_lbl.setStyleSheet(f"color:{T.SUCCESS}; padding:4px 0;")
            self.set_state_label("PASS — motor 10s", T.SUCCESS)
            self.cycle_ring.set_state(0.0, "PASS", "CONVEYING", T.SUCCESS)
            self.log_event(f"PASS [{severity}] — motor ON for 10s")
        else:
            self.failed += 1
            self.result_lbl.setText("FAIL")
            self.result_lbl.setStyleSheet(f"color:{T.DANGER}; padding:4px 0;")
            self.set_state_label("FAIL — motor 10s", T.DANGER)
            self.cycle_ring.set_state(0.0, "FAIL", "REJECTING", T.DANGER)
            self.log_event(f"FAIL [{severity}] — motor ON for 10s")

        self.action_timer.start(ACTION_DURATION_MS)
        self.update_stats()
        # Defer chart refresh so it doesn't block servo/motor commands
        QTimer.singleShot(300, self._refresh_charts)

    def finish_action(self):
        """Called after 10-second action timer fires."""
        # Send STOP 3 times with gaps — ensures it gets through even if Arduino just rebooted
        self.send("STOP")
        QTimer.singleShot(300, lambda: self.send("STOP"))
        QTimer.singleShot(700, lambda: self.send("STOP"))
        self.log_event("10s done — motor OFF")

        # Back to IDLE — but require sheet removal before next cycle
        self.state          = "IDLE"
        self.action_result  = None
        self.awaiting_clear = True   # must see empty camera before re-arming
        self.clear_count    = 0
        self.detect_count   = 0
        self.countdown_lbl.setText("")
        self.inspect_progress.setValue(0)
        self.set_state_label("IDLE — remove sheet", T.WARNING)
        self.cycle_ring.set_state(0.0, "◄", "REMOVE", T.WARNING)
        self.result_lbl.setText("REMOVE SHEET")
        self.result_lbl.setStyleSheet(f"color:{T.WARNING}; padding:4px 0;")
        self.log_event("Ready — remove sheet to continue")

    def update_stats(self):
        yld = (self.passed / self.total * 100) if self.total else 0
        self.tile_total[1].setText(str(self.total))
        self.tile_passed[1].setText(str(self.passed))
        self.tile_failed[1].setText(str(self.failed))
        self.tile_yield[1].setText(f"{yld:.1f}%")

    # ===============================
    # MAIN FRAME LOOP
    # ===============================
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        h, w = frame.shape[:2]

        # Run YOLO on every frame for live overlay
        results    = model(frame, conf=0.5)
        pills      = 0
        missing    = 0
        has_sheet  = False   # True if any detection exists

        for r in results:
            for b in r.boxes:
                cls  = int(b.cls[0])
                conf = float(b.conf[0])
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                has_sheet = True

                if cls == MISSING_ID:
                    missing += 1
                    color = (0, 0, 255)
                    lbl   = f"missing {conf:.2f}"
                else:
                    pills += 1
                    color = (0, 255, 0)
                    lbl   = f"pill {conf:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(frame, (x1, y1 - 22), (x1 + 150, y1), color, -1)
                cv2.putText(frame, lbl, (x1 + 4, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Pill / missing count overlay
        cv2.putText(frame, f"Pills: {pills}",   (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Missing: {missing}", (15, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # ---- STATE ACTIONS ----

        if self.state == "IDLE":
            if self.awaiting_clear:
                # Wait for sheet removal: 5 consecutive frames with NO detection
                if not has_sheet:
                    self.clear_count += 1
                else:
                    self.clear_count = 0
                if self.clear_count >= 5:
                    self.awaiting_clear = False
                    self.clear_count = 0
                    self.set_state_label("IDLE — waiting for sheet", T.TEXT_DIM)
                    self.result_lbl.setText("READY")
                    self.result_lbl.setStyleSheet(f"color:{T.PRIMARY}; padding:4px 0;")
                    self.cycle_ring.set_state(0.0, "—", "WAITING", T.PRIMARY)
                    self.inspect_progress.setValue(0)
                    self.log_event("Sheet removed — ready for next")
                else:
                    cv2.putText(frame, "REMOVE SHEET",
                                (w // 2 - 150, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 3)
            else:
                # Normal: require 3 consecutive frames with detection to confirm sheet
                if has_sheet:
                    self.detect_count += 1
                else:
                    self.detect_count = 0
                if self.detect_count >= self.detect_needed:
                    self.detect_count = 0
                    self.start_inspection()

        elif self.state == "INSPECTING":
            # Collect vote + frame-level data
            self.vote_buffer.append(missing > 0)
            self._frame_pills.append(pills)
            self._frame_missing.append(missing)
            if missing > self._frame_max_missing:
                self._frame_max_missing = missing
            # Gather confidence values from this frame
            for r in results:
                for b in r.boxes:
                    self._frame_confidences.append(float(b.conf[0]))

            # Show progress on frame
            elapsed_ms  = INSPECT_DURATION_MS - max(
                0, self.countdown_end_ms - int(time.time() * 1000)
            )
            remaining_s = max(0, (self.countdown_end_ms - int(time.time() * 1000)) / 1000)
            bar_w       = int((elapsed_ms / INSPECT_DURATION_MS) * (w - 40))
            cv2.rectangle(frame, (20, h - 30), (20 + bar_w, h - 10), (0, 200, 255), -1)
            cv2.rectangle(frame, (20, h - 30), (w - 20,     h - 10), (0, 200, 255),  1)
            cv2.putText(frame, f"Inspecting... {remaining_s:.1f}s",
                        (20, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
            self.countdown_lbl.setText(f"Inspect {remaining_s:.1f}s")
            # Qt progress bar + ring
            pct = int((elapsed_ms / INSPECT_DURATION_MS) * 100)
            self.inspect_progress.setValue(max(0, min(100, pct)))
            self.cycle_ring.set_state(
                elapsed_ms / INSPECT_DURATION_MS,
                f"{remaining_s:.1f}s", "INSPECTING", T.INFO,
            )

        elif self.state == "ACTING":
            # Show countdown on frame
            remaining_s = max(0, (self.countdown_end_ms - int(time.time() * 1000)) / 1000)
            color = (0, 255, 0) if self.action_result == "PASS" else (0, 0, 255)
            cv2.putText(frame, f"{'PASS' if self.action_result == 'PASS' else 'FAIL'}"
                               f" — motor running {remaining_s:.1f}s",
                        (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            self.countdown_lbl.setText(f"Action  {remaining_s:.1f}s")
            # Qt progress bar + ring
            elapsed_ms = ACTION_DURATION_MS - max(0, self.countdown_end_ms - int(time.time() * 1000))
            pct = int((elapsed_ms / ACTION_DURATION_MS) * 100)
            self.inspect_progress.setValue(max(0, min(100, pct)))
            ring_color = T.SUCCESS if self.action_result == "PASS" else T.DANGER
            ring_label = self.action_result or "—"
            self.cycle_ring.set_state(
                elapsed_ms / ACTION_DURATION_MS,
                f"{remaining_s:.1f}s", ring_label, ring_color,
            )

        # State text on frame bottom-right
        state_color = {
            "IDLE":       (150, 150, 150),
            "INSPECTING": (0, 200, 255),
            "ACTING":     (0, 255, 100),
        }.get(self.state, (150, 150, 150))
        cv2.putText(frame, f"State: {self.state}",
                    (w - 220, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)

        # Display
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        self.camera_view.setPixmap(QPixmap.fromImage(img))

    def closeEvent(self, event):
        self.inspect_timer.stop()
        self.action_timer.stop()
        self.serial_timer.stop()
        self.hb_timer.stop()
        self.timer.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.arduino and self.arduino.is_open:
            self.arduino.write(b"STOP\n")
            self.arduino.close()
        event.accept()

    # ===============================
    # DASHBOARD CHARTS
    # ===============================
    def _draw_empty_charts(self):
        """Draw placeholder charts before any data exists."""
        for fig, canvas, title in [
            (self.pie_fig, self.pie_canvas, "Pass / Fail"),
            (self.yield_fig, self.yield_canvas, "Yield Trend"),
            (self.sev_fig, self.sev_canvas, "Severity Distribution"),
            (self.hourly_fig, self.hourly_canvas, "Hourly Throughput"),
        ]:
            fig.clear()
            ax = fig.add_subplot(111)
            ax.set_facecolor(T.CARD)
            ax.text(0.5, 0.5, f"No data yet\n({title})",
                    ha='center', va='center', color=T.TEXT_MUTED, fontsize=10,
                    transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            fig.tight_layout(pad=1)
            canvas.draw()

    def _refresh_charts(self):
        """Update all 4 dashboard charts with current data."""
        if not self.inspection_log:
            return
        # Only draw charts if Dashboard tab is visible to avoid blocking
        if self.right_tabs.currentIndex() != 1:
            self._charts_stale = True
            return
        self._charts_stale = False
        self._draw_pie_chart()
        self._draw_yield_chart()
        self._draw_severity_chart()
        self._draw_hourly_chart()

    def _style_ax(self, ax, title):
        ax.set_facecolor(T.CARD)
        ax.set_title(title, color=T.TEXT, fontsize=10, fontweight='bold', pad=6)
        ax.tick_params(colors=T.TEXT_DIM, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(T.BORDER)

    def _draw_pie_chart(self):
        self.pie_fig.clear()
        ax = self.pie_fig.add_subplot(111)
        ax.set_facecolor(T.CARD)
        vals = [self.passed, self.failed]
        labels = [f"Pass ({self.passed})", f"Fail ({self.failed})"]
        colors = [T.SUCCESS, T.DANGER]
        if sum(vals) > 0:
            wedges, texts, autotexts = ax.pie(
                vals, labels=labels, colors=colors, autopct='%1.0f%%',
                startangle=90, textprops={'color': T.TEXT, 'fontsize': 9},
                wedgeprops={'edgecolor': T.CARD, 'linewidth': 2},
            )
            for at in autotexts:
                at.set_color('white')
                at.set_fontweight('bold')
        ax.set_title("Pass / Fail", color=T.TEXT, fontsize=10, fontweight='bold')
        self.pie_fig.tight_layout(pad=1)
        self.pie_canvas.draw()

    def _draw_yield_chart(self):
        self.yield_fig.clear()
        ax = self.yield_fig.add_subplot(111)
        self._style_ax(ax, "Yield Trend (%)")
        n = len(self.inspection_log)
        yields = []
        running_pass = 0
        for i, rec in enumerate(self.inspection_log):
            if rec["result"] == "PASS":
                running_pass += 1
            yields.append(running_pass / (i + 1) * 100)
        ax.fill_between(range(1, n + 1), yields, color=T.PRIMARY, alpha=0.15)
        ax.plot(range(1, n + 1), yields, color=T.PRIMARY, linewidth=2, marker='o', markersize=4)
        ax.set_xlabel("Pack #", color=T.TEXT_DIM, fontsize=8)
        ax.set_ylabel("Yield %", color=T.TEXT_DIM, fontsize=8)
        ax.set_ylim(0, 105)
        ax.axhline(y=95, color=T.SUCCESS, linestyle='--', linewidth=0.8, alpha=0.6)
        ax.grid(True, axis='y', color=T.BORDER, linestyle='-', linewidth=0.4, alpha=0.5)
        self.yield_fig.tight_layout(pad=1)
        self.yield_canvas.draw()

    def _draw_severity_chart(self):
        self.sev_fig.clear()
        ax = self.sev_fig.add_subplot(111)
        self._style_ax(ax, "Severity Distribution")
        sev_order = ["OK", "WARNING", "MINOR", "MAJOR", "CRITICAL"]
        counts = {s: 0 for s in sev_order}
        for rec in self.inspection_log:
            s = rec["severity"]
            if s in counts:
                counts[s] += 1
        labels = [s for s in sev_order if counts[s] > 0]
        vals = [counts[s] for s in labels]
        colors = [_SEV_COLORS.get(s, T.TEXT_MUTED) for s in labels]
        if labels:
            bars = ax.bar(labels, vals, color=colors, edgecolor=T.CARD, linewidth=1.5)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                        str(v), ha='center', va='bottom', color=T.TEXT, fontsize=8, fontweight='bold')
        ax.set_ylabel("Count", color=T.TEXT_DIM, fontsize=8)
        ax.grid(True, axis='y', color=T.BORDER, linestyle='-', linewidth=0.4, alpha=0.5)
        self.sev_fig.tight_layout(pad=1)
        self.sev_canvas.draw()

    def _draw_hourly_chart(self):
        self.hourly_fig.clear()
        ax = self.hourly_fig.add_subplot(111)
        self._style_ax(ax, "Hourly Throughput")
        hours = {}
        for rec in self.inspection_log:
            h = rec["timestamp"].split(":")[0]
            hours[h] = hours.get(h, 0) + 1
        if hours:
            sorted_h = sorted(hours.keys())
            vals = [hours[h] for h in sorted_h]
            labels = [f"{h}:00" for h in sorted_h]
            ax.bar(labels, vals, color=T.ACCENT, edgecolor=T.CARD, linewidth=1.5)
            for i, v in enumerate(vals):
                ax.text(i, v + 0.2, str(v), ha='center', va='bottom',
                        color=T.TEXT, fontsize=8, fontweight='bold')
        ax.set_ylabel("Inspections", color=T.TEXT_DIM, fontsize=8)
        ax.grid(True, axis='y', color=T.BORDER, linestyle='-', linewidth=0.4, alpha=0.5)
        self.hourly_fig.tight_layout(pad=1)
        self.hourly_canvas.draw()

    # ===============================
    # REPORTS
    # ===============================
    def _build_report_text(self):
        """Build a plain-text report summary from inspection data."""
        lines = []
        lines.append("INSPECT-OS  —  Inspection Report")
        lines.append("═" * 42)
        lines.append(f"Session Start : {self.session_start}")
        lines.append(f"Report Time   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        yld = (self.passed / self.total * 100) if self.total else 0
        lines.append(f"Total Sheets Checked : {self.total}")
        lines.append(f"Passed               : {self.passed}")
        lines.append(f"Failed               : {self.failed}")
        lines.append(f"Yield                : {yld:.1f}%")
        lines.append("")

        # Severity breakdown
        sev_counts = {}
        for rec in self.inspection_log:
            s = rec["severity"]
            sev_counts[s] = sev_counts.get(s, 0) + 1
        if sev_counts:
            lines.append("Severity Breakdown:")
            for sev, cnt in sev_counts.items():
                lines.append(f"  {sev:10s} : {cnt}")
            lines.append("")

        # Per-sheet details
        lines.append("─" * 42)
        lines.append("Per-Sheet Details:")
        lines.append(f"{'#':>3}  {'Result':6}  {'Severity':9}  {'Missing':7}  {'Conf':5}  {'Speed':5}  {'Time':8}")
        lines.append("─" * 42)
        for rec in self.inspection_log:
            lines.append(
                f"{rec['pack_id']:>3}  {rec['result']:6}  {rec['severity']:9}  "
                f"{rec['max_missing_in_frame']:>7}  {rec['avg_confidence']:.2f}   "
                f"{rec['belt_speed']:>4}%  {rec['timestamp']}"
            )
        lines.append("─" * 42)
        return "\n".join(lines)

    def preview_report(self):
        """Show report preview in the text area."""
        if not self.inspection_log:
            self.report_preview.setPlainText("No inspection data yet — run some inspections first.")
            return
        self.report_preview.setPlainText(self._build_report_text())
        self.report_status_lbl.setText("Preview updated.")

    def generate_report_pdf(self):
        """Generate and save PDF report to reports/ folder."""
        if not self.inspection_log:
            self.report_status_lbl.setText("No inspection data yet.")
            return
        try:
            from fpdf import FPDF

            if getattr(sys, '_MEIPASS', None):
                reports_dir = os.path.join(os.path.dirname(sys.executable), "reports")
            else:
                reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
            os.makedirs(reports_dir, exist_ok=True)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_path = os.path.join(reports_dir, f"INSPECT-OS_Report_{ts}.pdf")

            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()

            # Title
            pdf.set_font("Helvetica", "B", 20)
            pdf.set_text_color(0, 120, 215)
            pdf.cell(0, 15, "INSPECT-OS Inspection Report", new_x="LMARGIN", new_y="NEXT", align="C")
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(100, 100, 100)
            pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", new_x="LMARGIN", new_y="NEXT", align="C")
            pdf.cell(0, 8, f"Session: {self.session_start} to {datetime.now().isoformat()}", new_x="LMARGIN", new_y="NEXT", align="C")
            pdf.ln(5)

            # Summary
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_text_color(50, 50, 50)
            pdf.cell(0, 10, "Session Summary", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 11)
            yld = (self.passed / self.total * 100) if self.total else 0
            for line in [
                f"Total Sheets Checked: {self.total}",
                f"Passed: {self.passed}   |   Failed: {self.failed}",
                f"Yield: {yld:.1f}%",
            ]:
                pdf.cell(0, 7, line, new_x="LMARGIN", new_y="NEXT")

            # Severity breakdown
            sev_counts = {}
            for rec in self.inspection_log:
                s = rec["severity"]
                sev_counts[s] = sev_counts.get(s, 0) + 1
            if sev_counts:
                pdf.cell(0, 7, f"Severity: {', '.join(f'{k}: {v}' for k, v in sev_counts.items())}", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(3)

            # Embed charts
            for fig in [self.pie_fig, self.yield_fig, self.sev_fig]:
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                fig.savefig(tmp.name, dpi=120, facecolor='#ffffff', bbox_inches='tight')
                tmp.close()
                pdf.image(tmp.name, w=170)
                pdf.ln(3)
                os.unlink(tmp.name)

            # Per-sheet table
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_text_color(50, 50, 50)
            pdf.cell(0, 10, "Per-Sheet Results", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "B", 8)
            headers = ["#", "Result", "Severity", "Fail/Total", "Missing", "Conf", "Speed", "Time"]
            widths = [12, 18, 22, 28, 22, 18, 18, 22]
            pdf.set_fill_color(220, 220, 240)
            pdf.set_text_color(50, 50, 50)
            for h, w in zip(headers, widths):
                pdf.cell(w, 7, h, border=1, fill=True, align="C")
            pdf.ln()
            pdf.set_font("Helvetica", "", 8)
            for rec in self.inspection_log:
                row = [
                    str(rec["pack_id"]),
                    rec["result"],
                    rec["severity"],
                    f"{rec['fail_votes']}/{rec['total_votes']}",
                    str(rec["max_missing_in_frame"]),
                    f"{rec['avg_confidence']:.2f}",
                    f"{rec['belt_speed']}%",
                    rec["timestamp"],
                ]
                for val, w in zip(row, widths):
                    pdf.cell(w, 6, val, border=1, align="C")
                pdf.ln()

            pdf.output(pdf_path)
            self.report_status_lbl.setText(f"✅ Saved: {pdf_path}")
            self.report_preview.setPlainText(self._build_report_text())
            self.log_event(f"PDF report saved: {pdf_path}")

        except Exception as e:
            self.report_status_lbl.setText(f"Error: {e}")
            self.log_event(f"PDF generation error: {e}")


# ===============================
# RUN
# ===============================
app = QApplication(sys.argv)
window = InspectOS()
window.show()
sys.exit(app.exec())
