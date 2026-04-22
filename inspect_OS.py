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
    QTabWidget, QSizePolicy
)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap, QFont

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
        self.setStyleSheet("""
        QWidget  { background-color: #121212; color: #d0d0d0; font-family: Segoe UI; }
        QPushButton { padding: 10px; font-size: 14px; border-radius: 4px;
                      border: 1px solid #444; color: #d0d0d0; }
        QPushButton:hover { background-color: #3a3a3a; }
        QPushButton#start { background:#1f7a1f; color:white; border:1px solid #2a9a2a; font-weight:bold; }
        QPushButton#stop  { background:#8c1d18; color:white; border:1px solid #b22a24; font-weight:bold; }
        QLabel    { font-size: 13px; color: #d0d0d0; }
        QTextEdit { background:#050505; border:1px solid #333; color:#00ff00;
                    font-family:Consolas; font-size:11px; }
        QGroupBox { border:1px solid #444; border-radius:4px; margin-top:6px;
                    padding:8px; padding-top:16px; font-weight:bold;
                    color:#0078d7; font-size:12px; }
        QGroupBox::title { subcontrol-origin:margin; left:10px; padding:0 5px; }
        QComboBox { background:#2a2a2a; color:white; border:1px solid #555;
                    padding:5px; border-radius:3px; font-size:12px; }
        QSlider::groove:horizontal   { background:#333; height:6px; border-radius:3px; }
        QSlider::handle:horizontal   { background:#0078d7; width:16px; margin:-5px 0; border-radius:8px; }
        QSlider::sub-page:horizontal { background:#0078d7; border-radius:3px; }
        QTabWidget::pane { border:1px solid #333; background:#1a1a1a; }
        QTabBar::tab { background:#2a2a2a; color:#aaa; padding:8px 16px;
                       border:1px solid #333; border-bottom:none; border-top-left-radius:4px;
                       border-top-right-radius:4px; min-width:80px; }
        QTabBar::tab:selected { background:#1a1a1a; color:#0078d7; font-weight:bold; }
        QTabBar::tab:hover { background:#333; }
        """)

        # Header
        header = QHBoxLayout()
        title  = QLabel("INSPECT-OS")
        title.setFont(QFont("Segoe UI", 20, QFont.Bold))
        title.setStyleSheet("color: #0078d7;")
        header.addWidget(title)
        header.addStretch()
        self.clock_label = QLabel("00:00:00")
        self.clock_label.setFont(QFont("Consolas", 14))
        self.clock_label.setStyleSheet("color: #888;")
        header.addWidget(self.clock_label)

        # ---- LEFT PANEL ----
        left = QVBoxLayout()
        left.setSpacing(10)

        # Arduino connection
        conn_group  = QGroupBox("Arduino Connection")
        conn_layout = QVBoxLayout()
        port_row = QHBoxLayout()
        self.port_combo        = QComboBox()
        self.refresh_ports_btn = QPushButton("⟳")
        self.refresh_ports_btn.setFixedWidth(36)
        port_row.addWidget(self.port_combo, 1)
        port_row.addWidget(self.refresh_ports_btn)
        conn_layout.addLayout(port_row)
        btn_row = QHBoxLayout()
        self.connect_btn = QPushButton("CONNECT")
        self.connect_btn.setStyleSheet("background:#0078d7; color:white; font-weight:bold;")
        self.disconnect_btn = QPushButton("DISCONNECT")
        self.disconnect_btn.setStyleSheet("background:#555;")
        self.disconnect_btn.setEnabled(False)
        btn_row.addWidget(self.connect_btn)
        btn_row.addWidget(self.disconnect_btn)
        conn_layout.addLayout(btn_row)
        self.arduino_lbl = QLabel("● DISCONNECTED")
        self.arduino_lbl.setStyleSheet("color:#dc3545; font-weight:bold; font-size:13px;")
        conn_layout.addWidget(self.arduino_lbl)
        conn_group.setLayout(conn_layout)
        left.addWidget(conn_group)

        # System controls
        ctrl_group  = QGroupBox("System")
        ctrl_layout = QVBoxLayout()
        self.start_btn = QPushButton("▶  START")
        self.start_btn.setObjectName("start")
        self.stop_btn  = QPushButton("■  STOP")
        self.stop_btn.setObjectName("stop")
        self.test_servo_btn = QPushButton("⟳  TEST SERVO")
        self.test_servo_btn.setStyleSheet("background:#4a4a00; color:#ffd700; font-weight:bold;")
        ctrl_layout.addWidget(self.start_btn)
        ctrl_layout.addWidget(self.stop_btn)
        ctrl_layout.addWidget(self.test_servo_btn)
        ctrl_group.setLayout(ctrl_layout)
        left.addWidget(ctrl_group)

        # Hardware indicators
        hw_group  = QGroupBox("Hardware Status")
        hw_layout = QVBoxLayout()
        self.ind_arduino = self._make_indicator("Arduino")
        self.ind_camera  = self._make_indicator("Camera")
        self.ind_motor   = self._make_indicator("Motor")
        self.ind_servo   = self._make_indicator("Servo")
        for lbl, dot in [self.ind_arduino, self.ind_camera,
                         self.ind_motor,   self.ind_servo]:
            row = QHBoxLayout()
            row.addWidget(lbl)
            row.addStretch()
            row.addWidget(dot)
            hw_layout.addLayout(row)
        hw_group.setLayout(hw_layout)
        left.addWidget(hw_group)

        # Motor speed
        spd_group  = QGroupBox("Motor Speed")
        spd_layout = QVBoxLayout()
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(0, 100)
        self.speed_slider.setValue(self.motor_speed)
        self.speed_slider.setTickInterval(10)
        self.speed_slider.setTickPosition(QSlider.TicksBelow)
        self.speed_lbl = QLabel(f"{self.motor_speed}%")
        self.speed_lbl.setAlignment(Qt.AlignCenter)
        self.speed_lbl.setStyleSheet("color:#ff9800; font-weight:bold; font-size:13px;")
        spd_layout.addWidget(self.speed_slider)
        spd_layout.addWidget(self.speed_lbl)
        spd_group.setLayout(spd_layout)
        left.addWidget(spd_group)

        # Stats
        stats_group  = QGroupBox("Run Statistics")
        stats_layout = QVBoxLayout()
        self.stats_lbl = QLabel("Total: 0\nPassed: 0\nFailed: 0\nYield: 0.0%")
        self.stats_lbl.setStyleSheet("font-family:Consolas; font-size:13px; line-height:1.6;")
        stats_layout.addWidget(self.stats_lbl)
        stats_group.setLayout(stats_layout)
        left.addWidget(stats_group)

        left.addStretch()

        scroll_content = QWidget()
        scroll_content.setLayout(left)
        scroll = QScrollArea()
        scroll.setWidget(scroll_content)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border:none; background:#1a1a1a; }")

        left_panel = QFrame()
        lp_layout  = QVBoxLayout(left_panel)
        lp_layout.setContentsMargins(0, 0, 0, 0)
        lp_layout.addWidget(scroll)
        left_panel.setFixedWidth(300)
        left_panel.setStyleSheet("QFrame { background:#1a1a1a; border-right:2px solid #333; }")

        # Camera view
        self.camera_view = QLabel("LIVE CAMERA FEED")
        self.camera_view.setAlignment(Qt.AlignCenter)
        self.camera_view.setStyleSheet(
            "background:black; border:2px solid #0078d7; font-size:18px; color:#555;"
        )

        # ---- RIGHT PANEL — TABBED ----
        self.right_tabs = QTabWidget()
        self.right_tabs.setFixedWidth(380)

        # === TAB 1: LIVE ===
        live_tab = QWidget()
        live_layout = QVBoxLayout(live_tab)
        live_layout.setSpacing(12)

        self.result_lbl = QLabel("WAITING")
        self.result_lbl.setAlignment(Qt.AlignCenter)
        self.result_lbl.setFont(QFont("Segoe UI", 40, QFont.Bold))
        self.result_lbl.setStyleSheet(
            "color:cyan; background:#1a1a1a; border:2px solid #333; padding:20px;"
        )
        live_layout.addWidget(self.result_lbl)

        self.severity_lbl = QLabel("")
        self.severity_lbl.setAlignment(Qt.AlignCenter)
        self.severity_lbl.setFont(QFont("Segoe UI", 14, QFont.Bold))
        self.severity_lbl.setStyleSheet("color:#555; padding:4px;")
        live_layout.addWidget(self.severity_lbl)

        self.state_lbl = QLabel("State: IDLE")
        self.state_lbl.setAlignment(Qt.AlignCenter)
        self.state_lbl.setStyleSheet(
            "font-size:14px; font-weight:bold; color:#888; "
            "background:#0a0a0a; border:1px solid #333; padding:8px;"
        )
        live_layout.addWidget(self.state_lbl)

        self.countdown_lbl = QLabel("")
        self.countdown_lbl.setAlignment(Qt.AlignCenter)
        self.countdown_lbl.setFont(QFont("Consolas", 22, QFont.Bold))
        self.countdown_lbl.setStyleSheet("color:#ffc107;")
        live_layout.addWidget(self.countdown_lbl)

        self.hw_feedback_lbl = QLabel("Arduino: waiting...")
        self.hw_feedback_lbl.setAlignment(Qt.AlignCenter)
        self.hw_feedback_lbl.setWordWrap(True)
        self.hw_feedback_lbl.setStyleSheet(
            "color:#888; font-family:Consolas; font-size:11px; "
            "background:#0a0a0a; border:1px solid #333; padding:8px;"
        )
        live_layout.addWidget(self.hw_feedback_lbl)
        live_layout.addStretch()
        self.right_tabs.addTab(live_tab, "Live")

        # === TAB 2: DASHBOARD ===
        dash_tab = QWidget()
        dash_layout = QVBoxLayout(dash_tab)
        dash_layout.setContentsMargins(4, 4, 4, 4)
        dash_layout.setSpacing(4)

        # Pie chart: Pass/Fail
        self.pie_fig = Figure(figsize=(3.4, 2.2), dpi=80)
        self.pie_fig.patch.set_facecolor('#1a1a1a')
        self.pie_canvas = FigureCanvas(self.pie_fig)
        self.pie_canvas.setStyleSheet("background:#1a1a1a; border:none;")
        dash_layout.addWidget(self.pie_canvas)

        # Line chart: Yield trend
        self.yield_fig = Figure(figsize=(3.4, 2.2), dpi=80)
        self.yield_fig.patch.set_facecolor('#1a1a1a')
        self.yield_canvas = FigureCanvas(self.yield_fig)
        self.yield_canvas.setStyleSheet("background:#1a1a1a; border:none;")
        dash_layout.addWidget(self.yield_canvas)

        # Bar chart: Severity distribution
        self.sev_fig = Figure(figsize=(3.4, 2.2), dpi=80)
        self.sev_fig.patch.set_facecolor('#1a1a1a')
        self.sev_canvas = FigureCanvas(self.sev_fig)
        self.sev_canvas.setStyleSheet("background:#1a1a1a; border:none;")
        dash_layout.addWidget(self.sev_canvas)

        # Bar chart: Hourly throughput
        self.hourly_fig = Figure(figsize=(3.4, 2.2), dpi=80)
        self.hourly_fig.patch.set_facecolor('#1a1a1a')
        self.hourly_canvas = FigureCanvas(self.hourly_fig)
        self.hourly_canvas.setStyleSheet("background:#1a1a1a; border:none;")
        dash_layout.addWidget(self.hourly_canvas)

        # Wrap dashboard in scroll area
        dash_scroll = QScrollArea()
        dash_scroll.setWidget(dash_tab)
        dash_scroll.setWidgetResizable(True)
        dash_scroll.setStyleSheet("QScrollArea { border:none; background:#1a1a1a; }")
        self.right_tabs.addTab(dash_scroll, "Dashboard")

        self._draw_empty_charts()

        # === TAB 3: REPORTS ===
        report_tab = QWidget()
        report_layout = QVBoxLayout(report_tab)
        report_layout.setSpacing(10)

        report_title = QLabel("Inspection Reports")
        report_title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        report_title.setStyleSheet("color:#0078d7;")
        report_layout.addWidget(report_title)

        # Report preview area
        self.report_preview = QTextEdit()
        self.report_preview.setReadOnly(True)
        self.report_preview.setStyleSheet(
            "background:#0a0a0a; border:1px solid #333; color:#e0e0e0; "
            "font-family:Consolas; font-size:12px; padding:8px;"
        )
        self.report_preview.setPlaceholderText("Click 'Preview Report' to see inspection summary...")
        report_layout.addWidget(self.report_preview, 1)

        # Buttons
        report_btn_row = QHBoxLayout()
        self.preview_report_btn = QPushButton("🔍 Preview Report")
        self.preview_report_btn.setStyleSheet("background:#0078d7; color:white; font-weight:bold;")
        self.generate_pdf_btn = QPushButton("📄 Save as PDF")
        self.generate_pdf_btn.setStyleSheet("background:#1b5e20; color:white; font-weight:bold;")
        report_btn_row.addWidget(self.preview_report_btn)
        report_btn_row.addWidget(self.generate_pdf_btn)
        report_layout.addLayout(report_btn_row)

        self.report_status_lbl = QLabel("")
        self.report_status_lbl.setStyleSheet("color:#888; font-size:11px;")
        report_layout.addWidget(self.report_status_lbl)

        self.right_tabs.addTab(report_tab, "Reports")

        # Log
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(160)
        self.log.setPlaceholderText("System log...")

        # Assemble
        top = QHBoxLayout()
        top.addWidget(left_panel)
        top.addWidget(self.camera_view, 1)
        top.addWidget(self.right_tabs)

        main = QVBoxLayout(self)
        main.addLayout(header)
        main.addLayout(top)
        main.addWidget(self.log)

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

        self.refresh_ports()

    # ===============================
    # HELPERS
    # ===============================
    def _make_indicator(self, name):
        lbl = QLabel(name)
        lbl.setStyleSheet("font-size:12px; color:#ccc; font-weight:bold;")
        dot = QLabel("●")
        dot.setStyleSheet("color:#555; font-size:20px;")
        dot.setFixedWidth(28)
        dot.setAlignment(Qt.AlignCenter)
        return lbl, dot

    def _set_indicator(self, tup, state):
        colors = {"on": "#28a745", "warn": "#ffc107", "off": "#555555", "err": "#dc3545"}
        tup[1].setStyleSheet(f"color:{colors.get(state,'#555')}; font-size:20px;")

    def update_clock(self):
        self.clock_label.setText(datetime.now().strftime("%H:%M:%S"))

    def _on_tab_changed(self, index):
        """Refresh charts when switching to Dashboard tab if data changed."""
        if index == 1 and self._charts_stale:
            self._charts_stale = False
            QTimer.singleShot(50, self._refresh_charts)

    def log_event(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log.append(f"[{ts}] {msg}")

    def set_state_label(self, text, color="#888"):
        self.state_lbl.setText(f"State: {text}")
        self.state_lbl.setStyleSheet(
            f"font-size:14px; font-weight:bold; color:{color}; "
            f"background:#0a0a0a; border:1px solid #333; padding:8px;"
        )

    def update_hw_feedback(self, text):
        ts = datetime.now().strftime("%H:%M:%S")
        self.hw_feedback_lbl.setText(f"[{ts}] {text}")

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
            self.arduino_lbl.setText(f"● CONNECTED ({port})")
            self.arduino_lbl.setStyleSheet("color:#28a745; font-weight:bold; font-size:13px;")
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
        self.arduino_lbl.setStyleSheet("color:#dc3545; font-weight:bold; font-size:13px;")
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
            self.log_event("Camera opened — place sheet under camera to begin")
        else:
            self._set_indicator(self.ind_camera, "err")
            self.log_event("ERROR: camera not found")
            return

        # Make sure motor is OFF at start
        self.send("STOP")
        self.send(f"SET_SPEED:{self.motor_speed}")

        self.state = "IDLE"
        self.set_state_label("IDLE — waiting for sheet", "#888")
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
        self.camera_view.setText("LIVE CAMERA FEED")
        self.countdown_lbl.setText("")
        self.set_state_label("STOPPED", "#dc3545")
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
        self.set_state_label("INSPECTING — 5s", "#00bcd4")
        self.result_lbl.setText("...")
        self.result_lbl.setStyleSheet(
            "color:#00bcd4; background:#1a1a1a; border:2px solid #00bcd4; padding:20px;"
        )
        self.severity_lbl.setText("")
        self.log_event("Sheet detected — 5 second inspection started")

    def finish_inspection(self):
        """Called after 5-second inspection timer fires."""
        if not self.vote_buffer:
            self.state = "IDLE"
            self.set_state_label("IDLE — waiting for sheet", "#888")
            self.result_lbl.setText("WAITING")
            self.result_lbl.setStyleSheet(
                "color:cyan; background:#1a1a1a; border:2px solid #333; padding:20px;"
            )
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

        sev_color = _SEV_COLORS.get(severity, "#555")
        self.severity_lbl.setText(f"Severity: {severity}")
        self.severity_lbl.setStyleSheet(
            f"color:{sev_color}; font-weight:bold; font-size:14px; "
            f"background:#1a1a1a; border:1px solid {sev_color}; padding:4px; border-radius:3px;"
        )

        if self.action_result == "PASS":
            self.passed += 1
            self.result_lbl.setText("PASS")
            self.result_lbl.setStyleSheet(
                "color:#28a745; background:#0a2a0a; "
                "border:2px solid #28a745; padding:20px;"
            )
            self.set_state_label("PASS — motor running 10s", "#28a745")
            self.log_event(f"PASS [{severity}] — motor ON for 10s")
        else:
            self.failed += 1
            self.result_lbl.setText("FAIL")
            self.result_lbl.setStyleSheet(
                "color:#dc3545; background:#2a0a0a; "
                "border:2px solid #dc3545; padding:20px;"
            )
            self.set_state_label("FAIL — motor running 10s", "#dc3545")
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
        self.set_state_label("IDLE — remove sheet", "#ffc107")
        self.result_lbl.setText("REMOVE SHEET")
        self.result_lbl.setStyleSheet(
            "color:#ffc107; background:#1a1a1a; border:2px solid #ffc107; padding:20px;"
        )
        self.log_event("Ready — remove sheet to continue")

    def update_stats(self):
        yld = (self.passed / self.total * 100) if self.total else 0
        self.stats_lbl.setText(
            f"Total:  {self.total}\n"
            f"Passed: {self.passed}\n"
            f"Failed: {self.failed}\n"
            f"Yield:  {yld:.1f}%"
        )

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
                    self.set_state_label("IDLE — waiting for sheet", "#888")
                    self.result_lbl.setText("WAITING")
                    self.result_lbl.setStyleSheet(
                        "color:cyan; background:#1a1a1a; border:2px solid #333; padding:20px;"
                    )
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

        elif self.state == "ACTING":
            # Show countdown on frame
            remaining_s = max(0, (self.countdown_end_ms - int(time.time() * 1000)) / 1000)
            color = (0, 255, 0) if self.action_result == "PASS" else (0, 0, 255)
            cv2.putText(frame, f"{'PASS' if self.action_result == 'PASS' else 'FAIL'}"
                               f" — motor running {remaining_s:.1f}s",
                        (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            self.countdown_lbl.setText(f"Action  {remaining_s:.1f}s")

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
            ax.set_facecolor('#1a1a1a')
            ax.text(0.5, 0.5, f"No data yet\n({title})",
                    ha='center', va='center', color='#555', fontsize=10,
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
        ax.set_facecolor('#1a1a1a')
        ax.set_title(title, color='#ccc', fontsize=10, fontweight='bold', pad=6)
        ax.tick_params(colors='#888', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#333')

    def _draw_pie_chart(self):
        self.pie_fig.clear()
        ax = self.pie_fig.add_subplot(111)
        ax.set_facecolor('#1a1a1a')
        vals = [self.passed, self.failed]
        labels = [f"Pass ({self.passed})", f"Fail ({self.failed})"]
        colors = ['#28a745', '#dc3545']
        if sum(vals) > 0:
            wedges, texts, autotexts = ax.pie(
                vals, labels=labels, colors=colors, autopct='%1.0f%%',
                startangle=90, textprops={'color': '#ccc', 'fontsize': 9}
            )
            for at in autotexts:
                at.set_color('white')
                at.set_fontweight('bold')
        ax.set_title("Pass / Fail", color='#ccc', fontsize=10, fontweight='bold')
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
        ax.plot(range(1, n + 1), yields, color='#0078d7', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel("Pack #", color='#888', fontsize=8)
        ax.set_ylabel("Yield %", color='#888', fontsize=8)
        ax.set_ylim(0, 105)
        ax.axhline(y=95, color='#28a745', linestyle='--', linewidth=0.8, alpha=0.5)
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
        colors = [_SEV_COLORS.get(s, '#555') for s in labels]
        if labels:
            bars = ax.bar(labels, vals, color=colors, edgecolor='#333')
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                        str(v), ha='center', va='bottom', color='#ccc', fontsize=8, fontweight='bold')
        ax.set_ylabel("Count", color='#888', fontsize=8)
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
            ax.bar(labels, vals, color='#0078d7', edgecolor='#333')
            for i, v in enumerate(vals):
                ax.text(i, v + 0.2, str(v), ha='center', va='bottom',
                        color='#ccc', fontsize=8, fontweight='bold')
        ax.set_ylabel("Inspections", color='#888', fontsize=8)
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
