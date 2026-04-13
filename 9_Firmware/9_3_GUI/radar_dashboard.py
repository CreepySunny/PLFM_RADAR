#!/usr/bin/env python3
"""
AERIS-10 Radar Dashboard
===================================================
Real-time visualization and control for the AERIS-10 phased-array radar
via FT2232H USB 2.0 interface.

Features:
  - FT2232H USB reader with packet parsing (matches usb_data_interface_ft2232h.v)
  - Real-time range-Doppler magnitude heatmap (64x32)
  - CFAR detection overlay (flagged cells highlighted)
  - Range profile waterfall plot (range vs. time)
  - Host command sender (opcodes per radar_system_top.v:
    0x01-0x04, 0x10-0x16, 0x20-0x27, 0x30-0x31, 0xFF)
  - Configuration panel for all radar parameters
  - HDF5 data recording for offline analysis
  - Mock mode for development/testing without hardware

Usage:
  python radar_dashboard.py              # Launch with mock data
  python radar_dashboard.py --live       # Launch with FT2232H hardware
  python radar_dashboard.py --record     # Launch with HDF5 recording
"""

import os
import time
import queue
import logging
import argparse
import threading
import contextlib
from collections import deque

import numpy as np

import tkinter as tk
from tkinter import ttk, filedialog

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Import protocol layer (no GUI deps)
from radar_protocol import (
    RadarProtocol, FT2232HConnection, ReplayConnection,
    DataRecorder, RadarAcquisition,
    RadarFrame, StatusResponse,
    NUM_RANGE_BINS, NUM_DOPPLER_BINS, WATERFALL_DEPTH,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("radar_dashboard")



# ============================================================================
# Dashboard GUI
# ============================================================================

# Dark theme colors
BG = "#1e1e2e"
BG2 = "#282840"
FG = "#cdd6f4"
ACCENT = "#89b4fa"
GREEN = "#a6e3a1"
RED = "#f38ba8"
YELLOW = "#f9e2af"
SURFACE = "#313244"


class RadarDashboard:
    """Main tkinter application: real-time radar visualization and control."""

    UPDATE_INTERVAL_MS = 100  # 10 Hz display refresh

    # Radar parameters used for range-axis scaling.
    BANDWIDTH = 500e6        # Hz — chirp bandwidth
    C = 3e8                  # m/s — speed of light

    def __init__(self, root: tk.Tk, connection: FT2232HConnection,
                 recorder: DataRecorder, device_index: int = 0):
        self.root = root
        self.conn = connection
        self.recorder = recorder
        self.device_index = device_index

        self.root.title("AERIS-10 Radar Dashboard")
        self.root.geometry("1600x950")
        self.root.configure(bg=BG)

        # Frame queue (acquisition → display)
        self.frame_queue: queue.Queue[RadarFrame] = queue.Queue(maxsize=8)
        self._acq_thread: RadarAcquisition | None = None

        # Display state
        self._current_frame = RadarFrame()
        self._waterfall = deque(maxlen=WATERFALL_DEPTH)
        for _ in range(WATERFALL_DEPTH):
            self._waterfall.append(np.zeros(NUM_RANGE_BINS))

        self._frame_count = 0
        self._fps_ts = time.time()
        self._fps = 0.0

        # Stable colorscale — exponential moving average of vmax
        self._vmax_ema = 1000.0
        self._vmax_alpha = 0.15  # smoothing factor (lower = more stable)

        self._build_ui()
        self._schedule_update()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", background=BG, foreground=FG, fieldbackground=SURFACE)
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=FG)
        style.configure("TButton", background=SURFACE, foreground=FG)
        style.configure("TLabelframe", background=BG, foreground=ACCENT)
        style.configure("TLabelframe.Label", background=BG, foreground=ACCENT)
        style.configure("Accent.TButton", background=ACCENT, foreground=BG)
        style.configure("TNotebook", background=BG)
        style.configure("TNotebook.Tab", background=SURFACE, foreground=FG,
                         padding=[12, 4])
        style.map("TNotebook.Tab", background=[("selected", ACCENT)],
                  foreground=[("selected", BG)])

        # Top bar
        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=8, pady=(8, 0))

        self.lbl_status = ttk.Label(top, text="DISCONNECTED", foreground=RED,
                                     font=("Menlo", 11, "bold"))
        self.lbl_status.pack(side="left", padx=8)

        self.lbl_fps = ttk.Label(top, text="0.0 fps", font=("Menlo", 10))
        self.lbl_fps.pack(side="left", padx=16)

        self.lbl_detections = ttk.Label(top, text="Det: 0", font=("Menlo", 10))
        self.lbl_detections.pack(side="left", padx=16)

        self.lbl_frame = ttk.Label(top, text="Frame: 0", font=("Menlo", 10))
        self.lbl_frame.pack(side="left", padx=16)

        self.btn_connect = ttk.Button(top, text="Connect",
                                       command=self._on_connect,
                                       style="Accent.TButton")
        self.btn_connect.pack(side="right", padx=4)

        self.btn_record = ttk.Button(top, text="Record", command=self._on_record)
        self.btn_record.pack(side="right", padx=4)

        # -- Tabbed notebook layout --
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=8, pady=8)

        tab_display = ttk.Frame(nb)
        tab_control = ttk.Frame(nb)
        tab_log = ttk.Frame(nb)
        nb.add(tab_display, text="  Display  ")
        nb.add(tab_control, text="  Control  ")
        nb.add(tab_log, text="  Log  ")

        self._build_display_tab(tab_display)
        self._build_control_tab(tab_control)
        self._build_log_tab(tab_log)

    def _build_display_tab(self, parent):
        # Compute physical axis limits
        # Range resolution: dR = c / (2 * BW) per range bin
        # But we decimate 1024→64 bins, so each bin spans 16 FFT bins.
        # Range resolution derivation: c/(2*BW) gives ~0.3 m per FFT bin.
        # After 1024-to-64 decimation each displayed range bin spans 16 FFT bins.
        range_res = self.C / (2.0 * self.BANDWIDTH)  # ~0.3 m per FFT bin
        # After decimation 1024→64, each range bin = 16 FFT bins
        range_per_bin = range_res * 16
        max_range = range_per_bin * NUM_RANGE_BINS

        doppler_bin_lo = 0
        doppler_bin_hi = NUM_DOPPLER_BINS

        # Matplotlib figure with 3 subplots
        self.fig = Figure(figsize=(14, 7), facecolor=BG)
        self.fig.subplots_adjust(left=0.07, right=0.98, top=0.94, bottom=0.10,
                                  wspace=0.30, hspace=0.35)

        # Range-Doppler heatmap
        self.ax_rd = self.fig.add_subplot(1, 3, (1, 2))
        self.ax_rd.set_facecolor(BG2)
        self._rd_img = self.ax_rd.imshow(
            np.zeros((NUM_RANGE_BINS, NUM_DOPPLER_BINS)),
            aspect="auto", cmap="inferno", origin="lower",
            extent=[doppler_bin_lo, doppler_bin_hi, 0, max_range],
            vmin=0, vmax=1000,
        )
        self.ax_rd.set_title("Range-Doppler Map", color=FG, fontsize=12)
        self.ax_rd.set_xlabel("Doppler Bin (0-15: long PRI, 16-31: short PRI)", color=FG)
        self.ax_rd.set_ylabel("Range (m)", color=FG)
        self.ax_rd.tick_params(colors=FG)

        # Save axis limits for coordinate conversions
        self._max_range = max_range
        self._range_per_bin = range_per_bin

        # CFAR detection overlay (scatter)
        self._det_scatter = self.ax_rd.scatter([], [], s=30, c=GREEN,
                                                marker="x", linewidths=1.5,
                                                zorder=5, label="CFAR Det")

        # Waterfall plot (range profile vs time)
        self.ax_wf = self.fig.add_subplot(1, 3, 3)
        self.ax_wf.set_facecolor(BG2)
        wf_init = np.zeros((WATERFALL_DEPTH, NUM_RANGE_BINS))
        self._wf_img = self.ax_wf.imshow(
            wf_init, aspect="auto", cmap="viridis", origin="lower",
            extent=[0, max_range, 0, WATERFALL_DEPTH],
            vmin=0, vmax=5000,
        )
        self.ax_wf.set_title("Range Waterfall", color=FG, fontsize=12)
        self.ax_wf.set_xlabel("Range (m)", color=FG)
        self.ax_wf.set_ylabel("Frame", color=FG)
        self.ax_wf.tick_params(colors=FG)

        canvas = FigureCanvasTkAgg(self.fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self._canvas = canvas

    def _build_control_tab(self, parent):
        """Host command sender — organized by FPGA register groups.

        Layout: scrollable canvas with three columns:
          Left:   Quick Actions + Diagnostics (self-test)
          Center: Waveform Timing + Signal Processing
          Right:  Detection (CFAR) + Custom Command
        """
        # Scrollable wrapper for small screens
        canvas = tk.Canvas(parent, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        outer = ttk.Frame(canvas)
        outer.bind("<Configure>",
                   lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=outer, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True, padx=8, pady=8)
        scrollbar.pack(side="right", fill="y")

        self._param_vars: dict[str, tk.StringVar] = {}

        # ── Left column: Quick Actions + Diagnostics ──────────────────
        left = ttk.Frame(outer)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        # -- Radar Operation --
        grp_op = ttk.LabelFrame(left, text="Radar Operation", padding=10)
        grp_op.pack(fill="x", pady=(0, 8))

        ttk.Button(grp_op, text="Radar Mode On",
                   command=lambda: self._send_cmd(0x01, 1)).pack(fill="x", pady=2)
        ttk.Button(grp_op, text="Radar Mode Off",
                   command=lambda: self._send_cmd(0x01, 0)).pack(fill="x", pady=2)
        ttk.Button(grp_op, text="Trigger Chirp",
                   command=lambda: self._send_cmd(0x02, 1)).pack(fill="x", pady=2)

        # Stream Control (3-bit mask)
        sc_row = ttk.Frame(grp_op)
        sc_row.pack(fill="x", pady=2)
        ttk.Label(sc_row, text="Stream Control").pack(side="left")
        var_sc = tk.StringVar(value="7")
        self._param_vars["4"] = var_sc
        ttk.Entry(sc_row, textvariable=var_sc, width=6).pack(side="left", padx=6)
        ttk.Label(sc_row, text="0-7", foreground=ACCENT,
                  font=("Menlo", 9)).pack(side="left")
        ttk.Button(sc_row, text="Set",
                   command=lambda: self._send_validated(
                       0x04, var_sc, bits=3)).pack(side="right")

        ttk.Button(grp_op, text="Request Status",
                   command=lambda: self._send_cmd(0xFF, 0)).pack(fill="x", pady=2)

        # -- Signal Processing --
        grp_sp = ttk.LabelFrame(left, text="Signal Processing", padding=10)
        grp_sp.pack(fill="x", pady=(0, 8))

        sp_params = [
            # Format: label, opcode, default, bits, hint
            ("Detect Threshold",  0x03, "10000", 16, "0-65535"),
            ("Gain Shift",        0x16, "0",     4,  "0-15, dir+shift"),
            ("MTI Enable",        0x26, "0",     1,  "0=off, 1=on"),
            ("DC Notch Width",    0x27, "0",     3,  "0-7 bins"),
        ]
        for label, opcode, default, bits, hint in sp_params:
            self._add_param_row(grp_sp, label, opcode, default, bits, hint)

        # MTI quick toggle
        mti_row = ttk.Frame(grp_sp)
        mti_row.pack(fill="x", pady=2)
        ttk.Button(mti_row, text="Enable MTI",
                   command=lambda: self._send_cmd(0x26, 1)).pack(
                       side="left", expand=True, fill="x", padx=(0, 2))
        ttk.Button(mti_row, text="Disable MTI",
                   command=lambda: self._send_cmd(0x26, 0)).pack(
                       side="left", expand=True, fill="x", padx=(2, 0))

        # -- Diagnostics --
        grp_diag = ttk.LabelFrame(left, text="Diagnostics", padding=10)
        grp_diag.pack(fill="x", pady=(0, 8))

        ttk.Button(grp_diag, text="Run Self-Test",
                   command=lambda: self._send_cmd(0x30, 1)).pack(fill="x", pady=2)
        ttk.Button(grp_diag, text="Read Self-Test Result",
                   command=lambda: self._send_cmd(0x31, 0)).pack(fill="x", pady=2)

        st_frame = ttk.LabelFrame(grp_diag, text="Self-Test Results", padding=6)
        st_frame.pack(fill="x", pady=(4, 0))
        self._st_labels = {}
        for name, default_text in [
            ("busy", "Busy: --"),
            ("flags", "Flags: -----"),
            ("detail", "Detail: 0x--"),
            ("t0", "T0 BRAM: --"),
            ("t1", "T1 CIC:  --"),
            ("t2", "T2 FFT:  --"),
            ("t3", "T3 Arith: --"),
            ("t4", "T4 ADC:  --"),
        ]:
            lbl = ttk.Label(st_frame, text=default_text, font=("Menlo", 9))
            lbl.pack(anchor="w")
            self._st_labels[name] = lbl

        # ── Center column: Waveform Timing ────────────────────────────
        center = ttk.Frame(outer)
        center.grid(row=0, column=1, sticky="nsew", padx=6)

        grp_wf = ttk.LabelFrame(center, text="Waveform Timing", padding=10)
        grp_wf.pack(fill="x", pady=(0, 8))

        wf_params = [
            ("Long Chirp Cycles",   0x10, "3000",  16, "0-65535, rst=3000"),
            ("Long Listen Cycles",  0x11, "13700", 16, "0-65535, rst=13700"),
            ("Guard Cycles",        0x12, "17540", 16, "0-65535, rst=17540"),
            ("Short Chirp Cycles",  0x13, "50",    16, "0-65535, rst=50"),
            ("Short Listen Cycles", 0x14, "17450", 16, "0-65535, rst=17450"),
            ("Chirps Per Elevation", 0x15, "32",    6, "1-32, clamped"),
        ]
        for label, opcode, default, bits, hint in wf_params:
            self._add_param_row(grp_wf, label, opcode, default, bits, hint)

        # ── Right column: Detection (CFAR) + Custom ───────────────────
        right = ttk.Frame(outer)
        right.grid(row=0, column=2, sticky="nsew", padx=(6, 0))

        grp_cfar = ttk.LabelFrame(right, text="Detection (CFAR)", padding=10)
        grp_cfar.pack(fill="x", pady=(0, 8))

        cfar_params = [
            ("CFAR Enable",       0x25, "0",  1,  "0=off, 1=on"),
            ("CFAR Guard Cells",  0x21, "2",  4,  "0-15, rst=2"),
            ("CFAR Train Cells",  0x22, "8",  5,  "1-31, rst=8"),
            ("CFAR Alpha (Q4.4)", 0x23, "48", 8,  "0-255, rst=0x30=3.0"),
            ("CFAR Mode",         0x24, "0",  2,  "0=CA 1=GO 2=SO"),
        ]
        for label, opcode, default, bits, hint in cfar_params:
            self._add_param_row(grp_cfar, label, opcode, default, bits, hint)

        # CFAR quick toggle
        cfar_row = ttk.Frame(grp_cfar)
        cfar_row.pack(fill="x", pady=2)
        ttk.Button(cfar_row, text="Enable CFAR",
                   command=lambda: self._send_cmd(0x25, 1)).pack(
                       side="left", expand=True, fill="x", padx=(0, 2))
        ttk.Button(cfar_row, text="Disable CFAR",
                   command=lambda: self._send_cmd(0x25, 0)).pack(
                       side="left", expand=True, fill="x", padx=(2, 0))

        # ── AGC (Automatic Gain Control) ──────────────────────────────
        grp_agc = ttk.LabelFrame(right, text="AGC (Auto Gain)", padding=10)
        grp_agc.pack(fill="x", pady=(0, 8))

        agc_params = [
            ("AGC Enable",   0x28, "0",   1, "0=manual, 1=auto"),
            ("AGC Target",   0x29, "200", 8, "0-255, peak target"),
            ("AGC Attack",   0x2A, "1",   4, "0-15, atten step"),
            ("AGC Decay",    0x2B, "1",   4, "0-15, gain-up step"),
            ("AGC Holdoff",  0x2C, "4",   4, "0-15, frames"),
        ]
        for label, opcode, default, bits, hint in agc_params:
            self._add_param_row(grp_agc, label, opcode, default, bits, hint)

        # AGC quick toggle
        agc_row = ttk.Frame(grp_agc)
        agc_row.pack(fill="x", pady=2)
        ttk.Button(agc_row, text="Enable AGC",
                   command=lambda: self._send_cmd(0x28, 1)).pack(
                       side="left", expand=True, fill="x", padx=(0, 2))
        ttk.Button(agc_row, text="Disable AGC",
                   command=lambda: self._send_cmd(0x28, 0)).pack(
                       side="left", expand=True, fill="x", padx=(2, 0))

        # AGC status readback labels
        agc_st = ttk.LabelFrame(grp_agc, text="AGC Status", padding=6)
        agc_st.pack(fill="x", pady=(4, 0))
        self._agc_labels = {}
        for name, default_text in [
            ("enable", "AGC: --"),
            ("gain",   "Gain: --"),
            ("peak",   "Peak: --"),
            ("sat",    "Sat Count: --"),
        ]:
            lbl = ttk.Label(agc_st, text=default_text, font=("Menlo", 9))
            lbl.pack(anchor="w")
            self._agc_labels[name] = lbl

        # ── Custom Command (advanced / debug) ─────────────────────────
        grp_cust = ttk.LabelFrame(right, text="Custom Command", padding=10)
        grp_cust.pack(fill="x", pady=(0, 8))

        r0 = ttk.Frame(grp_cust)
        r0.pack(fill="x", pady=2)
        ttk.Label(r0, text="Opcode (hex)").pack(side="left")
        self._custom_op = tk.StringVar(value="01")
        ttk.Entry(r0, textvariable=self._custom_op, width=8).pack(
            side="left", padx=6)

        r1 = ttk.Frame(grp_cust)
        r1.pack(fill="x", pady=2)
        ttk.Label(r1, text="Value (dec)").pack(side="left")
        self._custom_val = tk.StringVar(value="0")
        ttk.Entry(r1, textvariable=self._custom_val, width=8).pack(
            side="left", padx=6)

        ttk.Button(grp_cust, text="Send",
                   command=self._send_custom).pack(fill="x", pady=2)

        # Column weights
        outer.columnconfigure(0, weight=1)
        outer.columnconfigure(1, weight=1)
        outer.columnconfigure(2, weight=1)
        outer.rowconfigure(0, weight=1)

    def _add_param_row(self, parent, label: str, opcode: int,
                       default: str, bits: int, hint: str):
        """Add a single parameter row: label, entry, hint, Set button with validation."""
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label).pack(side="left")
        var = tk.StringVar(value=default)
        self._param_vars[str(opcode)] = var
        ttk.Entry(row, textvariable=var, width=8).pack(side="left", padx=6)
        ttk.Label(row, text=hint, foreground=ACCENT,
                  font=("Menlo", 9)).pack(side="left")
        ttk.Button(row, text="Set",
                   command=lambda: self._send_validated(
                       opcode, var, bits=bits)).pack(side="right")

    def _send_validated(self, opcode: int, var: tk.StringVar, bits: int):
        """Parse, clamp to bit-width, send command, and update the entry."""
        try:
            raw = int(var.get())
        except ValueError:
            log.error(f"Invalid value for opcode 0x{opcode:02X}: {var.get()!r}")
            return
        max_val = (1 << bits) - 1
        clamped = max(0, min(raw, max_val))
        if clamped != raw:
            log.warning(f"Value {raw} clamped to {clamped} "
                        f"({bits}-bit max={max_val}) for opcode 0x{opcode:02X}")
            var.set(str(clamped))
        self._send_cmd(opcode, clamped)

    def _build_log_tab(self, parent):
        self.log_text = tk.Text(parent, bg=BG2, fg=FG, font=("Menlo", 10),
                                 insertbackground=FG, wrap="word")
        self.log_text.pack(fill="both", expand=True, padx=8, pady=8)

        # Redirect log handler to text widget
        handler = _TextHandler(self.log_text)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                                datefmt="%H:%M:%S"))
        logging.getLogger().addHandler(handler)

    # ------------------------------------------------------------ Actions
    def _on_connect(self):
        if self.conn.is_open:
            # Disconnect
            if self._acq_thread is not None:
                self._acq_thread.stop()
                self._acq_thread.join(timeout=2)
                self._acq_thread = None
            self.conn.close()
            self.lbl_status.config(text="DISCONNECTED", foreground=RED)
            self.btn_connect.config(text="Connect")
            log.info("Disconnected")
            return

        # Open connection in a background thread to avoid blocking the GUI
        self.lbl_status.config(text="CONNECTING...", foreground=YELLOW)
        self.btn_connect.config(state="disabled")
        self.root.update_idletasks()

        def _do_connect():
            ok = self.conn.open(self.device_index)
            # Schedule UI update back on the main thread
            self.root.after(0, lambda: self._on_connect_done(ok))

        threading.Thread(target=_do_connect, daemon=True).start()

    def _on_connect_done(self, success: bool):
        """Called on main thread after connection attempt completes."""
        self.btn_connect.config(state="normal")
        if success:
            self.lbl_status.config(text="CONNECTED", foreground=GREEN)
            self.btn_connect.config(text="Disconnect")
            self._acq_thread = RadarAcquisition(
                self.conn, self.frame_queue, self.recorder,
                status_callback=self._on_status_received)
            self._acq_thread.start()
            log.info("Connected and acquisition started")
        else:
            self.lbl_status.config(text="CONNECT FAILED", foreground=RED)
            self.btn_connect.config(text="Connect")

    def _on_record(self):
        if self.recorder.recording:
            self.recorder.stop()
            self.btn_record.config(text="Record")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".h5",
            filetypes=[("HDF5", "*.h5"), ("All", "*.*")],
            initialfile=f"radar_{time.strftime('%Y%m%d_%H%M%S')}.h5",
        )
        if filepath:
            self.recorder.start(filepath)
            self.btn_record.config(text="Stop Rec")

    def _send_cmd(self, opcode: int, value: int):
        cmd = RadarProtocol.build_command(opcode, value)
        ok = self.conn.write(cmd)
        log.info(f"CMD 0x{opcode:02X} val={value} ({'OK' if ok else 'FAIL'})")

    def _send_custom(self):
        try:
            op = int(self._custom_op.get(), 16)
            val = int(self._custom_val.get())
            self._send_cmd(op, val)
        except ValueError:
            log.error("Invalid custom command values")

    def _on_status_received(self, status: StatusResponse):
        """Called from acquisition thread — schedule UI update on main thread."""
        self.root.after(0, self._update_self_test_labels, status)

    def _update_self_test_labels(self, status: StatusResponse):
        """Update the self-test result labels and AGC status from a StatusResponse."""
        if not hasattr(self, '_st_labels'):
            return
        flags = status.self_test_flags
        detail = status.self_test_detail
        busy = status.self_test_busy

        busy_str = "RUNNING" if busy else "IDLE"
        busy_color = YELLOW if busy else FG
        self._st_labels["busy"].config(text=f"Busy: {busy_str}",
                                        foreground=busy_color)
        self._st_labels["flags"].config(text=f"Flags: {flags:05b}")
        self._st_labels["detail"].config(text=f"Detail: 0x{detail:02X}")

        # Individual test results (bit = 1 means PASS)
        test_names = [
            ("t0", "T0 BRAM"),
            ("t1", "T1 CIC"),
            ("t2", "T2 FFT"),
            ("t3", "T3 Arith"),
            ("t4", "T4 ADC"),
        ]
        for i, (key, name) in enumerate(test_names):
            if busy:
                result_str = "..."
                color = YELLOW
            elif flags & (1 << i):
                result_str = "PASS"
                color = GREEN
            else:
                result_str = "FAIL"
                color = RED
            self._st_labels[key].config(
                text=f"{name}: {result_str}", foreground=color)

        # AGC status readback
        if hasattr(self, '_agc_labels'):
            agc_str = "AUTO" if status.agc_enable else "MANUAL"
            agc_color = GREEN if status.agc_enable else FG
            self._agc_labels["enable"].config(
                text=f"AGC: {agc_str}", foreground=agc_color)
            self._agc_labels["gain"].config(
                text=f"Gain: {status.agc_current_gain}")
            self._agc_labels["peak"].config(
                text=f"Peak: {status.agc_peak_magnitude}")
            sat_color = RED if status.agc_saturation_count > 0 else FG
            self._agc_labels["sat"].config(
                text=f"Sat Count: {status.agc_saturation_count}",
                foreground=sat_color)

    # --------------------------------------------------------- Display loop
    def _schedule_update(self):
        self._update_display()
        self.root.after(self.UPDATE_INTERVAL_MS, self._schedule_update)

    def _update_display(self):
        """Pull latest frame from queue and update plots."""
        frame = None
        # Drain queue, keep latest
        while True:
            try:
                frame = self.frame_queue.get_nowait()
            except queue.Empty:
                break

        if frame is None:
            return

        self._current_frame = frame
        self._frame_count += 1

        # FPS calculation
        now = time.time()
        dt = now - self._fps_ts
        if dt > 0.5:
            self._fps = self._frame_count / dt
            self._frame_count = 0
            self._fps_ts = now

        # Update labels
        self.lbl_fps.config(text=f"{self._fps:.1f} fps")
        self.lbl_detections.config(text=f"Det: {frame.detection_count}")
        self.lbl_frame.config(text=f"Frame: {frame.frame_number}")

        # Update range-Doppler heatmap in raw dual-subframe bin order
        mag = frame.magnitude
        det_shifted = frame.detections

        # Stable colorscale via EMA smoothing of vmax
        frame_vmax = float(np.max(mag)) if np.max(mag) > 0 else 1.0
        self._vmax_ema = (self._vmax_alpha * frame_vmax +
                          (1.0 - self._vmax_alpha) * self._vmax_ema)
        stable_vmax = max(self._vmax_ema, 1.0)

        self._rd_img.set_data(mag)
        self._rd_img.set_clim(vmin=0, vmax=stable_vmax)

        # Update CFAR overlay in raw Doppler-bin coordinates
        det_coords = np.argwhere(det_shifted > 0)
        if len(det_coords) > 0:
            # det_coords[:, 0] = range bin, det_coords[:, 1] = Doppler bin
            range_m = (det_coords[:, 0] + 0.5) * self._range_per_bin
            doppler_bins = det_coords[:, 1] + 0.5
            offsets = np.column_stack([doppler_bins, range_m])
            self._det_scatter.set_offsets(offsets)
        else:
            self._det_scatter.set_offsets(np.empty((0, 2)))

        # Update waterfall
        self._waterfall.append(frame.range_profile.copy())
        wf_arr = np.array(list(self._waterfall))
        wf_max = max(np.max(wf_arr), 1.0)
        self._wf_img.set_data(wf_arr)
        self._wf_img.set_clim(vmin=0, vmax=wf_max)

        self._canvas.draw_idle()


class _TextHandler(logging.Handler):
    """Logging handler that writes to a tkinter Text widget."""

    def __init__(self, text_widget: tk.Text):
        super().__init__()
        self._text = text_widget

    def emit(self, record):
        msg = self.format(record)
        with contextlib.suppress(Exception):
            self._text.after(0, self._append, msg)

    def _append(self, msg: str):
        self._text.insert("end", msg + "\n")
        self._text.see("end")
        # Keep last 500 lines
        lines = int(self._text.index("end-1c").split(".")[0])
        if lines > 500:
            self._text.delete("1.0", f"{lines - 500}.0")


# ============================================================================
# Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="AERIS-10 Radar Dashboard")
    parser.add_argument("--live", action="store_true",
                        help="Use real FT2232H hardware (default: mock mode)")
    parser.add_argument("--replay", type=str, metavar="NPY_DIR",
                        help="Replay real data from .npy directory "
                             "(e.g. tb/cosim/real_data/hex/)")
    parser.add_argument("--no-mti", action="store_true",
                        help="With --replay, use non-MTI Doppler data")
    parser.add_argument("--record", action="store_true",
                        help="Start HDF5 recording immediately")
    parser.add_argument("--device", type=int, default=0,
                        help="FT2232H device index (default: 0)")
    args = parser.parse_args()

    if args.replay:
        npy_dir = os.path.abspath(args.replay)
        conn = ReplayConnection(npy_dir, use_mti=not args.no_mti)
        mode_str = f"REPLAY ({npy_dir}, MTI={'OFF' if args.no_mti else 'ON'})"
    elif args.live:
        conn = FT2232HConnection(mock=False)
        mode_str = "LIVE"
    else:
        conn = FT2232HConnection(mock=True)
        mode_str = "MOCK"

    recorder = DataRecorder()

    root = tk.Tk()

    dashboard = RadarDashboard(root, conn, recorder, device_index=args.device)

    if args.record:
        filepath = os.path.join(
            os.getcwd(),
            f"radar_{time.strftime('%Y%m%d_%H%M%S')}.h5"
        )
        recorder.start(filepath)

    def on_closing():
        if dashboard._acq_thread is not None:
            dashboard._acq_thread.stop()
            dashboard._acq_thread.join(timeout=2)
        if conn.is_open:
            conn.close()
        if recorder.recording:
            recorder.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)

    log.info(f"Dashboard started (mode={mode_str})")
    root.mainloop()


if __name__ == "__main__":
    main()
