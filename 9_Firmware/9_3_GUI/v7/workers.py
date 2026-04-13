"""
v7.workers — QThread-based workers and demo target simulator.

Classes:
  - RadarDataWorker      — reads from FT2232H via production RadarAcquisition,
                           parses 0xAA/0xBB packets, assembles 64x32 frames,
                           runs host-side DSP, emits PyQt signals.
  - RawIQReplayWorker    — reads raw IQ .npy frames from RawIQReplayController,
                           processes through RawIQFrameProcessor, emits same
                           signals as RadarDataWorker + playback state.
  - GPSDataWorker        — reads GPS frames from STM32 CDC, emits GPSData signals.
  - TargetSimulator      — QTimer-based demo target generator.

The old V6/V7 packet parsing (sync A5 C3 + type + CRC16) has been removed.
All packet parsing now uses the production radar_protocol.py which matches
the actual FPGA packet format (0xAA data 11-byte, 0xBB status 26-byte).
"""

import math
import time
import random
import queue
import struct
import logging

from PyQt6.QtCore import QThread, QObject, QTimer, pyqtSignal

from .models import RadarTarget, GPSData, RadarSettings
from .hardware import (
    RadarAcquisition,
    RadarFrame,
    StatusResponse,
    DataRecorder,
    STM32USBInterface,
)
from .processing import (
    RadarProcessor,
    RawIQFrameProcessor,
    USBPacketParser,
    extract_targets_from_frame,
)
from .raw_iq_replay import RawIQReplayController, PlaybackState

logger = logging.getLogger(__name__)


# =============================================================================
# Utility: polar → geographic
# =============================================================================

def polar_to_geographic(
    radar_lat: float,
    radar_lon: float,
    range_m: float,
    azimuth_deg: float,
) -> tuple:
    """
    Convert polar coordinates (range, azimuth) relative to radar
    to geographic (latitude, longitude).

    azimuth_deg: 0 = North, clockwise.
    Returns (lat, lon).
    """
    R = 6_371_000  # Earth radius in meters

    lat1 = math.radians(radar_lat)
    lon1 = math.radians(radar_lon)
    bearing = math.radians(azimuth_deg)

    lat2 = math.asin(
        math.sin(lat1) * math.cos(range_m / R)
        + math.cos(lat1) * math.sin(range_m / R) * math.cos(bearing)
    )
    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(range_m / R) * math.cos(lat1),
        math.cos(range_m / R) - math.sin(lat1) * math.sin(lat2),
    )
    return (math.degrees(lat2), math.degrees(lon2))


# =============================================================================
# Radar Data Worker (QThread) — production protocol
# =============================================================================

class RadarDataWorker(QThread):
    """
    Background worker that reads radar data from FT2232H (or ReplayConnection),
    parses 0xAA/0xBB packets via production RadarAcquisition, runs optional
    host-side DSP, and emits PyQt signals with results.

    This replaces the old V7 worker which used an incompatible packet format.
    Now uses production radar_protocol.py for all packet parsing and frame
    assembly (11-byte 0xAA data packets → 64x32 RadarFrame).

    Signals:
        frameReady(RadarFrame)    — a complete 64x32 radar frame
        statusReceived(object)    — StatusResponse from FPGA
        targetsUpdated(list)      — list of RadarTarget after host-side DSP
        errorOccurred(str)        — error message
        statsUpdated(dict)        — frame/byte counters
    """

    frameReady = pyqtSignal(object)       # RadarFrame
    statusReceived = pyqtSignal(object)   # StatusResponse
    targetsUpdated = pyqtSignal(list)     # List[RadarTarget]
    errorOccurred = pyqtSignal(str)
    statsUpdated = pyqtSignal(dict)

    def __init__(
        self,
        connection,  # FT2232HConnection or ReplayConnection
        processor: RadarProcessor | None = None,
        recorder: DataRecorder | None = None,
        gps_data_ref: GPSData | None = None,
        settings: RadarSettings | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._connection = connection
        self._processor = processor
        self._recorder = recorder
        self._gps = gps_data_ref
        self._settings = settings or RadarSettings()
        self._running = False

        # Frame queue for production RadarAcquisition → this thread
        self._frame_queue: queue.Queue = queue.Queue(maxsize=4)

        # Production acquisition thread (does the actual parsing)
        self._acquisition: RadarAcquisition | None = None

        # Counters
        self._frame_count = 0
        self._byte_count = 0
        self._error_count = 0

    def stop(self):
        self._running = False
        if self._acquisition:
            self._acquisition.stop()

    def run(self):
        """
        Start production RadarAcquisition thread, then poll its frame queue
        and emit PyQt signals for each complete frame.
        """
        self._running = True

        # Create and start the production acquisition thread
        self._acquisition = RadarAcquisition(
            connection=self._connection,
            frame_queue=self._frame_queue,
            recorder=self._recorder,
            status_callback=self._on_status,
        )
        self._acquisition.start()
        logger.info("RadarDataWorker started (production protocol)")

        while self._running:
            try:
                # Poll for complete frames from production acquisition
                frame: RadarFrame = self._frame_queue.get(timeout=0.1)
                self._frame_count += 1

                # Emit raw frame
                self.frameReady.emit(frame)

                # Run host-side DSP if processor is configured
                if self._processor is not None:
                    targets = self._run_host_dsp(frame)
                    if targets:
                        self.targetsUpdated.emit(targets)

                # Emit stats
                self.statsUpdated.emit({
                    "frames": self._frame_count,
                    "detection_count": frame.detection_count,
                    "errors": self._error_count,
                })

            except queue.Empty:
                continue
            except (ValueError, IndexError) as e:
                self._error_count += 1
                self.errorOccurred.emit(str(e))
                logger.error(f"RadarDataWorker error: {e}")

        # Stop acquisition thread
        if self._acquisition:
            self._acquisition.stop()
            self._acquisition.join(timeout=2.0)
            self._acquisition = None

        logger.info("RadarDataWorker stopped")

    def _on_status(self, status: StatusResponse):
        """Callback from production RadarAcquisition on status packet."""
        self.statusReceived.emit(status)

    def _run_host_dsp(self, frame: RadarFrame) -> list[RadarTarget]:
        """
        Run host-side DSP on a complete frame.
        This is where DBSCAN clustering, Kalman tracking, and other
        non-timing-critical processing happens.

        The FPGA already does: FFT, MTI, CFAR, DC notch.
        Host-side DSP adds: clustering, tracking, geo-coordinate mapping.

        Bin-to-physical conversion uses RadarSettings.range_resolution
        and velocity_resolution (should be calibrated to actual waveform).
        """
        cfg = self._processor.config
        if not (cfg.clustering_enabled or cfg.tracking_enabled):
            return []

        targets = extract_targets_from_frame(
            frame,
            self._settings.range_resolution,
            self._settings.velocity_resolution,
            gps=self._gps,
        )

        # DBSCAN clustering
        if cfg.clustering_enabled and len(targets) > 0:
            clusters = self._processor.clustering(
                targets, cfg.clustering_eps, cfg.clustering_min_samples)
            # Associate and track
            if cfg.tracking_enabled:
                targets = self._processor.association(targets, clusters)
                self._processor.tracking(targets)

        return targets


# =============================================================================
# Raw IQ Replay Worker (QThread) — processes raw .npy captures
# =============================================================================

class RawIQReplayWorker(QThread):
    """Background worker for raw IQ replay mode.

    Reads frames from a RawIQReplayController, processes them through
    RawIQFrameProcessor (quantize -> AGC -> FFT -> CFAR -> RadarFrame),
    and emits the same signals as RadarDataWorker so the dashboard can
    display them identically.

    Additional signal:
        playbackStateChanged(str)  — "playing", "paused", "stopped"
        frameIndexChanged(int, int) — (current_index, total_frames)

    Signals:
        frameReady(RadarFrame)
        statusReceived(object)
        targetsUpdated(list)
        errorOccurred(str)
        statsUpdated(dict)
        playbackStateChanged(str)
        frameIndexChanged(int, int)
    """

    frameReady = pyqtSignal(object)
    statusReceived = pyqtSignal(object)
    targetsUpdated = pyqtSignal(list)
    errorOccurred = pyqtSignal(str)
    statsUpdated = pyqtSignal(dict)
    playbackStateChanged = pyqtSignal(str)
    frameIndexChanged = pyqtSignal(int, int)

    def __init__(
        self,
        controller: RawIQReplayController,
        processor: RawIQFrameProcessor,
        host_processor: RadarProcessor | None = None,
        settings: RadarSettings | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._controller = controller
        self._processor = processor
        self._host_processor = host_processor
        self._settings = settings or RadarSettings()
        self._running = False
        self._frame_count = 0
        self._error_count = 0

    def stop(self):
        self._running = False
        self._controller.stop()

    def run(self):
        self._running = True
        self._frame_count = 0
        logger.info("RawIQReplayWorker started")

        info = self._controller.info
        total_frames = info.n_frames if info else 0

        while self._running:
            try:
                # Block until next frame or stop
                raw_frame = self._controller.next_frame()
                if raw_frame is None:
                    # Stopped or end of file
                    if self._running:
                        self.playbackStateChanged.emit("stopped")
                    break

                # Process through full signal chain
                import time as _time
                ts = _time.time()
                frame, status, _agc_result = self._processor.process_frame(
                    raw_frame, timestamp=ts)
                self._frame_count += 1

                # Emit signals
                self.frameReady.emit(frame)
                self.statusReceived.emit(status)

                # Emit frame index
                idx = self._controller.frame_index
                self.frameIndexChanged.emit(idx, total_frames)

                # Emit playback state
                state = self._controller.state
                self.playbackStateChanged.emit(state.name.lower())

                # Run host-side DSP if configured
                if self._host_processor is not None:
                    targets = self._extract_targets(frame)
                    if targets:
                        self.targetsUpdated.emit(targets)

                # Stats
                self.statsUpdated.emit({
                    "frames": self._frame_count,
                    "detection_count": frame.detection_count,
                    "errors": self._error_count,
                    "frame_index": idx,
                    "total_frames": total_frames,
                })

                # Rate limiting: sleep to match target FPS
                fps = self._controller.fps
                if fps > 0 and self._controller.state == PlaybackState.PLAYING:
                    self.msleep(int(1000.0 / fps))

            except (ValueError, IndexError) as e:
                self._error_count += 1
                self.errorOccurred.emit(str(e))
                logger.error(f"RawIQReplayWorker error: {e}")

        self._running = False
        logger.info("RawIQReplayWorker stopped")

    def _extract_targets(self, frame: RadarFrame) -> list[RadarTarget]:
        """Extract targets from detection mask using shared bin-to-physical conversion."""
        targets = extract_targets_from_frame(
            frame,
            self._settings.range_resolution,
            self._settings.velocity_resolution,
        )

        # Clustering + tracking
        if self._host_processor is not None:
            cfg = self._host_processor.config
            if cfg.clustering_enabled and len(targets) > 0:
                clusters = self._host_processor.clustering(
                    targets, cfg.clustering_eps, cfg.clustering_min_samples)
                if cfg.tracking_enabled:
                    targets = self._host_processor.association(targets, clusters)
                    self._host_processor.tracking(targets)

        return targets


# =============================================================================
# GPS Data Worker (QThread)
# =============================================================================

class GPSDataWorker(QThread):
    """
    Background worker that reads GPS frames from the STM32 USB CDC interface
    and emits parsed GPSData objects.

    Signals:
        gpsReceived(GPSData)
        errorOccurred(str)
    """

    gpsReceived = pyqtSignal(object)   # GPSData
    errorOccurred = pyqtSignal(str)

    def __init__(
        self,
        stm32: STM32USBInterface,
        usb_parser: USBPacketParser,
        parent=None,
    ):
        super().__init__(parent)
        self._stm32 = stm32
        self._parser = usb_parser
        self._running = False
        self._gps_count = 0

    @property
    def gps_count(self) -> int:
        return self._gps_count

    def stop(self):
        self._running = False

    def run(self):
        self._running = True
        while self._running:
            if not (self._stm32 and self._stm32.is_open):
                self.msleep(100)
                continue

            try:
                data = self._stm32.read_data(64, timeout=100)
                if data:
                    gps = self._parser.parse_gps_data(data)
                    if gps:
                        self._gps_count += 1
                        self.gpsReceived.emit(gps)
            except (ValueError, struct.error) as e:
                self.errorOccurred.emit(str(e))
                logger.error(f"GPSDataWorker error: {e}")
            self.msleep(100)


# =============================================================================
# Target Simulator (Demo Mode) — QTimer-based
# =============================================================================

class TargetSimulator(QObject):
    """
    Generates simulated radar targets for demo/testing.

    Uses a QTimer on the main thread (or whichever thread owns this object).
    Emits ``targetsUpdated`` with a list[RadarTarget] on each tick.
    """

    targetsUpdated = pyqtSignal(list)

    def __init__(self, radar_position: GPSData, parent=None):
        super().__init__(parent)
        self._radar_pos = radar_position
        self._targets: list[RadarTarget] = []
        self._next_id = 1
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._initialize_targets(8)

    # ---- public API --------------------------------------------------------

    def start(self, interval_ms: int = 500):
        self._timer.start(interval_ms)

    def stop(self):
        self._timer.stop()

    def set_radar_position(self, gps: GPSData):
        self._radar_pos = gps

    def add_random_target(self):
        self._add_random_target()

    # ---- internals ---------------------------------------------------------

    def _initialize_targets(self, count: int):
        for _ in range(count):
            self._add_random_target()

    def _add_random_target(self):
        range_m = random.uniform(5000, 40000)
        azimuth = random.uniform(0, 360)
        velocity = random.uniform(-100, 100)
        elevation = random.uniform(-5, 45)

        lat, lon = polar_to_geographic(
            self._radar_pos.latitude,
            self._radar_pos.longitude,
            range_m,
            azimuth,
        )

        target = RadarTarget(
            id=self._next_id,
            range=range_m,
            velocity=velocity,
            azimuth=azimuth,
            elevation=elevation,
            latitude=lat,
            longitude=lon,
            snr=random.uniform(10, 35),
            timestamp=time.time(),
            track_id=self._next_id,
            classification=random.choice(["aircraft", "drone", "bird", "unknown"]),
        )
        self._next_id += 1
        self._targets.append(target)

    def _tick(self):
        """Update all simulated targets and emit."""
        updated: list[RadarTarget] = []

        for t in self._targets:
            new_range = t.range - t.velocity * 0.5
            if new_range < 500 or new_range > 50000:
                continue  # target exits coverage — drop it

            new_vel = max(-150, min(150, t.velocity + random.uniform(-2, 2)))
            new_az = (t.azimuth + random.uniform(-0.5, 0.5)) % 360

            lat, lon = polar_to_geographic(
                self._radar_pos.latitude,
                self._radar_pos.longitude,
                new_range,
                new_az,
            )

            updated.append(RadarTarget(
                id=t.id,
                range=new_range,
                velocity=new_vel,
                azimuth=new_az,
                elevation=t.elevation + random.uniform(-0.1, 0.1),
                latitude=lat,
                longitude=lon,
                snr=t.snr + random.uniform(-1, 1),
                timestamp=time.time(),
                track_id=t.track_id,
                classification=t.classification,
            ))

        # Maintain a reasonable target count
        if len(updated) < 5 or (random.random() < 0.05 and len(updated) < 15):
            self._add_random_target()
            updated.append(self._targets[-1])

        self._targets = updated
        self.targetsUpdated.emit(updated)
