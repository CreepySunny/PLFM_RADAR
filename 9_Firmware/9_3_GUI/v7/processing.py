"""
v7.processing — Radar signal processing and GPS parsing.

Classes:
  - RadarProcessor       — dual-CPI fusion, multi-PRF unwrap, DBSCAN clustering,
                           association, Kalman tracking
  - RawIQFrameProcessor  — full signal chain for raw IQ replay:
                           quantize -> AGC -> Range FFT -> Doppler FFT ->
                           crop -> MTI -> DC notch -> mag -> CFAR
  - USBPacketParser      — parse GPS text/binary frames from STM32 CDC

Note: RadarPacketParser (old A5/C3 sync + CRC16 format) was removed.
      All packet parsing now uses production RadarProtocol (0xAA/0xBB format)
      from radar_protocol.py.
"""

import struct
import time
import logging
import math

import numpy as np

from .models import (
    RadarTarget, GPSData, ProcessingConfig,
    SCIPY_AVAILABLE, SKLEARN_AVAILABLE, FILTERPY_AVAILABLE,
)
from .agc_sim import (
    AGCConfig, AGCState, AGCFrameResult,
    quantize_iq, process_agc_frame,
)
from .hardware import RadarFrame, StatusResponse

if SKLEARN_AVAILABLE:
    from sklearn.cluster import DBSCAN

if FILTERPY_AVAILABLE:
    from filterpy.kalman import KalmanFilter

if SCIPY_AVAILABLE:
    from scipy.signal import windows as scipy_windows

logger = logging.getLogger(__name__)


# =============================================================================
# Utility: pitch correction (Bug #4 fix — was never defined in V6)
# =============================================================================

def apply_pitch_correction(raw_elevation: float, pitch: float) -> float:
    """
    Apply platform pitch correction to a raw elevation angle.

    Returns the corrected elevation = raw_elevation - pitch.
    """
    return raw_elevation - pitch


# =============================================================================
# Utility: bin-to-physical target extraction (shared by all workers)
# =============================================================================

def extract_targets_from_frame(
    frame: RadarFrame,
    range_resolution: float,
    velocity_resolution: float,
    *,
    gps: GPSData | None = None,
) -> list[RadarTarget]:
    """Extract RadarTargets from a RadarFrame's detection mask.

    Performs bin-to-physical conversion and optional GPS coordinate mapping.
    This is the shared implementation used by both RadarDataWorker (live mode)
    and RawIQReplayWorker (replay mode).

    Args:
        frame: RadarFrame with populated ``detections`` and ``magnitude`` arrays.
        range_resolution: Metres per range bin.
        velocity_resolution: m/s per Doppler bin.
        gps: Optional GPSData for pitch correction and geographic mapping.

    Returns:
        List of RadarTarget with physical-unit range, velocity, SNR, and
        (if GPS available) lat/lon/azimuth/elevation.
    """
    det_indices = np.argwhere(frame.detections > 0)
    if len(det_indices) == 0:
        return []

    n_doppler = frame.magnitude.shape[1]
    center_dbin = n_doppler // 2
    targets: list[RadarTarget] = []

    for idx in det_indices:
        rbin, dbin = idx
        mag = frame.magnitude[rbin, dbin]
        snr = 10 * np.log10(max(mag, 1)) if mag > 0 else 0.0

        range_m = float(rbin) * range_resolution
        velocity_ms = float(dbin - center_dbin) * velocity_resolution

        # GPS-dependent fields
        raw_elev = 0.0
        corr_elev = raw_elev
        lat, lon, azimuth = 0.0, 0.0, 0.0
        if gps is not None:
            corr_elev = apply_pitch_correction(raw_elev, gps.pitch)
            azimuth = gps.heading
            lat, lon = _polar_to_geographic(
                gps.latitude, gps.longitude, range_m, azimuth)

        targets.append(RadarTarget(
            id=len(targets),
            range=range_m,
            velocity=velocity_ms,
            azimuth=azimuth,
            elevation=corr_elev,
            latitude=lat,
            longitude=lon,
            snr=snr,
            timestamp=frame.timestamp,
        ))

    return targets


def _polar_to_geographic(
    radar_lat: float, radar_lon: float, range_m: float, bearing_deg: float,
) -> tuple[float, float]:
    """Convert polar (range, bearing) to geographic (lat, lon).

    Uses the spherical-Earth approximation (adequate for <50 km ranges).
    Duplicated from ``workers.polar_to_geographic`` to keep processing.py
    self-contained; the workers module still exports its own copy for
    backward-compat.
    """
    if range_m <= 0:
        return radar_lat, radar_lon
    earth_r = 6_371_000.0
    lat_r = math.radians(radar_lat)
    lon_r = math.radians(radar_lon)
    brg_r = math.radians(bearing_deg)
    d_r = range_m / earth_r

    new_lat = math.asin(
        math.sin(lat_r) * math.cos(d_r)
        + math.cos(lat_r) * math.sin(d_r) * math.cos(brg_r)
    )
    new_lon = lon_r + math.atan2(
        math.sin(brg_r) * math.sin(d_r) * math.cos(lat_r),
        math.cos(d_r) - math.sin(lat_r) * math.sin(new_lat),
    )
    return math.degrees(new_lat), math.degrees(new_lon)


# =============================================================================
# Radar Processor — signal-level processing & tracking pipeline
# =============================================================================

class RadarProcessor:
    """Full radar processing pipeline: fusion, clustering, association, tracking."""

    def __init__(self):
        self.range_doppler_map = np.zeros((1024, 32))
        self.detected_targets: list[RadarTarget] = []
        self.track_id_counter: int = 0
        self.tracks: dict[int, dict] = {}
        self.frame_count: int = 0
        self.config = ProcessingConfig()

        # MTI state: store previous frames for cancellation
        self._mti_history: list[np.ndarray] = []

    # ---- Configuration -----------------------------------------------------

    def set_config(self, config: ProcessingConfig):
        """Update the processing configuration and reset MTI history if needed."""
        old_order = self.config.mti_order
        self.config = config
        if config.mti_order != old_order:
            self._mti_history.clear()

    # ---- Windowing ----------------------------------------------------------

    @staticmethod
    def apply_window(data: np.ndarray, window_type: str) -> np.ndarray:
        """Apply a window function along each column (slow-time dimension).

        *data* shape: (range_bins, doppler_bins).  Window is applied along
        axis-1 (Doppler / slow-time).
        """
        if window_type == "None" or not window_type:
            return data

        n = data.shape[1]
        if n < 2:
            return data

        if SCIPY_AVAILABLE:
            wtype = window_type.lower()
            if wtype == "hann":
                w = scipy_windows.hann(n, sym=False)
            elif wtype == "hamming":
                w = scipy_windows.hamming(n, sym=False)
            elif wtype == "blackman":
                w = scipy_windows.blackman(n)
            elif wtype == "kaiser":
                w = scipy_windows.kaiser(n, beta=14)
            elif wtype == "chebyshev":
                w = scipy_windows.chebwin(n, at=80)
            else:
                w = np.ones(n)
        else:
            # Fallback: numpy Hann
            wtype = window_type.lower()
            if wtype == "hann":
                w = np.hanning(n)
            elif wtype == "hamming":
                w = np.hamming(n)
            elif wtype == "blackman":
                w = np.blackman(n)
            else:
                w = np.ones(n)

        return data * w[np.newaxis, :]

    # ---- DC Notch (zero-Doppler removal) ------------------------------------

    @staticmethod
    def dc_notch(data: np.ndarray) -> np.ndarray:
        """Remove the DC (zero-Doppler) component by subtracting the
        mean along the slow-time axis for each range bin."""
        return data - np.mean(data, axis=1, keepdims=True)

    # ---- MTI (Moving Target Indication) -------------------------------------

    def mti_filter(self, frame: np.ndarray) -> np.ndarray:
        """Apply MTI cancellation of order 1, 2, or 3.

        Order-1: y[n] = x[n] - x[n-1]
        Order-2: y[n] = x[n] - 2*x[n-1] + x[n-2]
        Order-3: y[n] = x[n] - 3*x[n-1] + 3*x[n-2] - x[n-3]

        The internal history buffer stores up to 3 previous frames.
        """
        order = self.config.mti_order
        self._mti_history.append(frame.copy())

        # Trim history to order + 1 frames
        max_len = order + 1
        if len(self._mti_history) > max_len:
            self._mti_history = self._mti_history[-max_len:]

        if len(self._mti_history) < order + 1:
            # Not enough history yet — return zeros (suppress output)
            return np.zeros_like(frame)

        h = self._mti_history
        if order == 1:
            return h[-1] - h[-2]
        if order == 2:
            return h[-1] - 2.0 * h[-2] + h[-3]
        if order == 3:
            return h[-1] - 3.0 * h[-2] + 3.0 * h[-3] - h[-4]
        return h[-1] - h[-2]

    # ---- CFAR (Constant False Alarm Rate) -----------------------------------

    @staticmethod
    def cfar_1d(signal_vec: np.ndarray, guard: int, train: int,
                threshold_factor: float, cfar_type: str = "CA-CFAR") -> np.ndarray:
        """1-D CFAR detector.

        Parameters
        ----------
        signal_vec : 1-D array (power in linear scale)
        guard      : number of guard cells on each side
        train      : number of training cells on each side
        threshold_factor : multiplier on estimated noise level
        cfar_type  : CA-CFAR, OS-CFAR, GO-CFAR, or SO-CFAR

        Returns
        -------
        detections : boolean array, True where target detected
        """
        n = len(signal_vec)
        detections = np.zeros(n, dtype=bool)
        half = guard + train

        for i in range(half, n - half):
            # Leading training cells
            lead = signal_vec[i - half: i - guard]
            # Lagging training cells
            lag = signal_vec[i + guard + 1: i + half + 1]

            if cfar_type == "CA-CFAR":
                noise = (np.sum(lead) + np.sum(lag)) / (2 * train)
            elif cfar_type == "GO-CFAR":
                noise = max(np.mean(lead), np.mean(lag))
            elif cfar_type == "SO-CFAR":
                noise = min(np.mean(lead), np.mean(lag))
            elif cfar_type == "OS-CFAR":
                all_train = np.concatenate([lead, lag])
                all_train.sort()
                k = int(0.75 * len(all_train))  # 75th percentile
                noise = all_train[min(k, len(all_train) - 1)]
            else:
                noise = (np.sum(lead) + np.sum(lag)) / (2 * train)

            threshold = noise * threshold_factor
            if signal_vec[i] > threshold:
                detections[i] = True

        return detections

    def cfar_2d(self, rdm: np.ndarray) -> np.ndarray:
        """Apply 1-D CFAR along each range bin (across Doppler dimension).

        Returns a boolean mask of the same shape as *rdm*.
        """
        cfg = self.config
        mask = np.zeros_like(rdm, dtype=bool)
        for r in range(rdm.shape[0]):
            row = rdm[r, :]
            if row.max() > 0:
                mask[r, :] = self.cfar_1d(
                    row, cfg.cfar_guard_cells, cfg.cfar_training_cells,
                    cfg.cfar_threshold_factor, cfg.cfar_type,
                )
        return mask

    # ---- Full processing pipeline -------------------------------------------

    def process_frame(self, raw_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run the full signal processing chain on a Range x Doppler frame.

        Parameters
        ----------
        raw_frame : 2-D array (range_bins x doppler_bins), complex or real

        Returns
        -------
        (processed_rdm, detection_mask)
            processed_rdm  — processed Range-Doppler map (power, linear)
            detection_mask — boolean mask of CFAR / threshold detections
        """
        cfg = self.config
        data = raw_frame.astype(np.float64)

        # 1. DC Notch
        if cfg.dc_notch_enabled:
            data = self.dc_notch(data)

        # 2. Windowing (before FFT — applied along slow-time axis)
        if cfg.window_type and cfg.window_type != "None":
            data = self.apply_window(data, cfg.window_type)

        # 3. MTI
        if cfg.mti_enabled:
            data = self.mti_filter(data)

        # 4. Power (magnitude squared)
        power = np.abs(data) ** 2
        power = np.maximum(power, 1e-20)  # avoid log(0)

        # 5. CFAR detection or simple threshold
        if cfg.cfar_enabled:
            detection_mask = self.cfar_2d(power)
        else:
            # Simple threshold: convert dB threshold to linear
            power_db = 10.0 * np.log10(power)
            noise_floor = np.median(power_db)
            detection_mask = power_db > (noise_floor + cfg.detection_threshold_db)

        # Update stored RDM
        self.range_doppler_map = power
        self.frame_count += 1

        return power, detection_mask

    # ---- Dual-CPI fusion ---------------------------------------------------

    @staticmethod
    def dual_cpi_fusion(range_profiles_1: np.ndarray,
                        range_profiles_2: np.ndarray) -> np.ndarray:
        """Dual-CPI fusion for better detection."""
        return np.mean(range_profiles_1, axis=0) + np.mean(range_profiles_2, axis=0)

    # ---- DBSCAN clustering -------------------------------------------------

    @staticmethod
    def clustering(detections: list[RadarTarget],
                   eps: float = 100, min_samples: int = 2) -> list:
        """DBSCAN clustering of detections (requires sklearn)."""
        if not SKLEARN_AVAILABLE or len(detections) == 0:
            return []

        points = np.array([[d.range, d.velocity] for d in detections])
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit(points).labels_

        clusters = []
        for label in set(labels):
            if label == -1:
                continue
            cluster_points = points[labels == label]
            clusters.append({
                "center": np.mean(cluster_points, axis=0),
                "points": cluster_points,
                "size": len(cluster_points),
            })
        return clusters

    # ---- Association -------------------------------------------------------

    def association(self, detections: list[RadarTarget],
                    _clusters: list) -> list[RadarTarget]:
        """Associate detections to existing tracks (nearest-neighbour)."""
        associated = []
        for det in detections:
            best_track = None
            min_dist = float("inf")
            for tid, track in self.tracks.items():
                dist = math.sqrt(
                    (det.range - track["state"][0]) ** 2
                    + (det.velocity - track["state"][2]) ** 2
                )
                if dist < min_dist and dist < 500:
                    min_dist = dist
                    best_track = tid

            if best_track is not None:
                det.track_id = best_track
            else:
                det.track_id = self.track_id_counter
                self.track_id_counter += 1

            associated.append(det)
        return associated

    # ---- Kalman tracking ---------------------------------------------------

    def tracking(self, associated_detections: list[RadarTarget]):
        """Kalman filter tracking (requires filterpy)."""
        if not FILTERPY_AVAILABLE:
            return

        now = time.time()

        for det in associated_detections:
            if det.track_id not in self.tracks:
                kf = KalmanFilter(dim_x=4, dim_z=2)
                kf.x = np.array([det.range, 0, det.velocity, 0])
                kf.F = np.array([
                    [1, 1, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 1],
                    [0, 0, 0, 1],
                ])
                kf.H = np.array([
                    [1, 0, 0, 0],
                    [0, 0, 1, 0],
                ])
                kf.P *= 1000
                kf.R = np.diag([10, 1])
                kf.Q = np.eye(4) * 0.1

                self.tracks[det.track_id] = {
                    "filter": kf,
                    "state": kf.x,
                    "last_update": now,
                    "hits": 1,
                }
            else:
                track = self.tracks[det.track_id]
                track["filter"].predict()
                track["filter"].update([det.range, det.velocity])
                track["state"] = track["filter"].x
                track["last_update"] = now
                track["hits"] += 1

        # Prune stale tracks (> 5 s without update)
        stale = [tid for tid, t in self.tracks.items()
                 if now - t["last_update"] > 5.0]
        for tid in stale:
            del self.tracks[tid]


# =============================================================================
# USB / GPS Packet Parser
# =============================================================================

class USBPacketParser:
    """
    Parse GPS (and general) data arriving from the STM32 via USB CDC.

    Supports:
      - Text format: ``GPS:lat,lon,alt,pitch\\r\\n``
      - Binary format: ``GPSB`` header, 30 bytes total
    """

    def __init__(self):
        pass

    def parse_gps_data(self, data: bytes) -> GPSData | None:
        """Attempt to parse GPS data from a raw USB CDC frame."""
        if not data:
            return None

        try:
            # Text format: "GPS:lat,lon,alt,pitch\r\n"
            text = data.decode("utf-8", errors="ignore").strip()
            if text.startswith("GPS:"):
                parts = text.split(":")[1].split(",")
                if len(parts) >= 4:
                    return GPSData(
                        latitude=float(parts[0]),
                        longitude=float(parts[1]),
                        altitude=float(parts[2]),
                        pitch=float(parts[3]),
                        timestamp=time.time(),
                    )

            # Binary format: [GPSB 4][lat 8][lon 8][alt 4][pitch 4][CRC 2] = 30 bytes
            if len(data) >= 30 and data[0:4] == b"GPSB":
                return self._parse_binary_gps(data)
        except (ValueError, struct.error) as e:
            logger.error(f"Error parsing GPS data: {e}")
        return None

    @staticmethod
    def _parse_binary_gps(data: bytes) -> GPSData | None:
        """Parse 30-byte binary GPS frame."""
        try:
            if len(data) < 30:
                return None

            # Simple checksum CRC
            crc_rcv = (data[28] << 8) | data[29]
            crc_calc = sum(data[0:28]) & 0xFFFF
            if crc_rcv != crc_calc:
                logger.warning("GPS binary CRC mismatch")
                return None

            lat = struct.unpack(">d", data[4:12])[0]
            lon = struct.unpack(">d", data[12:20])[0]
            alt = struct.unpack(">f", data[20:24])[0]
            pitch = struct.unpack(">f", data[24:28])[0]

            return GPSData(
                latitude=lat,
                longitude=lon,
                altitude=alt,
                pitch=pitch,
                timestamp=time.time(),
            )
        except (ValueError, struct.error) as e:
            logger.error(f"Error parsing binary GPS: {e}")
            return None


# =============================================================================
# Raw IQ Frame Processor — full signal chain for replay mode
# =============================================================================

class RawIQFrameProcessor:
    """Process raw complex IQ frames through the full radar signal chain.

    This replicates the FPGA processing pipeline in software so that
    raw ADI CN0566 captures (or similar) can be visualised in the V7
    dashboard exactly as they would appear from the FPGA.

    Pipeline per frame:
      1. Quantize raw complex → 16-bit signed I/Q
      2. AGC gain application (bit-accurate to rx_gain_control.v)
      3. Range FFT (across samples)
      4. Doppler FFT (across chirps) + fftshift + centre crop
      5. Optional MTI (2-pulse canceller using history)
      6. Optional DC notch (zero-Doppler removal)
      7. Magnitude (|I| + |Q| approximation matching FPGA, or true |.|)
      8. CFAR or simple threshold detection
      9. Build RadarFrame + synthetic StatusResponse
    """

    def __init__(
        self,
        n_range_out: int = 64,
        n_doppler_out: int = 32,
    ):
        self._n_range = n_range_out
        self._n_doppler = n_doppler_out

        # AGC state (persists across frames)
        self._agc_config = AGCConfig()
        self._agc_state = AGCState()

        # MTI history buffer (stores previous Range-Doppler maps)
        self._mti_history: list[np.ndarray] = []
        self._mti_enabled: bool = False

        # DC notch
        self._dc_notch_width: int = 0

        # CFAR / threshold config
        self._cfar_enabled: bool = False
        self._cfar_guard: int = 2
        self._cfar_train: int = 8
        self._cfar_alpha_q44: int = 0x30  # Q4.4 → 3.0
        self._cfar_mode: int = 0  # 0=CA, 1=GO, 2=SO
        self._detect_threshold: int = 10000

        # Frame counter
        self._frame_number: int = 0

        # Host-side processing (windowing, clustering, etc.)
        self._host_processor = RadarProcessor()

    # ---- Configuration setters ---------------------------------------------

    def set_agc_config(self, config: AGCConfig) -> None:
        self._agc_config = config

    def reset_agc_state(self) -> None:
        """Reset AGC state (e.g. on seek)."""
        self._agc_state = AGCState()
        self._mti_history.clear()

    def set_mti_enabled(self, enabled: bool) -> None:
        if self._mti_enabled != enabled:
            self._mti_history.clear()
        self._mti_enabled = enabled

    def set_dc_notch_width(self, width: int) -> None:
        self._dc_notch_width = max(0, min(7, width))

    def set_cfar_params(
        self,
        enabled: bool,
        guard: int = 2,
        train: int = 8,
        alpha_q44: int = 0x30,
        mode: int = 0,
    ) -> None:
        self._cfar_enabled = enabled
        self._cfar_guard = guard
        self._cfar_train = train
        self._cfar_alpha_q44 = alpha_q44
        self._cfar_mode = mode

    def set_detect_threshold(self, threshold: int) -> None:
        self._detect_threshold = threshold

    @property
    def agc_state(self) -> AGCState:
        return self._agc_state

    @property
    def agc_config(self) -> AGCConfig:
        return self._agc_config

    @property
    def frame_number(self) -> int:
        return self._frame_number

    # ---- Main processing entry point ---------------------------------------

    def process_frame(
        self,
        raw_frame: np.ndarray,
        timestamp: float = 0.0,
    ) -> tuple[RadarFrame, StatusResponse, AGCFrameResult]:
        """Process one raw IQ frame through the full chain.

        Parameters
        ----------
        raw_frame : complex array, shape (n_chirps, n_samples)
        timestamp : frame timestamp for RadarFrame

        Returns
        -------
        (radar_frame, status_response, agc_result)
        """
        n_chirps, _n_samples = raw_frame.shape
        self._frame_number += 1

        # 1. Quantize to 16-bit signed IQ
        frame_i, frame_q = quantize_iq(raw_frame)

        # 2. AGC
        agc_result = process_agc_frame(
            frame_i, frame_q, self._agc_config, self._agc_state)

        # Use AGC-shifted IQ for downstream processing
        iq = agc_result.shifted_i.astype(np.float64) + 1j * agc_result.shifted_q.astype(np.float64)

        # 3. Range FFT (across samples axis)
        range_fft = np.fft.fft(iq, axis=1)[:, :self._n_range]

        # 4. Doppler FFT (across chirps axis) + fftshift + centre crop
        doppler_fft = np.fft.fft(range_fft, axis=0)
        doppler_fft = np.fft.fftshift(doppler_fft, axes=0)
        # Centre-crop to n_doppler bins
        center = n_chirps // 2
        half_d = self._n_doppler // 2
        start_d = max(0, center - half_d)
        end_d = start_d + self._n_doppler
        if end_d > n_chirps:
            end_d = n_chirps
            start_d = max(0, end_d - self._n_doppler)
        rd_complex = doppler_fft[start_d:end_d, :]
        # shape: (n_doppler, n_range) → transpose to (n_range, n_doppler)
        rd_complex = rd_complex.T

        # 5. Optional MTI (2-pulse canceller)
        if self._mti_enabled:
            rd_complex = self._apply_mti(rd_complex)

        # 6. Optional DC notch (zero-Doppler bins)
        if self._dc_notch_width > 0:
            rd_complex = self._apply_dc_notch(rd_complex)

        # Extract I/Q for RadarFrame
        rd_i = np.round(rd_complex.real).astype(np.int16)
        rd_q = np.round(rd_complex.imag).astype(np.int16)

        # 7. Magnitude (FPGA uses |I|+|Q| approximation)
        magnitude = np.abs(rd_complex.real) + np.abs(rd_complex.imag)

        # Range profile (sum across Doppler)
        range_profile = np.sum(magnitude, axis=1)

        # 8. Detection (CFAR or simple threshold)
        if self._cfar_enabled:
            detections = self._run_cfar(magnitude)
        else:
            detections = self._run_threshold(magnitude)

        detection_count = int(np.sum(detections > 0))

        # 9. Build RadarFrame
        radar_frame = RadarFrame(
            timestamp=timestamp,
            range_doppler_i=rd_i,
            range_doppler_q=rd_q,
            magnitude=magnitude,
            detections=detections,
            range_profile=range_profile,
            detection_count=detection_count,
            frame_number=self._frame_number,
        )

        # 10. Build synthetic StatusResponse
        status = self._build_status(agc_result)

        return radar_frame, status, agc_result

    # ---- Internal helpers --------------------------------------------------

    def _apply_mti(self, rd: np.ndarray) -> np.ndarray:
        """2-pulse MTI canceller: y[n] = x[n] - x[n-1]."""
        self._mti_history.append(rd.copy())
        if len(self._mti_history) > 2:
            self._mti_history = self._mti_history[-2:]

        if len(self._mti_history) < 2:
            return np.zeros_like(rd)  # suppress first frame

        return self._mti_history[-1] - self._mti_history[-2]

    def _apply_dc_notch(self, rd: np.ndarray) -> np.ndarray:
        """Zero out centre Doppler bins (DC notch)."""
        n_doppler = rd.shape[1]
        center = n_doppler // 2
        w = self._dc_notch_width
        lo = max(0, center - w)
        hi = min(n_doppler, center + w + 1)
        result = rd.copy()
        result[:, lo:hi] = 0
        return result

    def _run_cfar(self, magnitude: np.ndarray) -> np.ndarray:
        """Run 1-D CFAR along each range bin (Doppler direction).

        Uses the host-side CFAR from RadarProcessor with alpha converted
        from Q4.4 to float.
        """
        alpha_float = self._cfar_alpha_q44 / 16.0
        cfar_types = {0: "CA-CFAR", 1: "GO-CFAR", 2: "SO-CFAR"}
        cfar_type = cfar_types.get(self._cfar_mode, "CA-CFAR")

        power = magnitude ** 2
        power = np.maximum(power, 1e-20)

        mask = np.zeros_like(magnitude, dtype=np.uint8)
        for r in range(magnitude.shape[0]):
            row = power[r, :]
            if row.max() > 0:
                det = RadarProcessor.cfar_1d(
                    row, self._cfar_guard, self._cfar_train,
                    alpha_float, cfar_type)
                mask[r, :] = det.astype(np.uint8)
        return mask

    def _run_threshold(self, magnitude: np.ndarray) -> np.ndarray:
        """Simple threshold detection (matches FPGA detect_threshold)."""
        return (magnitude > self._detect_threshold).astype(np.uint8)

    def _build_status(self, agc_result: AGCFrameResult) -> StatusResponse:
        """Build a synthetic StatusResponse from current processor state."""
        return StatusResponse(
            radar_mode=1,  # active
            stream_ctrl=0b111,
            cfar_threshold=self._detect_threshold,
            long_chirp=3000,
            long_listen=13700,
            guard=17540,
            short_chirp=50,
            short_listen=17450,
            chirps_per_elev=32,
            range_mode=0,
            agc_current_gain=agc_result.gain_enc,
            agc_peak_magnitude=agc_result.peak_mag_8bit,
            agc_saturation_count=agc_result.saturation_count,
            agc_enable=1 if self._agc_config.enabled else 0,
        )
