"""
v7.raw_iq_replay -- Raw IQ replay controller for the V7 dashboard.

Manages loading of raw complex IQ .npy captures, playback state
(play/pause/step/speed/loop), and delivers frames to a worker thread.

The controller is thread-safe: the worker calls ``next_frame()`` which
blocks until a frame is available or playback is stopped.

Supported file formats:
  - 3-D .npy: (n_frames, n_chirps, n_samples) complex
  - 2-D .npy: (n_chirps, n_samples) complex -> treated as single frame

Classes:
  - RawIQReplayController  -- playback state machine + frame delivery
  - RawIQFileInfo          -- metadata about the loaded file
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Playback state enum
# ---------------------------------------------------------------------------

class PlaybackState(Enum):
    """Playback state for the replay controller."""
    STOPPED = auto()
    PLAYING = auto()
    PAUSED = auto()


# ---------------------------------------------------------------------------
# File metadata
# ---------------------------------------------------------------------------

@dataclass
class RawIQFileInfo:
    """Metadata about a loaded raw IQ .npy file."""
    path: str
    n_frames: int
    n_chirps: int
    n_samples: int
    dtype: str
    file_size_mb: float


# ---------------------------------------------------------------------------
# Replay Controller
# ---------------------------------------------------------------------------

class RawIQReplayController:
    """Manages raw IQ file loading and playback state.

    Thread-safety: the controller uses a condition variable so the worker
    thread can block on ``next_frame()`` waiting for play/step events,
    while the GUI thread calls ``play()``, ``pause()``, ``step()``, etc.
    """

    def __init__(self) -> None:
        self._data: np.ndarray | None = None
        self._info: RawIQFileInfo | None = None

        # Playback state
        self._state = PlaybackState.STOPPED
        self._frame_index: int = 0
        self._fps: float = 10.0  # target frames per second
        self._loop: bool = True

        # Thread synchronisation
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

        # Step request flag (set by GUI, consumed by worker)
        self._step_requested: bool = False

        # Stop signal
        self._stop_requested: bool = False

    # ---- File loading ------------------------------------------------------

    def load_file(self, path: str) -> RawIQFileInfo:
        """Load a .npy file containing raw IQ data.

        Raises ValueError if the file is not a valid raw IQ capture.
        """
        p = Path(path)
        if not p.exists():
            msg = f"File not found: {path}"
            raise ValueError(msg)
        if p.suffix.lower() != ".npy":
            msg = f"Expected .npy file, got: {p.suffix}"
            raise ValueError(msg)

        # Memory-map for large files
        data = np.load(str(p), mmap_mode="r")

        if not np.iscomplexobj(data):
            msg = f"Expected complex data, got dtype={data.dtype}"
            raise ValueError(msg)

        # Normalise shape
        if data.ndim == 2:
            # Single frame: (chirps, samples) -> (1, chirps, samples)
            data = data[np.newaxis, :, :]
        elif data.ndim == 3:
            pass  # (frames, chirps, samples) — expected
        else:
            msg = f"Expected 2-D or 3-D array, got {data.ndim}-D"
            raise ValueError(msg)

        with self._lock:
            self._data = data
            self._frame_index = 0
            self._state = PlaybackState.PAUSED
            self._stop_requested = False

            self._info = RawIQFileInfo(
                path=str(p),
                n_frames=data.shape[0],
                n_chirps=data.shape[1],
                n_samples=data.shape[2],
                dtype=str(data.dtype),
                file_size_mb=p.stat().st_size / (1024 * 1024),
            )

        logger.info(
            f"Loaded raw IQ: {p.name} — {self._info.n_frames} frames, "
            f"{self._info.n_chirps} chirps, {self._info.n_samples} samples, "
            f"{self._info.file_size_mb:.1f} MB"
        )
        return self._info

    def unload(self) -> None:
        """Unload the current file and stop playback."""
        with self._lock:
            self._state = PlaybackState.STOPPED
            self._stop_requested = True
            self._data = None
            self._info = None
            self._frame_index = 0
            self._cond.notify_all()

    # ---- Playback control (called from GUI thread) -------------------------

    def play(self) -> None:
        with self._lock:
            if self._data is None:
                return
            self._state = PlaybackState.PLAYING
            self._stop_requested = False
            self._cond.notify_all()

    def pause(self) -> None:
        with self._lock:
            if self._state == PlaybackState.PLAYING:
                self._state = PlaybackState.PAUSED

    def stop(self) -> None:
        with self._lock:
            self._state = PlaybackState.STOPPED
            self._stop_requested = True
            self._cond.notify_all()

    def step_forward(self) -> None:
        """Advance one frame (works in PAUSED state)."""
        with self._lock:
            if self._data is None:
                return
            self._step_requested = True
            self._cond.notify_all()

    def seek(self, frame_index: int) -> None:
        """Jump to a specific frame index."""
        with self._lock:
            if self._data is None:
                return
            self._frame_index = max(0, min(frame_index, self._data.shape[0] - 1))

    def set_fps(self, fps: float) -> None:
        with self._lock:
            self._fps = max(0.1, min(60.0, fps))

    def set_loop(self, loop: bool) -> None:
        with self._lock:
            self._loop = loop

    # ---- State queries (thread-safe) ---------------------------------------

    @property
    def state(self) -> PlaybackState:
        with self._lock:
            return self._state

    @property
    def frame_index(self) -> int:
        with self._lock:
            return self._frame_index

    @property
    def info(self) -> RawIQFileInfo | None:
        with self._lock:
            return self._info

    @property
    def fps(self) -> float:
        with self._lock:
            return self._fps

    @property
    def is_loaded(self) -> bool:
        with self._lock:
            return self._data is not None

    # ---- Frame delivery (called from worker thread) ------------------------

    def next_frame(self) -> np.ndarray | None:
        """Block until the next frame is available, then return it.

        Returns None when playback is stopped or file is unloaded.
        The caller (worker thread) should use this in a loop.
        """
        with self._cond:
            while True:
                if self._stop_requested or self._data is None:
                    return None

                if self._state == PlaybackState.PLAYING:
                    return self._deliver_frame()

                if self._step_requested:
                    self._step_requested = False
                    return self._deliver_frame()

                # PAUSED or STOPPED — wait for signal
                self._cond.wait(timeout=0.1)

    def _deliver_frame(self) -> np.ndarray | None:
        """Return current frame and advance index. Must hold lock."""
        if self._data is None:
            return None

        n_frames = self._data.shape[0]
        if self._frame_index >= n_frames:
            if self._loop:
                self._frame_index = 0
            else:
                self._state = PlaybackState.PAUSED
                return None

        # Read the frame (memory-mapped, so this is cheap)
        frame = np.array(self._data[self._frame_index])  # copy from mmap
        self._frame_index += 1
        return frame
