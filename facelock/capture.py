"""CameraCapture (C1) -- V4L2 camera lifecycle via OpenCV.

Realizes REQ-F-05, REQ-NF-01/06, and the camera failure modes FM-01
(busy/unavailable), FM-07 (long-absence release), FM-09 (shutter/covered),
FM-13 (suspend/resume). The capture object:

  * opens the configured V4L2 node at the configured resolution / pixel format,
  * delivers frames at an active or idle rate,
  * releases the device (and thus the UVC activity LED, REQ-F-27) on long
    absence and re-acquires with exponential backoff,
  * NEVER raises past the read loop -- an unreadable device yields ``None`` +
    a :class:`~facelock.errors.CameraError`, and the caller fails closed.

Raw frames are held only in volatile memory and are never written to disk
(REQ-NF-13); nothing here calls ``imwrite``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

try:  # OpenCV is a hard runtime dependency but keep import failure legible.
    import cv2
except Exception as _exc:  # pragma: no cover - environment guaranteed to have cv2
    cv2 = None  # type: ignore[assignment]
    _CV2_IMPORT_ERROR = _exc
else:
    _CV2_IMPORT_ERROR = None

from .errors import CameraError

# Fraction of frame pixels below this luma that flags a "blocked/dark" frame.
_DARK_LUMA = 12
_DARK_FRACTION = 0.995


@dataclass
class Frame:
    """A single captured frame (BGR) plus provenance metadata."""

    bgr: np.ndarray  # HxWx3 uint8, BGR
    ts_monotonic: float
    seq: int
    w: int
    h: int

    @property
    def is_dark(self) -> bool:
        """Heuristic for a covered lens / closed shutter (FM-09)."""
        if self.bgr is None or self.bgr.size == 0:
            return True
        # Mean over channels -> luma proxy; cheap and dependency-free.
        luma = self.bgr.mean(axis=2)
        return float((luma < _DARK_LUMA).mean()) >= _DARK_FRACTION


class CameraCapture:
    """Owns the V4L2 device handle and the acquire/release lifecycle."""

    def __init__(
        self,
        device: str = "/dev/video0",
        *,
        width: int = 640,
        height: int = 480,
        pixel_format: str = "YUYV",
        fps: int = 5,
        max_backoff_s: float = 8.0,
    ) -> None:
        if cv2 is None:  # pragma: no cover
            raise CameraError(f"OpenCV unavailable: {_CV2_IMPORT_ERROR}", code="open")
        self.device = device
        self.width = width
        self.height = height
        self.pixel_format = pixel_format
        self.fps = fps
        self.max_backoff_s = max_backoff_s
        self._cap: "cv2.VideoCapture | None" = None
        self._seq = 0
        self._backoff = 0.5
        self._consecutive_read_failures = 0

    # -- lifecycle --------------------------------------------------------- #
    def _device_index(self) -> object:
        """Return an OpenCV-openable target from a device string."""
        if self.device.startswith("/dev/video"):
            try:
                return int(self.device.replace("/dev/video", ""))
            except ValueError:
                return self.device
        return self.device

    def open(self) -> None:
        """Open the device; raise CameraError with a code on failure (FM-01)."""
        assert cv2 is not None
        cap = cv2.VideoCapture(self._device_index(), cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            raise CameraError(f"cannot open camera {self.device}", code="open")
        fourcc = cv2.VideoWriter_fourcc(*self.pixel_format)
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        # Minimise latency: keep the driver buffer shallow.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self._cap = cap
        self._backoff = 0.5
        self._consecutive_read_failures = 0

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def release(self) -> bool:
        """Release the device (idempotent). Frees the handle + LED (FM-07)."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        return True

    def reacquire(self) -> bool:
        """Re-open after a loss/suspend with exponential backoff (FM-01/13)."""
        self.release()
        try:
            self.open()
            return True
        except CameraError:
            time.sleep(min(self._backoff, self.max_backoff_s))
            self._backoff = min(self._backoff * 2, self.max_backoff_s)
            return False

    def set_rate(self, fps: int) -> bool:
        """Change the target FPS (active vs idle throttle, REQ-NF-01/06)."""
        self.fps = fps
        if self._cap is not None:
            try:
                self._cap.set(cv2.CAP_PROP_FPS, fps)
                return True
            except Exception:
                return False
        return False

    # -- reading ----------------------------------------------------------- #
    def read(self) -> tuple[Frame | None, CameraError | None]:
        """Read one frame. Returns ``(Frame, None)`` or ``(None, CameraError)``.

        Never raises out of this method (SI): the perception loop treats a
        ``None`` frame as "no observation -> fail closed".
        """
        if self._cap is None or not self._cap.isOpened():
            return None, CameraError("camera not open", code="open")
        try:
            ok, bgr = self._cap.read()
        except Exception as exc:  # pragma: no cover - defensive
            return None, CameraError(f"camera read exception: {exc}", code="timeout")
        if not ok or bgr is None or getattr(bgr, "size", 0) == 0:
            self._consecutive_read_failures += 1
            code = "busy" if self._consecutive_read_failures <= 3 else "timeout"
            return None, CameraError("frame read failed", code=code)
        self._consecutive_read_failures = 0
        self._seq += 1
        h, w = bgr.shape[:2]
        return Frame(bgr=bgr, ts_monotonic=time.monotonic(), seq=self._seq, w=w, h=h), None

    def __enter__(self) -> "CameraCapture":
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()
