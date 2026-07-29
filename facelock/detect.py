"""FaceDetector (C2) -- YuNet detection via ``cv2.FaceDetectorYN``.

Realizes REQ-F-06. Detects zero, one, or many faces per frame with a bounding
box, detection score, and 5 landmarks (right eye, left eye, nose tip, right
mouth corner, left mouth corner). The landmarks feed both SFace alignment
(C3) and the active-liveness head-turn geometry (C5).

Fail-closed (I-3): a model load error raises :class:`ModelError` at
construction (FM-11); a per-frame inference error returns an empty list and
sets ``self.healthy = False`` -- an empty detection list is never an unlock.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import cv2
except Exception as _exc:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]
    _CV2_IMPORT_ERROR = _exc
else:
    _CV2_IMPORT_ERROR = None

from .errors import ModelError

# YuNet output row layout (15 columns): bbox(4) + 5 landmarks(10) + score(1).
_LANDMARK_NAMES = ("right_eye", "left_eye", "nose", "right_mouth", "left_mouth")


@dataclass
class Detection:
    """One detected face."""

    bbox: tuple[float, float, float, float]  # x, y, w, h (pixels)
    score: float
    landmarks: np.ndarray  # shape (5, 2), float32, pixel coords
    raw_row: np.ndarray  # shape (15,), float32 -- required by SFace alignCrop

    @property
    def size_px(self) -> float:
        """Min(width, height) of the box in pixels (face-size gate, FM-05)."""
        return float(min(self.bbox[2], self.bbox[3]))

    def landmark(self, name: str) -> tuple[float, float]:
        idx = _LANDMARK_NAMES.index(name)
        return float(self.landmarks[idx, 0]), float(self.landmarks[idx, 1])


class FaceDetector:
    """Wraps ``cv2.FaceDetectorYN`` with size/confidence gating."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        confidence_floor: float = 0.90,
        nms_threshold: float = 0.30,
        min_face_px: int = 80,
        top_k: int = 50,
    ) -> None:
        if cv2 is None:  # pragma: no cover
            raise ModelError(f"OpenCV unavailable: {_CV2_IMPORT_ERROR}")
        model_path = Path(model_path)
        if not model_path.exists():
            raise ModelError(f"YuNet model missing: {model_path} (FM-11)")
        self.model_path = str(model_path)
        self.confidence_floor = float(confidence_floor)
        self.nms_threshold = float(nms_threshold)
        self.min_face_px = int(min_face_px)
        self.healthy = True
        self._input_size = (0, 0)
        try:
            self._det = cv2.FaceDetectorYN.create(
                self.model_path,
                "",
                (320, 320),
                self.confidence_floor,
                self.nms_threshold,
                top_k,
            )
        except Exception as exc:
            raise ModelError(f"failed to load YuNet model: {exc} (FM-11)") from exc

    def _ensure_input_size(self, w: int, h: int) -> None:
        if self._input_size != (w, h):
            self._det.setInputSize((w, h))
            self._input_size = (w, h)

    def detect(self, bgr: np.ndarray) -> list[Detection]:
        """Return detections above the confidence + size floors.

        Never raises (I-3): inference errors return ``[]`` and flag unhealthy.
        """
        if bgr is None or getattr(bgr, "size", 0) == 0:
            return []
        h, w = bgr.shape[:2]
        try:
            self._ensure_input_size(w, h)
            _retval, faces = self._det.detect(bgr)
        except Exception:
            self.healthy = False
            return []
        self.healthy = True
        if faces is None:
            return []

        detections: list[Detection] = []
        for row in faces:
            row = np.asarray(row, dtype=np.float32).reshape(-1)
            if row.size < 15:
                continue
            score = float(row[14])
            if score < self.confidence_floor:
                continue
            x, y, bw, bh = (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
            if min(bw, bh) < self.min_face_px:
                continue
            landmarks = row[4:14].reshape(5, 2).astype(np.float32)
            detections.append(
                Detection(
                    bbox=(x, y, bw, bh),
                    score=score,
                    landmarks=landmarks,
                    raw_row=row.copy(),
                )
            )
        # Sort by descending box area so index 0 is the dominant face.
        detections.sort(key=lambda d: d.bbox[2] * d.bbox[3], reverse=True)
        return detections
