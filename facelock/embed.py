"""FaceEmbedder (C3) -- SFace 128-D embeddings via ``cv2.FaceRecognizerSF``.

Realizes REQ-F-07. Aligns the detected face using YuNet's 5 landmarks
(``alignCrop``) and produces an L2-normalized 128-D float32 embedding.

Fail-closed (I-4): a model load error raises :class:`ModelError` (FM-11); an
alignment/inference error returns ``None``, which the matcher treats as a
non-match (never an accept, FM-05/FM-11).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import cv2
except Exception as _exc:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]
    _CV2_IMPORT_ERROR = _exc
else:
    _CV2_IMPORT_ERROR = None

from .detect import Detection
from .errors import ModelError

EMBEDDING_DIM = 128


def l2_normalize(vec: np.ndarray) -> np.ndarray | None:
    """L2-normalize a vector; return ``None`` for a degenerate/NaN vector."""
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    if vec.size != EMBEDDING_DIM or not np.all(np.isfinite(vec)):
        return None
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return None
    return (vec / norm).astype(np.float32)


class FaceEmbedder:
    """Wraps ``cv2.FaceRecognizerSF`` (alignment + embedding)."""

    def __init__(self, model_path: str | Path) -> None:
        if cv2 is None:  # pragma: no cover
            raise ModelError(f"OpenCV unavailable: {_CV2_IMPORT_ERROR}")
        model_path = Path(model_path)
        if not model_path.exists():
            raise ModelError(f"SFace model missing: {model_path} (FM-11)")
        self.model_path = str(model_path)
        self.healthy = True
        try:
            self._rec = cv2.FaceRecognizerSF.create(self.model_path, "")
        except Exception as exc:
            raise ModelError(f"failed to load SFace model: {exc} (FM-11)") from exc

    def align(self, bgr: np.ndarray, detection: Detection) -> np.ndarray | None:
        """Return the aligned face crop (BGR) for a detection, or ``None``.

        Used by the passive-PAD liveness path (Hardening) so it does not reach
        into the private recognizer handle.
        """
        if bgr is None or getattr(bgr, "size", 0) == 0:
            return None
        try:
            return self._rec.alignCrop(bgr, detection.raw_row.reshape(1, -1))
        except Exception:
            return None

    def embed(self, bgr: np.ndarray, detection: Detection) -> np.ndarray | None:
        """Return an L2-normalized 128-D embedding, or ``None`` on any error."""
        if bgr is None or getattr(bgr, "size", 0) == 0:
            return None
        try:
            aligned = self._rec.alignCrop(bgr, detection.raw_row.reshape(1, -1))
            feature = self._rec.feature(aligned)
        except Exception:
            self.healthy = False
            return None
        self.healthy = True
        return l2_normalize(np.asarray(feature, dtype=np.float32))
