"""facelock.eval -- offline biometric accuracy evaluation harness.

This package produces the defensible FMR / FNMR / EER numbers for the *deployed*
facelock matcher (protocol: docs / phase3 fmr-fnmr-protocol). It is import-safe
with **no** camera, model, daemon, systemd, or network dependency: the core
metrics (:mod:`facelock.eval.metrics`) are pure functions on score arrays, and
the deployed-matcher scorer (:mod:`facelock.eval.scoring`) reuses the in-process
:class:`facelock.matcher.Matcher` -- no I/O is performed on import.

T1 (metrics), T2 (deployed-matcher scoring), T3 (dataset embedding) and T4
(report + ``facelock-eval`` CLI) are implemented here. Only genuine capture (T5,
which needs the user + a camera) is intentionally absent. Submodules import cv2 /
sklearn / matplotlib lazily *inside* functions, so importing this package (or
running the ``report`` path) requires none of them.
"""

from __future__ import annotations

__all__ = ["metrics", "scoring", "embed_dataset", "report", "cli"]
