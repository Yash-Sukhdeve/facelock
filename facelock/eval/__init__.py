"""facelock.eval -- offline biometric accuracy evaluation harness.

This package produces the defensible FMR / FNMR / EER numbers for the *deployed*
facelock matcher (protocol: docs / phase3 fmr-fnmr-protocol). It is import-safe
with **no** camera, model, daemon, systemd, or network dependency: the core
metrics (:mod:`facelock.eval.metrics`) are pure functions on score arrays, and
the deployed-matcher scorer (:mod:`facelock.eval.scoring`) reuses the in-process
:class:`facelock.matcher.Matcher` -- no I/O is performed on import.

Only T1 (metrics) and T2 (deployed-matcher scoring) are implemented here. The
dataset embedder (T3), report/CLI (T4), and genuine capture (T5) are separate,
later tasks and are intentionally absent.
"""

from __future__ import annotations

__all__ = ["metrics", "scoring"]
