"""facelock -- screensaver-only face-unlock prototype.

A lightweight, local, privacy-preserving face-unlock utility for a single
Linux workstation (Ubuntu 24.04 / X11 / GNOME, CPU-only). It engages the
screen lock when the owner walks away or an unknown face appears, and on the
owner's return face-verifies the owner and dismisses the tool's own shield with
a personalized greeting ("Welcome back, <name>").

Architecture (see docs/pilot-face-unlock/design.md):
  * Two cooperating user-space processes:
      - ``facelockd``          -- perception daemon (can only *request* an
                                  unlock grant; never holds lock authority).
      - ``facelock-guardian``  -- session guardian (sole lock authority +
                                  watchdog; keeps/engages the shield on any
                                  error, crash, or missed heartbeat).
  * Fail-closed Safety Invariant (SI-P1..P5): LOCKED is the default on every
    transition/error boundary; the OS password path is NEVER touched
    (screensaver-only; no PAM in the prototype).
  * Recognition: YuNet detector + SFace 128-D embeddings via OpenCV, cosine
    similarity, per-owner calibrated threshold, k-of-n voting.

This is the PROTOTYPE (``security.phase = P``). It is convenience-level
security only and is documentedly bypassable by a photo/video of the owner
(see README, REQ-F-17). Real anti-spoofing (PAD) and optional PAM live behind
clean hooks and are activated in the Hardening phase (``security.phase = H``).
"""

from __future__ import annotations

__all__ = [
    "__version__",
    "PROTOTYPE_SPOOF_DISCLOSURE",
]

# Single source of truth for the version is pyproject.toml. At runtime we read
# it back from the installed package metadata; the literal below is only a
# fallback for running straight from a source tree with no install, and is kept
# equal to the pyproject version.
try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("facelock")
except PackageNotFoundError:  # pragma: no cover - source tree without install
    __version__ = "0.2.1"

# REQ-F-17 / AC-F-17: the exact disclosure shown on first run and in the README.
PROTOTYPE_SPOOF_DISCLOSURE = (
    "facelock PROTOTYPE -- convenience-level security only.\n"
    "This build controls ONLY the screensaver/shield of an already-logged-in\n"
    "session. It is NOT a password replacement and is NOT wired into PAM,\n"
    "login, or sudo. With anti-spoofing disabled (the prototype default), it\n"
    "can be fooled by a printed photo or a phone/monitor video of the owner\n"
    "(the same limitation Howdy documents). Your OS password lock is never\n"
    "removed or weakened and always works. For presentation-attack resistance\n"
    "(liveness/PAD to ISO/IEC 30107-3 targets) and optional OS-auth\n"
    "integration, use the Hardening phase (security.phase = H)."
)
