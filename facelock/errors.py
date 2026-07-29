"""Typed exceptions for facelock.

All error handling in facelock is fail-closed (SI-P2): every error surfaces
as a locked/deny outcome, never an unlock. These exception types make the
failure mode explicit and let callers map errors to the correct fail-closed
state transition.
"""

from __future__ import annotations


class FacelockError(Exception):
    """Base class for all facelock errors."""


class ConfigError(FacelockError):
    """Configuration is invalid or a security-critical key is out of range.

    Raised by :mod:`facelock.config`. On this error the daemon/guardian refuse
    to start (fail-closed, REQ-F-23) rather than boot with a wrong security
    value.
    """

    def __init__(self, message: str, errors: list[str] | None = None) -> None:
        super().__init__(message)
        self.errors: list[str] = list(errors or [])


class CameraError(FacelockError):
    """Camera device could not be opened or a frame could not be read (FM-01).

    Attributes
    ----------
    code:
        A short machine-readable code: ``busy`` (EBUSY), ``open`` (open
        failed), ``timeout`` (frame read timed out), ``permission`` (EACCES),
        ``blocked`` (uniform-dark / shutter, FM-09).
    """

    def __init__(self, message: str, code: str = "open") -> None:
        super().__init__(message)
        self.code = code


class ModelError(FacelockError):
    """A model file is missing, corrupt, or failed a SHA-256 check (FM-11)."""


class TemplateError(FacelockError):
    """The owner template is missing, corrupt, tampered, or unreadable (FM-10)."""


class CalibrationError(FacelockError):
    """Threshold calibration could not meet the phase accuracy target (REQ-NF-10)."""


class ControlProtocolError(FacelockError):
    """A control-socket message was malformed or failed authentication (SI-P1)."""


class LockActuationError(FacelockError):
    """No lock backend could be confirmed-engaged (FM-16, SI-P5)."""
