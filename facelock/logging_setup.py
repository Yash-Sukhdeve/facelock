"""Logger / Audit (C15) -- structured, image-free event logging.

Realizes REQ-F-22, REQ-NF-09/24/25 and the privacy rule REQ-NF-13 (no raw
frames on disk). Two sinks:

  * ``events.log`` -- rotating, size-capped JSON-lines event log (both phases).
    Every accept/deny/lock/unlock/camera/liveness event is one JSON object with
    enough fields (score, tau, face_count, liveness_result, reason) to
    reconstruct any decision WITHOUT any image data (REQ-NF-24).
  * ``audit.log`` -- append-only, HMAC-chained audit trail (Hardening only,
    REQ-NF-25). Each line carries a running HMAC over (prev_hmac + payload) so
    truncation or edits are tamper-evident. Implemented as a clean hook that is
    inert unless ``security.audit`` is true; see :class:`AuditLog`.

On a log write failure (e.g. disk full, FM-12) logging degrades but never
raises into the locking logic (SI): the caller's lock decision is unaffected.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from . import paths as _paths

_IMAGE_KEYS = {"image", "frame", "img", "pixels", "raw", "bytes"}


class _JsonLineFormatter(logging.Formatter):
    """Format a record's ``msg`` dict (plus level/time) as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any]
        if isinstance(record.msg, dict):
            payload = dict(record.msg)
        else:
            payload = {"message": record.getMessage()}
        payload.setdefault("ts", round(time.time(), 3))
        payload.setdefault("level", record.levelname)
        payload.setdefault("logger", record.name)
        # Privacy guard (REQ-NF-13): strip anything that looks like image data.
        for key in list(payload.keys()):
            if key.lower() in _IMAGE_KEYS:
                payload[key] = "<redacted:no-image-persistence>"
        return json.dumps(payload, separators=(",", ":"), default=str)


def get_logger(
    name: str = "facelock",
    *,
    level: str = "INFO",
    max_size_mb: int = 10,
    rotate_count: int = 5,
    log_path: Path | None = None,
    to_stderr: bool = True,
) -> logging.Logger:
    """Return a configured structured logger (idempotent per name).

    A rotating file handler enforces REQ-NF-09's hard size cap; a stderr
    handler mirrors events for interactive use. Handler setup failures (e.g.
    an unwritable directory) degrade to stderr-only rather than raising.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level, logging.INFO))
    logger.propagate = False
    if getattr(logger, "_facelock_configured", False):
        return logger

    formatter = _JsonLineFormatter()

    path = log_path or _paths.events_log_path()
    try:
        _paths.ensure_dir(path.parent, 0o700)
        file_handler = RotatingFileHandler(
            path,
            maxBytes=max(1, max_size_mb) * 1024 * 1024,
            backupCount=max(1, rotate_count),
            delay=True,
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        try:
            os.chmod(path, 0o600)
        except FileNotFoundError:
            pass  # delay=True: file created on first write, chmod-ed then.
    except OSError:
        # FM-12: could not open the log file. Degrade to stderr only.
        to_stderr = True

    if to_stderr:
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        logger.addHandler(stream)

    logger._facelock_configured = True  # type: ignore[attr-defined]
    return logger


def event(logger: logging.Logger, kind: str, **fields: Any) -> None:
    """Emit one structured event line. Never raises (FM-12, SI)."""
    record: dict[str, Any] = {"event": kind}
    record.update(fields)
    try:
        logger.info(record)
    except Exception:  # pragma: no cover - logging must never break locking
        pass


# The exact loud CRITICAL dry-run banner (DES-DRYRUN, design section 4.1.2).
DRY_RUN_BANNER = (
    "================= facelock DRY-RUN — NOT PROTECTING THIS SESSION =================\n"
    "OS-lock actuation is DISABLED. Escalations are logged, never executed.\n"
    "loginctl/gdbus/xdg will NOT run. Your screen will NOT lock. Do NOT rely on this.\n"
    "================================================================================"
)


def emit_dry_run_banner(logger: Any, component: str) -> None:
    """Print the loud dry-run banner to stderr + log a CRITICAL event, every boot.

    A dry-run process must be impossible to start without a visible, logged
    declaration that the session is NOT protected (design section 4.1.2). Never
    raises: a logging failure must not stop the process from coming up.
    """
    try:
        print(DRY_RUN_BANNER, file=sys.stderr, flush=True)
    except Exception:  # pragma: no cover - a broken stderr must not crash startup
        pass
    try:
        logger.critical({
            "event": "dry_run_active",
            "component": component,
            "detail": "OS-lock actuation disabled; this session is NOT protected",
        })
    except Exception:  # pragma: no cover
        pass


class AuditLog:
    """Append-only HMAC-chained audit trail (Hardening, REQ-NF-25).

    Inert unless enabled. When enabled, each appended entry stores a running
    HMAC-SHA256 over ``prev_mac || canonical_json(entry)`` so that removing or
    editing any line breaks the chain and is detectable by :meth:`verify`.

    The chain key is derived from a per-install secret file (0600). This is the
    prototype's honest audit hook; the full Hardening design binds the key to
    the Secret Service keyring (design section 11.5) -- swapping the key source
    does not change the chaining logic.
    """

    def __init__(self, path: Path, key: bytes, enabled: bool) -> None:
        self.path = path
        self._key = key
        self.enabled = enabled

    @staticmethod
    def derive_key(state_dir: Path) -> bytes:
        """Load or create a 0600 audit chain key under the state dir."""
        key_path = state_dir / "audit.key"
        if key_path.exists():
            return key_path.read_bytes()
        key = os.urandom(32)
        _paths.secure_write_bytes(key_path, key, 0o600)
        return key

    def _mac(self, prev: str, payload: str) -> str:
        return hmac.new(
            self._key, (prev + payload).encode("utf-8"), hashlib.sha256
        ).hexdigest()

    def _last_mac(self) -> str:
        if not self.path.exists():
            return "GENESIS"
        last = "GENESIS"
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        last = json.loads(line)["mac"]
                    except (json.JSONDecodeError, KeyError):
                        continue
        except OSError:
            return "GENESIS"
        return last

    def append(self, kind: str, **fields: Any) -> None:
        """Append one audit entry. No-op when disabled; never raises (FM-12)."""
        if not self.enabled:
            return
        try:
            _paths.ensure_dir(self.path.parent, 0o700)
            prev = self._last_mac()
            entry = {"ts": round(time.time(), 3), "event": kind, **fields}
            payload = json.dumps(entry, separators=(",", ":"), sort_keys=True, default=str)
            entry_with_mac = dict(entry)
            entry_with_mac["mac"] = self._mac(prev, payload)
            entry_with_mac["prev"] = prev
            line = json.dumps(entry_with_mac, separators=(",", ":"), default=str)
            with open(os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600), "w") as fh:
                fh.write(line + "\n")
        except OSError:
            pass  # FM-12: degrade audit, never block locking.

    def verify(self) -> bool:
        """Re-walk the chain; return True iff every MAC links correctly."""
        if not self.path.exists():
            return True
        prev = "GENESIS"
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    mac = obj.pop("mac")
                    claimed_prev = obj.pop("prev")
                    if claimed_prev != prev:
                        return False
                    payload = json.dumps(obj, separators=(",", ":"), sort_keys=True, default=str)
                    if self._mac(prev, payload) != mac:
                        return False
                    prev = mac
        except (OSError, json.JSONDecodeError, KeyError):
            return False
        return True
