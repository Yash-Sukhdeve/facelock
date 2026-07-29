"""LockBackend (C10) + LockController (C9) -- OS-lock actuation, verified.

Realizes REQ-F-13, REQ-NF-19/20, and SI-P5 / FM-16. The lock actuator is behind
an abstract interface with multiple concrete backends (loginctl primary, GNOME
D-Bus and xdg-screensaver fallbacks), selectable at runtime. A Wayland backend
can be added here without touching the perception pipeline (ADR-8, OQ-10).

SI-P5 -- lock actuation is *verified, not assumed*: after calling a backend the
controller confirms the lock actually engaged (``is_locked()``); if it cannot be
confirmed it tries the next backend, and if none can be confirmed it reports
failure so the caller holds the shield and raises a critical alert. It NEVER
reports "locked" optimistically.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


def _run(cmd: list[str], timeout: float = 3.0) -> tuple[int, str, str]:
    """Run a command, returning ``(rc, stdout, stderr)``; never raises."""
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        return 255, "", str(exc)


class LockBackend(ABC):
    """Abstract lock backend."""

    name: str = "abstract"

    @abstractmethod
    def available(self) -> bool: ...

    @abstractmethod
    def lock(self) -> bool:
        """Request the OS lock. Returns True if the request was accepted."""

    @abstractmethod
    def is_locked(self) -> bool | None:
        """Return True/False if the lock state is known, else ``None``."""


class LoginctlBackend(LockBackend):
    """``loginctl lock-session`` -- session-manager-agnostic primary (ADR-8)."""

    name = "loginctl"

    def __init__(self) -> None:
        self._session = os.environ.get("XDG_SESSION_ID", "")

    def available(self) -> bool:
        return shutil.which("loginctl") is not None

    def _session_id(self) -> str:
        if self._session:
            return self._session
        rc, out, _ = _run(["loginctl", "list-sessions", "--no-legend"])
        if rc == 0:
            for line in out.splitlines():
                parts = line.split()
                if parts:
                    self._session = parts[0]
                    break
        return self._session

    def lock(self) -> bool:
        sid = self._session_id()
        cmd = ["loginctl", "lock-session"] + ([sid] if sid else [])
        rc, _, _ = _run(cmd)
        return rc == 0

    def is_locked(self) -> bool | None:
        sid = self._session_id()
        if not sid:
            return None
        rc, out, _ = _run(["loginctl", "show-session", sid, "-p", "LockedHint"])
        if rc != 0:
            return None
        return "LockedHint=yes" in out


class GnomeDbusBackend(LockBackend):
    """GNOME ``org.gnome.ScreenSaver`` via ``gdbus`` (fallback)."""

    name = "gnome_dbus"
    _DEST = "org.gnome.ScreenSaver"
    _PATH = "/org/gnome/ScreenSaver"

    def available(self) -> bool:
        return shutil.which("gdbus") is not None

    def lock(self) -> bool:
        rc, _, _ = _run([
            "gdbus", "call", "--session", "--dest", self._DEST,
            "--object-path", self._PATH, "--method", f"{self._DEST}.Lock",
        ])
        return rc == 0

    def is_locked(self) -> bool | None:
        rc, out, _ = _run([
            "gdbus", "call", "--session", "--dest", self._DEST,
            "--object-path", self._PATH, "--method", f"{self._DEST}.GetActive",
        ])
        if rc != 0:
            return None
        return "true" in out.lower()


class XdgScreensaverBackend(LockBackend):
    """``xdg-screensaver lock`` -- last-resort fallback (state unverifiable)."""

    name = "xdg"

    def available(self) -> bool:
        return shutil.which("xdg-screensaver") is not None

    def lock(self) -> bool:
        rc, _, _ = _run(["xdg-screensaver", "lock"])
        return rc == 0

    def is_locked(self) -> bool | None:
        # xdg-screensaver cannot reliably report *lock* state -> unknown.
        return None


_BACKEND_CLASSES = {
    "loginctl": LoginctlBackend,
    "gnome_dbus": GnomeDbusBackend,
    "xdg": XdgScreensaverBackend,
}
_AUTO_ORDER = ("loginctl", "gnome_dbus", "xdg")


def select_backends(backend_cfg: str = "auto") -> list[LockBackend]:
    """Build the ordered, available backend list from config (REQ-NF-19)."""
    if backend_cfg == "auto":
        names = _AUTO_ORDER
    else:
        # Explicit backend first, then the rest as safety fallbacks (SI-P5).
        names = (backend_cfg,) + tuple(n for n in _AUTO_ORDER if n != backend_cfg)
    backends: list[LockBackend] = []
    for name in names:
        cls = _BACKEND_CLASSES.get(name)
        if cls is None:
            continue
        inst = cls()
        try:
            if inst.available():
                backends.append(inst)
        except Exception:
            continue
    return backends


@dataclass
class LockOutcome:
    engaged: bool  # confirmed OR (last-resort) actuation accepted
    confirmed: bool  # is_locked() confirmed True
    backend: str | None
    detail: str


class LockController:
    """Orchestrates OS-lock actuation with verify-engaged + fallback (C9)."""

    def __init__(
        self,
        backends: list[LockBackend],
        *,
        verify_engaged_ms: int = 500,
        logger: Any = None,
    ) -> None:
        self.backends = backends
        self.verify_engaged_ms = int(verify_engaged_ms)
        self.logger = logger

    def is_any_locked(self) -> bool | None:
        """Return True if any backend confirms locked, False if all confirm not,
        or ``None`` if no backend can report."""
        seen_false = False
        for backend in self.backends:
            try:
                state = backend.is_locked()
            except Exception:
                state = None
            if state is True:
                return True
            if state is False:
                seen_false = True
        return False if seen_false else None

    def engage(self) -> LockOutcome:
        """Actuate the OS lock, verifying engagement and falling through (SI-P5)."""
        if not self.backends:
            return LockOutcome(False, False, None, "no lock backend available")
        deadline_step = max(0.02, self.verify_engaged_ms / 1000.0)
        last_detail = "no backend succeeded"
        for backend in self.backends:
            try:
                accepted = backend.lock()
            except Exception as exc:
                last_detail = f"{backend.name}: lock() raised {exc}"
                self._log("lock_backend_error", backend=backend.name, error=str(exc))
                continue
            if not accepted:
                last_detail = f"{backend.name}: lock() rejected"
                continue
            # Poll for confirmation up to verify_engaged_ms.
            confirmed = self._confirm(backend, deadline_step)
            if confirmed is True:
                self._log("lock_engaged", backend=backend.name, confirmed=True)
                return LockOutcome(True, True, backend.name, "confirmed engaged")
            if confirmed is None and backend is self.backends[-1]:
                # Last resort, unverifiable backend: actuation accepted but we
                # could not confirm. Report engaged-but-unconfirmed so the
                # caller keeps the shield up (fail-closed).
                self._log("lock_actuated_unverified", backend=backend.name)
                return LockOutcome(True, False, backend.name, "actuated, unverified")
            last_detail = f"{backend.name}: not confirmed engaged"
        self._log("lock_not_engaged", detail=last_detail)
        return LockOutcome(False, False, None, last_detail)

    def _confirm(self, backend: LockBackend, step_s: float) -> bool | None:
        deadline = time.monotonic() + max(step_s, 0.05)
        result: bool | None = None
        while time.monotonic() < deadline:
            try:
                result = backend.is_locked()
            except Exception:
                result = None
            if result is True:
                return True
            time.sleep(min(0.05, step_s))
        return result  # False or None

    def _log(self, event: str, **fields: Any) -> None:
        if self.logger is None:
            return
        try:
            self.logger.info({"event": event, **fields})
        except Exception:
            pass
