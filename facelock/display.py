"""DisplayController -- physical monitor power (DPMS) control for the guardian.

The prototype convenience lock (the shield, C11) covers the desktop, but the
monitor stays lit. For the "owner away / stranger present" experience the user
wants the *physical monitor* to go dark while locked and to come back on when the
owner face-unlocks. This controller turns the X11 display off/on via DPMS
(``xset dpms force off`` / ``xset dpms force on``) using ``$DISPLAY``.

Design rules (REQ-F-14, SI):
  * **Never raises, never blocks the lock.** Every public method is best-effort:
    on a missing ``$DISPLAY``, an absent ``xset``, a non-X11 (Wayland) session, or
    a subprocess failure it logs once and no-ops. Screen power is a comfort
    feature; it must never affect the fail-closed lock path.
  * **Does not touch perception.** Turning the monitor off is a *display-power*
    operation only. The camera + recognition run in the separate ``facelockd``
    process and keep running, so the daemon still detects the owner's return and
    asks the guardian to turn the screen back on. Nothing here releases the
    camera or pauses the perception loop.
  * **Re-assertable.** X wakes the display on activity/DPMS timers, so the
    guardian re-issues :meth:`screen_off` on a short cadence while locked
    (see :class:`facelock.guardian.Guardian`). This class exposes the idempotent
    primitive; the cadence lives in the guardian loop.

Wayland note: pure Wayland has no ``xset``/DPMS equivalent that we can drive from
here; on such sessions the controller degrades to a logged no-op and the shield
remains the (still fail-closed) barrier.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any

_UNSET = object()  # sentinel: distinguish "resolve xset" from an explicit None


class DisplayController:
    """Best-effort X11 DPMS monitor power control (off while locked, on to unlock)."""

    #: xset call budget; DPMS force is instantaneous, so a short timeout is safe.
    _TIMEOUT_S = 2.0

    def __init__(
        self,
        *,
        enabled: bool = True,
        logger: Any = None,
        xset_path: str | None | Any = _UNSET,
        env: dict[str, str] | None = None,
    ) -> None:
        self._cfg_enabled = bool(enabled)
        self.log = logger
        self._env = dict(os.environ if env is None else env)
        # Resolve the xset binary once. An explicit ``xset_path=None`` forces the
        # "unavailable" path (used by tests); the default resolves via PATH.
        self._xset = shutil.which("xset") if xset_path is _UNSET else xset_path
        self._warned = False  # log the "unavailable" reason at most once

    # -- capability ------------------------------------------------------- #
    @property
    def available(self) -> bool:
        """True if we can actually drive DPMS (X11 display + xset present)."""
        return bool(self._xset) and bool(self._env.get("DISPLAY"))

    @property
    def enabled(self) -> bool:
        """True if the feature is configured on AND the environment supports it."""
        return self._cfg_enabled and self.available

    def set_config_enabled(self, enabled: bool) -> None:
        """Hot-reload hook: update the configured on/off flag (REQ-F-23)."""
        self._cfg_enabled = bool(enabled)

    # -- power ops -------------------------------------------------------- #
    def screen_off(self) -> bool:
        """Blank the physical monitor (DPMS off). Returns True iff issued.

        Idempotent and safe to call repeatedly (used both on lock and on the
        guardian's re-assert cadence).
        """
        return self._dpms("off")

    def screen_on(self) -> bool:
        """Wake the physical monitor (DPMS on) and reset the screensaver.

        Returns True iff the wake command was issued. Called on unlock and on
        escalation to the OS password lock (so the password prompt is visible).
        """
        ok = self._dpms("on")
        if ok:
            # Also cancel any pending blank so the display stays awake for the
            # user (best-effort; failure here is irrelevant).
            self._run([self._xset, "s", "reset"])  # type: ignore[list-item]
        return ok

    # -- internals -------------------------------------------------------- #
    def _dpms(self, mode: str) -> bool:
        if not self._cfg_enabled:
            return False
        if not self.available:
            self._warn_once(mode)
            return False
        rc = self._run([self._xset, "dpms", "force", mode])  # type: ignore[list-item]
        if rc != 0:
            self._log("display_dpms_failed", mode=mode, rc=rc)
            return False
        self._log("display_dpms", mode=mode)
        return True

    def _run(self, cmd: list[str]) -> int:
        """Run an xset command; never raises. Returns the exit code (255 on error)."""
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._TIMEOUT_S,
                check=False,
                env=self._env,
            )
            return proc.returncode
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            self._log("display_subprocess_error", cmd=" ".join(cmd), error=str(exc))
            return 255

    def _warn_once(self, mode: str) -> None:
        if self._warned:
            return
        self._warned = True
        reason = "no xset binary" if not self._xset else "no $DISPLAY (X11 absent / Wayland)"
        self._log("display_unavailable", mode=mode, reason=reason,
                  detail="monitor power control disabled; shield remains the barrier")

    def _log(self, event: str, **fields: Any) -> None:
        if self.log is None:
            return
        try:
            self.log.info({"event": event, **fields})
        except Exception:
            pass
