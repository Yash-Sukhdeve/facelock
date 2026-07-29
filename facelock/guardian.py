"""facelock-guardian -- the Session Guardian process (C8..C12).

The guardian is the SOLE holder of lock authority (SI-P1). It:
  * owns the shield (C11) and the abstracted lock backend (C10 via C9),
  * runs the control server (C8) that receives decisions/CLI over the socket,
  * runs the watchdog: on a missed ``facelockd`` heartbeat it escalates to the
    real OS lock and keeps the shield up (SI-P4, FM-08),
  * dismisses the shield ONLY on a valid nonce-bound ``unlock_grant`` (SI-P1),
  * renders the greeting (C12).

The default on every boundary is LOCKED (SI-P2): the guardian starts with the
shield up and re-locks on start, on heartbeat miss, on disable, and on stop.

Threading: tkinter is not thread-safe, so ALL shield operations are enqueued by
the control-server thread and executed in the main thread; the ``GrantAuthority``
(its own lock) and the subprocess-based lock backend are safe to touch from the
handler thread.
"""

from __future__ import annotations

import argparse
import queue
import signal
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

from . import paths as _paths
from .config import Config, load_config
from .control import ControlServer, GrantAuthority
from .display import DisplayController
from .lock_backend import LockController, select_backends
from .logging_setup import AuditLog, event, get_logger
from .shield import Greeter, ShieldWindow

# Reasons that ALWAYS escalate to the real OS lock, regardless of config: these
# are security-mandatory (a disabled or unmonitored face-unlock must fall to the
# password path; panic must be a real lock).
# NOTE: "shutdown" is deliberately NOT hard-escalated. On a normal stop/restart
# (every ``systemctl restart`` sends SIGTERM) throwing the OS password lock would
# force the GNOME login page, which a face cannot clear (no PAM) -- exactly the
# "unlock stopped working after restart" symptom. Stop-time escalation is now
# governed by config (``escalate_os_lock_on``) and defaults OFF for the prototype.
_HARD_ESCALATE = frozenset({"panic", "heartbeat_miss", "disable", "error"})


def _sd_notify(state: str) -> None:
    """Best-effort systemd sd_notify (READY=1 / WATCHDOG=1)."""
    import os

    addr = os.environ.get("NOTIFY_SOCKET")
    if not addr:
        return
    if addr.startswith("@"):
        addr = "\0" + addr[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
            sock.connect(addr)
            sock.sendall(state.encode("utf-8"))
    except OSError:
        pass


class Guardian:
    """Lock-authority process. Injectable pieces make it unit-testable."""

    def __init__(
        self,
        config: Config,
        *,
        lock_controller: LockController | None = None,
        shield: ShieldWindow | None = None,
        greeter: Greeter | None = None,
        display: DisplayController | None = None,
        logger: Any = None,
        install_signals: bool = True,
    ) -> None:
        self.cfg = config
        self.log = logger or get_logger(
            "facelock.guardian",
            level=config.logging.level,
            max_size_mb=config.logging.max_size_mb,
            rotate_count=config.logging.rotate_count,
        )
        self.audit = AuditLog(
            _paths.audit_log_path(),
            AuditLog.derive_key(_paths.ensure_dir(_paths.state_home(), 0o700)),
            enabled=bool(config.security.audit),
        )
        self.grant = GrantAuthority(window_s=float(config.liveness.challenge_timeout_s))
        self.lock_ctl = lock_controller or LockController(
            select_backends(config.lock.backend),
            verify_engaged_ms=config.lock.verify_engaged_ms,
            logger=self.log,
        )
        self.shield_enabled = bool(config.lock.shield)
        self.shield = shield if shield is not None else ShieldWindow(
            owner_name=config.unlock.owner_name,
            on_password_escape=self._on_password_escape,
        )
        self.greeter = greeter or Greeter(enabled=bool(config.unlock.greeting))
        self.display = display if display is not None else DisplayController(
            enabled=bool(config.lock.screen_off), logger=self.log,
        )
        self.face_unlock_enabled = True

        self._escalate = set(config.lock.escalate_os_lock_on) | _HARD_ESCALATE
        self._heartbeat_sec = int(config.service.heartbeat_sec)
        self._owner_name = config.unlock.owner_name

        # On unlock, show a "Welcome back" splash on the shield for this long
        # before dismissing (0 => dismiss immediately). Non-blocking: scheduled
        # via the main loop, not a sleep in the handler thread.
        self._welcome_hold_s = float(config.unlock.welcome_hold_s)
        self._welcome_dismiss_at: float | None = None
        # Hold the UNAUTHORIZED verdict on screen this long before a re-check can
        # overwrite it, so a denied result is actually readable.
        self._denied_hold_s = 1.2
        self._denied_until = 0.0

        # Monitor-power (DPMS) re-assertion while locked. X wakes the display on
        # activity/DPMS timers, so while the shield is up we re-issue "force off"
        # on this cadence. This never touches the camera/perception (that runs in
        # facelockd), so the owner's return is still detected and wakes the screen.
        self._screen_off_active = False
        self._screen_reassert_s = 3.0
        self._last_screen_assert = 0.0

        self._shield_q: "queue.Queue[tuple[str, Any]]" = queue.Queue()
        self._server: ControlServer | None = None
        self._stop = threading.Event()
        self._started_at = time.monotonic()
        self._last_heartbeat = time.monotonic()
        self._heartbeat_seq = -1
        self._daemon_state = "INIT"
        self._daemon_health: dict[str, Any] = {}
        self._watchdog_tripped = False
        self._install_signals = install_signals
        # Enrollment coordination: when True the guardian tells the daemon (via
        # the heartbeat reply) to release the camera so `facelock enroll` can use
        # it, then resume. The daemon keeps heartbeating, so the watchdog does
        # NOT trip during enrollment.
        self._perception_paused = False

    # -- shield queue (executed in main thread) --------------------------- #
    def _enqueue_shield(self, op: str, arg: Any = None) -> None:
        self._shield_q.put((op, arg))

    def _drain_shield_queue(self) -> None:
        while True:
            try:
                op, arg = self._shield_q.get_nowait()
            except queue.Empty:
                break
            # Screen-power ops are independent of the shield and run even when the
            # shield is disabled (they never touch input/perception).
            if op == "screen_off":
                self._do_screen_off()
                continue
            if op == "screen_on":
                self._do_screen_on()
                continue
            if not self.shield_enabled:
                continue
            try:
                if op == "raise":
                    self.shield.raise_shield(arg or "Locked")
                elif op == "status":
                    self.shield.set_status(arg or "Locked")
                elif op == "checking":
                    prog, vk, vn = arg if arg else (0.0, 0, 0)
                    self.shield.set_recognizing(prog, vk, vn)
                elif op == "denied":
                    self.shield.set_denied(arg or "Unauthorized user")
                elif op == "welcome":
                    self.shield.set_welcome(arg or self._owner_name)
                elif op == "dismiss":
                    self.shield.dismiss()
            except Exception as exc:  # shield errors never crash the guardian
                event(self.log, "shield_error", op=op, error=str(exc))

    # -- monitor power (DPMS), executed in the main thread ---------------- #
    def _do_screen_off(self) -> None:
        """Blank the monitor and arm the re-assert cadence (main thread)."""
        try:
            issued = self.display.screen_off()
        except Exception as exc:  # never let display errors crash the guardian
            event(self.log, "display_error", op="off", error=str(exc))
            issued = False
        # Only arm the cadence if the command was actually issued; otherwise a
        # disabled/absent display would spin the re-assert loop for nothing.
        self._screen_off_active = bool(issued)
        self._last_screen_assert = time.monotonic()

    def _do_screen_on(self) -> None:
        """Wake the monitor and stop re-asserting (main thread)."""
        self._screen_off_active = False
        try:
            self.display.screen_on()
        except Exception as exc:
            event(self.log, "display_error", op="on", error=str(exc))

    def _maybe_finish_welcome(self, now: float) -> None:
        """Dismiss the shield once the post-unlock welcome hold elapses.

        Called every main-loop tick; a no-op until the scheduled time. This keeps
        the "Welcome back" splash non-blocking (no sleep in the handler thread).
        """
        if self._welcome_dismiss_at is not None and now >= self._welcome_dismiss_at:
            self._welcome_dismiss_at = None
            self._enqueue_shield("dismiss")

    def _on_password_escape(self) -> None:
        """Escape key on the shield -> engage the real OS lock (password)."""
        self.grant.force_locked()
        self._escalate_os_lock("panic")
        # The user deliberately wants the password screen: wake the monitor so
        # they can type, then drop our shield to expose the OS lock.
        self._enqueue_shield("screen_on")
        self._enqueue_shield("dismiss")

    # -- lock actuation --------------------------------------------------- #
    def _escalate_os_lock(self, reason: str) -> bool:
        outcome = self.lock_ctl.engage()
        event(self.log, "os_lock_escalation", reason=reason,
              engaged=outcome.engaged, confirmed=outcome.confirmed,
              backend=outcome.backend, detail=outcome.detail)
        self.audit.append("os_lock", reason=reason, engaged=outcome.engaged,
                          backend=outcome.backend)
        if not outcome.engaged:
            # SI-P5: no backend confirmed -> hold the shield + critical alert.
            event(self.log, "lock_critical", reason=reason,
                  detail="no lock backend confirmed engaged; holding shield")
        return outcome.engaged

    # -- control-server dispatch (handler thread) ------------------------- #
    def dispatch(self, message: dict[str, Any], peer_uid: int) -> dict[str, Any]:
        cmd = message.get("cmd")
        handlers = {
            "ping": lambda m: {"ok": True, "pong": True},
            "lock": self._cmd_lock,
            "unlock_grant": self._cmd_unlock_grant,
            "get_grant_nonce": self._cmd_get_nonce,
            "disable": self._cmd_disable,
            "enable": self._cmd_enable,
            "status": self._cmd_status,
            "reload_config": self._cmd_reload,
            "heartbeat": self._cmd_heartbeat,
            "greet": self._cmd_greet,
            "shield_status": self._cmd_shield_status,
            "pause_perception": self._cmd_pause_perception,
            "resume_perception": self._cmd_resume_perception,
        }
        handler = handlers.get(cmd or "")
        if handler is None:
            return {"ok": False, "reason": "unknown_cmd", "cmd": cmd}
        try:
            return handler(message)
        except Exception as exc:
            event(self.log, "dispatch_error", cmd=cmd, error=str(exc))
            return {"ok": False, "reason": "handler_error", "error": str(exc)}

    def _cmd_lock(self, message: dict[str, Any]) -> dict[str, Any]:
        reason = str(message.get("reason", "manual"))
        # Re-mint a fresh nonce/epoch and (re-)raise the shield. This is fully
        # idempotent and re-arms on every call, so re-locking after any number of
        # prior unlock cycles works indefinitely (requirement: reliable re-lock).
        epoch, nonce = self.grant.raise_shield()
        self._enqueue_shield("raise", _status_for(reason))
        escalated = False
        if reason in self._escalate:
            # Fail-closed OS password lock (heartbeat_miss/panic/suspend/shutdown/
            # disable). Keep the monitor ON so the password prompt is visible.
            escalated = self._escalate_os_lock(reason)
            self._enqueue_shield("screen_on")
        else:
            # Prototype face-dismissable lock (away/stranger/manual/cooldown):
            # blank the monitor; the owner's returning face wakes it.
            self._enqueue_shield("screen_off")
        event(self.log, "lock", reason=reason, lock_epoch=epoch, escalated=escalated,
              screen_off=(not escalated and reason not in self._escalate))
        self.audit.append("lock", reason=reason, epoch=epoch)
        return {"ok": True, "state": "LOCKED", "lock_epoch": epoch,
                "grant_nonce": nonce, "escalated": escalated}

    def _cmd_get_nonce(self, message: dict[str, Any]) -> dict[str, Any]:
        # Refresh the challenge window at request time (the daemon is about to
        # submit a grant) so a return long after locking is not "expired".
        locked, epoch, nonce = self.grant.refresh_challenge()
        expose_nonce = nonce if (locked and self.face_unlock_enabled) else None
        return {"ok": True, "locked": locked, "lock_epoch": epoch,
                "grant_nonce": expose_nonce, "face_unlock": self.face_unlock_enabled}

    def _cmd_unlock_grant(self, message: dict[str, Any]) -> dict[str, Any]:
        if not self.face_unlock_enabled:
            event(self.log, "unlock_denied", reason="face_unlock_disabled")
            return {"ok": False, "reason": "face_unlock_disabled"}
        nonce = message.get("grant_nonce")
        epoch = message.get("lock_epoch")
        if not isinstance(nonce, str) or not isinstance(epoch, int):
            return {"ok": False, "reason": "malformed_grant"}
        ok, why = self.grant.validate_grant(nonce, epoch)
        score = message.get("score")
        tau = message.get("tau")
        live = bool(message.get("live", False))
        if ok:
            # Wake the monitor first (so the desktop/greeting is visible).
            self._enqueue_shield("screen_on")
            # Desktop notification (both surfaces, REQ-F-15).
            self.greeter.show(self._owner_name, ttl_s=3)
            if self._welcome_hold_s > 0.0:
                # Flash "Welcome back, <name>" ON the shield, then dismiss after a
                # short, non-blocking hold scheduled in the main loop.
                self._enqueue_shield("welcome", self._owner_name)
                self._welcome_dismiss_at = time.monotonic() + self._welcome_hold_s
            else:
                # Instant dismiss (welcome shown via the notification only).
                self._enqueue_shield("dismiss")
                self._welcome_dismiss_at = None
            event(self.log, "unlock", score=score, tau=tau, live=live, lock_epoch=epoch)
            self.audit.append("unlock", score=score, tau=tau, live=live, epoch=epoch)
            return {"ok": True}
        event(self.log, "unlock_denied", reason=why, lock_epoch=epoch)
        self.audit.append("unlock_denied", reason=why, epoch=epoch)
        return {"ok": False, "reason": why}

    def _cmd_disable(self, message: dict[str, Any]) -> dict[str, Any]:
        self.face_unlock_enabled = False
        self.grant.force_locked()
        self._enqueue_shield("raise", "Face unlock disabled - use OS password")
        self._enqueue_shield("screen_on")  # password path: keep the monitor lit
        self._escalate_os_lock("disable")
        event(self.log, "face_unlock_disabled")
        self.audit.append("disable")
        return {"ok": True, "face_unlock": False}

    def _cmd_enable(self, message: dict[str, Any]) -> dict[str, Any]:
        self.face_unlock_enabled = True
        event(self.log, "face_unlock_enabled")
        self.audit.append("enable")
        return {"ok": True, "face_unlock": True}

    def _cmd_status(self, message: dict[str, Any]) -> dict[str, Any]:
        locked, epoch, _nonce = self.grant.current()
        return {
            "ok": True,
            "locked": locked,
            "lock_epoch": epoch,
            "face_unlock": self.face_unlock_enabled,
            "shield_up": bool(self.shield_enabled and self.shield.is_up),
            "daemon_state": self._daemon_state,
            "daemon_health": self._daemon_health,
            "last_heartbeat_age_s": round(time.monotonic() - self._last_heartbeat, 2),
            "watchdog_tripped": self._watchdog_tripped,
            "os_locked": self.lock_ctl.is_any_locked(),
            "audit_enabled": self.audit.enabled,
            "perception_paused": self._perception_paused,
        }

    def _cmd_reload(self, message: dict[str, Any]) -> dict[str, Any]:
        try:
            new_cfg = load_config(self.cfg.source_path)
        except Exception as exc:
            errors = getattr(exc, "errors", None) or [str(exc)]
            event(self.log, "config_reload_failed", errors=errors)
            return {"ok": False, "errors": errors}
        # Apply hot-reloadable settings (keep old on any doubt, REQ-F-23).
        self.cfg = new_cfg.resolve_model_paths(_paths.models_dir())
        self.greeter.enabled = bool(self.cfg.unlock.greeting)
        self.display.set_config_enabled(bool(self.cfg.lock.screen_off))
        self._escalate = set(self.cfg.lock.escalate_os_lock_on) | _HARD_ESCALATE
        self._heartbeat_sec = int(self.cfg.service.heartbeat_sec)
        self._owner_name = self.cfg.unlock.owner_name
        self._welcome_hold_s = float(self.cfg.unlock.welcome_hold_s)
        event(self.log, "config_reloaded", warnings=list(new_cfg.warnings))
        return {"ok": True, "warnings": list(new_cfg.warnings)}

    def _cmd_heartbeat(self, message: dict[str, Any]) -> dict[str, Any]:
        self._last_heartbeat = time.monotonic()
        self._heartbeat_seq = int(message.get("seq", self._heartbeat_seq))
        self._daemon_state = str(message.get("state", self._daemon_state))
        health = message.get("health")
        if isinstance(health, dict):
            self._daemon_health = health
        if self._watchdog_tripped:
            event(self.log, "heartbeat_recovered", seq=self._heartbeat_seq)
            self._watchdog_tripped = False
        # Surface disable + camera-pause state so the daemon can stop/resume
        # verifying and release/reacquire the camera for enrollment.
        return {"ok": True, "face_unlock": self.face_unlock_enabled,
                "pause_camera": self._perception_paused}

    def _cmd_pause_perception(self, message: dict[str, Any]) -> dict[str, Any]:
        """Ask the daemon to release the camera (for `facelock enroll`).

        The daemon keeps heartbeating while paused, so the watchdog does not trip.
        The session lock state is left untouched.
        """
        self._perception_paused = True
        event(self.log, "perception_pause_requested")
        return {"ok": True, "paused": True, "heartbeat_sec": self._heartbeat_sec}

    def _cmd_resume_perception(self, message: dict[str, Any]) -> dict[str, Any]:
        """Resume perception after enrollment; the daemon reacquires + reloads."""
        self._perception_paused = False
        event(self.log, "perception_resume_requested")
        return {"ok": True, "paused": False}

    def _cmd_greet(self, message: dict[str, Any]) -> dict[str, Any]:
        name = str(message.get("name", self._owner_name))
        self.greeter.show(name, ttl_s=3)
        return {"ok": True}

    def _cmd_shield_status(self, message: dict[str, Any]) -> dict[str, Any]:
        """Cosmetic on-shield feedback from the daemon (NO lock authority).

        The daemon reports what it's perceiving while the shield is up so the
        person in front of the screen gets feedback: an animated "Recognizing..."
        while verifying the owner, or a red "Unauthorized user" for a non-owner.
        This never unlocks anything and is ignored unless the shield is up.
        """
        phase = str(message.get("phase", ""))
        # Do not paint feedback over an unlocked/absent shield (e.g. a stale push
        # racing an unlock); only meaningful while locked.
        if not self.grant.current()[0]:
            return {"ok": True, "ignored": "not_locked"}
        # Someone is at the camera -> WAKE the monitor so the animated feedback is
        # actually visible (while locked-and-away the screen is DPMS-off). When
        # they leave, blank it again. This is what makes the graphics show.
        now = time.monotonic()
        if phase == "recognizing":  # CHECKING AUTHORIZATION (with a real progress bar)
            if now < self._denied_until:
                return {"ok": True, "held": "denied"}  # let the verdict linger
            progress = float(message.get("progress") or 0.0)
            votes_k = int(message.get("votes_k") or 0)
            votes_need = int(message.get("votes_need") or 0)
            self._enqueue_shield("screen_on")
            self._enqueue_shield("checking", (progress, votes_k, votes_need))
        elif phase == "denied":  # UNAUTHORIZED verdict
            self._denied_until = now + self._denied_hold_s
            self._enqueue_shield("screen_on")
            self._enqueue_shield("denied", "Unauthorized user")
        elif phase == "locked":
            reason = str(message.get("reason") or "away")
            self._enqueue_shield("status", _status_for(reason))
            self._enqueue_shield("screen_off")  # nobody present -> dark again
        else:
            return {"ok": False, "reason": "bad_phase", "phase": phase}
        return {"ok": True}

    # -- watchdog (main thread) ------------------------------------------- #
    def _check_watchdog(self) -> None:
        grace = self._heartbeat_sec * 3
        if (time.monotonic() - self._started_at) < grace:
            return  # startup grace: the daemon may still be coming up
        age = time.monotonic() - self._last_heartbeat
        if age > grace and not self._watchdog_tripped:
            self._watchdog_tripped = True
            event(self.log, "heartbeat_miss", age_s=round(age, 2))
            self.audit.append("heartbeat_miss", age_s=round(age, 2))
            # SI-P4: keep the shield up + escalate to the real OS lock. Wake the
            # monitor so the required password prompt is visible.
            self.grant.force_locked()
            self._enqueue_shield("raise", "Monitor lost - locked (password required)")
            self._enqueue_shield("screen_on")
            self._escalate_os_lock("heartbeat_miss")

    def _write_health(self) -> None:
        import json

        locked, epoch, _ = self.grant.current()
        snapshot = {
            "ts": round(time.time(), 2),
            "locked": locked,
            "lock_epoch": epoch,
            "face_unlock": self.face_unlock_enabled,
            "daemon_state": self._daemon_state,
            "last_heartbeat_age_s": round(time.monotonic() - self._last_heartbeat, 2),
            "watchdog_tripped": self._watchdog_tripped,
        }
        try:
            _paths.secure_write_bytes(
                _paths.health_path(),
                json.dumps(snapshot).encode("utf-8"), 0o600,
            )
        except OSError:
            pass

    # -- lifecycle -------------------------------------------------------- #
    def run(self) -> int:
        if self._install_signals:
            signal.signal(signal.SIGTERM, self._on_signal)
            signal.signal(signal.SIGINT, self._on_signal)
        # Fail-closed default: start LOCKED with the shield up (SI-P2).
        self.grant.raise_shield()
        if self.shield_enabled:
            self._enqueue_shield("raise", "Locked - facelock starting")
        self._server = ControlServer(handler=self.dispatch, logger=self.log)
        try:
            self._server.start()
        except OSError as exc:
            event(self.log, "control_server_failed", error=str(exc))
            return 1
        event(self.log, "guardian_started", socket=str(self._server.socket_path),
              shield=self.shield_enabled, phase=self.cfg.phase)
        _sd_notify("READY=1")

        last_health = 0.0
        try:
            while not self._stop.is_set():
                self._drain_shield_queue()
                if self.shield_enabled:
                    self.shield.pump()
                self._check_watchdog()
                now = time.monotonic()
                self._maybe_finish_welcome(now)
                # Re-assert monitor-off while locked (X wakes the display on its
                # own timers). This is display-only; perception keeps running in
                # facelockd, so the owner's return is still detected.
                if (self._screen_off_active
                        and (now - self._last_screen_assert) >= self._screen_reassert_s):
                    try:
                        self.display.screen_off()
                    except Exception as exc:
                        event(self.log, "display_error", op="reassert", error=str(exc))
                    self._last_screen_assert = now
                if now - last_health >= 2.0:
                    self._write_health()
                    _sd_notify("WATCHDOG=1")
                    last_health = now
                time.sleep(0.1)
        finally:
            self._shutdown()
        return 0

    def _on_signal(self, signum: int, _frame: Any) -> None:
        event(self.log, "signal", signum=signum)
        self._stop.set()

    def _shutdown(self) -> None:
        # Fail-closed on stop (SI-P2): ensure the grant state is LOCKED. But do
        # NOT throw the OS password lock on a normal stop/restart unless the
        # config explicitly opts in (prototype default OFF): that GNOME login page
        # cannot be face-cleared and is what broke "unlock after restart".
        # systemd restarts the guardian, which re-locks with the shield in ~1s.
        self.grant.force_locked()
        if "shutdown" in self._escalate:
            self._escalate_os_lock("shutdown")
        # Wake the monitor so the desktop / any password prompt is visible.
        self._screen_off_active = False
        try:
            self.display.screen_on()
        except Exception:
            pass
        if self._server is not None:
            self._server.stop()
        try:
            self.shield.dismiss()
        except Exception:
            pass
        event(self.log, "guardian_stopped")


def _status_for(reason: str) -> str:
    return {
        "away": "Locked - you stepped away",
        "stranger": "Locked - unrecognized face",
        "panic": "Locked",
        "camera_loss": "Locked - camera unavailable",
        "suspend": "Locked - resumed from suspend",
        "cooldown": "Locked - too many attempts",
        "startup": "Locked - facelock starting",
    }.get(reason, "Locked")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="facelock-guardian",
                                     description="facelock session guardian (lock authority)")
    parser.add_argument("--config", type=Path, default=None, help="path to config.toml")
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config).resolve_model_paths(_paths.models_dir())
    except Exception as exc:
        print(f"facelock-guardian: config error: {exc}", file=sys.stderr)
        for err in getattr(exc, "errors", []) or []:
            print(f"  - {err}", file=sys.stderr)
        return 2
    return Guardian(cfg).run()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
