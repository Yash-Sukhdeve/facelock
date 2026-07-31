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
from .config import (
    Config,
    DryRunUnderSystemdError,
    load_config,
    resolve_dry_run,
)
from .control import ControlServer, GrantAuthority
from .display import DisplayController
from .lock_backend import DryRunLockController, LockController, select_backends
from .logging_setup import AuditLog, emit_dry_run_banner, event, get_logger
from .shield import Greeter, ShieldWindow
from .store import TemplateStore

# Reasons that ALWAYS escalate to the real OS lock, regardless of config: these
# are security-mandatory (a disabled or unmonitored face-unlock must fall to the
# password path; panic must be a real lock).
# NOTE: "shutdown" is deliberately NOT hard-escalated. On a normal stop/restart
# (every ``systemctl restart`` sends SIGTERM) throwing the OS password lock would
# force the GNOME login page, which a face cannot clear (no PAM) -- exactly the
# "unlock stopped working after restart" symptom. Stop-time escalation is now
# governed by config (``escalate_os_lock_on``) and defaults OFF for the prototype.
_HARD_ESCALATE = frozenset({"panic", "heartbeat_miss", "disable", "error"})

# Lock reasons the USER explicitly requested: they act even with no owner enrolled
# (engaging the OS password lock, which the user can always clear). EVERY other
# reason is automatic/system and is gated on an enrolled owner -- with no owner,
# face-unlock is inert and must never trap the user behind a shield no face can
# clear (defense-in-depth for the "enabled before enrolled" case).
_EXPLICIT_LOCK = frozenset({"panic", "disable"})


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
        dry_run: bool | None = None,
    ) -> None:
        self.cfg = config
        # SAFE dry-run mode (DES-DRYRUN): effective value is the CLI-resolved
        # override (from main(), which OR-combines --dry-run with config and
        # applies the systemd gate) or, when unset, the config key directly so
        # EVERY construction path is covered. dry-run swaps only the OS-lock
        # ACTUATOR; all decision logic (shield/nonce/FSM/watchdog) is untouched.
        self._dry_run = bool(config.security.dry_run if dry_run is None else dry_run)
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
        # Single additive seam: an injected controller always wins (tests); else
        # in dry-run build the no-op DryRunLockController; else -- byte-for-byte
        # the original REAL controller (dry-run NEVER weakens real mode).
        if lock_controller is not None:
            self.lock_ctl = lock_controller
        elif self._dry_run:
            self.lock_ctl = DryRunLockController(logger=self.log)
        else:
            self.lock_ctl = LockController(
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
        # No-owner passive mode: with no enrolled owner, face-unlock is inert and
        # must NOT take over the screen. Checked at startup and kept in sync from
        # the daemon's heartbeat health (health["template"]), so enrollment
        # activates us live without a restart.
        self._owner_present = self._check_owner_enrolled()
        # Auto-resume deadline for `facelock pause --minutes N` (None => manual).
        self._resume_at: float | None = None

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
        # The cadence is deliberately SLOW: a 3s cadence spawned 42,336 `xset`
        # off events in one locked-away session (FM-DPMS). 30s keeps the OFF
        # intent (fights X's wake timers) at ~1/10th the subprocess volume.
        self._screen_off_active = False
        self._screen_reassert_s = 30.0
        self._last_screen_assert = 0.0
        # Last DPMS state we actually issued (None = unknown at boot). Guards
        # against re-spawning `xset` for a state the monitor is already in
        # (de-dupe of redundant same-state screen_on/screen_off calls).
        self._last_dpms_on: bool | None = None
        # Debounced perception DPMS desire, set by shield_status and reconciled
        # ONCE per shield-queue drain (one main-loop tick). This collapses a rapid
        # recognizing<->locked flap into at most one real transition per tick.
        # None = no pending desire. DPMS is cosmetic monitor power ONLY; this
        # never gates lock/shield authority (fail-closed posture is untouched).
        self._dpms_desired: bool | None = None

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

    @staticmethod
    def _check_owner_enrolled() -> bool:
        """True if an owner template exists (face-unlock has someone to match)."""
        try:
            return TemplateStore().try_load() is not None
        except Exception:
            return False

    # -- shield queue (executed in main thread) --------------------------- #
    def _enqueue_shield(self, op: str, arg: Any = None) -> None:
        # An authoritative DPMS decision (lock/unlock/escalation enqueues
        # screen_on/screen_off) supersedes any pending debounced perception
        # desire, so a stale shield_status frame cannot re-toggle the monitor
        # behind an explicit lock/unlock/password action.
        if op in ("screen_on", "screen_off"):
            self._dpms_desired = None
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
        # Reconcile the debounced perception DPMS desire ONCE per drain (one main
        # -loop tick). Collapsing here means a rapid recognizing<->locked flap
        # queued within a tick applies at most one real transition, and the
        # last-state de-dupe in _do_screen_* suppresses redundant same-state
        # calls. Cosmetic monitor power only -- never touches shield/grant state.
        self._reconcile_perception_dpms()

    # -- monitor power (DPMS), executed in the main thread ---------------- #
    def _reconcile_perception_dpms(self) -> None:
        """Apply the latest debounced perception DPMS desire (main thread)."""
        want = self._dpms_desired
        if want is None:
            return
        self._dpms_desired = None
        if want:
            self._do_screen_on()
        else:
            self._do_screen_off()

    def _do_screen_off(self) -> None:
        """Blank the monitor and arm the slow re-assert cadence (main thread).

        De-duped: if the monitor is already known-blanked we keep the OFF
        *intent* (the slow re-assert still fights X's wake timers) but do NOT
        re-spawn `xset`, so a stream of 'locked' frames cannot strobe the display.
        """
        if self._last_dpms_on is False:
            self._screen_off_active = True   # keep intent; no redundant xset
            return
        try:
            issued = self.display.screen_off()
        except Exception as exc:  # never let display errors crash the guardian
            event(self.log, "display_error", op="off", error=str(exc))
            issued = False
        # Only arm the cadence if the command was actually issued; otherwise a
        # disabled/absent display would spin the re-assert loop for nothing.
        self._screen_off_active = bool(issued)
        if issued:
            self._last_dpms_on = False
            self._last_screen_assert = time.monotonic()

    def _do_screen_on(self) -> None:
        """Wake the monitor and stop re-asserting (main thread).

        De-duped: a repeat 'recognizing' frame while already awake is a no-op
        (no redundant `xset`), but the re-assert is always disarmed so we never
        blank behind a present owner.
        """
        self._screen_off_active = False
        if self._last_dpms_on is True:
            return
        try:
            self.display.screen_on()
            self._last_dpms_on = True
        except Exception as exc:
            event(self.log, "display_error", op="on", error=str(exc))

    def _maybe_reassert_screen(self, now: float) -> bool:
        """Re-issue DPMS-off on a SLOW cadence while locked-away (main thread).

        Returns True iff it spawned an `xset`. Runs only while the screen is
        intentionally blanked (``_screen_off_active``) and at most once every
        ``_screen_reassert_s`` (>=30s). This deliberately re-issues OFF even
        though we believe the monitor is already off -- its whole job is to fight
        X waking the display on its own DPMS/activity timers -- so it bypasses the
        same-state de-dupe. It never blanks a present owner (guarded by
        ``_screen_off_active``, which ``_do_screen_on`` clears).
        """
        if not self._screen_off_active:
            return False
        if (now - self._last_screen_assert) < self._screen_reassert_s:
            return False
        try:
            self.display.screen_off()
        except Exception as exc:
            event(self.log, "display_error", op="reassert", error=str(exc))
        self._last_screen_assert = now
        return True

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
        # No-owner passive: ignore AUTOMATIC locks when nobody is enrolled -- a
        # shield with no enrolled face could only be cleared with the OS password,
        # so an inert face-unlock must not trap the user. Explicit user actions
        # (panic/disable) still engage the OS password lock.
        if not self._owner_present and reason not in _EXPLICIT_LOCK:
            event(self.log, "lock_ignored_no_owner", reason=reason)
            return {"ok": True, "state": "PASSIVE", "no_owner": True}
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
            "dry_run": self._dry_run,
            "perception_paused": self._perception_paused,
            "pause_resume_in_s": (round(self._resume_at - time.monotonic(), 1)
                                  if (self._perception_paused and self._resume_at is not None)
                                  else None),
            "no_owner": not self._owner_present,
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
            # Track owner presence live: enrollment flips template False->True,
            # which activates us out of no-owner passive mode without a restart.
            if "template" in health:
                now_present = bool(health.get("template"))
                if now_present and not self._owner_present:
                    event(self.log, "owner_enrolled_activating")
                self._owner_present = now_present
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
        minutes = message.get("minutes")
        if isinstance(minutes, (int, float)) and minutes > 0:
            self._resume_at = time.monotonic() + float(minutes) * 60.0
        else:
            self._resume_at = None
        event(self.log, "perception_pause_requested", auto_resume_min=minutes)
        return {"ok": True, "paused": True, "heartbeat_sec": self._heartbeat_sec,
                "auto_resume_min": minutes}

    def _cmd_resume_perception(self, message: dict[str, Any]) -> dict[str, Any]:
        """Resume perception after enrollment; the daemon reacquires + reloads."""
        self._perception_paused = False
        self._resume_at = None
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
        # DPMS side note: instead of enqueuing a screen_on/screen_off per frame
        # (which strobed the monitor under a camera flap), record the DESIRED
        # DPMS state; _drain_shield_queue reconciles it once per tick with the
        # last-state de-dupe. This debounces the perception-driven monitor power
        # while leaving the visual feedback (checking/denied/status) per-frame.
        if phase == "recognizing":  # CHECKING AUTHORIZATION (with a real progress bar)
            if now < self._denied_until:
                return {"ok": True, "held": "denied"}  # let the verdict linger
            progress = float(message.get("progress") or 0.0)
            votes_k = int(message.get("votes_k") or 0)
            votes_need = int(message.get("votes_need") or 0)
            self._dpms_desired = True  # someone is present -> wake (debounced)
            self._enqueue_shield("checking", (progress, votes_k, votes_need))
        elif phase == "denied":  # UNAUTHORIZED verdict
            self._denied_until = now + self._denied_hold_s
            self._dpms_desired = True  # show the verdict -> wake (debounced)
            self._enqueue_shield("denied", "Unauthorized user")
        elif phase == "locked":
            reason = str(message.get("reason") or "away")
            self._enqueue_shield("status", _status_for(reason))
            self._dpms_desired = False  # nobody present -> dark again (debounced)
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
            # No owner enrolled -> nothing to protect; stay passive (don't trap the
            # user behind an OS lock they never asked for).
            if not self._owner_present:
                return
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
            "dry_run": self._dry_run,
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
        # Fail-closed default: start LOCKED with the shield up (SI-P2) -- BUT only
        # when an owner is enrolled. With no owner, face-unlock is inert; starting
        # locked would trap the user at a shield no face can clear. Stay passive
        # until enrollment (the heartbeat then flips us active).
        if self._owner_present:
            self.grant.raise_shield()
            if self.shield_enabled:
                self._enqueue_shield("raise", "Locked - facelock starting")
        else:
            event(self.log, "no_owner_passive",
                  detail="no enrolled owner; face-unlock inactive -- run 'facelock enroll' to activate")
        self._server = ControlServer(handler=self.dispatch, logger=self.log)
        try:
            self._server.start()
        except OSError as exc:
            event(self.log, "control_server_failed", error=str(exc))
            return 1
        # Loud CRITICAL declaration BEFORE announcing readiness: a dry-run boot
        # must never be silent (DES-DRYRUN section 4.1.2).
        if self._dry_run:
            emit_dry_run_banner(self.log, "guardian")
        event(self.log, "guardian_started", socket=str(self._server.socket_path),
              shield=self.shield_enabled, phase=self.cfg.phase, dry_run=self._dry_run)
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
                # `facelock pause --minutes N` auto-resume.
                if (self._perception_paused and self._resume_at is not None
                        and now >= self._resume_at):
                    self._perception_paused = False
                    self._resume_at = None
                    event(self.log, "perception_auto_resumed")
                # Re-assert monitor-off while locked (X wakes the display on its
                # own timers). This is display-only; perception keeps running in
                # facelockd, so the owner's return is still detected. Runs on a
                # SLOW cadence (see _maybe_reassert_screen) to avoid the xset storm.
                self._maybe_reassert_screen(now)
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
    parser.add_argument(
        "--dry-run", action="store_true",
        help="SAFE test mode: log OS-lock escalations but NEVER actuate the OS lock "
             "(no loginctl/gdbus/xdg -- your screen will not lock)")
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config).resolve_model_paths(_paths.models_dir())
    except Exception as exc:
        print(f"facelock-guardian: config error: {exc}", file=sys.stderr)
        for err in getattr(exc, "errors", []) or []:
            print(f"  - {err}", file=sys.stderr)
        return 2
    try:
        dry_run = resolve_dry_run(args.dry_run, cfg)
    except DryRunUnderSystemdError as exc:
        # systemd hard-gate: config-only dry-run under a managed service is
        # fail-closed refused (exit 2, matching the config-refusal contract).
        print(f"facelock-guardian: {exc}", file=sys.stderr)
        return 2
    return Guardian(cfg, dry_run=dry_run).run()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
