"""SAFE dry-run mode -- the live-test safety harness (DES-DRYRUN, plan Task 3).

This is the security-contract regression guard for the ``--dry-run`` / no-OS-lock
mode. It exercises the FULL lock / escalation / unlock path of the guardian while
proving -- at the behavioural level -- that the sole OS-lock actuation subprocess
site (:func:`facelock.lock_backend._run`, which shells ``loginctl``/``gdbus``/
``xdg-screensaver``) is NEVER reached in dry-run. No camera, window, socket,
subprocess, or wall clock is touched: the guardian is driven directly through its
dispatch table with the shield / display / logger all faked.

Two halves, per the design:
  * POSITIVE: a real :class:`DryRunLockController` (built by the guardian at its
    single lock-controller seam when dry-run is effective) drives an
    away -> panic -> heartbeat-miss -> owner-return session. ``_run`` is
    monkeypatched to a spy that RAISES if ever called, so any real actuation
    fails the test loudly. We assert (A) ``_run`` never called, (B) escalation
    count with engaged=True/backend="dry-run", (C) an ordered decision log,
    (D) the grant unlocks + shield dismissed, (E) status surfaces dry_run=True.
  * NEGATIVE CONTROL: flip dry-run OFF with a REAL LockController whose backend
    routes through the ``_run`` spy, drive the same panic escalation, and prove
    ``_run`` WOULD have fired. This guarantees the positive assertions are
    meaningful -- if a future edit silently no-ops REAL mode (or dry-run stops
    protecting), CI fails.
"""

from __future__ import annotations

import os

import pytest

# These imports are the pre-implementation TRIPWIRE: the config helpers, the
# DryRunLockController, and the banner emitter do not exist until Task 3 lands,
# so this whole module fails to import (collection error) until the feature is
# implemented -- the TDD "red" state.
from facelock.config import (
    DryRunUnderSystemdError,
    load_config,
    resolve_dry_run,
)
from facelock.guardian import Guardian
from facelock.lock_backend import (
    DryRunLockController,
    LockController,
    LoginctlBackend,
)
from facelock.logging_setup import emit_dry_run_banner

# Reuse the proven hardware-free doubles from the re-lock suite.
from tests.test_relock import CountingShield, FakeDisplay

UID = os.getuid()


# --------------------------------------------------------------------------- #
# Test doubles local to this module.
# --------------------------------------------------------------------------- #
class RecordingLogger:
    """Captures every structured ``event()`` record so we can assert the log."""

    def __init__(self) -> None:
        self.records: list[dict] = []

    def _capture(self, record) -> None:
        if isinstance(record, dict):
            self.records.append(dict(record))

    # event() -> logger.info(dict); emit_dry_run_banner -> logger.critical(dict).
    def info(self, record) -> None:
        self._capture(record)

    def critical(self, record) -> None:
        self._capture(record)

    def warning(self, *a, **k) -> None:  # tolerated, unused
        pass

    def error(self, *a, **k) -> None:
        pass

    def debug(self, *a, **k) -> None:
        pass

    def events(self, kind: str) -> list[dict]:
        return [r for r in self.records if r.get("event") == kind]

    def kinds_in_order(self, *wanted: str) -> list[str]:
        want = set(wanted)
        return [r["event"] for r in self.records if r.get("event") in want]


class FakeGreeter:
    """No-op desktop greeter (avoid spawning notify-send)."""

    enabled = True

    def show(self, *a, **k) -> None:
        pass


def _dry_run_guardian(*, log: RecordingLogger, config_driven: bool = True) -> Guardian:
    """Build a REAL guardian with dry-run EFFECTIVE, no controller injected.

    With no ``lock_controller=`` injection the guardian must build a real
    :class:`DryRunLockController` at its single seam -- that is exactly what we
    are testing. ``config_driven`` toggles whether dry-run arrives via
    ``security.dry_run`` (the config seam) or via the ephemeral CLI-effective
    ``dry_run=`` constructor arg (the flag seam); both must produce the same
    no-actuation controller.
    """
    if config_driven:
        cfg = load_config(raw={"security": {"dry_run": True}})
        g = Guardian(
            cfg,
            shield=CountingShield(),
            display=FakeDisplay(),
            greeter=FakeGreeter(),
            logger=log,
            install_signals=False,
        )
    else:
        cfg = load_config(raw={})  # dry_run defaults False in config...
        g = Guardian(
            cfg,
            dry_run=True,  # ...but the CLI-effective flag forces it on
            shield=CountingShield(),
            display=FakeDisplay(),
            greeter=FakeGreeter(),
            logger=log,
            install_signals=False,
        )
    g._welcome_hold_s = 0.0   # immediate dismiss (welcome timing tested elsewhere)
    g._owner_present = True    # enrolled-owner scenario
    return g


# --------------------------------------------------------------------------- #
# PRIMARY simulation: full session, provably zero OS-lock actuation.
# --------------------------------------------------------------------------- #
def test_dryrun_full_session_never_actuates_os_lock(monkeypatch):
    # GLOBAL SAFETY GUARD (belt-and-suspenders): the ONLY subprocess site for
    # loginctl/gdbus/xdg. In dry-run it must never be reached; if it is, fail loud.
    def _exploding_run(cmd, timeout=3.0):
        raise AssertionError(f"real OS-lock backend invoked in dry-run! cmd={cmd!r}")

    monkeypatch.setattr("facelock.lock_backend._run", _exploding_run)

    log = RecordingLogger()
    g = _dry_run_guardian(log=log)

    # The seam wired the REAL product controller, not a test double.
    assert isinstance(g.lock_ctl, DryRunLockController)
    assert g._dry_run is True

    import time

    # 1) Owner steps away -> face-dismissable shield lock (NOT an escalation).
    away = g.dispatch({"cmd": "lock", "reason": "away"}, UID)
    g._drain_shield_queue()
    assert away["ok"] and away["escalated"] is False
    assert g.lock_ctl.engage_calls == 0          # away never escalates
    assert g.shield.is_up is True                 # shield raised
    assert g.display.off_calls == 1               # monitor blanked (face-dismissable)

    # 2) Panic (escape/panic-lock) -> escalate branch of _cmd_lock.
    panic = g.dispatch({"cmd": "lock", "reason": "panic"}, UID)
    g._drain_shield_queue()
    assert panic["escalated"] is True
    assert g.lock_ctl.engage_calls == 1

    # 3) Watchdog heartbeat-miss -> the watchdog escalation caller.
    g._started_at = time.monotonic() - 10_000     # clear the startup grace
    g._last_heartbeat = time.monotonic() - 10_000  # heartbeat long overdue
    g._check_watchdog()
    g._drain_shield_queue()
    assert g._watchdog_tripped is True
    assert g.lock_ctl.engage_calls == 2

    # (B) every escalation reported engaged via the dry-run backend.
    esc = log.events("os_lock_escalation")
    assert [e["reason"] for e in esc] == ["panic", "heartbeat_miss"]
    assert all(e["engaged"] is True for e in esc)
    assert all(e["backend"] == "dry-run" for e in esc)
    # The controller's dedicated "would-have" marker fired once per escalation.
    assert len(log.events("dry_run_would_lock")) == 2

    # 5) Owner returns and face-unlocks against the CURRENT (watchdog-minted)
    #    grant -- the daemon fetches the nonce then submits a bound grant.
    info = g.dispatch({"cmd": "get_grant_nonce"}, UID)
    assert info["locked"] is True and info["grant_nonce"]
    unlock = g.dispatch({
        "cmd": "unlock_grant", "grant_nonce": info["grant_nonce"],
        "lock_epoch": info["lock_epoch"], "score": 0.9, "tau": 0.5, "live": False,
    }, UID)
    g._drain_shield_queue()
    assert unlock["ok"] is True

    # (A) PRIMARY safety property: no loginctl/gdbus/xdg EVER (the spy never raised
    #     because it was never called -- assert its effect explicitly too).
    #     If _run had run, _exploding_run would have raised and failed the test.
    #     (D) the session is truly unlocked at the shield level.
    assert g.grant.current()[0] is False          # unlocked
    assert g.shield.is_up is False                 # shield dismissed

    # (C) ordered decision log: the away lock precedes the escalations, which
    #     precede the unlock, and the escalation reasons are ordered.
    ordered = log.kinds_in_order("lock", "os_lock_escalation", "unlock")
    assert ordered[0] == "lock"                    # the away lock first
    assert ordered[-1] == "unlock"                 # unlock last
    assert "os_lock_escalation" in ordered
    first_esc = ordered.index("os_lock_escalation")
    assert ordered[:first_esc].count("lock") >= 1  # a lock happened before escalating
    assert ordered.index("unlock") > first_esc     # unlock after the escalations

    # (E) dry-run is surfaced in status (fail-safe visibility).
    status = g._cmd_status({})
    assert status["dry_run"] is True


def test_dryrun_effective_via_cli_flag_builds_same_controller():
    """The CLI-effective flag (dry_run=) forces dry-run even with config False."""
    log = RecordingLogger()
    g = _dry_run_guardian(log=log, config_driven=False)
    assert isinstance(g.lock_ctl, DryRunLockController)
    assert g._dry_run is True
    assert g._cmd_status({})["dry_run"] is True


def test_dryrun_controller_is_pure_no_op():
    """The controller reports engaged-but-unconfirmed and knows nothing is locked."""
    ctl = DryRunLockController()
    out = ctl.engage()
    assert out.engaged is True          # avoids the false "no backend confirmed" alarm
    assert out.confirmed is False       # honest: nothing was really actuated
    assert out.backend == "dry-run"
    assert ctl.is_any_locked() is None  # unknown, like FakeController
    assert ctl.engage_calls == 1


# --------------------------------------------------------------------------- #
# NEGATIVE CONTROL: REAL mode WOULD actuate -- proves the guard has teeth.
# --------------------------------------------------------------------------- #
def test_negative_control_real_mode_routes_through_run(monkeypatch):
    calls: list[list[str]] = []

    def _recording_run(cmd, timeout=3.0):
        calls.append(list(cmd))
        # Emulate a confirmable loginctl lock so the controller reports engaged.
        return (0, "LockedHint=yes", "")

    monkeypatch.setattr("facelock.lock_backend._run", _recording_run)

    log = RecordingLogger()
    # REAL controller with a concrete backend that actuates via _run. Build the
    # backend directly (bypassing the host-availability filter) so the test is
    # deterministic on any machine.
    real_ctl = LockController([LoginctlBackend()], verify_engaged_ms=60)
    cfg = load_config(raw={})  # dry_run False (real mode)
    g = Guardian(
        cfg,
        lock_controller=real_ctl,
        shield=CountingShield(),
        display=FakeDisplay(),
        greeter=FakeGreeter(),
        logger=log,
        install_signals=False,
    )
    g._owner_present = True
    assert g._dry_run is False
    assert not isinstance(g.lock_ctl, DryRunLockController)

    g.dispatch({"cmd": "lock", "reason": "panic"}, UID)
    g._drain_shield_queue()

    # The whole point: flipping dry-run OFF DOES reach the subprocess site. If
    # this ever stops firing, REAL mode has silently stopped protecting.
    assert calls, "REAL mode must actuate the OS lock via _run (regression guard)"
    esc = log.events("os_lock_escalation")
    assert esc and esc[0]["backend"] != "dry-run"


# --------------------------------------------------------------------------- #
# Config validation: refuse-on-invalid, phase-independent default False.
# --------------------------------------------------------------------------- #
def test_security_dry_run_defaults_false():
    assert load_config(raw={}).security.dry_run is False
    # phase-independent (same default in Hardening).
    h = load_config(raw={"security": {"phase": "H"}, "liveness": {"mode": "full"}})
    assert h.security.dry_run is False


def test_security_dry_run_bad_value_refuses_even_with_default_policy():
    # security=True -> a bad dry_run refuses to start regardless of on_invalid,
    # exactly like tau / phase / stranger.policy.
    from facelock.errors import ConfigError

    with pytest.raises(ConfigError) as exc:
        load_config(raw={"config": {"on_invalid": "default"},
                         "security": {"dry_run": "yes"}})
    assert any("security.dry_run" in e for e in exc.value.errors)


# --------------------------------------------------------------------------- #
# systemd hard-gate: config-only dry-run under systemd is refused.
# --------------------------------------------------------------------------- #
def test_systemd_gate_refuses_config_only_dry_run(monkeypatch):
    monkeypatch.setenv("INVOCATION_ID", "deadbeef")  # simulate systemd unit
    monkeypatch.delenv("NOTIFY_SOCKET", raising=False)
    cfg = load_config(raw={"security": {"dry_run": True}})
    with pytest.raises(DryRunUnderSystemdError):
        resolve_dry_run(False, cfg)  # config-only under systemd -> refuse


def test_systemd_gate_honours_explicit_cli_flag(monkeypatch):
    # An EXPLICIT --dry-run is a deliberate, banner-visible act; honour it even
    # under systemd (the intended developer escape hatch).
    monkeypatch.setenv("NOTIFY_SOCKET", "/run/systemd/notify")
    cfg = load_config(raw={})  # config dry_run False
    assert resolve_dry_run(True, cfg) is True


def test_guardian_main_exits_2_on_config_only_dry_run_under_systemd(tmp_path, monkeypatch):
    """main() fail-closes (exit 2) and NEVER constructs the guardian under the gate.

    The real Guardian is stubbed to a tripwire so that even a broken gate could
    not start a real lock-authority process (SAFETY: nothing may risk locking).
    """
    import facelock.guardian as guardian_mod

    class _Tripwire:
        def __init__(self, *a, **k):
            raise AssertionError("guardian must NOT be constructed when the gate refuses")

    monkeypatch.setattr(guardian_mod, "Guardian", _Tripwire)
    monkeypatch.setenv("INVOCATION_ID", "deadbeef")
    monkeypatch.delenv("NOTIFY_SOCKET", raising=False)
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text("[security]\ndry_run = true\n")

    rc = guardian_mod.main(["--config", str(cfg_file)])
    assert rc == 2


def test_daemon_main_exits_2_on_config_only_dry_run_under_systemd(tmp_path, monkeypatch):
    import facelock.daemon as daemon_mod

    class _Tripwire:
        def __init__(self, *a, **k):
            raise AssertionError("daemon must NOT be constructed when the gate refuses")

    monkeypatch.setattr(daemon_mod, "PerceptionDaemon", _Tripwire)
    monkeypatch.setenv("INVOCATION_ID", "deadbeef")
    monkeypatch.delenv("NOTIFY_SOCKET", raising=False)
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text("[security]\ndry_run = true\n")

    rc = daemon_mod.main(["--config", str(cfg_file)])
    assert rc == 2


def test_resolve_dry_run_interactive_config_only_ok(monkeypatch):
    # No systemd env -> a config-only dry_run=true is fine (interactive/CI use).
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    monkeypatch.delenv("NOTIFY_SOCKET", raising=False)
    cfg = load_config(raw={"security": {"dry_run": True}})
    assert resolve_dry_run(False, cfg) is True
    # And a plain real-mode config resolves to False.
    assert resolve_dry_run(False, load_config(raw={})) is False


# --------------------------------------------------------------------------- #
# Fail-safe surfacing: loud banner + config-check warning.
# --------------------------------------------------------------------------- #
def test_dry_run_banner_is_loud_on_stderr_and_logged(capsys):
    log = RecordingLogger()
    emit_dry_run_banner(log, "guardian")
    err = capsys.readouterr().err
    assert "DRY-RUN" in err
    assert "NOT PROTECTING THIS SESSION" in err
    assert "will NOT run" in err  # loginctl/gdbus/xdg disclosure
    crit = log.events("dry_run_active")
    assert crit and crit[0]["component"] == "guardian"


def test_config_check_warns_when_dry_run_true(tmp_path, capsys):
    from facelock.cli import cmd_config_check

    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text("[security]\ndry_run = true\n")

    class _Args:
        config = cfg_file

    rc = cmd_config_check(_Args())
    out = capsys.readouterr().out
    assert rc == 0
    assert "config OK" in out
    assert "WARNING" in out and "security.dry_run" in out
    assert "does NOT protect" in out


def test_status_and_health_surface_dry_run():
    log = RecordingLogger()
    g = _dry_run_guardian(log=log)
    assert g._cmd_status({})["dry_run"] is True
    g._write_health()
    from facelock import paths as _paths
    import json

    snapshot = json.loads(_paths.health_path().read_text())
    assert snapshot["dry_run"] is True
