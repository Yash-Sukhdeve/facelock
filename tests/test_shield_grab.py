"""Shield input-grab verification -- fail-closed on an ungrabbed shield.

Requirement: REQ-F-14 (the face-dismissible convenience lock must be an effective
barrier) and SI-P5 (lock actuation is VERIFIED, not assumed). Audit gap #2.

Before this fix, ``ShieldWindow.raise_shield`` reported success whenever the Tk
window merely MAPPED -- even if both ``grab_set_global()`` and the ``grab_set()``
fallback silently failed. A shield that captures no keyboard/pointer is just a
picture: a stranger could click and type on the desktop behind it while the
guardian believed the session was safely locked. The OS-lock path already
verifies-and-escalates (``lock_backend.LockController`` + ``_escalate_os_lock``);
the shield path did not.

These tests drive the guardian with a FAKE shield whose grab result is
controllable, and unit-test the real ``ShieldWindow`` grab-confirmation logic with
an injected fake Tk root. NO real window, camera, socket, or screen is ever
touched (SAFETY: a real fullscreen input grab would lock the developer out).

Fail-closed contract asserted here:
  * grab UNCONFIRMED -> the guardian must NOT trust the shield; it escalates to
    the real OS lock (``lock_ctl.engage`` / ``os_lock_escalation`` logged) and
    wakes the monitor so the required password prompt is visible.
  * grab CONFIRMED   -> the normal face-dismissible shield path; NO escalation,
    monitor blanked (the returning face wakes it).

Trade-off (plan Task 5): an away/stranger lock is NORMALLY face-dismissible, but a
shield that cannot grab input offers zero protection, so losing face-convenience
(falling to the password-required OS lock) is the correct fail-closed choice.
"""

from __future__ import annotations

import os

from facelock.config import load_config
from facelock.guardian import Guardian

# Reuse the proven hardware-free doubles from the re-lock suite.
from tests.test_relock import FakeController, FakeDisplay

UID = os.getuid()


class _GrabReportingShield:
    """Fake shield whose ``raise_shield`` reports whether the grab was confirmed.

    ``grab_ok=False`` models the audit gap: the window maps (``is_up`` True) but
    the X11 input grab did NOT take, so ``raise_shield`` returns ``False`` (==
    "not grabbed / no protection"). ``grab_ok=True`` is a normal confirmed grab.
    """

    def __init__(self, grab_ok: bool = True) -> None:
        self.grab_ok = grab_ok
        self.is_up = False
        self.raises = 0
        self.dismisses = 0
        self.phase = None
        self.last_status = None

    def raise_shield(self, status: str = "Locked") -> bool:
        self.raises += 1
        self.is_up = True          # the window maps regardless...
        self.phase = "locked"
        self.last_status = status
        return self.grab_ok        # ...but the RESULT == grab CONFIRMED

    def set_status(self, s: str) -> None:
        self.last_status = s
        self.phase = "locked"

    def set_recognizing(self, progress: float = 0.0, votes_k: int = 0,
                        votes_need: int = 0) -> None:
        self.phase = "recognizing"

    def set_denied(self, text: str = "Unauthorized user") -> None:
        self.phase = "denied"

    def set_welcome(self, name: str) -> None:
        self.phase = "welcome"

    def dismiss(self) -> None:
        self.is_up = False
        self.dismisses += 1

    def pump(self) -> None:
        pass


class _RecordingLogger:
    """Captures structured ``event()`` records so we can assert what was logged."""

    def __init__(self) -> None:
        self.records: list[dict] = []

    def info(self, rec: object) -> None:
        if isinstance(rec, dict):
            self.records.append(dict(rec))

    def critical(self, rec: object) -> None:
        self.info(rec)

    def warning(self, *a: object, **k: object) -> None:
        pass

    def error(self, *a: object, **k: object) -> None:
        pass

    def debug(self, *a: object, **k: object) -> None:
        pass

    def events(self, kind: str) -> list[dict]:
        return [r for r in self.records if r.get("event") == kind]


def _make_guardian(grab_ok: bool, log: _RecordingLogger | None = None) -> Guardian:
    cfg = load_config(raw={})   # prototype defaults: away/stranger NOT escalated
    g = Guardian(
        cfg,
        lock_controller=FakeController(),
        shield=_GrabReportingShield(grab_ok=grab_ok),
        display=FakeDisplay(),
        logger=log,
        install_signals=False,
    )
    g._welcome_hold_s = 0.0
    g._owner_present = True      # enrolled-owner scenario
    return g


# --------------------------------------------------------------------------- #
# Guardian-level: an UNCONFIRMED grab must fail-closed (escalate).
# --------------------------------------------------------------------------- #
def test_away_lock_with_unconfirmed_grab_escalates_to_os_lock():
    """FAIL-CLOSED: an away lock whose shield cannot grab input must not be
    trusted -- the guardian escalates to the real OS lock (SI-P5 / REQ-F-14).

    This is the RED test: on the pre-fix guardian the ``raise_shield`` result is
    discarded, so ``engage_calls`` stays 0 and this assertion fails.
    """
    log = _RecordingLogger()
    g = _make_guardian(grab_ok=False, log=log)
    resp = g.dispatch({"cmd": "lock", "reason": "away"}, UID)
    assert resp["ok"] and resp["state"] == "LOCKED"
    # The grab result is only known when the shield actually raises (main thread).
    g._drain_shield_queue()
    # An ungrabbed shield offers NO protection -> the real OS lock is engaged.
    assert g.lock_ctl.engage_calls >= 1, "ungrabbed shield was trusted -- NOT escalated"
    assert log.events("os_lock_escalation"), "no os_lock_escalation logged"
    assert log.events("shield_grab_unconfirmed"), "grab failure not logged"
    # The monitor is woken so the required OS password prompt is visible.
    assert g.display.on_calls >= 1


def test_stranger_lock_with_unconfirmed_grab_escalates_to_os_lock():
    """Same fail-closed contract for the stranger case (the common auto-lock):
    a see-through shield in front of a stranger must escalate to the OS lock."""
    log = _RecordingLogger()
    g = _make_guardian(grab_ok=False, log=log)
    g.dispatch({"cmd": "lock", "reason": "stranger"}, UID)
    g._drain_shield_queue()
    assert g.lock_ctl.engage_calls >= 1
    assert log.events("os_lock_escalation")


# --------------------------------------------------------------------------- #
# Guardian-level companion: a CONFIRMED grab keeps the normal (no-escalation) path.
# --------------------------------------------------------------------------- #
def test_confirmed_grab_uses_shield_and_does_not_escalate():
    """When the grab IS confirmed the away lock stays on the face-dismissible
    shield path: the monitor blanks and the OS lock is NOT engaged. This pins that
    the fix does not weaken the normal path (no spurious escalation)."""
    log = _RecordingLogger()
    g = _make_guardian(grab_ok=True, log=log)
    g.dispatch({"cmd": "lock", "reason": "away"}, UID)
    g._drain_shield_queue()
    assert g.shield.is_up is True
    assert g.lock_ctl.engage_calls == 0            # normal path -- no escalation
    assert not log.events("shield_grab_unconfirmed")
    assert g.display.off_calls == 1                # monitor blanked (dismissible)
    assert g.display.on_calls == 0


# --------------------------------------------------------------------------- #
# Real ShieldWindow: grab confirmation is derived from Tk grab_status (no window).
# --------------------------------------------------------------------------- #
class _FakeRoot:
    """Minimal Tk-root stand-in: no real window, just the grab surface we query."""

    def __init__(self, status: str | None, global_raises: bool = False) -> None:
        self._status = status
        self._global_raises = global_raises
        self.global_calls = 0
        self.local_calls = 0

    def grab_set_global(self) -> None:
        self.global_calls += 1
        if self._global_raises:
            raise RuntimeError("grab failed: another application has grab")

    def grab_set(self) -> None:
        self.local_calls += 1

    def grab_status(self) -> str | None:
        return self._status


def test_shieldwindow_reports_confirmed_global_grab():
    from facelock.shield import ShieldWindow

    s = ShieldWindow()
    s._root = _FakeRoot("global")           # grab_set_global ok + status confirms
    assert s._grab() is True
    assert s._grab_confirmed() is True


def test_shieldwindow_reports_unconfirmed_grab_as_failure():
    from facelock.shield import ShieldWindow

    s = ShieldWindow()
    # grab_set_global raises AND grab_status never confirms -> NOT grabbed even
    # though the (weaker) local grab_set() did not itself raise.
    s._root = _FakeRoot(None, global_raises=True)
    assert s._grab() is False
    assert s._grab_confirmed() is False
    assert s._root.global_calls == 1 and s._root.local_calls == 1  # tried both


def test_shieldwindow_local_grab_fallback_confirmed():
    from facelock.shield import ShieldWindow

    s = ShieldWindow()
    # Global grab denied, but the local fallback takes AND status confirms "local".
    s._root = _FakeRoot("local", global_raises=True)
    assert s._grab() is True
    assert s._grab_confirmed() is True
