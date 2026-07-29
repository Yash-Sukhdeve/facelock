"""Guardian re-lock reliability + prototype stranger policy + monitor power.

These are guardian-level (C8/C11) tests with the shield, lock backend, and
display all faked so NO window, subprocess, socket, or real screen is touched.
They exercise the three behaviours the pilot requires:

  * a stranger raises the face-dismissable SHIELD, NOT the OS password lock;
  * the monitor is turned OFF on a shield-lock and back ON on face-unlock;
  * re-locking works across many lock->unlock->lock cycles (re-arm + re-mint).
"""

from __future__ import annotations

import os

from facelock.config import load_config
from facelock.guardian import Guardian
from facelock.lock_backend import LockOutcome

UID = os.getuid()


class CountingShield:
    def __init__(self):
        self.is_up = False
        self.raises = 0
        self.dismisses = 0
        self.last_status = None
        self.phase = None

    def raise_shield(self, status="Locked"):
        self.is_up = True
        self.raises += 1
        self.last_status = status
        self.phase = "locked"
        return True

    def set_status(self, s):
        self.last_status = s
        self.phase = "locked"

    def set_recognizing(self, progress=0.0, votes_k=0, votes_need=0):
        self.phase = "recognizing"
        self.progress = progress

    def set_denied(self, text="Unauthorized user"):
        self.phase = "denied"
        self.last_status = text

    def set_welcome(self, name):
        self.phase = "welcome"
        self.last_status = f"Welcome back, {name}"

    def dismiss(self):
        self.is_up = False
        self.dismisses += 1

    def pump(self):
        pass


class FakeController:
    def __init__(self):
        self.engage_calls = 0

    def engage(self):
        self.engage_calls += 1
        return LockOutcome(True, True, "fake", "confirmed")

    def is_any_locked(self):
        return None


class FakeDisplay:
    def __init__(self, off_raises=False):
        self.off_calls = 0
        self.on_calls = 0
        self._off_raises = off_raises

    def screen_off(self):
        self.off_calls += 1
        if self._off_raises:
            raise RuntimeError("xset exploded")
        return True

    def screen_on(self):
        self.on_calls += 1
        return True

    def set_config_enabled(self, enabled):
        pass


def make_guardian(display=None, welcome_hold=0.0):
    cfg = load_config(raw={})  # SCHEMA defaults (prototype: stranger NOT escalated)
    g = Guardian(
        cfg,
        lock_controller=FakeController(),
        shield=CountingShield(),
        display=display or FakeDisplay(),
        install_signals=False,
    )
    # Lock-focused tests want the pre-splash immediate-dismiss behaviour; the
    # welcome-splash timing is exercised separately in test_feedback.py.
    g._welcome_hold_s = welcome_hold
    return g


# --- B. prototype stranger policy: shield, not OS password lock ------------- #
def test_default_config_does_not_escalate_stranger_or_shutdown():
    cfg = load_config(raw={})
    # Neither a stranger NOR a normal stop/restart may throw the OS password lock
    # (both would leave a page a face cannot clear).
    assert "stranger" not in cfg.lock.escalate_os_lock_on
    assert "shutdown" not in cfg.lock.escalate_os_lock_on
    # But the genuine fail-closed reasons are still escalated.
    for reason in ("panic", "heartbeat_miss", "suspend"):
        assert reason in cfg.lock.escalate_os_lock_on


def test_stop_does_not_engage_os_lock_by_default():
    # A normal guardian stop/restart must NOT run the OS lock backend (that would
    # force the GNOME login page). The shield-based lock is enough for the prototype.
    g = make_guardian()
    g._shutdown()
    assert g.lock_ctl.engage_calls == 0


def test_stop_escalates_when_shutdown_opted_in():
    g = make_guardian()
    g._escalate = set(g._escalate) | {"shutdown"}   # hardening opt-in
    g._shutdown()
    assert g.lock_ctl.engage_calls == 1


def test_stranger_lock_uses_shield_not_os_backend():
    g = make_guardian()
    resp = g.dispatch({"cmd": "lock", "reason": "stranger"}, UID)
    assert resp["ok"] and resp["state"] == "LOCKED"
    assert resp["escalated"] is False
    assert g.lock_ctl.engage_calls == 0  # OS password path NOT touched
    g._drain_shield_queue()
    assert g.shield.is_up is True         # shield raised
    assert g.display.off_calls == 1       # monitor blanked


def test_away_lock_uses_shield_not_os_backend():
    g = make_guardian()
    resp = g.dispatch({"cmd": "lock", "reason": "away"}, UID)
    assert resp["escalated"] is False
    assert g.lock_ctl.engage_calls == 0
    g._drain_shield_queue()
    assert g.display.off_calls == 1


def test_heartbeat_miss_still_escalates_and_keeps_screen_on():
    # A hard fail-closed reason MUST still hit the OS lock and keep the monitor on.
    g = make_guardian()
    resp = g.dispatch({"cmd": "lock", "reason": "heartbeat_miss"}, UID)
    assert resp["escalated"] is True
    assert g.lock_ctl.engage_calls == 1
    g._drain_shield_queue()
    assert g.display.on_calls >= 1        # password prompt visible
    assert g.display.off_calls == 0


# --- A. monitor off on lock, on on unlock ---------------------------------- #
def test_unlock_turns_monitor_back_on():
    g = make_guardian()
    lock = g.dispatch({"cmd": "lock", "reason": "away"}, UID)
    g._drain_shield_queue()
    assert g.display.off_calls == 1 and g.display.on_calls == 0
    ok = g.dispatch({
        "cmd": "unlock_grant", "grant_nonce": lock["grant_nonce"],
        "lock_epoch": lock["lock_epoch"], "score": 0.9, "tau": 0.5, "live": False,
    }, UID)
    assert ok["ok"]
    g._drain_shield_queue()
    assert g.display.on_calls == 1        # monitor woken on unlock
    assert g.shield.is_up is False        # shield dismissed


def test_face_present_while_locked_wakes_monitor_for_feedback():
    # While locked-and-away the monitor is off; when a face appears the guardian
    # must wake it so the "Recognizing"/"Unauthorized" graphics are visible.
    g = make_guardian()
    g.dispatch({"cmd": "lock", "reason": "away"}, UID)
    g._drain_shield_queue()
    off_before = g.display.off_calls
    g.dispatch({"cmd": "shield_status", "phase": "recognizing"}, UID)
    g._drain_shield_queue()
    assert g.display.on_calls >= 1 and g._screen_off_active is False
    # When the face leaves (back to locked) the monitor blanks again.
    g.dispatch({"cmd": "shield_status", "phase": "locked", "reason": "away"}, UID)
    g._drain_shield_queue()
    assert g.display.off_calls > off_before and g._screen_off_active is True


def test_denied_phase_also_wakes_monitor():
    g = make_guardian()
    g.dispatch({"cmd": "lock", "reason": "stranger"}, UID)
    g._drain_shield_queue()
    g.dispatch({"cmd": "shield_status", "phase": "denied"}, UID)
    g._drain_shield_queue()
    assert g.display.on_calls >= 1
    assert g.shield.phase == "denied"


def test_screen_off_failure_never_crashes_guardian():
    g = make_guardian(display=FakeDisplay(off_raises=True))
    g.dispatch({"cmd": "lock", "reason": "away"}, UID)
    # Draining must swallow the display error and not arm the re-assert cadence.
    g._drain_shield_queue()
    assert g._screen_off_active is False
    assert g.shield.is_up is True         # lock still succeeded despite display error


def test_reassert_cadence_only_active_while_screen_off():
    g = make_guardian()
    g.dispatch({"cmd": "lock", "reason": "away"}, UID)
    g._drain_shield_queue()
    assert g._screen_off_active is True    # cadence armed while locked
    lock_epoch = g.grant.current()[1]
    # Unlock -> cadence disarmed.
    nonce = g.grant.current()[2]
    g.dispatch({"cmd": "unlock_grant", "grant_nonce": nonce,
                "lock_epoch": lock_epoch, "score": 0.9, "tau": 0.5, "live": False}, UID)
    g._drain_shield_queue()
    assert g._screen_off_active is False


# --- C. reliable re-lock across cycles (re-arm + re-mint) ------------------- #
def test_guardian_relocks_across_many_cycles():
    g = make_guardian()
    seen_nonces = set()
    for cycle in range(4):
        lock = g.dispatch({"cmd": "lock", "reason": "away"}, UID)
        g._drain_shield_queue()
        assert g.shield.is_up is True, f"cycle {cycle}: shield did not raise"
        assert g.grant.current()[0] is True, f"cycle {cycle}: not locked"
        # Every lock mints a FRESH nonce+epoch (replay-safe re-arm).
        assert lock["grant_nonce"] not in seen_nonces, f"cycle {cycle}: nonce reused"
        seen_nonces.add(lock["grant_nonce"])
        # Owner returns and face-unlocks with the current grant.
        resp = g.dispatch({
            "cmd": "unlock_grant", "grant_nonce": lock["grant_nonce"],
            "lock_epoch": lock["lock_epoch"], "score": 0.9, "tau": 0.5, "live": False,
        }, UID)
        assert resp["ok"], f"cycle {cycle}: unlock failed"
        g._drain_shield_queue()
        assert g.shield.is_up is False, f"cycle {cycle}: shield did not dismiss"
    # 4 raises + 4 dismisses, and the monitor toggled off/on each cycle.
    assert g.shield.raises == 4 and g.shield.dismisses == 4
    assert g.display.off_calls == 4 and g.display.on_calls == 4


def test_relock_after_unlock_with_stale_grant_stays_locked():
    """After a re-lock, the PREVIOUS cycle's grant must not unlock (re-mint)."""
    g = make_guardian()
    first = g.dispatch({"cmd": "lock", "reason": "away"}, UID)
    g.dispatch({
        "cmd": "unlock_grant", "grant_nonce": first["grant_nonce"],
        "lock_epoch": first["lock_epoch"], "score": 0.9, "tau": 0.5, "live": False,
    }, UID)
    # Re-lock (cycle 2) mints a new epoch/nonce.
    g.dispatch({"cmd": "lock", "reason": "away"}, UID)
    # Replaying cycle-1's grant must be rejected; session stays locked.
    replay = g.dispatch({
        "cmd": "unlock_grant", "grant_nonce": first["grant_nonce"],
        "lock_epoch": first["lock_epoch"], "score": 0.9, "tau": 0.5, "live": False,
    }, UID)
    assert not replay["ok"]
    assert g.grant.current()[0] is True   # still locked (fail-closed)
