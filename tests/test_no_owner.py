"""No-owner passive mode + pause/resume auto-timer (REQ-F no-owner safety).

When no face is enrolled, face-unlock is INERT: it must never raise a shield or
escalate the OS lock on an AUTOMATIC reason (away/stranger/heartbeat_miss) --
otherwise it would trap the user behind a lock no face can clear, forcing the OS
password on a machine the user never asked to protect. Explicit user actions
(panic/disable) still lock. Enrolling flips the guardian active live, via the
heartbeat's `template` health flag, with no restart.

These tests inject a dummy shield + fake controller and set `_owner_present`
explicitly, so they are hermetic (independent of any template on disk).
"""

from __future__ import annotations

import os
import time

from facelock.config import load_config
from facelock.guardian import Guardian
from facelock.lock_backend import LockOutcome

UID = os.getuid()


class DummyShield:
    def __init__(self):
        self.is_up = False

    def raise_shield(self, status="Locked"):
        self.is_up = True
        return True

    def set_status(self, s):
        pass

    def dismiss(self):
        self.is_up = False

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


def make_guardian(owner: bool):
    g = Guardian(
        load_config(raw={}),
        lock_controller=FakeController(),
        shield=DummyShield(),
        install_signals=False,
    )
    g._owner_present = owner  # hermetic: don't depend on a template on disk
    return g


# --- A. automatic locks are no-ops when nobody is enrolled ------------------ #
def test_no_owner_away_lock_is_passive_noop():
    g = make_guardian(owner=False)
    resp = g.dispatch({"cmd": "lock", "reason": "away"}, UID)
    assert resp["ok"] is True
    assert resp.get("state") == "PASSIVE"
    assert resp.get("no_owner") is True
    assert "grant_nonce" not in resp          # no shield armed
    assert g.lock_ctl.engage_calls == 0       # no OS escalation


def test_no_owner_stranger_lock_is_passive_noop():
    g = make_guardian(owner=False)
    resp = g.dispatch({"cmd": "lock", "reason": "stranger"}, UID)
    assert resp.get("state") == "PASSIVE" and resp.get("no_owner") is True
    assert g.lock_ctl.engage_calls == 0


# --- B. explicit user actions still lock even with no owner ----------------- #
def test_no_owner_panic_still_locks():
    g = make_guardian(owner=False)
    resp = g.dispatch({"cmd": "lock", "reason": "panic"}, UID)
    assert resp.get("state") == "LOCKED"
    assert "grant_nonce" in resp              # shield armed
    assert resp.get("escalated") is True      # panic escalates the OS lock
    assert g.lock_ctl.engage_calls >= 1


# --- C. the enrolled-owner path is unaffected ------------------------------- #
def test_owner_present_away_locks_normally():
    g = make_guardian(owner=True)
    resp = g.dispatch({"cmd": "lock", "reason": "away"}, UID)
    assert resp.get("state") == "LOCKED"
    assert "grant_nonce" in resp


# --- D. enrollment activates the guardian live via the heartbeat ------------ #
def test_heartbeat_template_true_activates_owner():
    g = make_guardian(owner=False)
    assert g._owner_present is False
    g.dispatch({"cmd": "heartbeat", "seq": 1, "state": "UNLOCKED_PRESENT",
                "health": {"template": True}}, UID)
    assert g._owner_present is True           # activated without a restart


def test_heartbeat_template_false_stays_passive():
    g = make_guardian(owner=False)
    g.dispatch({"cmd": "heartbeat", "seq": 1, "state": "UNLOCKED_PRESENT",
                "health": {"template": False}}, UID)
    assert g._owner_present is False


def test_status_reports_no_owner_flag():
    g = make_guardian(owner=False)
    st = g.dispatch({"cmd": "status"}, UID)
    assert st["no_owner"] is True
    g2 = make_guardian(owner=True)
    assert g2.dispatch({"cmd": "status"}, UID)["no_owner"] is False


# --- E. watchdog does not trap the user when there is no owner --------------- #
def test_watchdog_no_owner_does_not_escalate():
    g = make_guardian(owner=False)
    # Force the heartbeat-miss condition (past startup grace, stale heartbeat).
    g._started_at = time.monotonic() - 10_000
    g._last_heartbeat = time.monotonic() - 10_000
    g._check_watchdog()
    assert g.lock_ctl.engage_calls == 0       # never escalated the OS lock


def test_watchdog_with_owner_escalates():
    g = make_guardian(owner=True)
    g._started_at = time.monotonic() - 10_000
    g._last_heartbeat = time.monotonic() - 10_000
    g._check_watchdog()
    assert g._watchdog_tripped is True
    assert g.lock_ctl.engage_calls >= 1       # owner present -> real OS lock


# --- F. pause --minutes N arms an auto-resume timer ------------------------- #
def test_pause_with_minutes_sets_resume_timer():
    g = make_guardian(owner=True)
    p = g.dispatch({"cmd": "pause_perception", "minutes": 5}, UID)
    assert p["paused"] is True
    assert p["auto_resume_min"] == 5
    assert g._resume_at is not None
    st = g.dispatch({"cmd": "status"}, UID)
    assert isinstance(st["pause_resume_in_s"], (int, float))
    assert st["pause_resume_in_s"] > 0


def test_pause_without_minutes_has_no_timer():
    g = make_guardian(owner=True)
    p = g.dispatch({"cmd": "pause_perception"}, UID)
    assert p["paused"] is True
    assert g._resume_at is None
    st = g.dispatch({"cmd": "status"}, UID)
    assert st["pause_resume_in_s"] is None


def test_resume_clears_timer():
    g = make_guardian(owner=True)
    g.dispatch({"cmd": "pause_perception", "minutes": 5}, UID)
    assert g._resume_at is not None
    r = g.dispatch({"cmd": "resume_perception"}, UID)
    assert r["paused"] is False
    assert g._resume_at is None
