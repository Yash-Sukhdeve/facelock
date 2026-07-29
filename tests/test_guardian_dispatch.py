"""Guardian command-dispatch tests (C8): nonce-bound unlock, disable/enable.

Exercises the guardian's authority logic without sockets, tkinter, or
subprocess: a dummy shield and a fake lock controller are injected.
"""

from __future__ import annotations

import os

from facelock.config import load_config
from facelock.guardian import Guardian
from facelock.lock_backend import LockOutcome


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


def make_guardian():
    cfg = load_config(raw={})
    return Guardian(
        cfg,
        lock_controller=FakeController(),
        shield=DummyShield(),
        install_signals=False,
    )


UID = os.getuid()


def test_lock_returns_nonce_and_epoch():
    g = make_guardian()
    resp = g.dispatch({"cmd": "lock", "reason": "away"}, UID)
    assert resp["ok"] and resp["state"] == "LOCKED"
    assert isinstance(resp["grant_nonce"], str) and isinstance(resp["lock_epoch"], int)
    assert g.grant.current()[0] is True  # locked


def test_valid_unlock_grant_dismisses():
    g = make_guardian()
    lock = g.dispatch({"cmd": "lock", "reason": "away"}, UID)
    resp = g.dispatch({
        "cmd": "unlock_grant", "grant_nonce": lock["grant_nonce"],
        "lock_epoch": lock["lock_epoch"], "score": 0.9, "tau": 0.5, "live": False,
    }, UID)
    assert resp["ok"]
    assert g.grant.current()[0] is False  # unlocked


def test_stale_grant_rejected():
    g = make_guardian()
    lock = g.dispatch({"cmd": "lock", "reason": "away"}, UID)
    g.dispatch({"cmd": "lock", "reason": "away"}, UID)  # new epoch/nonce
    resp = g.dispatch({
        "cmd": "unlock_grant", "grant_nonce": lock["grant_nonce"],
        "lock_epoch": lock["lock_epoch"], "score": 0.9, "tau": 0.5, "live": False,
    }, UID)
    assert not resp["ok"] and resp["reason"] == "epoch_mismatch"


def test_malformed_grant():
    g = make_guardian()
    g.dispatch({"cmd": "lock", "reason": "away"}, UID)
    resp = g.dispatch({"cmd": "unlock_grant"}, UID)
    assert not resp["ok"] and resp["reason"] == "malformed_grant"


def test_disable_blocks_unlock_and_hides_nonce():
    g = make_guardian()
    g.dispatch({"cmd": "lock", "reason": "away"}, UID)
    dis = g.dispatch({"cmd": "disable"}, UID)
    assert dis["ok"] and dis["face_unlock"] is False
    assert g._escalate  # disable escalated to OS lock via fake controller
    assert g.lock_ctl.engage_calls >= 1
    nonce_info = g.dispatch({"cmd": "get_grant_nonce"}, UID)
    assert nonce_info["grant_nonce"] is None and nonce_info["face_unlock"] is False
    # A grant while disabled is refused even with a syntactically valid shape.
    resp = g.dispatch({
        "cmd": "unlock_grant", "grant_nonce": "x" * 32, "lock_epoch": 1,
        "score": 0.9, "tau": 0.5, "live": False,
    }, UID)
    assert not resp["ok"] and resp["reason"] == "face_unlock_disabled"


def test_enable_restores():
    g = make_guardian()
    g.dispatch({"cmd": "disable"}, UID)
    resp = g.dispatch({"cmd": "enable"}, UID)
    assert resp["ok"] and resp["face_unlock"] is True


def test_get_grant_nonce_exposes_when_enabled_and_locked():
    g = make_guardian()
    g.dispatch({"cmd": "lock", "reason": "away"}, UID)
    info = g.dispatch({"cmd": "get_grant_nonce"}, UID)
    assert info["locked"] and info["grant_nonce"] and info["face_unlock"]


def test_heartbeat_returns_face_unlock():
    g = make_guardian()
    resp = g.dispatch({"cmd": "heartbeat", "seq": 1, "state": "LOCKED_ABSENT",
                       "health": {"healthy": True}}, UID)
    assert resp["ok"] and resp["face_unlock"] is True


def test_status_snapshot():
    g = make_guardian()
    g.dispatch({"cmd": "lock", "reason": "away"}, UID)
    st = g.dispatch({"cmd": "status"}, UID)
    for key in ("locked", "lock_epoch", "face_unlock", "daemon_state"):
        assert key in st


def test_unknown_command():
    g = make_guardian()
    resp = g.dispatch({"cmd": "frobnicate"}, UID)
    assert not resp["ok"] and resp["reason"] == "unknown_cmd"
