"""LockController tests (C9, SI-P5/FM-16): verify-engaged + fallback."""

from __future__ import annotations

from facelock.lock_backend import LockController, select_backends


class FakeBackend:
    def __init__(self, name, lock_ok=True, locked=True):
        self.name = name
        self._lock_ok = lock_ok
        self._locked = locked
        self.lock_calls = 0

    def available(self):
        return True

    def lock(self):
        self.lock_calls += 1
        return self._lock_ok

    def is_locked(self):
        return self._locked


def ctl(backends):
    return LockController(backends, verify_engaged_ms=60)


def test_engage_confirmed():
    b = FakeBackend("loginctl", lock_ok=True, locked=True)
    out = ctl([b]).engage()
    assert out.engaged and out.confirmed and out.backend == "loginctl"
    assert b.lock_calls == 1


def test_fallback_to_next_backend():
    b1 = FakeBackend("gnome_dbus", lock_ok=True, locked=False)   # cannot confirm
    b2 = FakeBackend("loginctl", lock_ok=True, locked=True)      # confirms
    out = ctl([b1, b2]).engage()
    assert out.engaged and out.confirmed and out.backend == "loginctl"
    assert b1.lock_calls == 1 and b2.lock_calls == 1


def test_none_confirmed_fails_closed():
    b1 = FakeBackend("a", lock_ok=True, locked=False)
    b2 = FakeBackend("b", lock_ok=True, locked=False)
    out = ctl([b1, b2]).engage()
    assert not out.engaged and not out.confirmed


def test_last_resort_unverifiable_reports_unconfirmed():
    b = FakeBackend("xdg", lock_ok=True, locked=None)  # unknown lock state
    out = ctl([b]).engage()
    assert out.engaged and not out.confirmed and out.backend == "xdg"


def test_lock_rejected_moves_on():
    b1 = FakeBackend("a", lock_ok=False, locked=False)
    b2 = FakeBackend("b", lock_ok=True, locked=True)
    out = ctl([b1, b2]).engage()
    assert out.backend == "b" and out.confirmed


def test_no_backends():
    out = ctl([]).engage()
    assert not out.engaged and out.backend is None


def test_is_any_locked():
    assert ctl([FakeBackend("a", locked=True)]).is_any_locked() is True
    assert ctl([FakeBackend("a", locked=False)]).is_any_locked() is False
    assert ctl([FakeBackend("a", locked=None)]).is_any_locked() is None


def test_select_backends_auto_returns_list():
    # Availability depends on the host; the call must never raise and returns a list.
    assert isinstance(select_backends("auto"), list)
    assert isinstance(select_backends("loginctl"), list)
