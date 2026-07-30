"""Regression tests for the hardening pass (audit fixes).

Each test pins a specific fixed weakness so it cannot silently regress:
  * config refuses a below-floor recognition.tau override (REQ-NF-22),
  * the guardian refuses a self-inconsistent grant (score < tau),
  * the guardian holds the shield when the OS lock cannot be confirmed on Escape,
  * a failed shield raise escalates to the OS lock,
  * ensure_dir tightens EVERY created directory and rejects symlinked/foreign dirs,
  * secure_write_bytes refuses to follow a pre-planted symlink,
  * lock backends resolve binaries to absolute trusted paths (never $PATH).
"""

from __future__ import annotations

import os
import stat

import pytest

from facelock import paths as _paths
from facelock.config import load_config
from facelock.errors import ConfigError


# --------------------------------------------------------------------------- #
# Config: tau floor cannot be bypassed by an override.
# --------------------------------------------------------------------------- #
def test_tau_override_below_floor_is_refused():
    with pytest.raises(ConfigError):
        load_config(raw={"recognition": {"tau": 0.05, "tau_floor": 0.363}})


def test_tau_override_at_or_above_floor_is_allowed():
    cfg = load_config(raw={"recognition": {"tau": 0.40, "tau_floor": 0.363}})
    assert cfg.recognition.tau == 0.40


def test_tau_zero_means_calibrated_and_is_allowed():
    cfg = load_config(raw={"recognition": {"tau": 0.0, "tau_floor": 0.363}})
    assert cfg.recognition.tau == 0.0


# --------------------------------------------------------------------------- #
# Guardian: independent grant sanity + fail-closed Escape / shield-raise.
# --------------------------------------------------------------------------- #
class _DummyShield:
    def __init__(self, raise_ok=True):
        self.is_up = False
        self._raise_ok = raise_ok
        self.raises = 0

    def raise_shield(self, status="Locked"):
        self.raises += 1
        self.is_up = self._raise_ok
        return self._raise_ok

    def set_status(self, s): ...
    def dismiss(self): self.is_up = False
    def pump(self): ...


class _FakeController:
    def __init__(self, engaged=True):
        self.engage_calls = 0
        self._engaged = engaged

    def engage(self):
        self.engage_calls += 1
        from facelock.lock_backend import LockOutcome
        return LockOutcome(self._engaged, self._engaged, "fake",
                           "confirmed" if self._engaged else "not engaged")

    def is_any_locked(self):
        return None


def _guardian(shield, controller):
    from facelock.guardian import Guardian
    return Guardian(load_config(raw={}), lock_controller=controller,
                    shield=shield, install_signals=False)


UID = os.getuid()


def test_grant_with_score_below_tau_is_refused():
    g = _guardian(_DummyShield(), _FakeController())
    lock = g.dispatch({"cmd": "lock", "reason": "away"}, UID)
    resp = g.dispatch({
        "cmd": "unlock_grant", "grant_nonce": lock["grant_nonce"],
        "lock_epoch": lock["lock_epoch"], "score": 0.10, "tau": 0.40, "live": False,
    }, UID)
    assert not resp["ok"] and resp["reason"] == "score_below_tau"
    assert g.grant.current()[0] is True  # still locked; nonce not consumed


def test_grant_with_score_above_tau_still_unlocks():
    g = _guardian(_DummyShield(), _FakeController())
    lock = g.dispatch({"cmd": "lock", "reason": "away"}, UID)
    resp = g.dispatch({
        "cmd": "unlock_grant", "grant_nonce": lock["grant_nonce"],
        "lock_epoch": lock["lock_epoch"], "score": 0.55, "tau": 0.40, "live": True,
    }, UID)
    assert resp["ok"] and g.grant.current()[0] is False


def test_escape_holds_shield_when_os_lock_not_confirmed():
    shield = _DummyShield()
    g = _guardian(shield, _FakeController(engaged=False))
    g._on_password_escape()
    g._drain_shield_queue()
    # OS lock could not be confirmed -> shield must remain up (fail-closed).
    assert shield.is_up is True


def test_escape_dismisses_shield_when_os_lock_confirmed():
    shield = _DummyShield()
    shield.is_up = True
    g = _guardian(shield, _FakeController(engaged=True))
    g._on_password_escape()
    g._drain_shield_queue()
    assert shield.is_up is False


def test_failed_shield_raise_escalates_to_os_lock():
    shield = _DummyShield(raise_ok=False)
    ctl = _FakeController()
    g = _guardian(shield, ctl)
    g._enqueue_shield("raise", "Locked - away")
    g._drain_shield_queue()
    assert ctl.engage_calls >= 1  # raise failed -> OS lock escalation


# --------------------------------------------------------------------------- #
# Filesystem hardening.
# --------------------------------------------------------------------------- #
def test_ensure_dir_tightens_every_created_component(tmp_path):
    root = tmp_path / "a" / "b" / "c"
    _paths.ensure_dir(root, 0o700)
    for p in (tmp_path / "a", tmp_path / "a" / "b", root):
        assert stat.S_IMODE(os.stat(p).st_mode) == 0o700, p


def test_ensure_dir_rejects_symlinked_directory(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(OSError):
        _paths.ensure_dir(link)


def test_secure_write_refuses_symlinked_target(tmp_path):
    _paths.ensure_dir(tmp_path, 0o700)
    victim = tmp_path / "victim"
    target = tmp_path / "secret"
    # Pre-plant a symlink at the *final* path; O_NOFOLLOW on the temp plus the
    # atomic rename means the secret bytes never land on the symlink's target.
    target_link = tmp_path / "secret"
    target_link.symlink_to(victim)
    _paths.secure_write_bytes(target, b"top-secret")
    # The real file now holds the data and is a regular 0600 file, not a link.
    assert not os.path.islink(target)
    assert target.read_bytes() == b"top-secret"
    assert not victim.exists()  # the symlink target was never written through


# --------------------------------------------------------------------------- #
# Lock backends resolve to absolute trusted paths only.
# --------------------------------------------------------------------------- #
def test_run_refuses_non_absolute_command():
    from facelock.lock_backend import _run
    rc, _out, err = _run(["loginctl", "lock-session"])
    assert rc == 255 and "non-absolute" in err


def test_resolve_trusted_ignores_path(monkeypatch, tmp_path):
    from facelock import lock_backend
    # A trojan earlier on $PATH must NOT be resolved.
    trojan = tmp_path / "loginctl"
    trojan.write_text("#!/bin/sh\n")
    trojan.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")
    resolved = lock_backend.resolve_trusted("loginctl")
    assert resolved is None or resolved.startswith(("/usr/", "/bin", "/sbin"))
    assert resolved != str(trojan)
