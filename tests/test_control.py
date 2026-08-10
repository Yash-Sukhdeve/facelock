"""Control IPC tests (design section 10.3): GrantAuthority + ControlServer."""

from __future__ import annotations

import pytest

from facelock.control import ControlServer, GrantAuthority, send_command


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


# --------------------------------------------------------------------------- #
# GrantAuthority (nonce-bound, SI-P1).
# --------------------------------------------------------------------------- #
def test_starts_locked_with_nonce():
    ga = GrantAuthority()
    locked, epoch, nonce = ga.current()
    assert locked and epoch >= 1 and nonce


def test_valid_grant_consumes_and_unlocks():
    ga = GrantAuthority()
    _locked, epoch, nonce = ga.current()
    ok, why = ga.validate_grant(nonce, epoch)
    assert ok and why == "ok"
    locked, _e, n = ga.current()
    assert not locked and n is None


def test_replay_after_consume_fails():
    ga = GrantAuthority()
    _l, epoch, nonce = ga.current()
    assert ga.validate_grant(nonce, epoch)[0]
    ok, why = ga.validate_grant(nonce, epoch)  # replay
    assert not ok and why == "not_locked"


def test_stale_nonce_rejected_after_new_shield():
    ga = GrantAuthority()
    _l, old_epoch, old_nonce = ga.current()
    ga.raise_shield()  # mint a new nonce/epoch
    ok, why = ga.validate_grant(old_nonce, old_epoch)
    assert not ok and why == "epoch_mismatch"


def test_wrong_nonce_rejected():
    ga = GrantAuthority()
    _l, epoch, _nonce = ga.current()
    ok, why = ga.validate_grant("0" * 32, epoch)
    assert not ok and why == "stale_nonce"


def test_expired_grant_rejected():
    clk = Clock()
    ga = GrantAuthority(window_s=1.0, now_fn=clk)
    _l, epoch, nonce = ga.current()
    clk.t += 5.0  # move past the window
    ok, why = ga.validate_grant(nonce, epoch)
    assert not ok and why == "expired"


def test_refresh_challenge_makes_late_unlock_succeed():
    # Regression: returning long after the shield was raised must still unlock.
    # The window bounds the daemon's RESPONSE latency, not the age of the lock.
    clk = Clock()
    ga = GrantAuthority(window_s=4.0, now_fn=clk)
    _l, epoch, nonce = ga.current()
    clk.t += 300.0  # user walks away and comes back 5 minutes later
    # Stale path (what the guardian used to do): expired.
    assert ga.validate_grant(nonce, epoch) == (False, "expired")
    # Fixed path: the guardian refreshes the challenge when the daemon asks for
    # the nonce, so the grant submitted right after is inside the window.
    locked, epoch2, nonce2 = ga.refresh_challenge()
    assert locked and nonce2 is not None
    ok, why = ga.validate_grant(nonce2, epoch2)
    assert ok, why
    # And it consumed: replay fails.
    assert ga.validate_grant(nonce2, epoch2)[0] is False


def test_grant_expired_if_submitted_long_after_last_refresh_challenge():
    """Task-4 gap fill: the anti-lockout fix (refresh_challenge re-stamping
    issued_at) must NOT become a way to bypass expiry altogether. A grant
    fetched via refresh_challenge is still bound to window_s measured from that
    LAST refresh -- if the daemon (or a replayed/delayed submission) is slower
    than window_s AFTER the refresh, it must still be rejected "expired".

    This is not covered by test_expired_grant_rejected (which never calls
    refresh_challenge) nor by test_refresh_challenge_makes_late_unlock_succeed
    (which validates immediately after refreshing, with no further delay). A
    plausible bad "fix" for the expired-grant lockout bug is to make a
    refreshed nonce unconditionally fresh (skip the window_s check once
    refresh_challenge has been called) -- that would quietly reopen a replay/
    stale-grant window. See scratch proof in the Task-4 report: a
    GrantAuthority subclass that drops the expiry check post-refresh makes
    this exact assertion fail (ok=True, why="ok" instead of expired).
    """
    clk = Clock()
    ga = GrantAuthority(window_s=4.0, now_fn=clk)
    _l, epoch, nonce = ga.current()
    clk.t += 300.0  # owner away a long time
    locked, epoch2, nonce2 = ga.refresh_challenge()
    assert locked and nonce2 == nonce  # refresh re-stamps issued_at, not the nonce/epoch
    clk.t += 10.0  # slower than window_s (4s) AFTER the refresh itself
    ok, why = ga.validate_grant(nonce2, epoch2)
    assert not ok and why == "expired"


def test_force_locked_relocks():
    ga = GrantAuthority()
    _l, epoch, nonce = ga.current()
    ga.validate_grant(nonce, epoch)               # unlock
    assert not ga.current()[0]
    ga.force_locked()
    assert ga.current()[0]                         # locked again


# --------------------------------------------------------------------------- #
# ControlServer round-trip (owner-only Unix socket).
# --------------------------------------------------------------------------- #
@pytest.fixture
def server(tmp_path):
    sock = tmp_path / "control.sock"
    calls = []

    def handler(msg, uid):
        calls.append((msg, uid))
        if msg["cmd"] == "boom":
            raise RuntimeError("kaboom")
        if msg["cmd"] == "echo":
            return {"ok": True, "echo": msg.get("data")}
        return {"ok": False, "reason": "unknown_cmd"}

    srv = ControlServer(sock, handler=handler)
    srv.start()
    try:
        yield sock, calls
    finally:
        srv.stop()


def test_roundtrip_ok(server):
    sock, calls = server
    resp = send_command(sock, {"cmd": "echo", "data": 42})
    assert resp == {"ok": True, "echo": 42}
    assert calls[0][1] >= 0  # peer uid captured


def test_handler_exception_is_contained(server):
    sock, _ = server
    resp = send_command(sock, {"cmd": "boom"})
    assert resp["ok"] is False and resp["reason"] == "handler_error"


def test_unknown_cmd(server):
    sock, _ = server
    resp = send_command(sock, {"cmd": "nope"})
    assert resp["ok"] is False


def test_send_to_missing_socket_fails_closed(tmp_path):
    resp = send_command(tmp_path / "nonexistent.sock", {"cmd": "echo"})
    assert resp["ok"] is False and resp["reason"] == "transport"


def test_socket_permissions_owner_only(server):
    import os
    import stat

    sock, _ = server
    mode = stat.S_IMODE(os.stat(sock).st_mode)
    assert mode == 0o600
