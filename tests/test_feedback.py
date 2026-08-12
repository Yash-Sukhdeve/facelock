"""On-shield feedback tests: 'Recognizing...', 'Unauthorized user', 'Welcome back'.

Covers the four surfaces the user asked for:
  * shield status phases + dot animation (headless-safe: no window opened);
  * guardian ``shield_status`` command routing (recognizing/denied/locked);
  * the non-blocking post-unlock welcome splash + delayed dismiss (both the
    on-shield splash AND the desktop notification fire);
  * the daemon's (state, observation) -> phase mapping and debounced push;
  * the emitter message shape + short timeout.
No real window, socket, subprocess, or screen is touched.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import facelock.control as control_mod
from facelock.config import load_config
from facelock.daemon import PerceptionDaemon
from facelock.fsm import Observation, State
from facelock.guardian import Guardian
from facelock.lock_backend import LockOutcome
from facelock.matcher import Matcher, MatchResult
from facelock import shield as shield_mod
from facelock.shield import ShieldWindow


def _unit(seed: int, dim: int = 128) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)

UID = os.getuid()


# --------------------------------------------------------------------------- #
# Shield status/animation (headless: force no display so nothing is created).
# --------------------------------------------------------------------------- #
@pytest.fixture
def _no_display(monkeypatch):
    monkeypatch.setattr(shield_mod, "has_display", lambda: False)


def test_dots_animation_cycles():
    d = ShieldWindow._dots
    assert [d(t) for t in (0, 3, 6, 9, 12)] == ["", ".", "..", "...", ""]


def test_status_phase_setters_update_state(_no_display):
    s = ShieldWindow(owner_name="Yash")
    s.set_recognizing(0.5, 2, 3)
    assert s._anim_kind == "recognizing" and s._phase == "recognizing"
    assert s._progress == 0.5 and s._votes == (2, 3)
    s.set_denied()
    assert s._anim_kind == "denied" and s._base_text == "Unauthorized user"
    s.set_welcome("Yash")
    assert s._anim_kind is None and s._base_text == "AUTHORIZED - Welcome back, Yash"
    s.set_status("Locked - you stepped away")
    assert s._anim_kind is None and s._base_text == "Locked - you stepped away"


def test_status_setters_are_safe_without_window(_no_display):
    s = ShieldWindow()
    # No root/label exist -> must not raise.
    s.set_recognizing()
    s.set_denied("Unauthorized user")
    s.set_welcome("Yash")
    s.pump()  # animation step is a no-op when down
    assert s.is_up is False


# --------------------------------------------------------------------------- #
# Guardian shield_status routing + welcome splash timing.
# --------------------------------------------------------------------------- #
class FakeShield:
    def __init__(self):
        self.is_up = False
        self.phase = None
        self.text = None

    def raise_shield(self, status="Locked"):
        self.is_up = True
        self.phase = "locked"
        self.text = status
        return True

    def set_status(self, s):
        self.phase = "locked"
        self.text = s

    def set_recognizing(self, progress=0.0, votes_k=0, votes_need=0):
        self.phase = "recognizing"
        self.progress = progress
        self.votes = (votes_k, votes_need)

    def set_denied(self, text="Unauthorized user"):
        self.phase = "denied"
        self.text = text

    def set_welcome(self, name):
        self.phase = "welcome"
        self.text = f"Welcome back, {name}"

    def dismiss(self):
        self.is_up = False
        self.phase = None

    def pump(self):
        pass


class FakeController:
    def engage(self):
        return LockOutcome(True, True, "fake", "confirmed")

    def is_any_locked(self):
        return None


class FakeDisplay:
    def screen_off(self):
        return True

    def screen_on(self):
        return True

    def set_config_enabled(self, e):
        pass


class FakeGreeter:
    def __init__(self):
        self.enabled = True
        self.shown = []

    def show(self, name, ttl_s=3):
        self.shown.append(name)


def make_guardian(welcome_hold=0.6):
    g = Guardian(
        load_config(raw={}),
        lock_controller=FakeController(),
        shield=FakeShield(),
        greeter=FakeGreeter(),
        display=FakeDisplay(),
        install_signals=False,
    )
    g._welcome_hold_s = welcome_hold
    g._owner_present = True  # enrolled-owner scenario (see no-owner passive gate)
    return g


def _lock(g, reason="away"):
    return g.dispatch({"cmd": "lock", "reason": reason}, UID)


def test_shield_status_recognizing_routes_to_shield():
    g = make_guardian()
    _lock(g)
    g.dispatch({"cmd": "shield_status", "phase": "recognizing"}, UID)
    g._drain_shield_queue()
    assert g.shield.phase == "recognizing"


def test_shield_status_denied_shows_unauthorized():
    g = make_guardian()
    _lock(g)
    resp = g.dispatch({"cmd": "shield_status", "phase": "denied"}, UID)
    assert resp["ok"]
    g._drain_shield_queue()
    assert g.shield.phase == "denied"
    assert g.shield.text == "Unauthorized user"


def test_shield_status_locked_restores_text():
    g = make_guardian()
    _lock(g)
    g.dispatch({"cmd": "shield_status", "phase": "recognizing"}, UID)
    g.dispatch({"cmd": "shield_status", "phase": "locked", "reason": "stranger"}, UID)
    g._drain_shield_queue()
    assert g.shield.phase == "locked"
    assert g.shield.text == "Locked - unrecognized face"


def test_shield_status_ignored_when_not_locked():
    g = make_guardian()
    lock = _lock(g)
    g.dispatch({
        "cmd": "unlock_grant", "grant_nonce": lock["grant_nonce"],
        "lock_epoch": lock["lock_epoch"], "score": 0.9, "tau": 0.5, "live": False,
    }, UID)
    resp = g.dispatch({"cmd": "shield_status", "phase": "denied"}, UID)
    assert resp["ok"] and resp.get("ignored") == "not_locked"


def test_shield_status_bad_phase_rejected():
    g = make_guardian()
    _lock(g)
    resp = g.dispatch({"cmd": "shield_status", "phase": "party"}, UID)
    assert not resp["ok"] and resp["reason"] == "bad_phase"


def test_checking_progress_reaches_the_shield():
    g = make_guardian()
    _lock(g)
    g.dispatch({"cmd": "shield_status", "phase": "recognizing",
                "progress": 0.66, "votes_k": 2, "votes_need": 3}, UID)
    g._drain_shield_queue()
    assert g.shield.phase == "recognizing"
    assert g.shield.progress == pytest.approx(0.66)


def test_denied_verdict_holds_over_a_recheck():
    g = make_guardian()
    _lock(g)
    # UNAUTHORIZED verdict arrives.
    g.dispatch({"cmd": "shield_status", "phase": "denied"}, UID)
    g._drain_shield_queue()
    assert g.shield.phase == "denied"
    # An immediate re-check must NOT overwrite the verdict (held ~1.2 s).
    resp = g.dispatch({"cmd": "shield_status", "phase": "recognizing",
                       "progress": 0.3}, UID)
    g._drain_shield_queue()
    assert resp.get("held") == "denied"
    assert g.shield.phase == "denied"
    # After the hold window, checking can paint again.
    g._denied_until = 0.0
    g.dispatch({"cmd": "shield_status", "phase": "recognizing", "progress": 0.3}, UID)
    g._drain_shield_queue()
    assert g.shield.phase == "recognizing"


def test_unlock_shows_welcome_splash_then_dismisses_after_hold():
    g = make_guardian(welcome_hold=0.6)
    lock = _lock(g)
    g._drain_shield_queue()
    g.dispatch({
        "cmd": "unlock_grant", "grant_nonce": lock["grant_nonce"],
        "lock_epoch": lock["lock_epoch"], "score": 0.9, "tau": 0.5, "live": False,
    }, UID)
    g._drain_shield_queue()
    # Both surfaces: on-shield welcome splash AND the desktop notification.
    assert g.shield.phase == "welcome"
    assert g.shield.text == "Welcome back, User"
    assert g.greeter.shown == ["User"]
    assert g.shield.is_up is True                  # still up during the hold
    assert g._welcome_dismiss_at is not None
    # Before the hold elapses -> still up.
    g._maybe_finish_welcome(g._welcome_dismiss_at - 0.01)
    g._drain_shield_queue()
    assert g.shield.is_up is True
    # After the hold elapses -> dismissed (non-blocking, main-loop scheduled).
    g._maybe_finish_welcome(g._welcome_dismiss_at + 0.01)
    g._drain_shield_queue()
    assert g.shield.is_up is False


def test_unlock_with_zero_hold_dismisses_immediately():
    g = make_guardian(welcome_hold=0.0)
    lock = _lock(g)
    g._drain_shield_queue()
    g.dispatch({
        "cmd": "unlock_grant", "grant_nonce": lock["grant_nonce"],
        "lock_epoch": lock["lock_epoch"], "score": 0.9, "tau": 0.5, "live": False,
    }, UID)
    g._drain_shield_queue()
    assert g.shield.is_up is False                 # instant dismiss
    assert g.greeter.shown == ["User"]             # notification still fires
    assert g._welcome_dismiss_at is None


# --------------------------------------------------------------------------- #
# Daemon phase mapping + debounced push.
# --------------------------------------------------------------------------- #
class RecordingEmitter:
    def __init__(self):
        self.calls = []
        self.kw = []

    def shield_status(self, phase, reason, **kw):
        self.calls.append((phase, reason))
        self.kw.append(kw)
        return {"ok": True}


def make_daemon():
    return PerceptionDaemon(load_config(raw={}), emitter=RecordingEmitter(),
                            install_signals=False)


def test_shield_phase_mapping():
    P = PerceptionDaemon._shield_phase
    # While verifying we always report "recognizing" (checking) -- the verdict is
    # decided on the matcher result, never guessed from a stranger being co-present.
    assert P(State.VERIFYING, Observation(now=0, owner_visible=True)) == "recognizing"
    assert P(State.VERIFYING, Observation(now=0, stranger_visible=True)) == "recognizing"
    assert P(State.LIVENESS_CHALLENGE, Observation(now=0, owner_visible=True)) == "recognizing"
    assert P(State.LOCKED_STRANGER, Observation(now=0)) == "denied"
    assert P(State.LOCKED_ABSENT, Observation(now=0)) == "locked"
    assert P(State.COOLDOWN, Observation(now=0)) == "locked"
    assert P(State.UNLOCKED_PRESENT, Observation(now=0, owner_visible=True)) is None
    assert P(State.GRANT, Observation(now=0)) is None


def test_push_feedback_debounces_and_pushes_on_phase_change():
    d = make_daemon()
    # Enter VERIFYING -> one 'recognizing' (checking) push.
    d._push_shield_feedback(0.0, State.VERIFYING, Observation(now=0.0, owner_visible=True))
    # Same phase, no progress change -> no push.
    d._push_shield_feedback(0.1, State.VERIFYING, Observation(now=0.1, owner_visible=True))
    assert d.emitter.calls == [("recognizing", None)]
    # A phase change is pushed promptly (bypasses the progress rate-limit).
    d._push_shield_feedback(0.2, State.LOCKED_ABSENT, Observation(now=0.2))
    assert d.emitter.calls[-1][0] == "locked"
    # Shield down -> no push.
    d._push_shield_feedback(0.3, State.UNLOCKED_PRESENT, Observation(now=0.3, owner_visible=True))
    assert len(d.emitter.calls) == 2


def test_checking_progress_is_derived_from_matcher_votes():
    d = make_daemon()
    d.matcher = Matcher(_unit(0), tau=0.5, k=3, n=5)  # k-of-n = 3-of-5
    m = MatchResult(is_owner=False, score=0.7, tau=0.5, votes_k=2, votes_n=2, face_count=1)
    obs = Observation(now=0.0, face_count=1, match=m)
    d._push_shield_feedback(0.0, State.VERIFYING, obs)
    phase, _reason = d.emitter.calls[-1]
    kw = d.emitter.kw[-1]
    assert phase == "recognizing"
    # progress = max(votes_k/k, votes_n/n) = max(2/3, 2/5) = 0.666...
    assert kw["progress"] == pytest.approx(2 / 3)
    assert kw["votes_k"] == 2 and kw["votes_need"] == 3
    assert kw["frames"] == 2 and kw["frames_need"] == 5


def test_reject_bumps_fail_count_and_pushes_unauthorized():
    d = make_daemon()
    d._last_fail = 0
    d.fsm.fail_count = 1            # a verification just failed
    d._push_shield_feedback(0.0, State.LOCKED_ABSENT, Observation(now=0.0))
    assert d.emitter.calls[-1][0] == "denied"   # UNAUTHORIZED verdict


def test_emitter_shield_status_message_and_timeout(monkeypatch):
    captured = {}

    def fake_send(path, msg, timeout=3.0):
        captured["msg"] = msg
        captured["timeout"] = timeout
        return {"ok": True}

    monkeypatch.setattr(control_mod, "send_command", fake_send)
    em = control_mod.DecisionEmitter(socket_path="/tmp/facelock-test.sock", timeout=3.0)
    em.shield_status("denied", "stranger")
    assert captured["msg"] == {"cmd": "shield_status", "phase": "denied", "reason": "stranger"}
    assert captured["timeout"] <= 0.5    # never stalls the perception loop
