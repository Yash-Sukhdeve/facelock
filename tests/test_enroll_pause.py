"""Enrollment camera-coordination tests: auto-pause the daemon, enroll, resume.

The perception daemon holds the single-access UVC camera, so `facelock enroll`
must ask the guardian to pause perception (release the camera), capture, then
resume -- which live-reloads the new template. No real camera/socket is used.
"""

from __future__ import annotations

import os

import facelock.control as control_mod
from facelock.config import load_config
from facelock.enroll import EnrollmentTool
from facelock.errors import CameraError
from facelock.guardian import Guardian
from facelock.lock_backend import LockOutcome

UID = os.getuid()


# --- guardian side: pause/resume commands + heartbeat flag ----------------- #
class _Shield:
    is_up = False

    def raise_shield(self, status="Locked"):
        self.is_up = True
        return True

    def set_status(self, s): ...
    def set_recognizing(self, progress=0.0, votes_k=0, votes_need=0): ...
    def set_denied(self, text="Unauthorized user"): ...
    def set_welcome(self, name): ...
    def dismiss(self): self.is_up = False
    def pump(self): ...


class _Ctl:
    def engage(self):
        return LockOutcome(True, True, "fake", "ok")

    def is_any_locked(self):
        return None


def _guardian():
    return Guardian(load_config(raw={}), lock_controller=_Ctl(), shield=_Shield(),
                    install_signals=False)


def test_pause_resume_commands_toggle_flag_and_heartbeat():
    g = _guardian()
    # Heartbeat before pause: not paused.
    hb0 = g.dispatch({"cmd": "heartbeat", "seq": 1, "state": "UNLOCKED_PRESENT",
                      "health": {}}, UID)
    assert hb0["pause_camera"] is False
    # Pause -> flag set, heartbeat tells the daemon to release the camera.
    p = g.dispatch({"cmd": "pause_perception"}, UID)
    assert p["ok"] and p["paused"] is True and g._perception_paused is True
    hb1 = g.dispatch({"cmd": "heartbeat", "seq": 2, "state": "UNLOCKED_PRESENT",
                      "health": {}}, UID)
    assert hb1["pause_camera"] is True
    # Resume -> cleared.
    r = g.dispatch({"cmd": "resume_perception"}, UID)
    assert r["ok"] and r["paused"] is False and g._perception_paused is False
    assert g.dispatch({"cmd": "status"}, UID)["perception_paused"] is False


# --- daemon side: enter/exit pause ----------------------------------------- #
class _FakeCam:
    def __init__(self):
        self.released = 0
        self.is_open = False

    def release(self):
        self.released += 1


class _HBEmitter:
    """Emitter whose heartbeat reply carries a settable pause flag."""

    def __init__(self):
        self.pause = False

    def heartbeat(self, seq, state, health):
        return {"ok": True, "face_unlock": True, "pause_camera": self.pause}

    def request_lock(self, reason):
        return {"ok": True}


def _daemon(emitter):
    from facelock.daemon import PerceptionDaemon
    return PerceptionDaemon(load_config(raw={}), emitter=emitter, install_signals=False)


def test_daemon_enters_and_exits_pause_via_heartbeat():
    from facelock.fsm import State

    em = _HBEmitter()
    d = _daemon(em)
    d.camera = _FakeCam()
    d.fsm.state = State.UNLOCKED_PRESENT   # enrolling happens while unlocked
    d._last_hb = -1000.0            # force the (rate-limited) heartbeat to fire

    em.pause = True
    d._heartbeat(now=100.0)
    assert d._paused is True and d.camera.released == 1

    em.pause = False
    d._last_hb = -1000.0
    d._heartbeat(now=200.0)
    assert d._paused is False
    # Resume must NOT force a re-lock: the owner was already authenticated.
    assert d.fsm.state == State.UNLOCKED_PRESENT


def test_exit_pause_reloads_template_without_crashing():
    d = _daemon(_HBEmitter())
    d._paused = True
    d._exit_pause()                # no template on disk in the sandbox -> matcher None
    assert d._paused is False and d.matcher is not None  # rebuilt (no-template matcher)


# --- enroll orchestration: pause -> retry-open -> resume ------------------- #
class _BusyCam:
    def __init__(self, fail_times):
        self._fail = fail_times
        self.opens = 0

    def open(self):
        self.opens += 1
        if self.opens <= self._fail:
            raise CameraError("device busy", code="busy")

    def release(self):
        ...


def _tool():
    return EnrollmentTool(load_config(raw={}))


def test_acquire_opens_directly_when_camera_free(monkeypatch):
    calls = []
    monkeypatch.setattr(control_mod, "send_command",
                        lambda s, m, **k: calls.append(m) or {"ok": True})
    opened, paused = _tool()._acquire_camera(_BusyCam(fail_times=0))
    assert opened is True and paused is False
    assert calls == []             # never contacted the daemon


def test_acquire_pauses_daemon_then_opens(monkeypatch):
    sent = []

    def fake_send(sock, msg, **kw):
        sent.append(msg["cmd"])
        return {"ok": True, "heartbeat_sec": 0.01}

    monkeypatch.setattr(control_mod, "send_command", fake_send)
    opened, paused = _tool()._acquire_camera(_BusyCam(fail_times=1), wait_s=1.0)
    assert opened is True and paused is True
    assert "pause_perception" in sent
    # On success, resume is deferred to enroll's finally (not sent here).
    assert "resume_perception" not in sent


def test_acquire_gives_up_and_resumes_when_still_busy(monkeypatch):
    sent = []

    def fake_send(sock, msg, **kw):
        sent.append(msg["cmd"])
        return {"ok": True, "heartbeat_sec": 0.01}

    monkeypatch.setattr(control_mod, "send_command", fake_send)
    opened, paused = _tool()._acquire_camera(_BusyCam(fail_times=999), wait_s=0.5)
    assert opened is False
    assert "pause_perception" in sent and "resume_perception" in sent  # undo the pause


def test_acquire_prints_hint_when_no_guardian(monkeypatch, capsys):
    monkeypatch.setattr(control_mod, "send_command",
                        lambda s, m, **k: {"ok": False, "reason": "transport"})
    opened, paused = _tool()._acquire_camera(_BusyCam(fail_times=999), wait_s=0.5)
    assert opened is False and paused is False
    assert "systemctl --user stop" in capsys.readouterr().out
