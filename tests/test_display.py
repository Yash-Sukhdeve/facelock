"""DisplayController tests (monitor DPMS power).

Every test injects a fake ``xset`` path + ``DISPLAY`` env and monkeypatches
``subprocess.run`` so NO real ``xset`` runs and the live screen is never touched.
Covers: off/on issue the right commands, graceful degradation (never raises),
and disabled/absent-environment no-ops.
"""

from __future__ import annotations

import subprocess

import pytest

from facelock import display as display_mod
from facelock.display import DisplayController


class FakeRun:
    """Records argv of each subprocess.run call; returns a chosen return code."""

    def __init__(self, rc: int = 0, raises: Exception | None = None):
        self.calls: list[list[str]] = []
        self._rc = rc
        self._raises = raises

    def __call__(self, cmd, **kwargs):
        self.calls.append(list(cmd))
        if self._raises is not None:
            raise self._raises
        return subprocess.CompletedProcess(cmd, self._rc, "", "")

    def cmds(self) -> list[str]:
        return [" ".join(c) for c in self.calls]


def controller(**kw):
    """An 'available' controller (fake xset + DISPLAY), enabled by default."""
    kw.setdefault("xset_path", "/usr/bin/xset")
    kw.setdefault("env", {"DISPLAY": ":1"})
    kw.setdefault("enabled", True)
    return DisplayController(**kw)


def test_available_requires_xset_and_display():
    assert controller().available is True
    assert DisplayController(xset_path=None, env={"DISPLAY": ":1"}).available is False
    assert DisplayController(xset_path="/usr/bin/xset", env={}).available is False


def test_screen_off_issues_dpms_force_off(monkeypatch):
    run = FakeRun()
    monkeypatch.setattr(display_mod.subprocess, "run", run)
    dc = controller()
    assert dc.screen_off() is True
    assert run.cmds() == ["/usr/bin/xset dpms force off"]


def test_screen_on_issues_force_on_and_saver_reset(monkeypatch):
    run = FakeRun()
    monkeypatch.setattr(display_mod.subprocess, "run", run)
    dc = controller()
    assert dc.screen_on() is True
    assert run.cmds() == ["/usr/bin/xset dpms force on", "/usr/bin/xset s reset"]


def test_off_then_on_cycle_repeatably(monkeypatch):
    run = FakeRun()
    monkeypatch.setattr(display_mod.subprocess, "run", run)
    dc = controller()
    for _ in range(3):
        assert dc.screen_off() is True
        assert dc.screen_on() is True
    # 3 offs + 3 * (on + reset) = 9 calls.
    assert run.cmds().count("/usr/bin/xset dpms force off") == 3
    assert run.cmds().count("/usr/bin/xset dpms force on") == 3


def test_disabled_is_noop(monkeypatch):
    run = FakeRun()
    monkeypatch.setattr(display_mod.subprocess, "run", run)
    dc = controller(enabled=False)
    assert dc.enabled is False
    assert dc.screen_off() is False
    assert dc.screen_on() is False
    assert run.calls == []  # never shells out when disabled


def test_no_display_is_noop_and_never_raises(monkeypatch):
    run = FakeRun()
    monkeypatch.setattr(display_mod.subprocess, "run", run)
    dc = DisplayController(enabled=True, xset_path="/usr/bin/xset", env={})  # no DISPLAY
    assert dc.enabled is False
    assert dc.screen_off() is False
    assert dc.screen_on() is False
    assert run.calls == []


def test_no_xset_is_noop(monkeypatch):
    run = FakeRun()
    monkeypatch.setattr(display_mod.subprocess, "run", run)
    dc = DisplayController(enabled=True, xset_path=None, env={"DISPLAY": ":1"})
    assert dc.screen_off() is False
    assert run.calls == []


def test_subprocess_failure_does_not_raise(monkeypatch):
    run = FakeRun(raises=OSError("boom"))
    monkeypatch.setattr(display_mod.subprocess, "run", run)
    dc = controller()
    # Must degrade to False, never propagate.
    assert dc.screen_off() is False
    assert dc.screen_on() is False


def test_nonzero_exit_reports_false(monkeypatch):
    run = FakeRun(rc=1)
    monkeypatch.setattr(display_mod.subprocess, "run", run)
    dc = controller()
    assert dc.screen_off() is False


def test_set_config_enabled_hot_toggle(monkeypatch):
    run = FakeRun()
    monkeypatch.setattr(display_mod.subprocess, "run", run)
    dc = controller(enabled=False)
    assert dc.screen_off() is False
    dc.set_config_enabled(True)
    assert dc.screen_off() is True


def test_timeout_is_handled(monkeypatch):
    run = FakeRun(raises=subprocess.TimeoutExpired(cmd="xset", timeout=2.0))
    monkeypatch.setattr(display_mod.subprocess, "run", run)
    dc = controller()
    assert dc.screen_off() is False
