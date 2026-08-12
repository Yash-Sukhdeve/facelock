"""Multi-monitor + native-resolution enrollment preview (offline).

These tests cover the *pure* pieces added to wire the premium enrollment
renderer into the live loop at the right resolution on the right screen:

  * ``parse_xrandr_monitors`` -- turns ``xrandr --listmonitors`` text into a
    list of ``Monitor(name, w, h, x, y)`` (real 2-monitor + 1-monitor samples,
    and empty/garbage -> ``[]``).
  * ``list_monitors`` -- wraps xrandr and falls back to a single 1920x1080@0,0
    monitor when xrandr is absent / unparseable (never raises).
  * ``pick_monitor`` -- clamps a ``--screen`` index into range.
  * ``letterbox`` -- resizes a camera frame to the display WxH preserving aspect
    (so ``render`` output is 1:1 with the monitor -- no fullscreen upscaling).
  * CLI arg threading -- ``--screen`` / ``--windowed`` reach ``enroll(...)``.

No camera is opened and no window is created anywhere in this module.
"""

from __future__ import annotations

import subprocess

import numpy as np
import pytest

from facelock import enroll as enroll_mod
from facelock.enroll import (
    Monitor,
    letterbox,
    list_monitors,
    parse_xrandr_monitors,
    pick_monitor,
)

# Real `xrandr --listmonitors` captures (laptop panel + external display).
XRANDR_TWO = (
    "Monitors: 2\n"
    " 0: +*eDP-1 1920/344x1080/193+0+0  eDP-1\n"
    " 1: +HDMI-1 2560/597x1440/336+1920+0  HDMI-1\n"
)
XRANDR_ONE = (
    "Monitors: 1\n"
    " 0: +*eDP-1 1920/309x1200/193+0+0  eDP-1\n"
)
# Some drivers/older xrandr omit the physical-mm parts: WxH+X+Y.
XRANDR_NO_MM = (
    "Monitors: 1\n"
    " 0: +*DP-1 3440x1440+0+0  DP-1\n"
)


# --- parser ---------------------------------------------------------------- #
def test_parse_two_monitors():
    mons = parse_xrandr_monitors(XRANDR_TWO)
    assert len(mons) == 2
    assert mons[0] == Monitor(name="eDP-1", w=1920, h=1080, x=0, y=0)
    assert mons[1] == Monitor(name="HDMI-1", w=2560, h=1440, x=1920, y=0)
    # Tuple-friendly unpacking (name, w, h, x, y).
    name, w, h, x, y = mons[1]
    assert (name, w, h, x, y) == ("HDMI-1", 2560, 1440, 1920, 0)


def test_parse_single_monitor():
    mons = parse_xrandr_monitors(XRANDR_ONE)
    assert mons == [Monitor(name="eDP-1", w=1920, h=1200, x=0, y=0)]


def test_parse_tolerates_missing_mm_geometry():
    mons = parse_xrandr_monitors(XRANDR_NO_MM)
    assert mons == [Monitor(name="DP-1", w=3440, h=1440, x=0, y=0)]


def test_parse_empty_and_garbage_yield_empty():
    assert parse_xrandr_monitors("") == []
    assert parse_xrandr_monitors("Monitors: 0\n") == []
    assert parse_xrandr_monitors("total nonsense\nnot a monitor line\n") == []


# --- list_monitors (subprocess wrapper + fallback) ------------------------- #
_FALLBACK = Monitor(name="default", w=1920, h=1080, x=0, y=0)


def test_list_monitors_parses_real_output(monkeypatch):
    def fake_run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 0, XRANDR_TWO, "")

    monkeypatch.setattr(enroll_mod.subprocess, "run", fake_run)
    mons = list_monitors()
    assert [m.name for m in mons] == ["eDP-1", "HDMI-1"]


def test_list_monitors_falls_back_when_xrandr_absent(monkeypatch):
    def boom(cmd, **kw):
        raise FileNotFoundError("xrandr")

    monkeypatch.setattr(enroll_mod.subprocess, "run", boom)
    assert list_monitors() == [_FALLBACK]


def test_list_monitors_falls_back_on_garbage(monkeypatch):
    def garbage(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 0, "kaboom\n", "")

    monkeypatch.setattr(enroll_mod.subprocess, "run", garbage)
    assert list_monitors() == [_FALLBACK]


def test_list_monitors_falls_back_on_timeout(monkeypatch):
    def slow(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 2.0)

    monkeypatch.setattr(enroll_mod.subprocess, "run", slow)
    assert list_monitors() == [_FALLBACK]


# --- pick_monitor (index clamping) ----------------------------------------- #
def test_pick_monitor_selects_and_clamps():
    mons = parse_xrandr_monitors(XRANDR_TWO)
    assert pick_monitor(mons, 0).name == "eDP-1"
    assert pick_monitor(mons, 1).name == "HDMI-1"
    assert pick_monitor(mons, 9).name == "HDMI-1"    # clamp high -> last
    assert pick_monitor(mons, -3).name == "eDP-1"    # clamp low -> first
    # Empty list -> the safe fallback, never an IndexError.
    assert pick_monitor([], 0) == _FALLBACK


# --- letterbox (aspect-preserving fit to display WxH) ---------------------- #
def test_letterbox_matches_target_shape_and_dtype():
    src = np.full((720, 1280, 3), 200, np.uint8)
    out = letterbox(src, 1920, 1080)
    assert out.shape == (1080, 1920, 3)
    assert out.dtype == np.uint8


def test_letterbox_16x9_into_16x9_fills_without_bars():
    # 1280x720 -> 1920x1080 is a pure 1.5x scale: no padding, corners painted.
    src = np.full((720, 1280, 3), 180, np.uint8)
    out = letterbox(src, 1920, 1080)
    assert int(out[0, 0].sum()) > 0          # corner is real content, not a bar
    assert int(out[540, 960].sum()) > 0      # centre content


def test_letterbox_4x3_into_16x9_pads_sides_preserves_aspect():
    src = np.full((480, 640, 3), 210, np.uint8)
    out = letterbox(src, 1920, 1080)
    assert out.shape == (1080, 1920, 3)
    # 4:3 scaled to 1080 tall -> 1440 wide, centred: ~240px black bars L/R.
    assert int(out[540, 0].sum()) == 0       # left bar is black
    assert int(out[540, 1919].sum()) == 0    # right bar is black
    assert int(out[540, 960].sum()) > 0      # centre is content


def test_letterbox_handles_degenerate_frame():
    # An empty frame must not crash -- returns a black target-sized image.
    out = letterbox(np.zeros((0, 0, 3), np.uint8), 640, 480)
    assert out.shape == (480, 640, 3)


# --- CLI arg threading ----------------------------------------------------- #
def _spy_cli(monkeypatch):
    """Patch cli so ``enroll`` verbs hit a spy tool with no side effects."""
    import facelock.cli as cli

    captured: dict = {}

    class _SpyTool:
        def __init__(self, cfg):
            captured["cfg_built"] = True

        def enroll(self, name, **kw):
            captured["name"] = name
            captured.update(kw)
            return 0

    # cmd_enroll resolves EnrollmentTool via a lazy ``from .enroll import ...``,
    # so patch the source module (keeps CLI startup light for control verbs).
    monkeypatch.setattr(enroll_mod, "EnrollmentTool", _SpyTool)
    monkeypatch.setattr(cli, "_maybe_show_disclosure", lambda: None)
    monkeypatch.setattr(cli, "_load_cfg", lambda _p: object())
    return cli, captured


def test_cli_threads_screen_and_windowed(monkeypatch):
    cli, captured = _spy_cli(monkeypatch)
    args = cli.build_parser().parse_args(
        ["enroll", "--name", "Yash", "--screen", "2", "--windowed"])
    assert args.func(args) == 0
    assert captured["name"] == "Yash"
    assert captured["screen"] == 2
    assert captured["windowed"] is True


def test_cli_defaults_screen_zero_not_windowed(monkeypatch):
    cli, captured = _spy_cli(monkeypatch)
    args = cli.build_parser().parse_args(["enroll"])
    assert args.func(args) == 0
    assert captured["screen"] == 0
    assert captured["windowed"] is False
