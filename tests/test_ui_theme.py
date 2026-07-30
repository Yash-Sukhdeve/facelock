"""Pure UI-engine tests (the futuristic shield's animation + colour maths).

No tkinter, no display -- this is exactly why the animation logic lives in
``ui_theme``: the whole look is deterministic and unit-testable headless.
"""

from __future__ import annotations

import re

import pytest

from facelock import ui_theme as ui

_HEX = re.compile(r"^#[0-9a-f]{6}$")


def test_theme_for_known_and_unknown():
    assert ui.theme_for(ui.RECOGNIZING).ring == ui.PHASE_THEMES[ui.RECOGNIZING].ring
    # Unknown phase falls back to the LOCKED palette (never crashes).
    assert ui.theme_for("nonsense") is ui.PHASE_THEMES[ui.LOCKED]


def test_every_phase_theme_is_valid_hex():
    for th in ui.PHASE_THEMES.values():
        for colour in (th.ring, th.glow, th.text, th.accent):
            assert _HEX.match(colour), colour


def test_triangle_wave_shape():
    assert ui.triangle(0, 20) == 0.0
    assert ui.triangle(10, 20) == pytest.approx(1.0)   # peak at half period
    assert ui.triangle(20, 20) == 0.0                  # back to trough
    assert ui.triangle(5, 20) == pytest.approx(0.5)
    assert ui.triangle(3, 0) == 0.0                    # guard: period 0


def test_pulse_stays_in_range():
    for t in range(0, 40):
        v = ui.pulse(t, 17, lo=0.2, hi=0.9)
        assert 0.2 <= v <= 0.9


def test_blink_alternates():
    assert ui.blink_on(0, 4) is True
    assert ui.blink_on(3, 4) is True
    assert ui.blink_on(4, 4) is False
    assert ui.blink_on(7, 4) is False
    assert ui.blink_on(8, 4) is True


def test_ring_sweep_wraps_and_clamps_extent():
    s0, e0 = ui.ring_sweep(0, 8.0, extent=100)
    assert s0 == 0.0 and e0 == 100.0
    s1, _ = ui.ring_sweep(45, 8.0)          # 45*8 = 360 -> wraps to 0
    assert s1 == pytest.approx(0.0)
    # extent is clamped away from a full circle / invisible sliver.
    assert ui.ring_sweep(0, 8.0, extent=999)[1] == 355.0
    assert ui.ring_sweep(0, 8.0, extent=0)[1] == 5.0


def test_progress_extent():
    assert ui.progress_extent(0, 5) == 0.0
    assert ui.progress_extent(5, 5) == 360.0
    assert ui.progress_extent(2, 4) == 180.0
    assert ui.progress_extent(9, 4) == 360.0   # clamps at full
    assert ui.progress_extent(1, 0) == 0.0     # guard: total 0


def test_scanline_frac_bounds():
    for t in range(0, 60):
        assert 0.0 <= ui.scanline_frac(t) <= 1.0


def test_hex_lerp_endpoints_and_midpoint():
    assert ui.hex_lerp("#000000", "#ffffff", 0.0) == "#000000"
    assert ui.hex_lerp("#000000", "#ffffff", 1.0) == "#ffffff"
    assert ui.hex_lerp("#000000", "#ffffff", 0.5) == "#808080"
    # t is clamped.
    assert ui.hex_lerp("#000000", "#ffffff", -3) == "#000000"
    assert ui.hex_lerp("#000000", "#ffffff", 9) == "#ffffff"


def test_hex_lerp_rejects_bad_colour():
    with pytest.raises(ValueError):
        ui.hex_lerp("blue", "#ffffff", 0.5)


def test_glow_ramp_returns_valid_colours():
    ramp = ui.glow_ramp("#00e5ff", tick=7, period=20, steps=4)
    assert len(ramp) == 4
    assert all(_HEX.match(c) for c in ramp)


def test_corner_bracket_segments_shape_and_bounds():
    segs = ui.corner_bracket_segments(1920, 1080)
    assert len(segs) == 8  # two arms per corner, four corners
    for (x0, y0, x1, y1) in segs:
        assert 0 <= x0 <= 1920 and 0 <= x1 <= 1920
        assert 0 <= y0 <= 1080 and 0 <= y1 <= 1080


def test_corner_brackets_do_not_invert_on_tiny_display():
    # Clamped so arms/margins never exceed the display half/third.
    segs = ui.corner_bracket_segments(60, 40)
    assert len(segs) == 8
    for (x0, y0, x1, y1) in segs:
        assert 0 <= x0 <= 60 and 0 <= x1 <= 60
        assert 0 <= y0 <= 40 and 0 <= y1 <= 40


def test_telemetry_lines_are_stable_strings():
    for phase in (ui.LOCKED, ui.RECOGNIZING, ui.DENIED, ui.WELCOME, ui.ENROLLING):
        lines = ui.telemetry_lines(phase, tick=13, owner_name="Yash")
        assert len(lines) == 4
        assert all(isinstance(s, str) and s for s in lines)
    # Deterministic in tick (resume-safe: no wall clock).
    assert ui.telemetry_lines(ui.LOCKED, 7, "x") == ui.telemetry_lines(ui.LOCKED, 7, "x")


def test_phase_captions():
    assert ui.phase_caption(ui.LOCKED) == "LOCKED"
    assert ui.phase_caption(ui.RECOGNIZING) == "CHECKING AUTHORIZATION"
    assert ui.phase_caption(ui.DENIED) == "UNAUTHORIZED"
    assert ui.phase_caption(ui.WELCOME, "Yash") == "AUTHORIZED  -  WELCOME BACK, YASH"
    assert ui.phase_caption(ui.WELCOME, "") == "AUTHORIZED"
    assert ui.phase_caption("bogus") == "LOCKED"
    # Sub-captions exist for every phase.
    for p in (ui.LOCKED, ui.RECOGNIZING, ui.DENIED, ui.WELCOME, ui.ENROLLING):
        assert isinstance(ui.phase_subcaption(p), str)
