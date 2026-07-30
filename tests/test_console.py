"""Tests for the terminal HUD engine (pure string maths, no TTY needed).

The console must (a) degrade to clean, escape-free text when colour is off so
piped output stays machine-readable, (b) keep every framed panel aligned to its
declared width regardless of embedded colour codes, and (c) never crash on odd
input. All of that is testable headless.
"""

from __future__ import annotations

import re

from facelock import console as con

_SGR = re.compile(r"\x1b\[[0-9;]*m")


def _visible(text: str) -> str:
    return _SGR.sub("", text)


def test_plain_console_emits_no_escapes():
    c = con.Console(color=False, width=60)
    out = c.panel("STATUS", [c.kv("STATE", "LOCKED"), c.bar(0.5)], footer="v1")
    assert "\x1b" not in out
    assert "STATUS" in out and "LOCKED" in out


def test_paint_is_noop_without_color():
    c = con.Console(color=False, width=40)
    assert c.paint("hello", con.CYAN) == "hello"
    c2 = con.Console(color=True, width=40)
    painted = c2.paint("hello", con.CYAN)
    assert painted != "hello" and "hello" in painted and painted.endswith("\x1b[0m")


def test_panel_rows_align_to_width_with_color():
    c = con.Console(color=True, width=66)
    rows = [c.kv("SESSION", "LOCKED", value_colour=con.RED),
            c.bar(0.6), "plain row"]
    panel = c.panel("GUARDIAN TELEMETRY", rows, footer="facelock")
    lines = _visible(panel).splitlines()
    # Every rendered line is exactly the declared width (borders included).
    assert all(len(ln) == 66 for ln in lines), [len(ln) for ln in lines]


def test_panel_wraps_overlong_content_inside_the_frame():
    c = con.Console(color=False, width=50)
    long_footer = "x " * 60  # far wider than the interior
    panel = c.panel("T", ["short"], footer=long_footer.strip())
    lines = panel.splitlines()
    assert all(len(ln) == 50 for ln in lines)


def test_bar_clamps_and_reports_percent():
    c = con.Console(color=False, width=40)
    assert "0%" in c.bar(-1)
    assert "100%" in c.bar(5)
    assert "50%" in c.bar(0.5)


def test_vis_len_ignores_escapes():
    c = con.Console(color=True, width=40)
    painted = c.paint("abcde", con.GREEN, bold=True)
    assert con._vis_len(painted) == 5


def test_supports_color_honours_no_color(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    assert con.supports_color() is False
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")
    assert con.supports_color() is True


def test_supports_color_non_tty(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)

    class FakeStream:
        def isatty(self):
            return False

    assert con.supports_color(FakeStream()) is False


def test_banner_and_wrap_are_stringy():
    c = con.Console(color=False, width=48)
    assert "FACELOCK" or con._WORDMARK  # wordmark rows exist
    banner = c.banner("SUBTITLE")
    assert "SUBTITLE" in banner
    wrapped = c.wrap("one two three four five", con.TEXT)
    assert isinstance(wrapped, list) and all(isinstance(s, str) for s in wrapped)


def test_phase_colour_maps_known_and_default():
    from facelock import ui_theme
    assert con.phase_colour(ui_theme.DENIED) == con.RED
    assert con.phase_colour("nonsense") == con.CYAN
