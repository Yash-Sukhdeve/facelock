"""ShieldWindow headless-safety tests.

CRITICAL: these tests must NEVER open a real shield (a fullscreen input-grabbing
window would lock the developer out). They force ``has_display`` to False so the
reworked create-once/reuse logic exercises only its fail-closed, no-display path.
The reuse-with-display path is validated live, not in unit tests (per task: mock
X, do not touch the screen).
"""

from __future__ import annotations

import pytest

from facelock import shield as shield_mod
from facelock.shield import ShieldWindow


@pytest.fixture(autouse=True)
def _no_display(monkeypatch):
    # Guarantee no window is ever created regardless of the host's $DISPLAY.
    monkeypatch.setattr(shield_mod, "has_display", lambda: False)


def test_raise_without_display_reports_failure():
    s = ShieldWindow()
    assert s.raise_shield("Locked") is False
    assert s.is_up is False


def test_raise_dismiss_cycle_is_safe_and_idempotent():
    s = ShieldWindow()
    for _ in range(3):
        assert s.raise_shield("Locked - you stepped away") is False
        assert s.is_up is False
        # dismiss must be safe even when nothing was ever shown.
        s.dismiss()
        assert s.is_up is False


def test_set_status_and_pump_are_noops_without_window():
    s = ShieldWindow()
    # Should not raise even though no root/status var exists.
    s.set_status("Locked - unrecognized face")
    s.pump()
    assert s.is_up is False


def test_phase_setters_track_phase_headless():
    s = ShieldWindow(owner_name="Yash")
    s.set_recognizing()
    assert s._phase == "recognizing"
    s.set_denied()
    assert s._phase == "denied"
    s.set_welcome("Yash")
    assert s._phase == "welcome" and s._caption_name == "Yash"
    s.set_status("Locked")
    assert s._phase == "locked"


def test_render_is_safe_without_canvas():
    # _render must be a no-op (never raise) when no Canvas exists (headless).
    s = ShieldWindow()
    s._render()
    assert s._canvas is None
