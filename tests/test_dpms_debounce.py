"""DPMS (monitor-power) debounce + re-assert cadence — Task 2 stabilization.

Reproduces the display churn seen in one real session (root-cause §2):

  * 42,419 ``display_dpms`` events, 42,336 of them a 3-second "force off"
    re-assert that ran the whole locked-away session (mechanism a);
  * an undebounced ``screen_on``/``screen_off`` toggled once per shield-status
    frame, so a rapid recognizing<->locked flap strobed the monitor (mechanism b).

These are guardian-level (C8/C11) tests with the shield, lock backend and display
faked (the same ``CountingShield``/``FakeDisplay`` used by ``test_relock``), so NO
window, subprocess, socket, camera or real screen is touched.

The fix must (1) DE-DUPE redundant same-state DPMS calls, (2) collapse rapid
flapping into a BOUNDED number of real transitions (hysteresis), and (3) run the
OFF re-assert on a slow cadence — while PRESERVING prompt owner-return wake and a
stable-away blank, and never weakening the fail-closed lock/shield posture.
"""

from __future__ import annotations

from tests.test_relock import UID, FakeDisplay, make_guardian


# --------------------------------------------------------------------------- #
# (b) per-frame screen_on/screen_off strobe: debounce + de-dupe
# --------------------------------------------------------------------------- #
def test_shield_status_flap_does_not_strobe_the_monitor():
    """A rapid recognizing<->locked flap (surfaced by an upstream camera flap)
    must NOT strobe DPMS. The real main loop drains the shield queue once per
    tick, so a burst of alternating phases within one tick must collapse to at
    most one real on/off transition — not one per frame (current code: 10)."""
    g = make_guardian()
    g.dispatch({"cmd": "lock", "reason": "away"}, UID)
    g._drain_shield_queue()
    on0, off0 = g.display.on_calls, g.display.off_calls  # off0 == 1 (the lock)
    # 10 flap cycles enqueued, then ONE drain (as the guardian loop drains/tick).
    for _ in range(10):
        g.dispatch({"cmd": "shield_status", "phase": "recognizing"}, UID)
        g.dispatch({"cmd": "shield_status", "phase": "locked", "reason": "away"}, UID)
    g._drain_shield_queue()
    assert g.display.on_calls - on0 <= 1, "monitor strobed ON per flap frame"
    assert g.display.off_calls - off0 <= 1, "monitor strobed OFF per flap frame"
    # Fail-closed invariant: cosmetic DPMS churn must never touch lock/shield.
    assert g.grant.current()[0] is True
    assert g.shield.is_up is True


def test_redundant_locked_frames_do_not_respawn_screen_off():
    """Once blanked, a stream of 'locked' frames must be de-duped — no extra
    `xset dpms force off` per frame (current code: one off per frame)."""
    g = make_guardian()
    g.dispatch({"cmd": "lock", "reason": "away"}, UID)
    g._drain_shield_queue()
    assert g.display.off_calls == 1
    for _ in range(5):
        g.dispatch({"cmd": "shield_status", "phase": "locked", "reason": "away"}, UID)
        g._drain_shield_queue()
    assert g.display.off_calls == 1        # de-duped, not 6
    assert g._screen_off_active is True    # OFF intent preserved (stays dark)


def test_redundant_recognizing_frames_do_not_respawn_screen_on():
    """A continuous 'recognizing' stream (owner standing at the camera) must wake
    the monitor ONCE, not re-issue `xset dpms force on` every frame."""
    g = make_guardian()
    g.dispatch({"cmd": "lock", "reason": "away"}, UID)
    g._drain_shield_queue()
    g.dispatch({"cmd": "shield_status", "phase": "recognizing"}, UID)
    g._drain_shield_queue()
    assert g.display.on_calls == 1
    for _ in range(5):
        g.dispatch({"cmd": "shield_status", "phase": "recognizing"}, UID)
        g._drain_shield_queue()
    assert g.display.on_calls == 1         # de-duped, not 6


# --------------------------------------------------------------------------- #
# Preserved behaviour: prompt wake on return, stays dark on stable absence.
# --------------------------------------------------------------------------- #
def test_owner_return_wakes_monitor_promptly():
    """Genuine owner-return: a single 'recognizing' frame wakes the monitor
    immediately (no debounce delay), so feedback shows promptly."""
    g = make_guardian()
    g.dispatch({"cmd": "lock", "reason": "away"}, UID)
    g._drain_shield_queue()
    assert g._screen_off_active is True
    g.dispatch({"cmd": "shield_status", "phase": "recognizing"}, UID)
    g._drain_shield_queue()
    assert g.display.on_calls == 1
    assert g._screen_off_active is False


def test_stable_away_keeps_monitor_off():
    """A stable locked-away stream never wakes the monitor (stays dark)."""
    g = make_guardian()
    g.dispatch({"cmd": "lock", "reason": "away"}, UID)
    g._drain_shield_queue()
    for _ in range(20):
        g.dispatch({"cmd": "shield_status", "phase": "locked", "reason": "away"}, UID)
        g._drain_shield_queue()
    assert g._screen_off_active is True
    assert g.display.on_calls == 0


# --------------------------------------------------------------------------- #
# (a) OFF re-assert cadence: slow, not every 3s
# --------------------------------------------------------------------------- #
def test_reassert_cadence_is_slow_not_every_3s():
    """The locked-away DPMS re-assert must use a SLOW cadence. At 3s it spawned
    42,336 `xset` off events in one session (root-cause §2a)."""
    g = make_guardian()
    assert g._screen_reassert_s >= 30.0


def test_reassert_screen_off_is_bounded_over_a_locked_window():
    """Over a 5-minute locked-away window the re-assert must spawn a BOUNDED
    number of `xset dpms force off` calls (slow cadence) — not ~100 (3s cadence).
    Exercises the guardian's real main-loop re-assert unit."""
    g = make_guardian()
    g.dispatch({"cmd": "lock", "reason": "away"}, UID)
    g._drain_shield_queue()
    assert g._screen_off_active is True
    before = g.display.off_calls
    for i in range(1, 3001):               # 300s of 0.1s ticks
        g._maybe_reassert_screen(i * 0.1)
    spawns = g.display.off_calls - before
    assert spawns <= 12, f"re-assert storm: {spawns} xset-off spawns / 5 min"


def test_reassert_is_a_noop_while_screen_on():
    """The re-assert fights X's wake timers only while intentionally blanked;
    once the monitor is ON (owner present) it must never blank behind them."""
    g = make_guardian(display=FakeDisplay())
    g._screen_off_active = False
    before = g.display.off_calls
    for i in range(1, 200):
        g._maybe_reassert_screen(i * 100.0)
    assert g.display.off_calls == before
