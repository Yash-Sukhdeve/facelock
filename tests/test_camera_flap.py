"""Camera-flap regression (PRIMARY Phase-1 reliability defect).

Reproduces the unbounded camera release<->reacquire cycling that the daemon
exhibited under continuous owner-absence (29,494 release/reacquire cycles in a
single 43 h session -- see scratchpad/investigation/flapping-rootcause.md #2).

Root cause: ``_manage_camera`` released the device on a *level* trigger --
``long_gone = (now - _absent_since) >= long_absence_release_s`` -- while a
duty-cycle probe reacquire cleared ``_camera_released`` but never touched
``_absent_since`` and did not move the FSM out of LOCKED_ABSENT (owner still
gone). So ``_absent_since`` stayed stale, ``long_gone`` was *permanently* true,
and the device re-released on the very next tick: a self-sustaining flap whose
period was the ~3 s recheck interval, not the ``long_absence_release_s``
threshold.

The fix makes exit-from-released edge-triggered on an *actual* owner return by
resetting the settle/absence clock on reacquire, so a re-release cannot fire
again within ``long_absence_release_s``. Under continuous absence the camera
must therefore cycle at most ~once per ``long_absence_release_s`` -- bounded --
instead of once per tick.

SAFETY: no daemon is started, no real camera/socket/systemd is touched. We drive
``PerceptionDaemon._manage_camera`` directly with a counting fake camera and a
forced (already long-absent) FSM state. Unit-only.
"""

from __future__ import annotations

from facelock.config import load_config
from facelock.daemon import PerceptionDaemon
from facelock.fsm import State


class CountingCamera:
    """Fake camera that counts release()/reacquire() and tracks open state.

    Duck-types the subset of ``CameraCapture`` that ``_manage_camera`` uses:
    ``release()``, ``reacquire()``, ``is_open`` and ``set_rate()``. ``read()``
    always yields a no-face frame so, if a caller ever observes, the owner is
    (correctly) seen as absent.
    """

    def __init__(self) -> None:
        self.releases = 0
        self.reacquires = 0
        self.is_open = True
        self.rates: list[int] = []

    def release(self) -> bool:
        self.releases += 1
        self.is_open = False
        return True

    def reacquire(self) -> bool:
        self.reacquires += 1
        self.is_open = True
        return True

    def set_rate(self, fps: int) -> bool:
        self.rates.append(fps)
        return True

    def read(self):
        return None, "no-frame"  # never surfaces a face


class _InertEmitter:
    """Never contacts a socket; ``_manage_camera`` doesn't touch it anyway."""

    def request_lock(self, reason):  # pragma: no cover - defensive
        return {"ok": True}

    def heartbeat(self, seq, state, health):  # pragma: no cover - defensive
        return {"ok": True}


def _daemon() -> PerceptionDaemon:
    return PerceptionDaemon(load_config(raw={}), emitter=_InertEmitter(),
                            install_signals=False)


def test_camera_does_not_flap_under_continuous_absence():
    """Owner leaves -> long-absence release -> periodic probes with NO face.

    The device must NOT re-release+reacquire on every tick. With the fix the
    cycling is bounded to at most ~once per ``long_absence_release_s``.
    """
    d = _daemon()
    cam = CountingCamera()
    d.camera = cam

    long_s = d.cfg.camera.long_absence_release_s  # default 120 s
    recheck = max(3.0, 1.0 / max(d.cfg.camera.fps_idle, 1))  # default 3.0 s

    # Force "already long-absent": FSM parked in LOCKED_ABSENT and the absence
    # clock set past the release threshold, exactly as after the owner walks off.
    d.fsm.state = State.LOCKED_ABSENT
    t0 = 1_000.0
    d._absent_since = t0 - long_s - 1.0

    # Drive 60 s of ticks (0.1 s apart) with no returning face. 60 s spans ~20
    # recheck intervals but well under a second ``long_absence_release_s`` window.
    dt = 0.1
    ticks = 600
    for i in range(ticks):
        d._manage_camera(t0 + i * dt)

    span_s = ticks * dt
    # Bound: at most one release per long-absence window, plus one for the
    # initial release edge. The buggy level-trigger produces ~span/recheck (~20).
    max_cycles = int(span_s // long_s) + 2
    assert cam.releases <= max_cycles, (
        f"camera flapped: {cam.releases} releases over {span_s:.0f}s "
        f"(one per ~{recheck:.0f}s recheck instead of per "
        f"~{long_s:.0f}s absence window); expected <= {max_cycles}")
    assert cam.reacquires <= max_cycles, (
        f"camera flapped: {cam.reacquires} reacquires over {span_s:.0f}s; "
        f"expected <= {max_cycles}")


def test_flap_stays_bounded_over_a_long_absence():
    """Over a 10-minute continuous absence the cycling stays ~once per window.

    Defence-in-depth beyond the 60 s reproduction: the buggy code produces
    ~200 cycles (span/recheck); the fix keeps it to ~span/long_absence_release_s.
    """
    d = _daemon()
    cam = CountingCamera()
    d.camera = cam
    long_s = d.cfg.camera.long_absence_release_s
    recheck = max(3.0, 1.0 / max(d.cfg.camera.fps_idle, 1))

    d.fsm.state = State.LOCKED_ABSENT
    t0 = 5_000.0
    d._absent_since = t0 - long_s - 1.0

    span_s = 600.0
    dt = 0.5
    for i in range(int(span_s / dt)):
        d._manage_camera(t0 + i * dt)

    tick_bound = int(span_s // recheck)          # ~200: what the flap would do
    window_bound = int(span_s // long_s) + 2     # ~7: bounded target
    assert cam.releases <= window_bound
    assert cam.releases < tick_bound // 4        # unambiguously NOT per-tick


def test_owner_return_reacquires_and_leaves_released_state():
    """Preserved behaviour: when the owner returns the camera comes back open.

    Passes on both the buggy and the fixed code -- it pins the guarantee that
    the fix must not regress: leaving LOCKED_ABSENT reacquires the device and
    clears the released flag so perception resumes promptly (owner-return path).
    """
    d = _daemon()
    cam = CountingCamera()
    d.camera = cam
    long_s = d.cfg.camera.long_absence_release_s

    # Start long-absent and released (as after a power-saving release).
    d.fsm.state = State.LOCKED_ABSENT
    t0 = 9_000.0
    d._absent_since = t0 - long_s - 1.0
    d._manage_camera(t0)              # releases (long-absence)
    assert d._camera_released is True and cam.is_open is False

    # Recheck interval elapses -> probe reacquire brings the device back open.
    recheck = max(3.0, 1.0 / max(d.cfg.camera.fps_idle, 1))
    d._manage_camera(t0 + recheck + 0.01)
    assert cam.is_open is True and d._camera_released is False

    # Owner is now present: the FSM has left LOCKED_ABSENT. The camera stays
    # open and observing (no release), so the returning face is seen at once.
    d.fsm.state = State.VERIFYING
    releases_before = cam.releases
    d._manage_camera(t0 + recheck + 1.0)
    assert cam.is_open is True and d._camera_released is False
    assert cam.releases == releases_before  # no spurious release on return
    assert d._absent_since is None          # absence clock cleared on presence
