"""Dynamic enrollment UI tests: head-pose geometry + interactive ring HUD.

Drawing is real OpenCV on NumPy frames (testable headless -- no window opened).
We assert the pure head-direction/segment/guidance maths and that ``render``
draws every state without raising and preserves the frame shape/dtype.
"""

from __future__ import annotations

import numpy as np
import pytest

from facelock.enroll_ui import (
    RingView,
    coverage_fraction,
    head_offset,
    nearest_uncovered,
    progress_fraction,
    render,
    segment_of,
)

cv2 = pytest.importorskip("cv2")  # drawing needs OpenCV (a hard runtime dep)


def _frame(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


# --- pure geometry --------------------------------------------------------- #
def test_head_offset_center_is_zero():
    # Nose at the box centre -> looking straight ahead.
    bbox = (100, 100, 200, 200)
    lm = np.array([[150, 160], [250, 160], [200, 200], [160, 250], [240, 250]],
                  dtype=np.float32)  # nose (idx 2) at (200,200) = centre
    nx, ny = head_offset(lm, bbox)
    assert abs(nx) < 1e-6 and abs(ny) < 1e-6


def test_head_offset_points_toward_turn():
    bbox = (0, 0, 200, 200)             # centre (100,100), half-extent 100
    lm = np.zeros((5, 2), dtype=np.float32)
    lm[2] = (160, 100)                  # nose right of centre
    nx, ny = head_offset(lm, bbox)
    assert nx == pytest.approx(0.6) and abs(ny) < 1e-6


def test_head_offset_handles_degenerate():
    assert head_offset(np.zeros((1, 2)), (0, 0, 10, 10)) == (0.0, 0.0)


def test_segment_of_frontal_deadzone():
    assert segment_of(0.0, 0.0, 16) is None
    assert segment_of(0.05, 0.05, 16, deadzone=0.12) is None   # inside deadzone


def test_segment_of_buckets_angles():
    # +x (right) is angle 0 -> segment 0; +y (down) is 90deg -> N/4.
    assert segment_of(1.0, 0.0, 16) == 0
    assert segment_of(0.0, 1.0, 16) == 4
    assert segment_of(-1.0, 0.0, 16) == 8
    assert segment_of(0.0, -1.0, 16) == 12
    assert segment_of(1.0, 0.0, 0) is None   # guard


def test_nearest_uncovered_prefers_closest():
    covered = {0, 1, 2, 3}
    # current=2 covered; nearest uncovered is 4 (dist 2) vs 15 (dist 3).
    assert nearest_uncovered(2, covered, 16) == 4
    assert nearest_uncovered(None, {0, 1}, 16) == 2
    assert nearest_uncovered(0, set(range(16)), 16) is None   # all covered


def test_progress_and_coverage_fractions():
    assert progress_fraction(0, 15) == 0.0
    assert progress_fraction(30, 15) == 1.0        # clamps
    assert coverage_fraction(set(range(8)), 16) == pytest.approx(0.5)
    assert coverage_fraction(set(), 0) == 0.0      # guard


# --- rendering ------------------------------------------------------------- #
def _view(**kw):
    base = dict(owner="Yash", captured=5, target=18, n_segments=16)
    base.update(kw)
    return RingView(**base)


@pytest.mark.parametrize("phase", ["capture", "done"])
def test_render_every_phase(phase):
    img = render(_frame(), _view(phase=phase, covered=frozenset({0, 1, 2}),
                                 current=3, target_segment=4, tick=7))
    assert img.shape == (480, 640, 3) and img.dtype == np.uint8
    assert int(img.sum()) > 0


def test_render_with_flash_and_reject_and_bbox():
    img = render(_frame(), _view(bbox=(80, 60, 200, 200), quality_ok=False,
                                 reject="too_blurry", flash=1.0, current=2,
                                 target_segment=5, status="Hold still"))
    assert img.shape == (480, 640, 3) and int(img.sum()) > 0


def test_render_does_not_mutate_input():
    src = _frame()
    _ = render(src, _view())
    assert int(src.sum()) == 0     # drew on a copy


def test_render_covered_ring_has_green():
    img = render(_frame(), _view(covered=frozenset(range(16))))
    assert int(img[:, :, 1].max()) > 150   # green channel present (BGR)


def test_render_small_frame():
    img = render(_frame(160, 120), _view(current=1, target_segment=2))
    assert img.shape == (160, 120, 3)
