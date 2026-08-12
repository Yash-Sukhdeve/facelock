"""Dynamic enrollment UI tests: head-pose geometry + the Face-ID-style HUD.

Drawing is real OpenCV/Pillow on NumPy frames (testable headless -- no window
opened). We assert the pure head-direction/segment/guidance maths, and that the
polished ``render`` draws every phase without raising, preserves the frame
shape/dtype, never mutates the input, lights green as coverage fills, shows the
completion state, and -- critically -- still returns a valid image when Pillow
is unavailable (the cv2.putText fallback for a bare machine).
"""

from __future__ import annotations

import builtins

import numpy as np
import pytest

import facelock.enroll_ui as ui
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
                                 current=3, target_segment=4, tick=7,
                                 instruction="Set Up Face Unlock",
                                 status="Looking great", quality_ok=True))
    assert img.shape == (480, 640, 3) and img.dtype == np.uint8
    assert int(img.sum()) > 0            # something was actually drawn


def test_render_with_flash_and_reject_and_bbox():
    img = render(_frame(), _view(bbox=(80, 60, 200, 200), quality_ok=False,
                                 reject="too_blurry", flash=1.0, current=2,
                                 target_segment=5, status="Hold still"))
    assert img.shape == (480, 640, 3) and img.dtype == np.uint8
    assert int(img.sum()) > 0


def test_render_does_not_mutate_input():
    src = _frame()
    _ = render(src, _view(covered=frozenset({0, 1}), flash=0.5))
    assert int(src.sum()) == 0     # drew on a fresh canvas, not the input


def test_render_returns_new_array():
    src = _frame()
    out = render(src, _view())
    assert out is not src


def test_render_covered_ring_has_green():
    # A fully covered ring lights green with a glow: the green (BGR idx 1)
    # channel must dominate -- and clearly exceed the red/blue channels there.
    img = render(_frame(), _view(covered=frozenset(range(16)), tick=3))
    assert int(img[:, :, 1].max()) > 150
    assert int(img[:, :, 1].max()) > int(img[:, :, 2].max())  # greener than red


def test_render_capture_state_not_all_green():
    # Early in capture (nothing covered) the ring should NOT be flooded green:
    # far fewer strong-green pixels than when fully covered.
    def _green_pixels(cov):
        im = render(_frame(), _view(covered=frozenset(cov), tick=1))
        g = im[:, :, 1].astype(int)
        return int(((g > 140) & (g > im[:, :, 2].astype(int) + 30)).sum())

    assert _green_pixels(range(16)) > _green_pixels([]) + 200


def test_render_done_state_shows_completion():
    img = render(_frame(), _view(phase="done", covered=frozenset(range(16)),
                                 frontal_done=True, instruction="All set",
                                 status="Your face is enrolled", tick=12))
    assert img.shape == (480, 640, 3) and img.dtype == np.uint8
    # The checkmark + "enrolled" copy render in green.
    assert int(img[:, :, 1].max()) > 150


def test_render_no_face_bbox_none_does_not_crash():
    img = render(_frame(), _view(bbox=None, reject="no_face", quality_ok=False,
                                 covered=frozenset(), current=None,
                                 target_segment=0))
    assert img.shape == (480, 640, 3) and img.dtype == np.uint8


def test_render_small_frame():
    img = render(_frame(160, 120), _view(current=1, target_segment=2,
                                         covered=frozenset({0})))
    assert img.shape == (160, 120, 3) and img.dtype == np.uint8


def test_render_tiny_frame_no_crash():
    # A degenerate frame must still return a same-size image, not raise.
    img = render(_frame(8, 8), _view(covered=frozenset({0, 1})))
    assert img.shape == (8, 8, 3) and img.dtype == np.uint8


# --- typography: Inter present, and the graceful cv2 fallback -------------- #
def test_bundled_inter_font_loads():
    # The Inter weights we ship must be reachable via importlib.resources and
    # decode through Pillow (proves the assets are on the package path).
    ImageFont = pytest.importorskip("PIL.ImageFont")
    for weight in ("regular", "medium", "semibold"):
        font = ui._load_font(ImageFont, weight, 24)
        assert font is not None


def test_render_with_pillow_present_is_crisp():
    # Sanity: with Pillow available the title text is rasterised (non-empty).
    pytest.importorskip("PIL")
    img = render(_frame(), _view(instruction="Set Up Face Unlock", tick=2))
    assert img.shape == (480, 640, 3) and int(img.sum()) > 0


def test_render_falls_back_when_pillow_absent(monkeypatch):
    """render() MUST NOT crash if Pillow is unavailable -- it uses cv2.putText.

    We block only ``import PIL`` (everything else imports normally) and confirm
    the font stack reports absent and render still returns a valid image with
    legible text (non-zero content in the title band).
    """
    ui._font_cache.clear()
    real_import = builtins.__import__

    def _no_pil(name, *args, **kwargs):
        if name == "PIL" or name.startswith("PIL."):
            raise ImportError("PIL disabled for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_pil)

    assert ui._pil() is None                       # the fallback branch is live

    src = _frame()
    img = render(src, _view(phase="capture", covered=frozenset({0, 1, 2}),
                            current=3, target_segment=4, status="Looking great",
                            quality_ok=True, instruction="Set Up Face Unlock",
                            flash=0.5, tick=5))
    assert img.shape == (480, 640, 3) and img.dtype == np.uint8
    assert int(img.sum()) > 0
    assert int(src.sum()) == 0                     # still no input mutation
    # Hershey text drew into the top title band.
    assert int(img[0:60, :, :].sum()) > 0

    # The done phase must also survive the Pillow-absent path.
    done = render(_frame(), _view(phase="done", covered=frozenset(range(16)),
                                  instruction="All set", tick=12))
    assert done.shape == (480, 640, 3) and int(done[:, :, 1].max()) > 150
