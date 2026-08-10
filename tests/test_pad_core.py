"""Model-free RGB PAD core (T1-T4): cropper, preprocessing, decode.

These tests exercise the *security-critical* decode and the crop/preprocess
geometry with **synthetic numpy arrays and mock logits only** -- no model is
loaded, no camera is opened, nothing touches disk. The golden-vector tests
(T4) are a regression gate: a future change that flips the live class index or
the two-scale fusion must FAIL loudly here.

Realizes design tasks T1 (pad_crop), T2 (build_liveness_observation), T3
(pad_blob), T4 (decode + fusion + fail-closed), per rgb-pad-design.md.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from facelock.liveness import (
    PAD_INPUT_SIZE,
    PAD_LIVE_INDEX,
    PAD_SCALES,
    _PassivePAD,
    _pad_crop_box,
    build_liveness_observation,
    pad_blob,
    pad_crop,
    pad_fuse_live_prob,
    pad_live_prob,
    pad_softmax,
)

# --------------------------------------------------------------------------- #
# Pinned golden vectors (computed once from softmax(index=1); see design T4).
# A regression that flips the index or fusion changes these -> test fails.
# --------------------------------------------------------------------------- #
# scale 2.7 logits [0,3,0], scale 4.0 logits [0,1,0]
GOLDEN_BONAFIDE_LOGITS = ([0.0, 3.0, 0.0], [0.0, 1.0, 0.0])
GOLDEN_BONAFIDE_LIVE = 0.7427799416392855
# scale 2.7 logits [3,0,0] (attack class 0), scale 4.0 logits [0,0,3] (attack class 2)
GOLDEN_SPOOF_LOGITS = ([3.0, 0.0, 0.0], [0.0, 0.0, 3.0])
GOLDEN_SPOOF_LIVE = 0.045278500743629074
# What a WRONG decode (softmax[-1] == index 2) would have fused for bona-fide.
GOLDEN_BONAFIDE_FLIPPED_INDEX = 0.12861002918035724
GOLDEN_TAU = 0.5


# ============================ T1: pad_crop geometry ======================== #
def test_pad_crop_box_geometry_centered():
    """A centered bbox enlarged by ``scale`` gives a box scale-x larger,
    centered on the same point (Silent-Face CropImage semantics)."""
    box = _pad_crop_box(1000, 1000, (400, 400, 100, 100), 2.0)
    assert box == (350, 350, 550, 550)  # 200x200, centered on (450,450)


def test_pad_crop_box_clamps_at_top_left_edge():
    """A bbox in the corner slides fully inside the frame (never negative)."""
    box = _pad_crop_box(200, 200, (0, 0, 50, 50), 4.0)
    x1, y1, x2, y2 = box
    assert x1 >= 0 and y1 >= 0
    assert x2 <= 199 and y2 <= 199
    assert (x1, y1) == (0, 0)


def test_pad_crop_box_scale_clamped_to_frame():
    """An over-large scale is clamped so the crop never exceeds the frame."""
    box = _pad_crop_box(200, 200, (50, 50, 100, 100), 99.0)
    x1, y1, x2, y2 = box
    assert x1 >= 0 and y1 >= 0 and x2 <= 199 and y2 <= 199


def test_pad_crop_box_degenerate_returns_none():
    assert _pad_crop_box(200, 200, (10, 10, 0, 50), 2.7) is None
    assert _pad_crop_box(200, 200, (10, 10, 50, 0), 2.7) is None


def test_pad_crop_larger_scale_covers_more_area():
    """Scale 4.0 must enclose strictly more pixels than scale 2.7 (more
    context), the whole point of the bbox-context crop vs the SFace warp."""
    small = _pad_crop_box(2000, 2000, (900, 900, 100, 100), 2.7)
    large = _pad_crop_box(2000, 2000, (900, 900, 100, 100), 4.0)
    area_s = (small[2] - small[0]) * (small[3] - small[1])
    area_l = (large[2] - large[0]) * (large[3] - large[1])
    assert area_l > area_s


def test_pad_crop_output_shape_and_dtype():
    frame = np.zeros((480, 640, 3), np.uint8)
    out = pad_crop(frame, (200, 150, 120, 120), 2.7, size=PAD_INPUT_SIZE)
    assert out is not None
    assert out.shape == (80, 80, 3)
    assert out.dtype == np.uint8


def test_pad_crop_uniform_color_preserved():
    """Cropping+resizing a constant-colour region yields that colour (proves it
    crops real pixels, not a fixed blank)."""
    frame = np.zeros((400, 400, 3), np.uint8)
    frame[:, :] = (50, 100, 150)  # BGR
    out = pad_crop(frame, (100, 100, 80, 80), 2.7)
    assert out is not None
    assert abs(float(out[..., 0].mean()) - 50) < 2
    assert abs(float(out[..., 1].mean()) - 100) < 2
    assert abs(float(out[..., 2].mean()) - 150) < 2


def test_pad_crop_context_includes_background():
    """A larger scale pulls in surrounding background: black frame with a white
    face-box -> mean pixel value DROPS as the scale grows (more black context)."""
    frame = np.zeros((400, 400, 3), np.uint8)
    frame[150:250, 150:250] = 255  # white face region
    tight = pad_crop(frame, (150, 150, 100, 100), 1.0)
    wide = pad_crop(frame, (150, 150, 100, 100), 4.0)
    assert tight is not None and wide is not None
    assert float(tight.mean()) > float(wide.mean())


def test_pad_crop_degenerate_bbox_none():
    frame = np.zeros((400, 400, 3), np.uint8)
    assert pad_crop(frame, (10, 10, 0, 50), 2.7) is None
    assert pad_crop(frame, None, 2.7) is None


def test_pad_crop_none_or_empty_frame_none():
    assert pad_crop(None, (10, 10, 50, 50), 2.7) is None
    assert pad_crop(np.zeros((0, 0, 3), np.uint8), (10, 10, 50, 50), 2.7) is None


# ============================ T3: pad_blob preprocess ====================== #
def test_pad_blob_shape_is_nchw_80():
    crop = np.zeros((80, 80, 3), np.uint8)
    blob = pad_blob(crop)
    assert blob is not None
    assert blob.shape == (1, 3, 80, 80)


def test_pad_blob_scale_1_over_255_golden():
    """Constant crop -> every element equals pixel/255 (scale 1/255, no mean)."""
    crop = np.full((80, 80, 3), 128, np.uint8)
    blob = pad_blob(crop)
    assert np.allclose(blob, 128.0 / 255.0, atol=1e-6)


def test_pad_blob_no_channel_swap_golden():
    """swapRB=False: the BGR channel order MUST be preserved into the blob.
    A regression to swapRB=True would swap planes 0 and 2 and feed the model a
    channel-flipped image (silent PAD degradation) -- pinned here."""
    crop = np.zeros((80, 80, 3), np.uint8)
    crop[:, :, 0] = 10  # B
    crop[:, :, 1] = 20  # G
    crop[:, :, 2] = 30  # R
    blob = pad_blob(crop)
    assert abs(float(blob[0, 0].mean()) - 10.0 / 255.0) < 1e-6  # plane 0 == B
    assert abs(float(blob[0, 1].mean()) - 20.0 / 255.0) < 1e-6  # plane 1 == G
    assert abs(float(blob[0, 2].mean()) - 30.0 / 255.0) < 1e-6  # plane 2 == R


def test_pad_blob_resizes_non_80_input():
    crop = np.full((37, 51, 3), 200, np.uint8)
    blob = pad_blob(crop)
    assert blob.shape == (1, 3, 80, 80)


def test_pad_blob_none_input_none():
    assert pad_blob(None) is None


# =================== T4: decode + fusion + golden gate ===================== #
def test_pad_softmax_sums_to_one():
    p = pad_softmax([1.0, 2.0, 3.0])
    assert abs(float(p.sum()) - 1.0) < 1e-12
    assert p.shape == (3,)


def test_pad_live_index_is_one():
    """Guards the security-critical constant: live == class 1 (NOT [-1])."""
    assert PAD_LIVE_INDEX == 1


def test_pad_live_prob_selects_index_1_not_last():
    """For an asymmetric logit the index-1 probability differs from the last
    ([-1]) probability, so an accidental ``[-1]`` decode is observable."""
    logits = [0.0, 3.0, 0.0]
    live = pad_live_prob(logits)
    last = float(pad_softmax(logits)[-1])
    assert live is not None
    assert abs(live - 0.909442998512742) < 1e-9
    assert abs(live - last) > 0.5  # index 1 and index 2 are far apart here


def test_golden_vector_bonafide_passes():
    """GOLDEN: pinned bona-fide logits fuse to a known live-score ABOVE tau."""
    fused = pad_fuse_live_prob(GOLDEN_BONAFIDE_LOGITS)
    assert fused is not None
    assert abs(fused - GOLDEN_BONAFIDE_LIVE) < 1e-9
    assert fused >= GOLDEN_TAU  # decision: LIVE / grant-eligible


def test_golden_vector_spoof_denied():
    """GOLDEN: pinned print/replay logits fuse BELOW tau -> DENY."""
    fused = pad_fuse_live_prob(GOLDEN_SPOOF_LOGITS)
    assert fused is not None
    assert abs(fused - GOLDEN_SPOOF_LIVE) < 1e-9
    assert fused < GOLDEN_TAU  # decision: SPOOF / deny


def test_flipped_index_would_flip_bonafide_verdict():
    """SECURITY: if the decode regressed to ``softmax[-1]`` (index 2), the same
    bona-fide logits would fuse BELOW tau and the live user would be DENIED.
    This documents exactly what the golden gate protects against."""
    flipped = (
        float(pad_softmax(GOLDEN_BONAFIDE_LOGITS[0])[-1])
        + float(pad_softmax(GOLDEN_BONAFIDE_LOGITS[1])[-1])
    ) / 2.0
    assert abs(flipped - GOLDEN_BONAFIDE_FLIPPED_INDEX) < 1e-9
    assert flipped < GOLDEN_TAU  # the wrong index inverts the correct verdict
    # And the correct decode is on the other side of tau:
    assert pad_fuse_live_prob(GOLDEN_BONAFIDE_LOGITS) >= GOLDEN_TAU


def test_fuse_is_mean_of_two_scales():
    """Fusion averages the two per-scale live probs (pred[1]/2), so the result
    lies strictly between the two single-scale probs when they differ."""
    l1, l2 = GOLDEN_BONAFIDE_LOGITS
    p1, p2 = pad_live_prob(l1), pad_live_prob(l2)
    fused = pad_fuse_live_prob((l1, l2))
    assert min(p1, p2) < fused < max(p1, p2)
    assert abs(fused - (p1 + p2) / 2.0) < 1e-12


# ------------------------- T4 fail-closed (DENY) --------------------------- #
def test_fuse_fail_closed_when_a_scale_is_none():
    """A degraded/missing scale output -> fused None -> caller denies."""
    assert pad_fuse_live_prob(([0.0, 3.0, 0.0], None)) is None
    assert pad_fuse_live_prob((None, [0.0, 3.0, 0.0])) is None


def test_fuse_fail_closed_on_empty():
    assert pad_fuse_live_prob(()) is None
    assert pad_fuse_live_prob(None) is None


def test_live_prob_fail_closed_on_malformed_single_class():
    """A malformed 1-class output cannot address the live index -> None."""
    assert pad_live_prob([0.7]) is None


def test_live_prob_fail_closed_on_empty():
    assert pad_live_prob([]) is None
    assert pad_live_prob(None) is None


def test_live_prob_fail_closed_on_nonfinite():
    assert pad_live_prob([float("inf"), 0.0, 0.0]) is None
    assert pad_live_prob([float("nan"), 1.0, 2.0]) is None


def test_passive_pad_unavailable_scores_none():
    """No provisioned model -> both single-crop and fused scoring fail closed."""
    pad = _PassivePAD("/nonexistent-model.onnx", 0.5)
    assert not pad.available
    crop = np.full((80, 80, 3), 128, np.uint8)
    assert pad.score(crop) is None
    assert pad.score_crops({2.7: crop, 4.0: crop}) is None


def test_passive_pad_score_crops_none_input():
    pad = _PassivePAD("/nonexistent-model.onnx", 0.5)
    assert pad.score_crops(None) is None
    assert pad.score_crops({}) is None


# ==================== T2: observation build (no camera) ==================== #
def _fake_detection(bbox=(200, 150, 120, 120)):
    """A duck-typed detection exposing ONLY bbox + landmarks -- proving the PAD
    path needs neither the SFace raw_row nor the aligned warp (recognition)."""
    lm = np.array(
        [[220, 190], [300, 190], [260, 220], [230, 260], [290, 260]],
        dtype=np.float32,
    )
    return SimpleNamespace(bbox=bbox, landmarks=lm)


def test_build_observation_passive_populates_pad_crops():
    frame = np.zeros((480, 640, 3), np.uint8)
    frame[150:270, 200:320] = 200
    det = _fake_detection()
    obs = build_liveness_observation(frame, ts=1.5, detection=det, mode="full")
    assert obs.pad_crops is not None
    assert set(obs.pad_crops.keys()) == set(PAD_SCALES)
    for scale in PAD_SCALES:
        assert obs.pad_crops[scale].shape == (80, 80, 3)
    assert np.array_equal(obs.landmarks, det.landmarks)
    assert obs.ts == 1.5


def test_build_observation_passive_mode_also_populates():
    frame = np.zeros((480, 640, 3), np.uint8)
    obs = build_liveness_observation(frame, ts=0.0, detection=_fake_detection(), mode="passive")
    assert obs.pad_crops is not None and set(obs.pad_crops) == set(PAD_SCALES)


def test_build_observation_turn_mode_no_pad_crops():
    """`turn` needs no model/crop -> pad_crops stays None (mode unaffected)."""
    frame = np.zeros((480, 640, 3), np.uint8)
    obs = build_liveness_observation(frame, ts=0.0, detection=_fake_detection(), mode="turn")
    assert obs.pad_crops is None


def test_build_observation_off_mode_no_pad_crops():
    frame = np.zeros((480, 640, 3), np.uint8)
    obs = build_liveness_observation(frame, ts=0.0, detection=_fake_detection(), mode="off")
    assert obs.pad_crops is None


def test_build_observation_degenerate_bbox_no_crops_no_crash():
    """A degenerate bbox yields no crops (fail-closed) rather than raising."""
    frame = np.zeros((480, 640, 3), np.uint8)
    det = _fake_detection(bbox=(10, 10, 0, 0))
    obs = build_liveness_observation(frame, ts=0.0, detection=det, mode="full")
    assert obs.pad_crops is None
