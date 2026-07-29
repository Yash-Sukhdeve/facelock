"""Enrollment core tests (C16): quality gate + template building (no camera)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from facelock.enroll import (
    CAMERA_BUSY_HINT,
    POSES,
    assess_quality,
    build_template,
    pose_plan,
)
from facelock.errors import CalibrationError
from facelock.store import generate_synthetic_impostors
from tests.conftest import owner_cluster


@dataclass
class FakeDet:
    bbox: tuple
    score: float = 0.99


def test_quality_no_face():
    assert assess_quality([], sharpness=100, brightness=120).reason == "no_face"


def test_quality_multiple_faces():
    dets = [FakeDet((0, 0, 100, 100)), FakeDet((0, 0, 90, 90))]
    assert assess_quality(dets, sharpness=100, brightness=120).reason == "multiple_faces"


def test_quality_face_too_small():
    dets = [FakeDet((0, 0, 50, 50))]
    r = assess_quality(dets, sharpness=100, brightness=120, min_face_px=80)
    assert not r.ok and r.reason == "face_too_small"


def test_quality_too_blurry():
    dets = [FakeDet((0, 0, 120, 120))]
    r = assess_quality(dets, sharpness=5, brightness=120, sharpness_floor=40)
    assert not r.ok and r.reason == "too_blurry"


def test_quality_bad_brightness():
    dets = [FakeDet((0, 0, 120, 120))]
    assert assess_quality(dets, sharpness=100, brightness=5).reason == "bad_brightness"
    assert assess_quality(dets, sharpness=100, brightness=250).reason == "bad_brightness"


def test_quality_ok():
    dets = [FakeDet((0, 0, 120, 120))]
    r = assess_quality(dets, sharpness=100, brightness=120)
    assert r.ok and r.reason == "ok" and r.face_px == 120


def test_camera_busy_hint_tells_user_to_stop_both_services():
    # The hint must name both units so the watchdog can't trip mid-enroll.
    assert "systemctl --user stop facelockd.service facelock-guardian.service" in CAMERA_BUSY_HINT
    assert "start facelockd.service facelock-guardian.service" in CAMERA_BUSY_HINT


def test_pose_plan_default_covers_all_poses():
    plan = pose_plan(min_samples=5, samples_per_pose=3)
    assert len(plan) == len(POSES)
    hints = [h for h, _, _ in plan]
    assert hints == [h for h, _ in POSES]
    assert all(n == 3 for _, _, n in plan)
    assert sum(n for _, _, n in plan) == 15   # 5 poses x 3


def test_pose_plan_tops_up_to_min_samples():
    plan = pose_plan(min_samples=20, samples_per_pose=3)
    assert sum(n for _, _, n in plan) >= 20
    # Distribution stays balanced (top-up spreads across poses).
    counts = [n for _, _, n in plan]
    assert max(counts) - min(counts) <= 1


def test_pose_plan_single_pose_mode():
    plan = pose_plan(min_samples=5, multipose=False)
    assert len(plan) == 1
    hint, _instr, count = plan[0]
    assert hint == "center" and count >= 5


def test_pose_plan_respects_samples_per_pose_floor():
    plan = pose_plan(min_samples=1, samples_per_pose=0)   # coerced to >=1
    assert all(n >= 1 for _, _, n in plan)


def test_build_template_produces_calibrated_template():
    samples = owner_cluster(jitter=0.03)
    impostors = generate_synthetic_impostors(n=1500, seed=11)
    meta = [{"pose_hint": "auto"} for _ in range(samples.shape[0])]
    t = build_template(
        "Yash", samples, meta, impostors,
        model_id="abc123", phase="P", metric="cosine",
        fmr_target=0.01, fnmr_target=0.05, tau_floor=0.363,
    )
    assert t.owner_name == "Yash"
    assert t.tau >= 0.363                      # floor enforced (REQ-NF-22)
    assert t.model_id == "abc123"
    assert t.samples.shape == samples.shape
    assert abs(np.linalg.norm(t.centroid) - 1.0) < 1e-5
    assert "meets_target" in t.calibration


def test_build_template_too_few_samples():
    impostors = generate_synthetic_impostors(n=200, seed=1)
    with pytest.raises(CalibrationError):
        build_template("Y", owner_cluster(n=1), [{}], impostors,
                       model_id="", phase="P", metric="cosine",
                       fmr_target=0.01, fnmr_target=0.05, tau_floor=0.363)
