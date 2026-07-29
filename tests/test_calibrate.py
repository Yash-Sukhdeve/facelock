"""Threshold calibration tests (design section 3.2, REQ-NF-10/22)."""

from __future__ import annotations

import numpy as np
import pytest

from facelock.calibrate import calibrate, centroid_of, wilson_interval
from tests.conftest import owner_cluster


def test_centroid_is_unit_norm():
    samples = owner_cluster()
    c = centroid_of(samples)
    assert abs(np.linalg.norm(c) - 1.0) < 1e-5


def test_wilson_interval_bounds():
    lo, hi = wilson_interval(0, 1000)
    assert 0.0 <= lo <= hi <= 1.0
    lo2, hi2 = wilson_interval(10, 1000)
    assert lo2 < 0.01 < hi2


def test_calibration_enforces_floor(impostors):
    # A tight owner cluster vs random impostors -> impostor-derived tau would be
    # low, so the floor (0.363) MUST be enforced (never ship weaker than seed).
    samples = owner_cluster(jitter=0.03)
    res = calibrate(samples, impostors, fmr_target=0.01, fnmr_target=0.05, tau_floor=0.363)
    assert res.tau >= 0.363
    assert res.tau_floor == 0.363


def test_calibration_meets_target_for_tight_cluster(impostors):
    samples = owner_cluster(jitter=0.03)
    res = calibrate(samples, impostors, fmr_target=0.01, fnmr_target=0.05, tau_floor=0.363)
    assert res.fmr_measured <= 0.01
    assert res.fnmr_measured <= 0.05
    assert res.meets_target
    # Confidence intervals present and well-formed.
    assert res.fmr_ci[0] <= res.fmr_ci[1]
    assert res.impostor_n == impostors.shape[0]


def test_calibration_never_relaxes_below_floor_even_if_target_needs_it(impostors):
    # Even when the genuine spread is wide, tau must not drop below the floor.
    samples = owner_cluster(jitter=0.4)
    res = calibrate(samples, impostors, fmr_target=0.01, fnmr_target=0.05, tau_floor=0.363)
    assert res.tau >= 0.363


def test_calibration_requires_min_samples(impostors):
    with pytest.raises(ValueError):
        calibrate(owner_cluster(n=1), impostors)


def test_calibration_requires_impostors():
    with pytest.raises(ValueError):
        calibrate(owner_cluster(), np.zeros((10, 128), dtype=np.float32))


def test_meta_serialisable(impostors):
    res = calibrate(owner_cluster(), impostors)
    meta = res.as_meta()
    assert set(["tau_floor", "fmr_measured", "fnmr_measured", "meets_target"]).issubset(meta)
