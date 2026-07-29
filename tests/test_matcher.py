"""Matcher tests (C4, REQ-F-07/08): cosine math + k-of-n voting, fail-closed."""

from __future__ import annotations

import numpy as np

from facelock.matcher import Matcher, MatchResult, cosine_similarity, l2_distance
from tests.conftest import unit_vec


def test_cosine_basic():
    a = np.array([1.0, 0.0, 0.0])
    assert cosine_similarity(a, a) == 1.0
    assert cosine_similarity(a, np.array([0.0, 1.0, 0.0])) == 0.0
    assert cosine_similarity(a, np.array([-1.0, 0.0, 0.0])) == -1.0


def test_cosine_fail_closed_on_bad_input():
    a = np.array([1.0, 0.0, 0.0])
    assert cosine_similarity(a, np.array([0.0, 0.0, 0.0])) == -1.0  # zero norm
    assert cosine_similarity(a, np.array([np.nan, 0.0, 0.0])) == -1.0  # NaN
    assert cosine_similarity(a, np.array([1.0, 0.0])) == -1.0  # shape mismatch


def test_l2_distance():
    a = np.array([0.0, 0.0]); b = np.array([3.0, 4.0])
    assert l2_distance(a, b) == 5.0
    assert l2_distance(a, np.array([np.inf, 0.0])) == float("inf")


def _matcher(tau=0.5, k=3, n=5):
    centroid = unit_vec(7)
    return Matcher(centroid, tau, k=k, n=n), centroid


def test_k_of_n_requires_k_votes():
    m, centroid = _matcher()
    owner = centroid  # cosine 1.0 >= 0.5
    # First two owner frames: not enough votes yet.
    r1 = m.verify(owner, face_count=1)
    r2 = m.verify(owner, face_count=1)
    assert not r1.is_owner and not r2.is_owner
    r3 = m.verify(owner, face_count=1)
    assert r3.is_owner and r3.votes_k == 3


def test_current_frame_must_pass_even_with_history():
    m, centroid = _matcher()
    owner = centroid
    stranger = unit_vec(99)  # near-orthogonal -> score < tau
    for _ in range(3):
        m.verify(owner, face_count=1)  # accumulate 3 owner votes
    # A stranger single-face frame must NOT grant despite owner history.
    res = m.verify(stranger, face_count=1)
    assert not res.is_owner


def test_multiface_never_owner():
    m, centroid = _matcher()
    owner = centroid
    for _ in range(3):
        m.verify(owner, face_count=1)
    res = m.verify(owner, face_count=2)  # two faces present
    assert not res.is_owner and res.face_count == 2


def test_no_template_never_owner():
    m = Matcher(None, 0.5, k=1, n=3)
    assert not m.loaded
    res = m.verify(unit_vec(3), face_count=1)
    assert not res.is_owner


def test_reset_clears_votes():
    m, centroid = _matcher(k=2, n=3)
    m.verify(centroid, 1)
    m.verify(centroid, 1)
    m.reset()
    res = m.verify(centroid, 1)
    assert res.votes_n == 1 and not res.is_owner


def test_passes_direction_cosine_and_l2():
    mc = Matcher(unit_vec(1), 0.5, k=1, n=1, metric="cosine")
    assert mc.passes(0.6) and not mc.passes(0.4)
    ml = Matcher(unit_vec(1), 1.0, k=1, n=1, metric="l2")
    assert ml.passes(0.9) and not ml.passes(1.1)  # lower distance is better


def test_matchresult_is_frozen_dataclass():
    r = MatchResult(True, 0.9, 0.5, 3, 5, 1)
    assert r.is_owner and r.score == 0.9 and r.tau == 0.5
