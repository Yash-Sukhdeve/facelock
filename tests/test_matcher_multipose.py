"""Multi-pose (max-over-templates) matcher tests.

The matcher scores each probe against the BEST of the enrolled pose
sub-templates, so an off-angle face clears tau via its nearest pose -- the
dependency-light path to "easy" authentication. These tests prove the pose bank
raises the genuine off-angle score, still rejects a stranger, and never lowers
tau (REQ-NF-22).
"""

from __future__ import annotations

import numpy as np

import pytest

from facelock.matcher import EMBEDDING_DIM, Matcher, verification_progress


def test_verification_progress_reflects_real_votes():
    # No votes yet -> 0; k votes -> full (about to accept).
    assert verification_progress(0, 0, 3, 5) == 0.0
    assert verification_progress(3, 3, 3, 5) == 1.0
    # Partial: max(votes_k/k, votes_n/n).
    assert verification_progress(1, 1, 3, 5) == pytest.approx(1 / 3)
    assert verification_progress(2, 4, 3, 5) == pytest.approx(4 / 5)
    # Window full without enough votes -> 1.0 (about to REJECT).
    assert verification_progress(0, 5, 3, 5) == 1.0
    # Clamps + guards against zero/negative denominators.
    assert verification_progress(9, 9, 3, 5) == 1.0
    assert verification_progress(-1, -1, 0, 0) == 0.0


def _unit(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


def _make_poses(seed: int = 0):
    rng = np.random.default_rng(seed)
    pose_a = _unit(rng.standard_normal(EMBEDDING_DIM))
    # A clearly different direction (near-orthogonal) = a different head pose.
    pose_b = _unit(rng.standard_normal(EMBEDDING_DIM))
    centroid = _unit(pose_a + pose_b)  # the averaged single-template
    return rng, pose_a, pose_b, centroid


def test_pose_bank_raises_offangle_score():
    rng, pose_a, pose_b, centroid = _make_poses(1)
    probe_b = _unit(pose_b + 0.03 * rng.standard_normal(EMBEDDING_DIM))

    single = Matcher(centroid, tau=0.5, k=1, n=1)
    multi = Matcher(centroid, tau=0.5, k=1, n=1,
                    extra_templates=np.stack([pose_a, pose_b]))

    s_single = single.score_only(probe_b)
    s_multi = multi.score_only(probe_b)
    # The pose closest to the probe lifts the score well above the centroid-only
    # score (that is exactly what makes off-angle auth "easy").
    assert s_multi > s_single
    assert s_multi > 0.95
    assert multi.pose_count == 3   # centroid + 2 poses
    assert single.pose_count == 1


def test_stranger_still_rejected_with_pose_bank():
    rng, pose_a, pose_b, centroid = _make_poses(2)
    multi = Matcher(centroid, tau=0.5, k=1, n=1,
                    extra_templates=np.stack([pose_a, pose_b]))
    stranger = _unit(rng.standard_normal(EMBEDDING_DIM))  # unrelated direction
    score = multi.score_only(stranger)
    assert score < 0.5
    assert multi.passes(score) is False


def test_tau_is_never_lowered_by_pose_mode():
    _, pose_a, pose_b, centroid = _make_poses(3)
    multi = Matcher(centroid, tau=0.42, k=1, n=1,
                    extra_templates=np.stack([pose_a, pose_b]))
    assert multi.tau == 0.42   # pose mode only widens acceptance, never relaxes tau


def test_verify_grants_offangle_owner_via_pose():
    rng, pose_a, pose_b, centroid = _make_poses(4)
    m = Matcher(centroid, tau=0.6, k=2, n=3,
                extra_templates=np.stack([pose_a, pose_b]))
    probe = _unit(pose_a + 0.02 * rng.standard_normal(EMBEDDING_DIM))
    r1 = m.verify(probe, face_count=1)
    r2 = m.verify(probe, face_count=1)
    assert r2.is_owner is True and r2.votes_k >= 2


def test_select_diverse_picks_the_outlier():
    _, pose_a, pose_b, _ = _make_poses(5)
    samples = np.stack([pose_a, pose_a, pose_b])  # A duplicated + B outlier
    sel = Matcher._select_diverse(samples, 2)
    assert sel.shape == (2, EMBEDDING_DIM)
    # The far pose (B) must be among the two selected.
    assert float(np.max(sel @ pose_b)) > 0.999


def test_select_diverse_caps_and_empty():
    _, pose_a, pose_b, _ = _make_poses(6)
    samples = np.stack([pose_a, pose_b])
    assert Matcher._select_diverse(samples, 5).shape[0] == 2   # k >= m -> all
    empty = np.empty((0, EMBEDDING_DIM), dtype=np.float32)
    assert Matcher._select_diverse(empty, 3).shape == (0, EMBEDDING_DIM)


def test_pose_max_caps_bank_size():
    rng, pose_a, pose_b, centroid = _make_poses(7)
    samples = _unit(rng.standard_normal((10, EMBEDDING_DIM)).T).T  # 10 poses
    samples = np.stack([_unit(samples[i]) for i in range(10)])
    m = Matcher(centroid, tau=0.5, k=1, n=1, extra_templates=samples, pose_max=3)
    assert m.pose_count == 1 + 3   # centroid + capped poses


def test_no_template_disables_bank():
    m = Matcher(None, 0.363, k=1, n=1,
                extra_templates=np.zeros((3, EMBEDDING_DIM), dtype=np.float32))
    assert m.loaded is False and m.pose_count == 0
    r = m.verify(np.ones(EMBEDDING_DIM, dtype=np.float32), face_count=1)
    assert r.is_owner is False   # fail-closed with no centroid (FM-10)
