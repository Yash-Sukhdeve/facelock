"""Deployed-matcher scoring tests (T2, the D-1 core).

The audit finding (protocol Gap B) is that calibration scores impostors against
the *centroid only*, while deployment scores against the *max over a pose bank*.
Because the centroid is itself a bank member,

    S_deployed(x) = max_j cos(x, bank_j) >= cos(x, centroid) = S_centroid(x)

for EVERY probe, so on the same probe set FMR_deployed(tau) >= FMR_centroid(tau).
These tests prove that superset property directly on the shipped matcher path,
so a future refactor cannot silently regress the harness back to the understated
centroid-only FMR. Offline: no camera, no models, no daemon.
"""

from __future__ import annotations

import numpy as np
import pytest

from facelock.calibrate import centroid_of
from facelock.store import Template, generate_synthetic_impostors
from facelock.matcher import Matcher, cosine_similarity
from facelock.eval import metrics as M
from facelock.eval import scoring as S
from tests.conftest import owner_cluster, unit_vec


def _make_template(jitter: float = 0.05, tau: float = 0.363, n: int = 6) -> Template:
    samples = owner_cluster(n=n, seed=1, jitter=jitter)
    return Template(
        owner_name="test-owner",
        centroid=centroid_of(samples),
        samples=samples,
        tau=tau,
        metric="cosine",
    )


def test_score_probes_returns_finite_1d_array():
    tmpl = _make_template()
    probes = np.stack([unit_vec(s) for s in range(50, 60)])
    scores = S.score_probes(tmpl, probes)
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (10,)
    assert np.all(np.isfinite(scores))
    # a single 1-D probe is accepted too.
    one = S.score_probes(tmpl, probes[0])
    assert one.shape == (1,)


def test_score_probes_equals_matcher_score_only_exactly():
    # The harness MUST use the exact production scoring path, not a re-derivation.
    tmpl = _make_template(jitter=0.15)
    m = Matcher(
        tmpl.centroid, tmpl.tau, k=1, n=1,
        metric=tmpl.metric, extra_templates=tmpl.samples, pose_max=5,
    )
    probes = np.stack([unit_vec(s) for s in range(200, 215)])
    got = S.score_probes(tmpl, probes)
    ref = np.array([m.score_only(p) for p in probes], dtype=float)
    assert np.array_equal(got, ref)
    assert S.deployed_scores is S.score_probes  # protocol alias


def test_deployed_score_ge_centroid_score_for_every_probe():
    # The superset property, elementwise -- the security-correctness guard.
    tmpl = _make_template(jitter=0.20)
    rng = np.random.default_rng(9)
    probes = np.stack([
        (v / np.linalg.norm(v)).astype(np.float32)
        for v in rng.standard_normal((300, 128))
    ])
    deployed = S.score_probes(tmpl, probes)
    centroid = S.centroid_scores(tmpl, probes)
    assert np.all(deployed >= centroid - 1e-12)
    # And it is NOT vacuous: with spread poses some probes score strictly higher
    # against a pose than against the centroid (deterministic under fixed seeds).
    assert np.any(deployed > centroid + 1e-9)


def test_centroid_scores_match_direct_cosine():
    tmpl = _make_template(jitter=0.1)
    probes = np.stack([unit_vec(s) for s in range(10, 20)])
    got = S.centroid_scores(tmpl, probes)
    ref = np.array([cosine_similarity(p, tmpl.centroid) for p in probes])
    assert np.allclose(got, ref, atol=0.0, rtol=0.0)


def test_fmr_deployed_dominates_fmr_centroid_on_impostors():
    # The audit finding, demonstrated: on the SAME impostor set and the SAME tau,
    # the deployed max-of-bank FMR is >= the centroid-only FMR the calibrator
    # currently reports. This is the number that could be silently understated.
    tmpl = _make_template(jitter=0.25)
    impostors = generate_synthetic_impostors(n=2000, seed=42)
    dep = S.score_probes(tmpl, impostors)
    cen = S.centroid_scores(tmpl, impostors)
    for tau in np.linspace(0.1, 0.6, 26):
        fmr_dep = M.fmr_at_tau(dep, tau)
        fmr_cen = M.fmr_at_tau(cen, tau)
        assert fmr_dep >= fmr_cen - 1e-12, f"regression at tau={tau}"
    # Strictly greater somewhere in the operating band -> the gap is real.
    gaps = [
        M.fmr_at_tau(dep, tau) - M.fmr_at_tau(cen, tau)
        for tau in np.linspace(0.2, 0.5, 31)
    ]
    assert max(gaps) > 0.0


def test_pose_max_bounds_the_bank_used_for_scoring():
    # pose_max is the deployment knob; the scorer must honor it (bank = 1+pose_max).
    tmpl = _make_template(jitter=0.3, n=10)
    m_small = S.build_matcher(tmpl, pose_max=2)
    m_big = S.build_matcher(tmpl, pose_max=5)
    assert m_small.pose_count == 1 + 2
    assert m_big.pose_count == 1 + 5


def test_scoring_l2_metric_uses_min_over_bank():
    # For an l2 template the deployed score is the MIN distance over the bank,
    # so deployed_score <= centroid_score (better = smaller) -- acceptance is
    # still a superset. Guards the metric-aware direction.
    samples = owner_cluster(n=6, seed=3, jitter=0.2)
    tmpl = Template(
        owner_name="l2-owner",
        centroid=centroid_of(samples),
        samples=samples,
        tau=0.9,
        metric="l2",
    )
    rng = np.random.default_rng(5)
    probes = np.stack([
        (v / np.linalg.norm(v)).astype(np.float32)
        for v in rng.standard_normal((100, 128))
    ])
    dep = S.score_probes(tmpl, probes)
    cen = S.centroid_scores(tmpl, probes)
    assert np.all(dep <= cen + 1e-12)
