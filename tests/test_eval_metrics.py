"""Known-answer + property tests for facelock.eval.metrics (T1).

These are the *science* of the eval harness: every rate here becomes a claim in
the accuracy report, so each core value is checked against a hand-computed
reference, not just an internal round-trip. Pure functions only -- no cv2, no
models, no I/O.

Hand-worked reference set (used throughout):

    gen (genuine/owner scores) = [0.4, 0.5, 0.6, 0.7, 0.8]   (n=5)
    imp (impostor scores)      = [0.2, 0.3, 0.4, 0.5, 0.6]   (n=5)

Sweeping tau (cosine, accept <=> s >= tau) gives, at the observed scores:

    tau  FMR=mean(imp>=tau)  FNMR=mean(gen<tau)  d=FMR-FNMR
    0.2  1.0                 0.0                 +1.0
    0.3  0.8                 0.0                 +0.8
    0.4  0.6                 0.0                 +0.6
    0.5  0.4                 0.2                 +0.2
    0.6  0.2                 0.4                 -0.2
    0.7  0.0                 0.6                 -0.6
    0.8  0.0                 0.8                 -0.8

d changes sign between tau=0.5 (+0.2) and tau=0.6 (-0.2); the linear crossing is
at alpha = 0.2/(0.2-(-0.2)) = 0.5 -> tau*=0.55, FMR=FNMR=0.3 -> EER = 0.30.
"""

from __future__ import annotations

import numpy as np
import pytest

from facelock.eval import metrics as M
from facelock import calibrate


# Hand-worked reference score set.
GEN = np.array([0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float64)
IMP = np.array([0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float64)


# --------------------------------------------------------------------------- #
# FMR / FNMR at a fixed tau -- direct definitions.
# --------------------------------------------------------------------------- #
def test_fmr_at_tau_known_values():
    assert M.fmr_at_tau(IMP, 0.2) == pytest.approx(1.0)
    assert M.fmr_at_tau(IMP, 0.5) == pytest.approx(0.4)   # {0.5,0.6}/5
    assert M.fmr_at_tau(IMP, 0.6) == pytest.approx(0.2)   # {0.6}/5
    assert M.fmr_at_tau(IMP, 0.7) == pytest.approx(0.0)   # none >= 0.7


def test_fnmr_at_tau_known_values():
    assert M.fnmr_at_tau(GEN, 0.4) == pytest.approx(0.0)  # none < 0.4
    assert M.fnmr_at_tau(GEN, 0.5) == pytest.approx(0.2)  # {0.4}/5
    assert M.fnmr_at_tau(GEN, 0.6) == pytest.approx(0.4)  # {0.4,0.5}/5
    assert M.fnmr_at_tau(GEN, 0.9) == pytest.approx(1.0)  # all < 0.9


def test_empty_arrays_are_zero_rate():
    # No comparisons -> a rate of 0 (never a spurious accept/reject).
    assert M.fmr_at_tau(np.array([]), 0.5) == 0.0
    assert M.fnmr_at_tau(np.array([]), 0.5) == 0.0


# --------------------------------------------------------------------------- #
# Monotonicity -- the load-bearing shape guarantee.
# --------------------------------------------------------------------------- #
def test_fmr_is_nonincreasing_and_fnmr_nondecreasing_in_tau():
    taus, fmr, fnmr = M.det_points(GEN, IMP)
    # taus are returned in ascending order.
    assert np.all(np.diff(taus) >= 0)
    # cosine: raising tau can only reject more -> FMR down, FNMR up.
    assert np.all(np.diff(fmr) <= 1e-12)
    assert np.all(np.diff(fnmr) >= -1e-12)
    # DET spans the full corner-to-corner range.
    assert fmr[0] == pytest.approx(1.0) and fnmr[0] == pytest.approx(0.0)
    assert fmr[-1] == pytest.approx(0.0) and fnmr[-1] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# EER -- interpolated crossing, hand-computed reference.
# --------------------------------------------------------------------------- #
def test_eer_known_answer_interpolated_crossing():
    res = M.eer(GEN, IMP, bootstrap=0)
    value, tau, ci = res
    assert value == pytest.approx(0.30, abs=1e-9)
    assert tau == pytest.approx(0.55, abs=1e-9)
    # named-tuple access mirrors tuple access.
    assert res.value == pytest.approx(0.30)
    assert res.tau == pytest.approx(0.55)


def test_eer_zero_for_perfectly_separable_scores():
    gen = np.array([0.7, 0.8, 0.9])
    imp = np.array([0.1, 0.2, 0.3])
    value, tau, _ = M.eer(gen, imp, bootstrap=0)
    assert value == pytest.approx(0.0, abs=1e-12)


def test_overlap_has_larger_eer_than_separable():
    rng = np.random.default_rng(0)
    sep_gen = rng.normal(3.0, 0.3, 400)
    sep_imp = rng.normal(-3.0, 0.3, 400)
    ov_gen = rng.normal(0.5, 1.0, 400)
    ov_imp = rng.normal(-0.5, 1.0, 400)
    eer_sep = M.eer(sep_gen, sep_imp, bootstrap=0).value
    eer_ov = M.eer(ov_gen, ov_imp, bootstrap=0).value
    assert 0.0 <= eer_sep < 0.05
    assert eer_ov > eer_sep
    assert 0.0 <= eer_ov <= 0.5


def test_eer_bootstrap_ci_brackets_point_and_is_deterministic():
    r1 = M.eer(GEN, IMP, bootstrap=500, seed=12345)
    r2 = M.eer(GEN, IMP, bootstrap=500, seed=12345)
    # Fixed seed -> identical CI (R5 reproducibility).
    assert r1.ci == r2.ci
    lo, hi = r1.ci
    assert 0.0 <= lo <= hi <= 1.0
    # The point estimate lies inside its own bootstrap interval.
    assert lo - 1e-9 <= r1.value <= hi + 1e-9


# --------------------------------------------------------------------------- #
# Operating points -- both requirement-aligned readings.
# --------------------------------------------------------------------------- #
def test_fnmr_at_fmr_target_known_answer():
    # Smallest-FMR-bounding tau at FMR<=0.2 sits just above 0.5:
    #   FMR = {0.6}/5 = 0.2 (<= target);  FNMR = {0.4,0.5}/5 = 0.4.
    op = M.fnmr_at_fmr(GEN, IMP, fmr_target=0.2)
    assert op.rate == pytest.approx(0.4)
    assert M.fmr_at_tau(IMP, op.tau) <= 0.2 + 1e-12
    lo, hi = op.ci
    assert lo <= op.rate <= hi


def test_fmr_at_fnmr_target_known_answer():
    # Largest tau with FNMR<=0.2 is tau=0.5 (FNMR={0.4}/5=0.2);
    #   FMR at 0.5 = {0.5,0.6}/5 = 0.4.
    op = M.fmr_at_fnmr(GEN, IMP, fnmr_target=0.2)
    assert op.tau == pytest.approx(0.5)
    assert op.rate == pytest.approx(0.4)
    assert M.fnmr_at_tau(GEN, op.tau) <= 0.2 + 1e-12


def test_tau_at_fmr_matches_shipped_calibrator_logic():
    # metrics.tau_at_fmr must not silently diverge from the deployed calibrator
    # (calibrate._tau_at_fmr_cosine); drift here would move the shipped tau.
    rng = np.random.default_rng(7)
    imp = rng.uniform(-0.2, 0.5, 500)
    for target in (0.01, 0.05, 0.1, 0.2):
        assert M.tau_at_fmr(imp, target) == pytest.approx(
            calibrate._tau_at_fmr_cosine(imp, target), abs=0.0, rel=0.0
        )


# --------------------------------------------------------------------------- #
# Wilson score interval -- reuse the in-repo estimator + a known value.
# --------------------------------------------------------------------------- #
def test_wilson_delegates_to_calibrate_estimator():
    for x, n in [(0, 100), (7, 50), (10, 1000), (30, 3000)]:
        assert M.wilson(x, n) == calibrate.wilson_interval(x, n)


def test_wilson_zero_successes_rule_of_three_regime():
    # Wilson 95% CI for 0/100: lower bound clamps to 0, upper ~ 0.036994
    # (hand-derivable: upper = z^2/n / (1+z^2/n), with z=1.959963984540054).
    lo, hi = M.wilson(0, 100)
    assert lo == pytest.approx(0.0, abs=1e-12)
    assert hi == pytest.approx(0.036994, abs=1e-4)


# --------------------------------------------------------------------------- #
# k-of-n system-rate transform -- the voting compounding.
# --------------------------------------------------------------------------- #
def test_system_rate_kofn_known_answers():
    # Protocol worked example: per-frame FMR 1e-2, 3-of-5 -> ~9.85e-6.
    assert M.system_rate_kofn(0.01, 3, 5) == pytest.approx(9.8506e-6, abs=1e-9)
    # Fair-coin sanity: P(Binom(5,0.5) >= 3) = 16/32 = 0.5.
    assert M.system_rate_kofn(0.5, 3, 5) == pytest.approx(0.5, abs=1e-12)


def test_system_rate_kofn_edges():
    assert M.system_rate_kofn(0.0, 3, 5) == pytest.approx(0.0)
    assert M.system_rate_kofn(1.0, 3, 5) == pytest.approx(1.0)
    assert M.system_rate_kofn(0.3, 0, 5) == pytest.approx(1.0)   # >=0 votes always
    assert M.system_rate_kofn(0.3, 5, 5) == pytest.approx(0.3 ** 5)


def test_system_rate_below_frame_rate_for_kofn_majority():
    # k-of-n voting with k>1 makes the *system* accept rarer than a single frame
    # (this is why per-frame FMR is the honest headline; system rate is lower).
    p = 0.01
    assert M.system_rate_kofn(p, 3, 5) < p


def test_system_rate_kofn_validates_k_n():
    with pytest.raises(ValueError):
        M.system_rate_kofn(0.1, 6, 5)     # k > n
    with pytest.raises(ValueError):
        M.system_rate_kofn(0.1, 3, 0)     # n < 1
    with pytest.raises(ValueError):
        M.system_rate_kofn(1.5, 3, 5)     # p out of [0,1]


# --------------------------------------------------------------------------- #
# l2 (lower-is-better) direction -- the metric the store may carry.
# --------------------------------------------------------------------------- #
def test_l2_direction_flips_accept_rule():
    # For l2 distance, accept <=> s <= tau; genuine are SMALL, impostors LARGE.
    gen = np.array([0.1, 0.2, 0.3])       # small distances = owner
    imp = np.array([0.7, 0.8, 0.9])       # large distances = strangers
    # A tau between the clusters separates them perfectly -> EER 0.
    value, _, _ = M.eer(gen, imp, higher_is_better=False, bootstrap=0)
    assert value == pytest.approx(0.0, abs=1e-12)
    # FMR for l2 counts impostors at/below tau.
    assert M.fmr_at_tau(imp, 0.5, higher_is_better=False) == pytest.approx(0.0)   # none <= 0.5
    assert M.fmr_at_tau(imp, 0.75, higher_is_better=False) == pytest.approx(1 / 3)  # {0.7} <= 0.75
    assert M.fnmr_at_tau(gen, 0.15, higher_is_better=False) == pytest.approx(2 / 3)  # {0.2,0.3} > 0.15
