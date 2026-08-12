"""Pure verification-error metrics for the facelock eval harness (T1).

No cv2, no models, no file/network I/O -- every function operates on plain
NumPy score arrays so the correctness-critical maths is unit-testable offline.

Definitions follow ISO/IEC 19795-1:2021 §9 for single-comparison (1:1) error
rates. Cosine convention (the deployed default): higher score = more similar,
``accept <=> s >= tau`` (matches ``facelock.matcher.Matcher._passes``). An
``l2`` distance template flips this to ``accept <=> s <= tau``; pass
``higher_is_better=False`` for that path.

    * FMR(tau)  = fraction of IMPOSTOR comparisons accepted   (false match)
    * FNMR(tau) = fraction of GENUINE comparisons rejected     (false non-match)

Interval estimation reuses the in-repo Wilson score interval
(:func:`facelock.calibrate.wilson_interval`, Wilson 1927 [C2]); EER carries a
nonparametric bootstrap CI (seeded, R5). The k-of-n voting transform maps a
per-frame rate to the per-decision (system) rate under majority voting.

References
----------
[C1] ISO/IEC 19795-1:2021, Biometric performance testing and reporting, §9.
[C2] Wilson, E. B. (1927). Probable Inference, the Law of Succession, and
     Statistical Inference. JASA 22(158): 209-212.
[C3] Brown, Cai & DasGupta (2001). Interval Estimation for a Binomial
     Proportion. Statistical Science 16(2): 101-133.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from .. import calibrate as _calibrate

__all__ = [
    "EERResult",
    "OperatingPoint",
    "higher_is_better_for_metric",
    "fmr_at_tau",
    "fnmr_at_tau",
    "fmr_count_at_tau",
    "fnmr_count_at_tau",
    "sweep_thresholds",
    "det_points",
    "tau_at_fmr",
    "tau_at_fnmr",
    "fnmr_at_fmr",
    "fmr_at_fnmr",
    "eer",
    "wilson",
    "rate_with_ci",
    "system_rate_kofn",
]

_Z95 = _calibrate._Z95  # 1.959963984540054, shared with the shipped calibrator


class EERResult(NamedTuple):
    """Equal-error-rate estimate plus its interpolation threshold and CI."""

    value: float
    tau: float
    ci: tuple[float, float]


class OperatingPoint(NamedTuple):
    """A requirement-aligned operating point: the paired rate at a chosen tau."""

    tau: float
    rate: float
    ci: tuple[float, float]


def higher_is_better_for_metric(metric: str) -> bool:
    """Direction of the score for a template ``metric`` (cosine up, l2 down)."""
    if metric == "cosine":
        return True
    if metric == "l2":
        return False
    raise ValueError(f"unknown metric {metric!r} (expected 'cosine' or 'l2')")


def _as_1d(scores) -> np.ndarray:
    return np.asarray(scores, dtype=np.float64).reshape(-1)


# --------------------------------------------------------------------------- #
# Per-comparison rates at a fixed threshold.
# --------------------------------------------------------------------------- #
def fmr_count_at_tau(impostor_scores, tau: float, higher_is_better: bool = True) -> int:
    """Number of impostor comparisons ACCEPTED at ``tau`` (the raw false-match count).

    Carrying the integer count (rather than reconstructing it as
    ``round(rate * n)``) is what lets the Wilson CI use the exact success count,
    avoiding a float round-trip that can be off by one at large ``n`` (the A1
    minor fix).
    """
    imp = _as_1d(impostor_scores)
    if imp.size == 0:
        return 0
    accept = imp >= tau if higher_is_better else imp <= tau
    return int(np.count_nonzero(accept))


def fnmr_count_at_tau(genuine_scores, tau: float, higher_is_better: bool = True) -> int:
    """Number of genuine comparisons REJECTED at ``tau`` (the raw false-non-match count)."""
    gen = _as_1d(genuine_scores)
    if gen.size == 0:
        return 0
    reject = gen < tau if higher_is_better else gen > tau
    return int(np.count_nonzero(reject))


def fmr_at_tau(impostor_scores, tau: float, higher_is_better: bool = True) -> float:
    """False Match Rate at ``tau``: fraction of impostor scores that ACCEPT.

    Empty input -> 0.0 (no comparison can be a false match).
    """
    imp = _as_1d(impostor_scores)
    if imp.size == 0:
        return 0.0
    return fmr_count_at_tau(imp, tau, higher_is_better) / float(imp.size)


def fnmr_at_tau(genuine_scores, tau: float, higher_is_better: bool = True) -> float:
    """False Non-Match Rate at ``tau``: fraction of genuine scores that REJECT.

    Empty input -> 0.0 (no comparison can be a false non-match).
    """
    gen = _as_1d(genuine_scores)
    if gen.size == 0:
        return 0.0
    return fnmr_count_at_tau(gen, tau, higher_is_better) / float(gen.size)


# --------------------------------------------------------------------------- #
# Threshold sweep + DET/ROC points.
# --------------------------------------------------------------------------- #
def sweep_thresholds(
    genuine_scores, impostor_scores, higher_is_better: bool = True
) -> np.ndarray:
    """Ascending candidate thresholds spanning both score distributions.

    The observed distinct scores, bracketed by two sentinels just past the
    global min/max, so the DET curve reaches both corners (FMR 1 / FNMR 0 at the
    permissive end, FMR 0 / FNMR 1 at the strict end) and any FMR=FNMR crossing
    is guaranteed to be interior.
    """
    gen = _as_1d(genuine_scores)
    imp = _as_1d(impostor_scores)
    both = np.concatenate([gen, imp]) if (gen.size or imp.size) else np.array([0.0])
    taus = np.unique(both)
    lo = math.nextafter(float(taus[0]), -math.inf)
    hi = math.nextafter(float(taus[-1]), math.inf)
    return np.concatenate([[lo], taus, [hi]])


def det_points(
    genuine_scores, impostor_scores, higher_is_better: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(taus, fmr, fnmr)`` over the threshold sweep, tau ascending.

    ``fmr`` is monotone non-increasing and ``fnmr`` monotone non-decreasing in
    ``tau`` for the cosine (higher-is-better) convention.
    """
    taus = sweep_thresholds(genuine_scores, impostor_scores, higher_is_better)
    imp = _as_1d(impostor_scores)
    gen = _as_1d(genuine_scores)
    fmr = np.array([fmr_at_tau(imp, t, higher_is_better) for t in taus])
    fnmr = np.array([fnmr_at_tau(gen, t, higher_is_better) for t in taus])
    return taus, fmr, fnmr


# --------------------------------------------------------------------------- #
# Threshold selection for the two requirement-aligned operating points.
# --------------------------------------------------------------------------- #
def tau_at_fmr(impostor_scores, fmr_target: float, higher_is_better: bool = True) -> float:
    """Tightest ``tau`` whose impostor FMR is <= ``fmr_target``.

    For cosine this is the *smallest* accepting threshold (highest tau); the
    logic mirrors the shipped calibrator (:func:`calibrate._tau_at_fmr_cosine`)
    so the eval harness and the deployed tau never diverge.
    """
    if higher_is_better:
        return _calibrate._tau_at_fmr_cosine(_as_1d(impostor_scores), fmr_target)
    return _calibrate._tau_at_fmr_l2(_as_1d(impostor_scores), fmr_target)


def tau_at_fnmr(genuine_scores, fnmr_target: float, higher_is_better: bool = True) -> float:
    """``tau`` giving the LOWEST FMR while keeping genuine FNMR <= ``fnmr_target``.

    Cosine: the largest tau with ``mean(gen < tau) <= target`` (raising tau
    lowers FMR but raises FNMR, so we take the largest tau still honouring the
    FNMR bound). ``+inf`` when the target permits rejecting every genuine probe.
    """
    gen = np.sort(_as_1d(genuine_scores))
    n = gen.size
    if n == 0:
        return math.inf if higher_is_better else -math.inf
    allowed = int(math.floor(fnmr_target * n))  # max genuine we may reject
    if allowed >= n:
        return math.inf if higher_is_better else -math.inf
    if higher_is_better:
        # FNMR = mean(gen < tau); largest tau with at most `allowed` below it is
        # the (allowed)-th smallest genuine score (genuine at tau still accept).
        return float(gen[allowed])
    # l2: FNMR = mean(gen > tau); smallest tau with at most `allowed` above it is
    # the (allowed)-th largest genuine score.
    return float(gen[n - 1 - allowed])


def fnmr_at_fmr(
    genuine_scores, impostor_scores, fmr_target: float, higher_is_better: bool = True
) -> OperatingPoint:
    """FNMR (with Wilson CI) at the tau that bounds FMR <= ``fmr_target``.

    The headline REQ-NF-10 reading: false-reject rate at the security FMR bound.
    """
    gen = _as_1d(genuine_scores)
    tau = tau_at_fmr(impostor_scores, fmr_target, higher_is_better)
    # A1 fix: feed the raw reject COUNT to Wilson (never round(rate * n), which
    # can drift by one at large n and mis-state the interval).
    count = fnmr_count_at_tau(gen, tau, higher_is_better)
    rate = 0.0 if gen.size == 0 else count / float(gen.size)
    ci = wilson(count, int(gen.size))
    return OperatingPoint(tau=float(tau), rate=float(rate), ci=ci)


def fmr_at_fnmr(
    genuine_scores, impostor_scores, fnmr_target: float, higher_is_better: bool = True
) -> OperatingPoint:
    """FMR (with Wilson CI) at the tau that bounds FNMR <= ``fnmr_target``.

    The ASM-05 convenience reading: false-match rate at the usability FNMR bound.
    """
    imp = _as_1d(impostor_scores)
    tau = tau_at_fnmr(genuine_scores, fnmr_target, higher_is_better)
    # A1 fix: exact accept COUNT into Wilson (not round(rate * n)).
    count = fmr_count_at_tau(imp, tau, higher_is_better)
    rate = 0.0 if imp.size == 0 else count / float(imp.size)
    ci = wilson(count, int(imp.size))
    return OperatingPoint(tau=float(tau), rate=float(rate), ci=ci)


# --------------------------------------------------------------------------- #
# Equal Error Rate -- interpolated crossing + bootstrap CI.
# --------------------------------------------------------------------------- #
def _eer_point(gen: np.ndarray, imp: np.ndarray, higher_is_better: bool) -> tuple[float, float]:
    """Interpolated EER and its threshold: the first FMR=FNMR crossing.

    ``d(tau) = FMR(tau) - FNMR(tau)`` is monotone across the sweep; the crossing
    is found where ``d`` changes sign and the rates are linearly interpolated
    between the two bracketing thresholds. Falls back to the sweep point
    minimising ``max(FMR, FNMR)`` if no sign change occurs (e.g. degenerate
    input).
    """
    taus, fmr, fnmr = det_points(gen, imp, higher_is_better)
    d = fmr - fnmr
    for i in range(d.size - 1):
        a, b = d[i], d[i + 1]
        if (a >= 0.0 >= b) or (a <= 0.0 <= b):
            denom = a - b
            alpha = 0.0 if denom == 0.0 else float(a / denom)
            alpha = min(1.0, max(0.0, alpha))
            eer_fmr = fmr[i] + alpha * (fmr[i + 1] - fmr[i])
            eer_fnmr = fnmr[i] + alpha * (fnmr[i + 1] - fnmr[i])
            tau = taus[i] + alpha * (taus[i + 1] - taus[i])
            return 0.5 * (eer_fmr + eer_fnmr), float(tau)
    idx = int(np.argmin(np.maximum(fmr, fnmr)))
    return 0.5 * (fmr[idx] + fnmr[idx]), float(taus[idx])


def eer(
    genuine_scores,
    impostor_scores,
    higher_is_better: bool = True,
    *,
    bootstrap: int = 2000,
    seed: int = 20260730,
) -> EERResult:
    """Equal Error Rate with an interpolated crossing and a bootstrap 95% CI.

    ``bootstrap`` resamples (with replacement) both score arrays ``B`` times,
    recomputes EER, and reports the 2.5/97.5 percentiles; ``bootstrap=0``
    returns the point estimate with a degenerate ``(value, value)`` CI. The RNG
    is seeded for exact reproducibility (R5).
    """
    gen = _as_1d(genuine_scores)
    imp = _as_1d(impostor_scores)
    value, tau = _eer_point(gen, imp, higher_is_better)
    if bootstrap <= 0 or gen.size == 0 or imp.size == 0:
        return EERResult(value=float(value), tau=float(tau), ci=(float(value), float(value)))
    rng = np.random.default_rng(seed)
    boots = np.empty(int(bootstrap), dtype=np.float64)
    for b in range(int(bootstrap)):
        g = gen[rng.integers(0, gen.size, gen.size)]
        m = imp[rng.integers(0, imp.size, imp.size)]
        boots[b], _ = _eer_point(g, m, higher_is_better)
    lo = float(np.percentile(boots, 2.5))
    hi = float(np.percentile(boots, 97.5))
    return EERResult(value=float(value), tau=float(tau), ci=(lo, hi))


# --------------------------------------------------------------------------- #
# Interval estimation.
# --------------------------------------------------------------------------- #
def wilson(successes: int, trials: int, z: float = _Z95) -> tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion.

    Thin delegation to the shipped estimator
    (:func:`facelock.calibrate.wilson_interval`) so the harness and the
    calibrator share one, verified implementation (Wilson 1927 [C2]).
    """
    return _calibrate.wilson_interval(int(successes), int(trials), z)


def rate_with_ci(count: int, total: int, z: float = _Z95) -> tuple[float, tuple[float, float]]:
    """Convenience: a proportion and its Wilson CI from a raw count/total."""
    rate = 0.0 if total <= 0 else float(count) / float(total)
    return rate, wilson(int(count), int(total), z)


# --------------------------------------------------------------------------- #
# k-of-n voting transform (per-frame rate -> per-decision system rate).
# --------------------------------------------------------------------------- #
def system_rate_kofn(p_frame: float, k: int, n: int) -> float:
    """System rate under k-of-n voting: ``P(Binomial(n, p_frame) >= k)``.

    Treats frames as independent (protocol §1): the matcher accepts only when at
    least ``k`` of the last ``n`` frames pass tau, so a per-frame FMR of
    ``p_frame`` compounds to this per-decision system FMR (e.g. p=1e-2, k=3, n=5
    -> ~9.85e-6). Exact binomial-tail summation via :func:`math.comb`.
    """
    k = int(k)
    n = int(n)
    if n < 1:
        raise ValueError("n (probe_frames) must be >= 1")
    if not (0 <= k <= n):
        raise ValueError("k (match_votes) must satisfy 0 <= k <= n")
    if not (0.0 <= p_frame <= 1.0):
        raise ValueError("p_frame must lie in [0, 1]")
    if k == 0:
        return 1.0
    q = 1.0 - p_frame
    terms = [
        math.comb(n, j) * (p_frame ** j) * (q ** (n - j))
        for j in range(k, n + 1)
    ]
    return float(min(1.0, math.fsum(terms)))
