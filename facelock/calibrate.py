"""Per-owner threshold (tau) calibration -- design section 3.2.

Given the accepted enrollment sample embeddings and a bundled impostor
embedding set, compute the operating threshold tau so that the Prototype
accuracy target is met (FMR <= fmr_target, then verify FNMR <= fnmr_target),
with the following invariants:

  * SEED / FLOOR: tau is never set below ``tau_floor`` (default 0.363, SFace's
    published cosine operating point). The impostor-derived threshold can only
    make tau *tighter*, never weaker (REQ-NF-22).
  * NEVER SILENTLY WEAK: if the target cannot be met, calibration warns and
    records the *achieved* operating point with confidence intervals; it never
    ships a weak tau silently (design 3.2 step 4, R1).
  * tau is calibrated at enrollment and NEVER auto-relaxed at runtime.

All computation is on embeddings only (no images, REQ-NF-13).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

EMBEDDING_DIM = 128
_Z95 = 1.959963984540054  # z for a two-sided 95% interval


@dataclass
class CalibrationResult:
    tau: float
    fmr_target: float
    fnmr_target: float
    fmr_measured: float
    fnmr_measured: float
    fmr_ci: tuple[float, float]
    fnmr_ci: tuple[float, float]
    impostor_n: int
    genuine_n: int
    tau_from_impostor: float
    tau_floor: float
    meets_target: bool
    warnings: list[str] = field(default_factory=list)
    calibrated_at: str = ""
    metric: str = "cosine"

    def as_meta(self) -> dict[str, Any]:
        """Serialisable calibration metadata for the template (§11.2)."""
        return {
            "fmr_target": self.fmr_target,
            "fnmr_target": self.fnmr_target,
            "fmr_measured": self.fmr_measured,
            "fnmr_measured": self.fnmr_measured,
            "fmr_ci": list(self.fmr_ci),
            "fnmr_ci": list(self.fnmr_ci),
            "impostor_n": self.impostor_n,
            "genuine_n": self.genuine_n,
            "tau_from_impostor": self.tau_from_impostor,
            "tau_floor": self.tau_floor,
            "meets_target": self.meets_target,
            "metric": self.metric,
            "calibrated_at": self.calibrated_at,
            "warnings": list(self.warnings),
        }


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


def centroid_of(samples: np.ndarray) -> np.ndarray:
    """L2-normalized mean embedding (the owner centroid, REQ-F-07)."""
    samples = _l2_normalize(samples)
    mean = samples.mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm == 0.0:
        return mean.astype(np.float32)
    return (mean / norm).astype(np.float32)


def wilson_interval(successes: int, trials: int, z: float = _Z95) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion."""
    if trials <= 0:
        return (0.0, 1.0)
    p = successes / trials
    denom = 1.0 + z * z / trials
    center = (p + z * z / (2 * trials)) / denom
    half = (z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _genuine_scores_loo(samples: np.ndarray, metric: str) -> np.ndarray:
    """Leave-one-out genuine scores: each sample vs the centroid of the rest."""
    samples = _l2_normalize(samples)
    n = samples.shape[0]
    scores = np.empty(n, dtype=np.float64)
    for i in range(n):
        rest = np.delete(samples, i, axis=0)
        c = centroid_of(rest).astype(np.float64)
        if metric == "cosine":
            scores[i] = float(np.dot(samples[i], c))  # both unit-norm
        else:
            scores[i] = float(np.linalg.norm(samples[i] - c))
    return scores


def _impostor_scores(centroid: np.ndarray, impostors: np.ndarray, metric: str) -> np.ndarray:
    centroid = np.asarray(centroid, dtype=np.float64).reshape(-1)
    impostors = _l2_normalize(impostors)
    if metric == "cosine":
        return impostors @ centroid
    return np.linalg.norm(impostors - centroid, axis=1)


def _tau_at_fmr_cosine(impostor_scores: np.ndarray, fmr_target: float) -> float:
    """Smallest cosine tau with impostor FMR <= target (higher-is-better)."""
    scores = np.sort(np.asarray(impostor_scores, dtype=np.float64))[::-1]
    m = scores.size
    allowed = int(math.floor(fmr_target * m))
    if allowed >= m:
        return float(scores[-1])  # everything allowed -> lowest impostor score
    # Threshold must exceed the (allowed)-th largest impostor score so that at
    # most `allowed` impostors sit at/above tau.
    boundary = float(scores[allowed])
    return math.nextafter(boundary, 1.0)


def _tau_at_fmr_l2(impostor_scores: np.ndarray, fmr_target: float) -> float:
    """Largest l2 tau with impostor FMR <= target (lower-is-better)."""
    scores = np.sort(np.asarray(impostor_scores, dtype=np.float64))
    m = scores.size
    allowed = int(math.floor(fmr_target * m))
    if allowed >= m:
        return float(scores[-1])
    boundary = float(scores[allowed])
    return math.nextafter(boundary, 0.0)


def calibrate(
    samples: np.ndarray,
    impostors: np.ndarray,
    *,
    fmr_target: float = 0.01,
    fnmr_target: float = 0.05,
    tau_floor: float = 0.363,
    metric: str = "cosine",
) -> CalibrationResult:
    """Calibrate tau from genuine (LOO) and impostor score distributions."""
    samples = np.asarray(samples, dtype=np.float64).reshape(-1, EMBEDDING_DIM)
    impostors = np.asarray(impostors, dtype=np.float64).reshape(-1, EMBEDDING_DIM)
    if samples.shape[0] < 2:
        raise ValueError("calibration needs >= 2 samples for leave-one-out")
    if impostors.shape[0] < 100:
        raise ValueError("calibration needs >= 100 impostor embeddings")

    centroid = centroid_of(samples)
    imp_scores = _impostor_scores(centroid, impostors, metric)
    gen_scores = _genuine_scores_loo(samples, metric)
    warnings: list[str] = []

    if metric == "cosine":
        tau_fmr = _tau_at_fmr_cosine(imp_scores, fmr_target)
        tau = max(tau_fmr, float(tau_floor))  # enforce safety floor
        if tau > tau_fmr:
            warnings.append(
                f"impostor-derived tau {tau_fmr:.4f} below floor {tau_floor:.4f}; "
                f"floor enforced (never ship a weaker-than-seed tau)"
            )
        fmr_measured = float(np.mean(imp_scores >= tau))
        fnmr_measured = float(np.mean(gen_scores < tau))
    else:  # l2 distance: lower is better
        tau_fmr = _tau_at_fmr_l2(imp_scores, fmr_target)
        tau = min(tau_fmr, float(tau_floor)) if tau_floor > 0 else tau_fmr
        fmr_measured = float(np.mean(imp_scores <= tau))
        fnmr_measured = float(np.mean(gen_scores > tau))

    n_imp = imp_scores.size
    n_gen = gen_scores.size
    fmr_ci = wilson_interval(int(round(fmr_measured * n_imp)), n_imp)
    fnmr_ci = wilson_interval(int(round(fnmr_measured * n_gen)), n_gen)

    meets = fmr_measured <= fmr_target and fnmr_measured <= fnmr_target
    if fmr_measured > fmr_target:
        warnings.append(
            f"achieved FMR {fmr_measured:.4f} exceeds target {fmr_target:.4f}"
        )
    if fnmr_measured > fnmr_target:
        warnings.append(
            f"achieved FNMR {fnmr_measured:.4f} exceeds target {fnmr_target:.4f}; "
            f"consider re-enrolling with more/better samples (tau NOT relaxed)"
        )

    return CalibrationResult(
        tau=float(tau),
        fmr_target=float(fmr_target),
        fnmr_target=float(fnmr_target),
        fmr_measured=fmr_measured,
        fnmr_measured=fnmr_measured,
        fmr_ci=fmr_ci,
        fnmr_ci=fnmr_ci,
        impostor_n=int(n_imp),
        genuine_n=int(n_gen),
        tau_from_impostor=float(tau_fmr),
        tau_floor=float(tau_floor),
        meets_target=bool(meets),
        warnings=warnings,
        calibrated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        metric=metric,
    )
