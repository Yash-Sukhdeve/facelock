"""Deployed-matcher scoring for the eval harness (T2, the D-1 core).

Every accuracy number in this harness is measured by pushing genuine *and*
impostor probe embeddings through the **exact deployed matcher** -- a
:class:`facelock.matcher.Matcher` built from the enrolled template's centroid
**and** ``template.samples`` as ``extra_templates`` (the multi-pose bank), read
via :meth:`Matcher.score_only` (the max-over-bank cosine, or min-over-bank l2).
This closes the audit gap (protocol Gap B) whereby calibration scored impostors
against the *centroid only* while deployment scores against the *bank*.

Superset property (the security-correctness guarantee this module exists to
protect): the centroid is itself the first bank member, so for cosine

    S_deployed(x) = max_j cos(x, bank_j) >= cos(x, centroid) = S_centroid(x)

for every probe ``x`` -- hence ``FMR_deployed(tau) >= FMR_centroid(tau)`` on any
fixed probe set. Never re-measure against the centroid; that path *understates*
the deployed false-match rate.

Offline only: constructs an in-process ``Matcher`` and does arithmetic. No
camera, model file, daemon, systemd, or network access.
"""

from __future__ import annotations

import numpy as np

from ..matcher import EMBEDDING_DIM, Matcher, cosine_similarity, l2_distance

__all__ = ["build_matcher", "score_probes", "deployed_scores", "centroid_scores"]

# The daemon's default pose bank cap (config recognition.pose_max default = 5).
# T4 (report/CLI) will thread the live ``cfg.recognition.pose_max`` through here;
# this module stays I/O-free and takes it as a parameter.
DEFAULT_POSE_MAX = 5


def _as_probe_matrix(probes) -> np.ndarray:
    """Coerce probes to an ``(n, 128)`` float array (a lone vector -> one row)."""
    arr = np.asarray(probes, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr.reshape(-1, EMBEDDING_DIM)


def build_matcher(template, *, pose_max: int | None = None) -> Matcher:
    """Construct the exact deployed ``Matcher`` for ``template``.

    ``k=1, n=1``: the k-of-n vote window is irrelevant to per-comparison scoring
    (we only call :meth:`Matcher.score_only`), so we use the minimal legal
    window and keep the pose bank -- ``{centroid} u select_diverse(samples,
    pose_max)`` -- identical to production. The k-of-n compounding is applied
    separately in :func:`facelock.eval.metrics.system_rate_kofn`.
    """
    pm = DEFAULT_POSE_MAX if pose_max is None else int(pose_max)
    return Matcher(
        template.centroid,
        float(template.tau),
        k=1,
        n=1,
        metric=getattr(template, "metric", "cosine"),
        extra_templates=template.samples,
        pose_max=pm,
    )


def score_probes(template, probes, *, pose_max: int | None = None) -> np.ndarray:
    """Deployed max-over-bank scores for ``probes`` -> ``float64`` array ``(n,)``.

    Identical to what the daemon computes per frame: ``Matcher.score_only`` over
    the pose bank. This is the single scoring entry point the whole harness uses,
    guaranteeing FMR/FNMR are properties of the system *as shipped*.
    """
    matcher = build_matcher(template, pose_max=pose_max)
    P = _as_probe_matrix(probes)
    return np.array([matcher.score_only(P[i]) for i in range(P.shape[0])], dtype=np.float64)


# Protocol alias: the harness spec names this ``deployed_scores``.
deployed_scores = score_probes


def centroid_scores(template, probes) -> np.ndarray:
    """Centroid-ONLY scores -- the *understated* calibration path, for contrast.

    Computed with the matcher's own ``cosine_similarity`` / ``l2_distance`` on
    ``template.centroid`` so the comparison against :func:`score_probes` is
    exact: the deployed score is a max/min over a bank whose first member is this
    very centroid, so the superset inequality holds elementwise with no
    numerical slack. This function exists ONLY to demonstrate/guard the audit
    gap; it is never used to produce a reported rate.
    """
    metric = getattr(template, "metric", "cosine")
    centroid = template.centroid
    P = _as_probe_matrix(probes)
    if metric == "l2":
        return np.array([l2_distance(P[i], centroid) for i in range(P.shape[0])], dtype=np.float64)
    return np.array([cosine_similarity(P[i], centroid) for i in range(P.shape[0])], dtype=np.float64)
