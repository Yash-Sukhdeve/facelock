"""Matcher (C4) -- cosine similarity, per-owner tau, k-of-n voting.

Realizes REQ-F-07/08 and design section 3.2. The matcher compares a probe
embedding against the owner centroid, applies the calibrated threshold tau, and
requires ``k`` matching frames out of the last ``n`` probes (default 3-of-5)
before it will report the owner. A single frame never decides, which suppresses
single-frame false accepts (FM-02) and false rejects (FM-03).

Fail-closed rules (I-5):
  * No template / degenerate embedding / more than one face -> ``is_owner`` is
    always ``False``. The matcher NEVER returns owner on error.
  * ``tau`` is NEVER lowered at runtime (REQ-NF-22). There is no code path that
    relaxes the threshold to overcome bad light or drift.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

EMBEDDING_DIM = 128


@dataclass(frozen=True)
class MatchResult:
    """Outcome of one k-of-n verification step."""

    is_owner: bool
    score: float
    tau: float
    votes_k: int
    votes_n: int
    face_count: int


def verification_progress(votes_k: int, votes_n: int, k: int, n: int) -> float:
    """Real progress toward a k-of-n verification DECISION, in [0, 1].

    A decision is reached either when the positive votes hit ``k`` (accept) or the
    probe window fills to ``n`` without enough votes (reject). So the bar is the
    fraction closest to *either* outcome: ``max(votes_k/k, votes_n/n)``. It hits
    1.0 exactly when the matcher is about to decide -- accept OR reject -- so the
    UI never claims progress the verifier hasn't actually made. Display-only: this
    reads the matcher's counters and changes no security logic.
    """
    k = max(1, int(k))
    n = max(1, int(n))
    vk = max(0, int(votes_k))
    vn = max(0, int(votes_n))
    return min(1.0, max(vk / k, vn / n))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two vectors, robust to non-normalized / zero input.

    Returns a value in [-1, 1]; returns ``-1.0`` (worst) for a degenerate
    (zero-norm or non-finite) input so a NaN embedding can never score a match
    (fail-closed against FM-05/FM-11).
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.shape != b.shape or a.size == 0:
        return -1.0
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return -1.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return -1.0
    return float(np.dot(a, b) / (na * nb))


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.shape != b.shape or a.size == 0:
        return float("inf")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return float("inf")
    return float(np.linalg.norm(a - b))


class Matcher:
    """Stateful k-of-n verifier over a sliding window of probe frames."""

    def __init__(
        self,
        centroid: np.ndarray | None,
        tau: float,
        *,
        k: int = 3,
        n: int = 5,
        metric: str = "cosine",
        extra_templates: np.ndarray | None = None,
        pose_max: int = 5,
    ) -> None:
        if n < 1:
            raise ValueError("n (probe_frames) must be >= 1")
        if not (1 <= k <= n):
            raise ValueError("k (match_votes) must satisfy 1 <= k <= n")
        if metric not in ("cosine", "l2"):
            raise ValueError("metric must be 'cosine' or 'l2'")
        self.metric = metric
        self.tau = float(tau)
        self.k = int(k)
        self.n = int(n)
        self.pose_max = max(1, int(pose_max))
        self._window: deque[bool] = deque(maxlen=n)
        self.centroid: np.ndarray | None = None
        if centroid is not None:
            c = np.asarray(centroid, dtype=np.float32).reshape(-1)
            if c.shape != (EMBEDDING_DIM,) or not np.all(np.isfinite(c)):
                # Corrupt centroid -> behave as if no template (I-5, FM-10).
                self.centroid = None
            else:
                self.centroid = c
        # Multi-pose bank: the centroid PLUS a diverse subset of the enrolled
        # per-pose sample embeddings. A probe is scored against the best-matching
        # sub-template (max cosine), so an off-angle face clears tau via the pose
        # closest to it -- the dependency-light, RGB-only path to "easy" auth.
        # tau is NEVER lowered (REQ-NF-22); adding poses only widens acceptance
        # geometrically, so k-of-n voting still bounds false accepts (FM-02).
        self._bank: np.ndarray | None = self._build_bank(self.centroid, extra_templates)

    def _build_bank(
        self, centroid: np.ndarray | None, extras: np.ndarray | None
    ) -> np.ndarray | None:
        if centroid is None:
            return None
        rows: list[np.ndarray] = [centroid]
        if extras is not None:
            diverse = self._select_diverse(extras, self.pose_max)
            for row in diverse:
                rows.append(row.astype(np.float32))
        return np.stack(rows).astype(np.float32)

    @staticmethod
    def _select_diverse(samples: np.ndarray, k: int) -> np.ndarray:
        """Farthest-point subset of ``samples`` (up to ``k``) for pose coverage.

        Greedy max-min selection on the (L2-normalized) embeddings so the chosen
        sub-templates are spread across the enrolled poses instead of clustered.
        Pure + deterministic (seeded from row 0), so it is unit-testable.
        """
        S = np.asarray(samples, dtype=np.float32).reshape(-1, EMBEDDING_DIM)
        if S.shape[0] > 0:
            S = S[np.all(np.isfinite(S), axis=1)]  # drop degenerate rows (FM-05)
        m = S.shape[0]
        if m == 0 or k <= 0:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        if k >= m:
            return S
        chosen = [0]
        while len(chosen) < k:
            sims = S @ S[chosen].T          # cosine (rows are unit-norm)
            nearest = sims.max(axis=1)      # similarity to the closest chosen
            nearest[chosen] = np.inf        # never re-pick
            chosen.append(int(np.argmin(nearest)))  # farthest = least similar
        return S[chosen]

    @property
    def loaded(self) -> bool:
        return self._bank is not None

    @property
    def pose_count(self) -> int:
        """Number of sub-templates in the bank (1 = centroid only)."""
        return 0 if self._bank is None else int(self._bank.shape[0])

    def _score(self, embedding: np.ndarray) -> float:
        """Best score over the multi-pose bank (max cosine / min l2)."""
        assert self._bank is not None
        if self.metric == "cosine":
            best = -1.0
            for row in self._bank:
                s = cosine_similarity(embedding, row)
                if s > best:
                    best = s
            return best
        best = float("inf")
        for row in self._bank:
            d = l2_distance(embedding, row)
            if d < best:
                best = d
        return best

    def _passes(self, score: float) -> bool:
        # Higher-is-better for cosine; lower-is-better for l2 distance.
        if self.metric == "cosine":
            return score >= self.tau
        return score <= self.tau

    def passes(self, score: float) -> bool:
        """Public: does ``score`` clear tau (metric-aware direction)?

        Used for per-frame presence (owner_visible / stranger_visible) WITHOUT
        touching the k-of-n vote window.
        """
        return self._passes(score)

    def score_only(self, embedding: np.ndarray) -> float:
        """Return the raw metric score without updating the vote window."""
        if not self.loaded:
            return -1.0 if self.metric == "cosine" else float("inf")
        return self._score(embedding)

    def verify(self, embedding: np.ndarray | None, face_count: int) -> MatchResult:
        """Record one probe and return the current k-of-n verdict.

        A probe counts as a positive vote ONLY when: a template is loaded,
        exactly one face is present, the embedding is valid, and the score
        passes tau. Anything else records a negative vote (fail-closed).
        """
        worst = -1.0 if self.metric == "cosine" else float("inf")
        vote = False
        score = worst

        if self.loaded and face_count == 1 and embedding is not None:
            score = self._score(embedding)
            vote = self._passes(score)

        self._window.append(vote)
        votes_k = sum(1 for v in self._window if v)
        # Window-based decision AND current-frame single-face guard (FM-06):
        is_owner = votes_k >= self.k and face_count == 1 and vote
        return MatchResult(
            is_owner=is_owner,
            score=score,
            tau=self.tau,
            votes_k=votes_k,
            votes_n=len(self._window),
            face_count=face_count,
        )

    def reset(self) -> None:
        """Clear the vote window (called on every lock / grant transition)."""
        self._window.clear()

    def update_template(
        self,
        centroid: np.ndarray | None,
        tau: float,
        *,
        extra_templates: np.ndarray | None = None,
    ) -> None:
        """Swap in a new template (re-enroll) and reset the window."""
        self.__init__(centroid, tau, k=self.k, n=self.n, metric=self.metric,
                      extra_templates=extra_templates, pose_max=self.pose_max)
