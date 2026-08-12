"""LivenessEngine (C5) -- anti-spoofing strategy object.

Realizes REQ-F-19 and design section 13. The engine is a *strategy* selected by
``liveness.mode`` so no perception/state-machine code differs across the
Prototype/Hardening boundary -- only the injected strategy differs.

Modes
-----
``off``     Prototype default. A documented NO-OP: returns ``passed=True``. This
            is the honest photo-spoofable prototype behaviour (REQ-F-17,
            ASM-04). Permitted ONLY when ``security.phase == P`` (config forbids
            ``off`` under phase H).
``turn``    FULLY IMPLEMENTED active challenge. A randomized head-turn computed
            purely from YuNet's 5 landmarks (yaw geometry). Defeats a *static*
            photo, which cannot produce the required frontal->turned motion
            (AC-F-19). No model, no extra dependency.
``blink``   Hardening hook: requires a tiny eye-state model. Without it, this
            mode is unavailable and fails closed. YuNet gives only eye *centre*
            points (not contours), so EAR-blink is impossible without a model
            (design 3.1 / ADR-7) -- hence a hook, not a stub.
``passive`` Hardening: MiniFASNet passive PAD over the aligned crop, run via
            ``cv2.dnn`` (already a dependency). A clean, runnable hook: it
            executes IF a compatible model is provisioned; if the model is
            absent it fails closed (``passed=False``). No stub.
``full``    Hardening: passive PAD AND the head-turn challenge -- both must pass.

Fail-closed (I-6): timeout / insufficient motion / model-unavailable in a mode
that needs it all yield ``passed=False``.
"""

from __future__ import annotations

import math
import secrets
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# MiniFASNet / Silent-Face passive-PAD constants (design 2.1-2.3, ADR-7).
#
# These are SECURITY-CRITICAL and pinned to the authoritative
# minivision-ai/Silent-Face-Anti-Spoofing inference recipe (Apache-2.0):
#   * two crop scales 2.7 (MiniFASNetV2) and 4.0 (MiniFASNetV1SE),
#   * 80x80 input, BGR order (NO channel swap), pixel scale 1/255, no mean,
#   * live/bona-fide class == softmax index 1 (classes 0 and 2 are attacks),
#   * two-scale fusion = mean of the per-scale live probabilities (pred[1]/2).
# A change to any of these silently disables anti-spoofing; the golden-vector
# tests in tests/test_pad_core.py exist to make such a regression fail loudly.
# --------------------------------------------------------------------------- #
PAD_INPUT_SIZE = 80
PAD_SCALE_V2 = 2.7  # MiniFASNetV2, 2.7_80x80
PAD_SCALE_V1SE = 4.0  # MiniFASNetV1SE, 4_0_0_80x80
PAD_SCALES = (PAD_SCALE_V2, PAD_SCALE_V1SE)
PAD_LIVE_INDEX = 1  # Silent-Face bona-fide class (NOT the last class)
PAD_CLASSES = 3


@dataclass
class Challenge:
    """A liveness challenge issued for one verification attempt."""

    type: str  # 'none' | 'turn' | 'blink' | 'passive' | 'full'
    nonce: str
    params: dict = field(default_factory=dict)


@dataclass
class LivenessResult:
    passed: bool
    method: str
    score: float


@dataclass
class LivenessObservation:
    """One frame's worth of liveness evidence (built by the daemon).

    ``pad_crops`` carries the bbox-context crops the passive MiniFASNet PAD
    consumes -- one 80x80 BGR crop per scale in :data:`PAD_SCALES`. This is a
    DIFFERENT distribution from the SFace similarity-warped ``aligned`` crop
    used for recognition (design 2.2, correction A1): PAD needs the bezel /
    paper-edge / background context that the recognition warp deliberately
    removes. ``aligned`` is retained only for the recognition path; passive PAD
    reads ``pad_crops`` exclusively.
    """

    landmarks: np.ndarray  # shape (5, 2)
    ts: float
    aligned: np.ndarray | None = None  # SFace aligned crop (recognition only)
    pad_crops: dict[float, np.ndarray] | None = None  # {scale: 80x80 BGR}


def estimate_yaw(landmarks: np.ndarray) -> float:
    """Estimate a normalized yaw proxy in [-1, 1] from 5 landmarks.

    Uses the horizontal position of the nose relative to the eye midpoint,
    normalized by half the inter-ocular distance. ~0 when frontal; magnitude
    grows as the head turns. This is a geometric proxy (enough to require
    motion and defeat a static photo), not a calibrated head-pose estimate.
    """
    landmarks = np.asarray(landmarks, dtype=np.float64).reshape(5, 2)
    right_eye, left_eye, nose = landmarks[0], landmarks[1], landmarks[2]
    eye_center_x = (right_eye[0] + left_eye[0]) / 2.0
    inter_ocular = abs(left_eye[0] - right_eye[0])
    half = inter_ocular / 2.0
    if half < 1e-6:
        return 0.0
    pos = (nose[0] - eye_center_x) / half
    return float(max(-1.0, min(1.0, pos)))


# --------------------------------------------------------------------------- #
# T1 -- PAD context cropper (pure, no I/O, no model).
#
# Reproduces Silent-Face's ``CropImage`` geometry (design 2.2, correction A1):
# enlarge the detector bbox by ``scale`` about its centre, clamp the scale so
# the enlarged box fits the frame, then SLIDE the box back inside the frame
# edges (never truncated asymmetrically) and resize to ``size`` x ``size``
# WITHOUT any geometric warp. This is the crop distribution MiniFASNet was
# trained on; feeding the SFace 112x112 warp instead would silently degrade the
# model (correction A1).
# --------------------------------------------------------------------------- #
def _pad_crop_box(
    src_w: int, src_h: int, bbox: tuple[float, float, float, float], scale: float
) -> tuple[int, int, int, int] | None:
    """Return the integer inclusive box ``(x1, y1, x2, y2)`` for a context crop.

    ``None`` for a degenerate bbox (non-positive width/height) so callers fail
    closed rather than cropping garbage.
    """
    try:
        x, y, box_w, box_h = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    except (TypeError, ValueError, IndexError):
        return None
    if box_w <= 0.0 or box_h <= 0.0 or src_w <= 1 or src_h <= 1:
        return None

    # Clamp the requested scale so the enlarged box cannot exceed the frame.
    scale = min((src_h - 1) / box_h, (src_w - 1) / box_w, float(scale))
    if scale <= 0.0:
        return None

    new_w = box_w * scale
    new_h = box_h * scale
    center_x = box_w / 2.0 + x
    center_y = box_h / 2.0 + y

    left_top_x = center_x - new_w / 2.0
    left_top_y = center_y - new_h / 2.0
    right_bottom_x = center_x + new_w / 2.0
    right_bottom_y = center_y + new_h / 2.0

    # Slide the whole box inside the frame (Silent-Face semantics), not truncate.
    if left_top_x < 0:
        right_bottom_x -= left_top_x
        left_top_x = 0.0
    if left_top_y < 0:
        right_bottom_y -= left_top_y
        left_top_y = 0.0
    if right_bottom_x > src_w - 1:
        left_top_x -= right_bottom_x - (src_w - 1)
        right_bottom_x = src_w - 1
    if right_bottom_y > src_h - 1:
        left_top_y -= right_bottom_y - (src_h - 1)
        right_bottom_y = src_h - 1

    x1 = int(max(0.0, left_top_x))
    y1 = int(max(0.0, left_top_y))
    x2 = int(min(float(src_w - 1), right_bottom_x))
    y2 = int(min(float(src_h - 1), right_bottom_y))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def pad_crop(
    frame_bgr: "np.ndarray | None",
    bbox: "tuple[float, float, float, float] | None",
    scale: float,
    size: int = PAD_INPUT_SIZE,
) -> "np.ndarray | None":
    """Return an ``size`` x ``size`` BGR context crop, or ``None`` (fail-closed).

    ``None`` is returned for a missing/empty frame, a degenerate/absent bbox, or
    any cv2 error -- never a partial or warped crop.
    """
    if frame_bgr is None or bbox is None or cv2 is None:
        return None
    if getattr(frame_bgr, "size", 0) == 0 or frame_bgr.ndim != 3:
        return None
    src_h, src_w = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
    box = _pad_crop_box(src_w, src_h, bbox, scale)
    if box is None:
        return None
    x1, y1, x2, y2 = box
    try:
        sub = frame_bgr[y1 : y2 + 1, x1 : x2 + 1]
        if sub.size == 0:
            return None
        return cv2.resize(sub, (int(size), int(size)))
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# T3 -- MiniFASNet preprocessing (pure). BGR, NO channel swap, scale 1/255, no
# mean (design 2.3, correction: swapRB must be False, not True).
# --------------------------------------------------------------------------- #
def pad_blob(crop_bgr: "np.ndarray | None", size: int = PAD_INPUT_SIZE) -> "np.ndarray | None":
    """Return the ``(1, 3, size, size)`` blob MiniFASNet expects, or ``None``.

    swapRB is **False** so the cv2 BGR channel order is preserved into the
    network (Silent-Face ``ToTensor`` keeps BGR); a True here would feed a
    channel-flipped image and silently degrade PAD.
    """
    if crop_bgr is None or cv2 is None:
        return None
    if getattr(crop_bgr, "size", 0) == 0:
        return None
    try:
        return cv2.dnn.blobFromImage(
            crop_bgr,
            scalefactor=1.0 / 255.0,
            size=(int(size), int(size)),
            mean=(0.0, 0.0, 0.0),
            swapRB=False,
            crop=False,
        )
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# T4 -- decode + two-scale fusion (the SECURITY GATE). live == softmax index 1;
# fuse two scales as the mean of the per-scale live probabilities (pred[1]/2).
# --------------------------------------------------------------------------- #
def pad_softmax(logits: "np.ndarray | list | tuple") -> np.ndarray:
    """Numerically stable softmax over a 1-D logit vector.

    Raises ``ValueError`` on an empty or non-finite input so the callers below
    can convert that to a fail-closed ``None``.
    """
    arr = np.asarray(logits, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("empty logits")
    if not np.all(np.isfinite(arr)):
        raise ValueError("non-finite logits")
    ex = np.exp(arr - arr.max())
    total = ex.sum()
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("degenerate softmax")
    return ex / total


def pad_live_prob(logits: "np.ndarray | list | tuple | None") -> float | None:
    """Single-scale bona-fide (live) probability = ``softmax(logits)[1]``.

    Enforces the Silent-Face **3-class** decode contract (I1, fail-closed):
    the head is exactly ``PAD_CLASSES`` == 3 outputs ``[attack, live, attack]``,
    so ANY output with a different class count is rejected with ``None`` --
    never decoded. The previous guard only rejected ``<= PAD_LIVE_INDEX`` (i.e.
    0- or 1-class) outputs, which let a wrong-head 2-class ``[0.1, 5.0]`` or
    4-class ``[0, 9, 0, 0]`` place a spurious ~0.99 at index 1 and read as
    'live' (a fail-OPEN). A missing/empty/non-finite output is likewise ``None``.
    """
    if logits is None:
        return None
    try:
        probs = pad_softmax(logits)
    except (ValueError, TypeError):
        return None
    # Exact 3-class contract: anything else cannot be trusted to address the
    # live class -> treat as spoof (deny). PAD_CLASSES pins the Silent-Face head.
    if probs.size != PAD_CLASSES:
        return None
    return float(probs[PAD_LIVE_INDEX])


def pad_fuse_live_prob(logits_by_scale: "object") -> float | None:
    """Two-scale fusion: the mean of the per-scale live probabilities.

    Equivalent to Silent-Face's ``(softmax(net_2.7) + softmax(net_4.0))[1] / 2``.
    Returns ``None`` (fail-closed) if the iterable is empty/absent or ANY scale
    yields a degraded output -- a partial fusion never grants.
    """
    if logits_by_scale is None:
        return None
    probs: list[float] = []
    for logits in logits_by_scale:
        p = pad_live_prob(logits)
        if p is None:
            return None
        probs.append(p)
    if not probs:
        return None
    return float(sum(probs) / len(probs))


# --------------------------------------------------------------------------- #
# T2 -- observation build (pure; injectable frame/detection, never a camera).
# --------------------------------------------------------------------------- #
def build_liveness_observation(
    frame_bgr: "np.ndarray | None",
    ts: float,
    detection: "object",
    mode: str,
    scales: "tuple[float, ...]" = PAD_SCALES,
) -> LivenessObservation:
    """Build one :class:`LivenessObservation` for the daemon's frame burst.

    For ``passive`` / ``full`` modes it attaches the bbox-context PAD crops (one
    per scale) built purely from the detector bbox -- it does NOT reach into the
    recognition aligner (``embedder.align`` stays recognition-only, design 2.2).
    For ``off`` / ``turn`` / ``blink`` no crops are built, so those modes are
    unaffected. A degenerate bbox simply yields no crops (fail-closed), never a
    crash.
    """
    pad_crops: dict[float, np.ndarray] | None = None
    if mode in ("passive", "full") and frame_bgr is not None:
        bbox = getattr(detection, "bbox", None)
        crops: dict[float, np.ndarray] = {}
        for scale in scales:
            crop = pad_crop(frame_bgr, bbox, scale)
            if crop is not None:
                crops[scale] = crop
        pad_crops = crops or None
    return LivenessObservation(
        landmarks=getattr(detection, "landmarks", None),
        ts=ts,
        pad_crops=pad_crops,
    )


class _PassivePAD:
    """MiniFASNet passive PAD hook (Hardening). Runnable if a model is present."""

    def __init__(self, model_path: str, threshold: float) -> None:
        self.threshold = float(threshold)
        self.available = False
        self._net = None
        path = Path(model_path) if model_path else None
        if path is not None and path.exists() and cv2 is not None:
            try:
                self._net = cv2.dnn.readNetFromONNX(str(path))
                self.available = True
            except Exception:
                self.available = False

    def score(self, crop_bgr: np.ndarray | None) -> float | None:
        """Return the single-scale bona-fide (live) probability, or ``None``.

        Uses the Silent-Face-exact preprocessing (:func:`pad_blob`: BGR, no
        channel swap, 1/255) and decode (:func:`pad_live_prob`: softmax index 1,
        NOT the last class). Any missing model, malformed output, or exception
        returns ``None`` -> the caller fails closed.
        """
        if not self.available or self._net is None or crop_bgr is None:
            return None
        try:
            blob = pad_blob(crop_bgr)
            if blob is None:
                return None
            self._net.setInput(blob)
            out = np.asarray(self._net.forward(), dtype=np.float64).reshape(-1)
            return pad_live_prob(out)
        except Exception:
            return None

    def score_crops(self, pad_crops: "dict[float, np.ndarray] | None") -> float | None:
        """Return the two-scale FUSED live probability, or ``None`` (deny).

        Runs the net over BOTH calibrated scale crops and fuses as the mean of
        the per-scale live probs (Silent-Face ``(pred_2.7 + pred_4.0)[1] / 2``).
        BOTH :data:`PAD_SCALES` crops are REQUIRED (M1, fail-closed): a
        single-scale fusion is uncalibrated -- the pinned operating point is the
        two-scale mean, so a lone scale must never grant. If either required
        scale is absent/degraded, or there are no crops, returns ``None`` so the
        verify path fails closed. (Two distinct per-scale nets are wired in task
        T5; this single-net path already applies the correct fusion math.)
        """
        if not self.available or self._net is None or not pad_crops:
            return None
        # Require BOTH calibrated scales before any live verdict (M1).
        if not set(pad_crops) >= set(PAD_SCALES):
            return None
        # Fuse over exactly the two calibrated scales (ignore any stray key),
        # in a fixed order so the mean is deterministic.
        per_scale = [self.score(pad_crops[scale]) for scale in sorted(PAD_SCALES)]
        if any(p is None for p in per_scale):
            return None
        return float(sum(per_scale) / len(per_scale))


class LivenessEngine:
    """Selects and runs the configured liveness strategy."""

    def __init__(
        self,
        *,
        mode: str = "off",
        phase: str = "P",
        challenge_timeout_s: int = 4,
        turn_yaw_deg: float = 15.0,
        pad_model_path: str = "",
        pad_threshold: float = 0.5,
        pad_min_live_frames: int = 3,
    ) -> None:
        self.mode = mode
        self.phase = phase
        self.challenge_timeout_s = int(challenge_timeout_s)
        self.turn_yaw_deg = float(turn_yaw_deg)
        # Passive-PAD temporal quorum: require this many frames in the burst to
        # INDEPENDENTLY clear the PAD threshold (k-of-n, mirrors recognition's
        # match_votes). >= 1; a burst with fewer valid frames can never satisfy
        # it and fails closed (I2). Replaces the old max-across-frames aggregate.
        self.pad_min_live_frames = max(1, int(pad_min_live_frames))
        # Normalized asymmetry threshold from the requested turn angle.
        self.pos_threshold = math.sin(math.radians(self.turn_yaw_deg))
        self.frontal_tol = 0.15
        self._pad = None
        if mode in ("passive", "full"):
            self._pad = _PassivePAD(pad_model_path, pad_threshold)

    @property
    def requires_challenge(self) -> bool:
        return self.mode in ("turn", "blink", "full")

    @property
    def requires_frames(self) -> bool:
        """Whether the daemon must collect a burst of frames for this mode."""
        return self.mode != "off"

    def new_challenge(self, rng: "secrets.SystemRandom | None" = None) -> Challenge:
        """Issue a fresh, randomized challenge bound to a nonce."""
        rng = rng or secrets.SystemRandom()
        nonce = secrets.token_hex(8)
        if self.mode in ("turn", "full"):
            direction = rng.choice(("left", "right"))
            return Challenge(type=self.mode, nonce=nonce, params={"direction": direction})
        if self.mode == "blink":
            return Challenge(type="blink", nonce=nonce, params={})
        if self.mode == "passive":
            return Challenge(type="passive", nonce=nonce, params={})
        return Challenge(type="none", nonce=nonce, params={})

    def _check_turn(self, observations: list[LivenessObservation], direction: str) -> LivenessResult:
        yaws = [estimate_yaw(o.landmarks) for o in observations if o.landmarks is not None]
        if len(yaws) < 2:
            return LivenessResult(False, "turn:insufficient-frames", 0.0)
        arr = np.asarray(yaws)
        delta = float(arr.max() - arr.min())
        frontal_present = bool((np.abs(arr) < self.frontal_tol).any())
        # Direction binding (best-effort): the extreme yaw must lie in the
        # requested direction. Sign convention: positive yaw = nose shifted
        # toward the (image) left-eye side.
        if direction == "left":
            directional = float(arr.max()) >= self.pos_threshold * 0.6
        else:
            directional = float(arr.min()) <= -self.pos_threshold * 0.6
        passed = delta >= self.pos_threshold and frontal_present and directional
        return LivenessResult(passed, f"turn:{direction}", delta)

    def _check_passive(self, observations: list[LivenessObservation]) -> LivenessResult:
        if self._pad is None or not self._pad.available:
            # Hardening mode selected but model not provisioned -> fail closed.
            return LivenessResult(False, "passive:model-unavailable", 0.0)
        # I2: aggregate per-frame fused-live scores with a k-of-n QUORUM, not
        # ``max``. ``max`` is the most permissive temporal estimator -- a single
        # lucky replay/glare frame above threshold would pass = fail-OPEN on the
        # time axis. Instead require ``pad_min_live_frames`` (k) frames to
        # INDEPENDENTLY clear the threshold. Equivalently, the k-th largest
        # per-frame score must be >= threshold; if fewer than k frames scored at
        # all, the quorum is unreachable -> deny.
        scores: list[float] = []
        for obs in observations:
            s = self._pad.score_crops(getattr(obs, "pad_crops", None))
            if s is not None:
                scores.append(float(s))
        if not scores:
            # No usable crops in any frame -> fail closed (design 2.4).
            return LivenessResult(False, "passive:no-crops", 0.0)
        k = self.pad_min_live_frames
        if len(scores) < k:
            # Fewer valid frames than the required live quorum -> fail closed.
            return LivenessResult(False, "passive:insufficient-live-frames", 0.0)
        # k-th largest per-frame score: >= threshold IFF at least k frames pass.
        quorum_score = sorted(scores, reverse=True)[k - 1]
        return LivenessResult(quorum_score >= self._pad.threshold, "passive", quorum_score)

    def check(
        self,
        observations: list[LivenessObservation],
        challenge: Challenge,
    ) -> LivenessResult:
        """Evaluate liveness for one attempt. Never raises (fail-closed)."""
        if self.mode == "off":
            if self.phase == "P":
                # Documented prototype no-op (photo-spoofable, REQ-F-17).
                return LivenessResult(True, "off:prototype-noop", 1.0)
            # Never reachable via config (phase H forbids off) -> fail closed.
            return LivenessResult(False, "off:forbidden-in-hardening", 0.0)

        try:
            if self.mode == "turn":
                return self._check_turn(observations, challenge.params.get("direction", "left"))
            if self.mode == "passive":
                return self._check_passive(observations)
            if self.mode == "blink":
                # Hook: requires an eye-state model not shipped in the prototype.
                return LivenessResult(False, "blink:model-required", 0.0)
            if self.mode == "full":
                turn = self._check_turn(observations, challenge.params.get("direction", "left"))
                passive = self._check_passive(observations)
                passed = turn.passed and passive.passed
                return LivenessResult(
                    passed, f"full[{turn.method}+{passive.method}]",
                    min(turn.score, passive.score),
                )
        except Exception:
            return LivenessResult(False, "error", 0.0)
        return LivenessResult(False, "unknown-mode", 0.0)
