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
    """One frame's worth of liveness evidence (built by the daemon)."""

    landmarks: np.ndarray  # shape (5, 2)
    ts: float
    aligned: np.ndarray | None = None  # aligned BGR crop for passive PAD


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

    def score(self, aligned_bgr: np.ndarray | None) -> float | None:
        """Return a bona-fide (live) probability in [0, 1], or ``None``.

        The exact output mapping must match the provisioned MiniFASNet variant;
        this generic path softmaxes the network output and returns the "real"
        class probability (last class by convention). It is a documented
        Hardening hook -- swap in the model-specific decoding when the model is
        pinned.
        """
        if not self.available or self._net is None or aligned_bgr is None:
            return None
        try:
            blob = cv2.dnn.blobFromImage(
                aligned_bgr, scalefactor=1.0 / 255.0, size=(80, 80),
                mean=(0, 0, 0), swapRB=True, crop=False,
            )
            self._net.setInput(blob)
            out = np.asarray(self._net.forward(), dtype=np.float64).reshape(-1)
            if out.size == 0:
                return None
            exp = np.exp(out - out.max())
            probs = exp / exp.sum()
            return float(probs[-1])
        except Exception:
            return None


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
    ) -> None:
        self.mode = mode
        self.phase = phase
        self.challenge_timeout_s = int(challenge_timeout_s)
        self.turn_yaw_deg = float(turn_yaw_deg)
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
        best = 0.0
        scored = 0
        for obs in observations:
            s = self._pad.score(obs.aligned)
            if s is not None:
                scored += 1
                best = max(best, s)
        if scored == 0:
            return LivenessResult(False, "passive:no-crops", 0.0)
        return LivenessResult(best >= self._pad.threshold, "passive", best)

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
