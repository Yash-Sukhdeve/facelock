"""LivenessEngine tests (C5, REQ-F-19). Geometry + fail-closed hooks."""

from __future__ import annotations

import numpy as np

from facelock.liveness import (
    Challenge,
    LivenessEngine,
    LivenessObservation,
    estimate_yaw,
)


def landmarks(pos: float):
    """5 landmarks with the nose offset by ``pos`` * half-inter-ocular."""
    right_eye = (30.0, 50.0)
    left_eye = (70.0, 50.0)
    half = 20.0
    nose = (50.0 + pos * half, 70.0)
    return np.array([right_eye, left_eye, nose, (40.0, 90.0), (60.0, 90.0)], dtype=np.float32)


def test_estimate_yaw_frontal_zero():
    assert abs(estimate_yaw(landmarks(0.0))) < 1e-6


def test_estimate_yaw_sign():
    assert estimate_yaw(landmarks(0.5)) > 0
    assert estimate_yaw(landmarks(-0.5)) < 0


def test_off_mode_prototype_passes():
    eng = LivenessEngine(mode="off", phase="P")
    res = eng.check([], Challenge("none", "n"))
    assert res.passed and "noop" in res.method
    assert not eng.requires_frames


def test_off_mode_hardening_fails_closed():
    eng = LivenessEngine(mode="off", phase="H")
    assert not eng.check([], Challenge("none", "n")).passed


def test_turn_pass_with_motion():
    eng = LivenessEngine(mode="turn", phase="P", turn_yaw_deg=15.0)
    obs = [
        LivenessObservation(landmarks(0.0), 0.0),
        LivenessObservation(landmarks(0.2), 0.1),
        LivenessObservation(landmarks(0.5), 0.2),
    ]
    res = eng.check(obs, Challenge("turn", "n", {"direction": "left"}))
    assert res.passed


def test_turn_fails_for_static_photo():
    eng = LivenessEngine(mode="turn", phase="P", turn_yaw_deg=15.0)
    obs = [LivenessObservation(landmarks(0.3), t / 10) for t in range(5)]  # constant
    res = eng.check(obs, Challenge("turn", "n", {"direction": "left"}))
    assert not res.passed


def test_turn_needs_frames():
    eng = LivenessEngine(mode="turn", phase="P")
    res = eng.check([LivenessObservation(landmarks(0.0), 0.0)], Challenge("turn", "n", {"direction": "left"}))
    assert not res.passed


def test_passive_unavailable_fails_closed():
    eng = LivenessEngine(mode="passive", phase="H", pad_model_path="/nonexistent.onnx")
    res = eng.check([LivenessObservation(landmarks(0.0), 0.0)], Challenge("passive", "n"))
    assert not res.passed and "unavailable" in res.method
    assert eng.requires_frames


def test_blink_hook_fails_without_model():
    eng = LivenessEngine(mode="blink", phase="H")
    res = eng.check([LivenessObservation(landmarks(0.0), 0.0)], Challenge("blink", "n"))
    assert not res.passed and "model-required" in res.method


def test_new_challenge_binds_direction_and_nonce():
    eng = LivenessEngine(mode="turn", phase="P")
    ch = eng.new_challenge()
    assert ch.type == "turn" and ch.params["direction"] in ("left", "right") and ch.nonce


# ===================== I2: passive PAD temporal quorum ===================== #
# The passive path was previously UNTESTED (no test built an *available* PAD)
# and aggregated per-frame scores with ``max`` -- the most permissive temporal
# estimator, so a single lucky replay/glare frame passed = fail-OPEN on the time
# axis. These tests build a MOCK-available PAD (no model, no camera) and pin the
# k-of-n live quorum that replaced ``max``.
class _FakePAD:
    """A mock-*available* MiniFASNet PAD returning a scripted per-frame fused
    live probability (a ``None`` entry == no usable crop that frame)."""

    def __init__(self, threshold: float, scores: list) -> None:
        self.available = True
        self.threshold = float(threshold)
        self._scores = list(scores)
        self._i = 0

    def score_crops(self, pad_crops):  # mirrors _PassivePAD.score_crops
        s = self._scores[self._i]
        self._i += 1
        return s


def _passive_engine(scores, *, k=3, threshold=0.5):
    """A passive LivenessEngine wired to a mock PAD scripted with ``scores``,
    plus one observation per scored frame."""
    eng = LivenessEngine(
        mode="passive", phase="H", pad_model_path="/nonexistent.onnx",
        pad_threshold=threshold, pad_min_live_frames=k,
    )
    eng._pad = _FakePAD(threshold, scores)
    obs = [
        LivenessObservation(landmarks(0.0), float(i), pad_crops={2.7: None, 4.0: None})
        for i in range(len(scores))
    ]
    return eng, obs


def test_passive_quorum_denies_single_lucky_frame():
    """I2 (SECURITY -- the critical case): a burst where a SINGLE frame beats tau
    but the k-of-n live quorum is not met MUST deny. The old ``max`` aggregation
    would fail OPEN on exactly this one replay/glare frame."""
    eng, obs = _passive_engine([0.95, 0.10, 0.12, 0.08, 0.11], k=3)
    res = eng.check(obs, Challenge("passive", "n"))
    assert not res.passed          # 1 live frame < 3-of-n quorum -> DENY
    assert res.method == "passive"  # decision path reached (not model-unavailable)


def test_passive_quorum_passes_when_enough_live_frames():
    """I2: at least k frames independently clear tau -> quorum met -> PASS."""
    eng, obs = _passive_engine([0.90, 0.88, 0.20, 0.91, 0.15], k=3)
    res = eng.check(obs, Challenge("passive", "n"))
    assert res.passed and res.method == "passive"


def test_passive_zero_valid_frames_denies():
    """I2: no frame yields a usable crop -> fail closed (scored == 0)."""
    eng, obs = _passive_engine([None, None, None], k=3)
    res = eng.check(obs, Challenge("passive", "n"))
    assert not res.passed and "no-crops" in res.method


def test_passive_insufficient_valid_frames_denies():
    """I2: only 2 frames score at all -- both strongly live -- but a k=3 quorum
    can never be reached, so DENY (fail-closed on the time axis, not a max)."""
    eng, obs = _passive_engine([0.99, 0.99, None, None, None], k=3)
    res = eng.check(obs, Challenge("passive", "n"))
    assert not res.passed


def test_passive_quorum_boundary_exactly_k_passes():
    """I2: exactly k frames at tau meet the quorum (k-th largest >= tau)."""
    eng, obs = _passive_engine([0.50, 0.50, 0.50, 0.10, 0.10], k=3, threshold=0.5)
    res = eng.check(obs, Challenge("passive", "n"))
    assert res.passed


def test_passive_full_mode_requires_passive_quorum():
    """I2: in ``full`` mode a single lucky passive frame must NOT satisfy the
    passive leg (turn is independently evaluated); the whole check denies."""
    eng = LivenessEngine(
        mode="full", phase="H", pad_model_path="/nonexistent.onnx",
        pad_threshold=0.5, pad_min_live_frames=3, turn_yaw_deg=15.0,
    )
    eng._pad = _FakePAD(0.5, [0.95, 0.10, 0.12, 0.08, 0.11])
    # Genuine head-turn motion so the turn leg alone would pass.
    obs = [
        LivenessObservation(landmarks(0.0), 0.0, pad_crops={2.7: None, 4.0: None}),
        LivenessObservation(landmarks(0.2), 0.1, pad_crops={2.7: None, 4.0: None}),
        LivenessObservation(landmarks(0.5), 0.2, pad_crops={2.7: None, 4.0: None}),
        LivenessObservation(landmarks(0.6), 0.3, pad_crops={2.7: None, 4.0: None}),
        LivenessObservation(landmarks(0.7), 0.4, pad_crops={2.7: None, 4.0: None}),
    ]
    res = eng.check(obs, Challenge("full", "n", {"direction": "left"}))
    assert not res.passed  # passive quorum unmet -> full denies despite good turn
