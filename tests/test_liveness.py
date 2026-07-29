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
