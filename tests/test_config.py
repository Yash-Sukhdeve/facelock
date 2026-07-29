"""Config loader tests (C13, REQ-F-23, design section 12)."""

from __future__ import annotations

import pytest

from facelock.config import load_config
from facelock.errors import ConfigError


def test_defaults_prototype():
    cfg = load_config(raw={})
    assert cfg.phase == "P"
    assert cfg.stranger.policy == "lenient"          # OQ-2 default
    assert cfg.liveness.mode == "off"                # prototype default
    assert cfg.security.phase == "P"
    assert cfg.recognition.fmr_target == 0.01        # P target
    assert cfg.recognition.probe_frames == 5 and cfg.recognition.match_votes == 3
    assert cfg.lock.shield is True


def test_phase_aliases_normalize():
    assert load_config(raw={"security": {"phase": "prototype"}}).phase == "P"
    assert load_config(raw={"security": {"phase": "hardening"},
                            "liveness": {"mode": "full"}}).phase == "H"
    assert load_config(raw={"security": {"phase": "screensaver-only"}}).phase == "P"


def test_hardening_defaults_differ():
    cfg = load_config(raw={"security": {"phase": "H"}, "liveness": {"mode": "full"}})
    assert cfg.recognition.fmr_target == 0.001       # H tighter target
    assert cfg.security.template_encryption == "keyring"
    assert cfg.security.audit is True
    assert cfg.liveness.mode == "full"


def test_security_key_always_refuses_even_with_default_policy():
    # config.on_invalid=default must NOT rescue a bad security-critical value.
    with pytest.raises(ConfigError) as exc:
        load_config(raw={"config": {"on_invalid": "default"},
                         "stranger": {"policy": "wide-open"}})
    assert any("stranger.policy" in e for e in exc.value.errors)


def test_nonsecurity_default_policy_substitutes():
    cfg = load_config(raw={"config": {"on_invalid": "default"},
                           "camera": {"fps_active": 999}})
    assert cfg.camera.fps_active == 5                 # substituted default
    assert any("camera.fps_active" in w for w in cfg.warnings)


def test_nonsecurity_refuse_policy_raises():
    with pytest.raises(ConfigError):
        load_config(raw={"config": {"on_invalid": "refuse"},
                         "camera": {"fps_active": 999}})


def test_cross_field_match_votes_le_probe_frames():
    with pytest.raises(ConfigError) as exc:
        load_config(raw={"recognition": {"match_votes": 9, "probe_frames": 5}})
    assert any("match_votes" in e for e in exc.value.errors)


def test_liveness_off_forbidden_in_hardening():
    with pytest.raises(ConfigError) as exc:
        load_config(raw={"security": {"phase": "H"}, "liveness": {"mode": "off"}})
    assert any("liveness.mode" in e for e in exc.value.errors)


def test_persist_frames_true_rejected():
    with pytest.raises(ConfigError):
        load_config(raw={"config": {"on_invalid": "default"},
                         "privacy": {"persist_frames": True}})


def test_bad_toml_file_fails_closed(tmp_path):
    bad = tmp_path / "config.toml"
    bad.write_text("this is = = not valid toml ][")
    with pytest.raises(ConfigError):
        load_config(bad)


def test_missing_file_uses_defaults(tmp_path):
    cfg = load_config(tmp_path / "does-not-exist.toml")
    assert cfg.phase == "P"


def test_resolve_model_paths(tmp_path):
    cfg = load_config(raw={}).resolve_model_paths(tmp_path / "models")
    assert cfg.detection.model_path.endswith("face_detection_yunet_2023mar.onnx")
    assert cfg.recognition.model_path.endswith("face_recognition_sface_2021dec.onnx")


def test_tau_floor_is_security_critical():
    with pytest.raises(ConfigError):
        load_config(raw={"config": {"on_invalid": "default"},
                         "recognition": {"tau_floor": 5.0}})
