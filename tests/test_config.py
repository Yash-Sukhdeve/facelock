"""Config loader tests (C13, REQ-F-23, design section 12)."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from facelock.config import load_config, update_owner_name
from facelock.errors import ConfigError

REPO_ROOT = Path(__file__).resolve().parent.parent
SHIPPED_CONFIG = REPO_ROOT / "config" / "facelock.toml"


def test_defaults_prototype():
    cfg = load_config(raw={})
    assert cfg.phase == "P"
    assert cfg.stranger.policy == "lenient"          # OQ-2 default
    assert cfg.liveness.mode == "off"                # prototype default
    assert cfg.security.phase == "P"
    assert cfg.recognition.fmr_target == 0.01        # P target
    assert cfg.recognition.probe_frames == 5 and cfg.recognition.match_votes == 3
    assert cfg.lock.shield is True


def test_default_owner_name_is_neutral_not_a_persons_name():
    # Pre-publish fix: the shipped/code default must not be the author's name
    # ("Yash") -- a public user who never customises owner_name (and never runs
    # enroll --name) must not see someone else's name in the greeting.
    cfg = load_config(raw={})
    assert cfg.unlock.owner_name == "User"
    assert cfg.unlock.owner_name != "Yash"


def test_shipped_config_toml_owner_name_is_neutral():
    # The packaged config/facelock.toml (installed by `facelock setup`) must
    # ship the same neutral default, not a person's name.
    cfg = load_config(SHIPPED_CONFIG)
    assert cfg.unlock.owner_name == "User"
    assert cfg.unlock.owner_name != "Yash"


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


# --------------------------------------------------------------------------- #
# update_owner_name (enroll --name persistence, REQ-F-15)
# --------------------------------------------------------------------------- #
def test_update_owner_name_writes_new_value(tmp_path):
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(SHIPPED_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")

    ok, msg = update_owner_name(cfg_path, "Bob")

    assert ok is True
    assert "Bob" in msg
    assert load_config(cfg_path).unlock.owner_name == "Bob"


def test_update_owner_name_preserves_other_keys_and_comments(tmp_path):
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(SHIPPED_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    before = load_config(cfg_path)

    ok, _msg = update_owner_name(cfg_path, "Bob")
    assert ok is True

    after = load_config(cfg_path)
    assert after.unlock.owner_name == "Bob"
    # Every OTHER key in every section is byte-for-byte unchanged.
    for section in before.sections:
        for key, val in before.sections[section].as_dict().items():
            if (section, key) == ("unlock", "owner_name"):
                continue
            assert after.get(section, key) == val, f"{section}.{key} changed"
    # Comments / formatting around unrelated keys survive (spot-check one).
    text = cfg_path.read_text(encoding="utf-8")
    assert "greeting name" in text
    assert "REQ-F-15, ASM-01" in text
    assert 'cooldown_s            = 30' in text


def test_update_owner_name_file_mode_is_0600(tmp_path):
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(SHIPPED_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    os.chmod(cfg_path, 0o644)  # start permissive to prove the writer tightens it

    ok, _msg = update_owner_name(cfg_path, "Bob")

    assert ok is True
    mode = stat.S_IMODE(os.stat(cfg_path).st_mode)
    assert mode == 0o600


def test_update_owner_name_missing_file_is_fail_safe_not_fatal(tmp_path):
    missing = tmp_path / "does-not-exist.toml"

    ok, msg = update_owner_name(missing, "Bob")

    assert ok is False
    assert isinstance(msg, str) and msg  # a human-readable note, no exception
    assert not missing.exists()  # never creates a file out of thin air


def test_update_owner_name_no_unlock_section_is_fail_safe(tmp_path):
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("[camera]\ndevice = \"/dev/video0\"\n", encoding="utf-8")

    ok, msg = update_owner_name(cfg_path, "Bob")

    assert ok is False
    assert "unlock" in msg.lower()
    # Original content untouched.
    assert cfg_path.read_text(encoding="utf-8") == "[camera]\ndevice = \"/dev/video0\"\n"


def test_update_owner_name_escapes_quotes_and_backslashes(tmp_path):
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(SHIPPED_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")

    ok, _msg = update_owner_name(cfg_path, 'O"Brien\\')

    assert ok is True
    assert load_config(cfg_path).unlock.owner_name == 'O"Brien\\'
