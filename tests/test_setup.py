"""Tests for `facelock setup` (setup_cmd) -- offline provisioning.

All network I/O is injected (a fake ``opener``) so these tests do NO real
network. They lock the fail-closed SHA-256 contract (R6 / FM-11): a good hash
installs, a bad hash refuses; an already-present valid model is skipped
(idempotent); the config is written 0600 and never clobbered; the systemd
units are installed but NEVER enabled (enroll-first safety).
"""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

import pytest

from facelock import setup_cmd

_REPO = Path(__file__).resolve().parents[1]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _mode(path: Path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


# --------------------------------------------------------------------------- #
# ensure_model: SHA-256 verify + idempotency (the security core).
# --------------------------------------------------------------------------- #
def test_ensure_model_downloads_and_verifies_good_hash(tmp_path):
    data = b"a-fake-onnx-model-blob"
    sha = _sha(data)
    dest = tmp_path / "models"
    dest.mkdir()
    calls = []

    def opener(url):
        calls.append(url)
        return data

    status = setup_cmd.ensure_model("m.onnx", "http://x/m.onnx", sha, dest, opener)
    assert status == "downloaded"
    assert calls == ["http://x/m.onnx"]
    target = dest / "m.onnx"
    assert target.read_bytes() == data
    # sidecar records the verified hash (traceable, R6).
    assert (dest / "m.onnx.sha256").read_text().split()[0] == sha


def test_ensure_model_bad_hash_raises_and_writes_nothing(tmp_path):
    data = b"tampered-blob"
    wrong_sha = _sha(b"the-blob-we-expected")
    dest = tmp_path / "models"
    dest.mkdir()

    def opener(url):
        return data

    with pytest.raises(setup_cmd.SetupError):
        setup_cmd.ensure_model("m.onnx", "http://x/m.onnx", wrong_sha, dest, opener)
    # Fail-closed: nothing written on mismatch.
    assert not (dest / "m.onnx").exists()
    assert not (dest / "m.onnx.sha256").exists()


def test_ensure_model_idempotent_skip_when_present_and_valid(tmp_path):
    data = b"already-here"
    sha = _sha(data)
    dest = tmp_path / "models"
    dest.mkdir()
    (dest / "m.onnx").write_bytes(data)

    def opener(url):
        raise AssertionError("network must not be touched for a valid, present model")

    status = setup_cmd.ensure_model("m.onnx", "http://x/m.onnx", sha, dest, opener)
    assert status == "skipped"


def test_ensure_model_present_but_wrong_hash_redownloads(tmp_path):
    good = b"the-good-model"
    sha = _sha(good)
    dest = tmp_path / "models"
    dest.mkdir()
    (dest / "m.onnx").write_bytes(b"stale-or-corrupt")

    def opener(url):
        return good

    status = setup_cmd.ensure_model("m.onnx", "http://x/m.onnx", sha, dest, opener)
    assert status == "downloaded"
    assert (dest / "m.onnx").read_bytes() == good


# --------------------------------------------------------------------------- #
# install_config: 0600, never clobber.
# --------------------------------------------------------------------------- #
def test_install_config_writes_0600(tmp_path):
    src = tmp_path / "facelock.toml"
    src.write_text("phase = 'P'\n")
    dest = tmp_path / "cfg" / "config.toml"

    status = setup_cmd.install_config(src, dest)
    assert status == "installed"
    assert dest.read_text() == "phase = 'P'\n"
    assert _mode(dest) == 0o600
    # parent dir is owner-only.
    assert _mode(dest.parent) == 0o700


def test_install_config_does_not_clobber_existing(tmp_path):
    src = tmp_path / "facelock.toml"
    src.write_text("phase = 'P'\n")
    dest = tmp_path / "cfg" / "config.toml"
    dest.parent.mkdir(parents=True)
    dest.write_text("USER-EDITED\n")

    status = setup_cmd.install_config(src, dest)
    assert status == "kept"
    assert dest.read_text() == "USER-EDITED\n"


# --------------------------------------------------------------------------- #
# systemd units: install both, never enable.
# --------------------------------------------------------------------------- #
def test_install_systemd_units_copies_both(tmp_path):
    src = tmp_path / "systemd"
    src.mkdir()
    (src / "facelockd.service").write_text("[Service]\n")
    (src / "facelock-guardian.service").write_text("[Service]\n")
    dest = tmp_path / "user"

    units = setup_cmd.install_systemd_units(src, dest)
    assert sorted(units) == ["facelock-guardian.service", "facelockd.service"]
    for u in units:
        assert (dest / u).exists()
        assert _mode(dest / u) == 0o644


# --------------------------------------------------------------------------- #
# Packaged-data locators resolve in the source checkout.
# --------------------------------------------------------------------------- #
def test_packaged_config_and_systemd_locatable():
    cfg = setup_cmd.packaged_config()
    assert cfg.exists() and cfg.name == "facelock.toml"
    unit_dir = setup_cmd.packaged_systemd_dir()
    assert (unit_dir / "facelockd.service").exists()
    assert (unit_dir / "facelock-guardian.service").exists()


def test_model_pins_match_scripts_file():
    """The Python model registry MUST equal scripts/models.sha256 (single
    traceable record of the verified pins -- R6). Guards against drift."""
    pin_file = _REPO / "scripts" / "models.sha256"
    pins: dict[str, str] = {}
    for line in pin_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        sha, name = line.split()
        pins[name] = sha
    assert setup_cmd.MODELS, "model registry must not be empty"
    for name, (url, sha) in setup_cmd.MODELS.items():
        assert name in pins, f"{name} missing from scripts/models.sha256"
        assert sha == pins[name], f"pin drift for {name}"
        assert url.startswith(setup_cmd.ZOO_BASE), f"{name} url not from ZOO_BASE"


# --------------------------------------------------------------------------- #
# run_setup end-to-end (no network): models + config, no systemd.
# --------------------------------------------------------------------------- #
def test_run_setup_downloads_config_no_systemd(tmp_path, monkeypatch):
    # Swap the real pinned registry for a test registry with fabricated bytes so
    # the whole flow runs offline with a fake opener.
    b1 = b"detector-model-bytes"
    b2 = b"recognizer-model-bytes"
    reg = {
        "yunet.onnx": ("http://zoo/yunet.onnx", _sha(b1)),
        "sface.onnx": ("http://zoo/sface.onnx", _sha(b2)),
    }
    monkeypatch.setattr(setup_cmd, "MODELS", reg)
    served = {"http://zoo/yunet.onnx": b1, "http://zoo/sface.onnx": b2}
    seen = []

    def opener(url):
        seen.append(url)
        return served[url]

    src = tmp_path / "facelock.toml"
    src.write_text("phase = 'P'\n")
    models_dir = tmp_path / "data" / "models"
    config_dest = tmp_path / "cfg" / "config.toml"
    out_lines: list[str] = []

    res = setup_cmd.run_setup(
        models_dir=models_dir,
        config_dest=config_dest,
        config_src=src,
        systemd=False,
        opener=opener,
        out=out_lines.append,
    )
    assert res.models == {"yunet.onnx": "downloaded", "sface.onnx": "downloaded"}
    assert res.config == "installed"
    assert res.systemd == []
    assert res.systemd_requested is False
    assert (models_dir / "yunet.onnx").read_bytes() == b1
    assert (models_dir / "sface.onnx").read_bytes() == b2
    assert _mode(config_dest) == 0o600
    assert set(seen) == set(served)
    # A clear "next: enroll" message is printed.
    joined = "\n".join(out_lines).lower()
    assert "enroll" in joined


def test_run_setup_is_idempotent_second_run_skips(tmp_path, monkeypatch):
    b1 = b"det"
    reg = {"yunet.onnx": ("http://zoo/yunet.onnx", _sha(b1))}
    monkeypatch.setattr(setup_cmd, "MODELS", reg)
    src = tmp_path / "facelock.toml"
    src.write_text("phase = 'P'\n")
    models_dir = tmp_path / "models"
    config_dest = tmp_path / "config.toml"

    setup_cmd.run_setup(models_dir=models_dir, config_dest=config_dest,
                        config_src=src, opener=lambda u: b1, out=lambda s: None)

    def opener_boom(url):
        raise AssertionError("second run must not re-download a valid model")

    res = setup_cmd.run_setup(models_dir=models_dir, config_dest=config_dest,
                              config_src=src, opener=opener_boom, out=lambda s: None)
    assert res.models == {"yunet.onnx": "skipped"}
    assert res.config == "kept"  # config already present


def test_run_setup_systemd_installs_units_but_does_not_enable(tmp_path, monkeypatch):
    b1 = b"det"
    reg = {"yunet.onnx": ("http://zoo/yunet.onnx", _sha(b1))}
    monkeypatch.setattr(setup_cmd, "MODELS", reg)
    src = tmp_path / "facelock.toml"
    src.write_text("phase = 'P'\n")
    unit_src = tmp_path / "systemd"
    unit_src.mkdir()
    (unit_src / "facelockd.service").write_text("[Service]\n")
    (unit_src / "facelock-guardian.service").write_text("[Service]\n")
    unit_dest = tmp_path / "user"

    reload_calls = []

    res = setup_cmd.run_setup(
        models_dir=tmp_path / "models",
        config_dest=tmp_path / "config.toml",
        config_src=src,
        systemd=True,
        unit_src_dir=unit_src,
        unit_dest_dir=unit_dest,
        opener=lambda u: b1,
        reloader=lambda: reload_calls.append("reload"),
        out=lambda s: None,
    )
    assert sorted(res.systemd) == ["facelock-guardian.service", "facelockd.service"]
    assert res.systemd_requested is True
    assert (unit_dest / "facelockd.service").exists()
    # daemon-reload was invoked, but NOTHING enables auto-start: the dest dir
    # holds only plain unit files, no *.wants symlinks (enroll-first safety).
    assert reload_calls == ["reload"]
    for entry in unit_dest.iterdir():
        assert entry.is_file()
        assert not entry.name.endswith(".wants")


# --------------------------------------------------------------------------- #
# CLI wiring.
# --------------------------------------------------------------------------- #
def test_cli_setup_dispatches_without_systemd(monkeypatch):
    captured = {}

    def fake_run_setup(**kwargs):
        captured.update(kwargs)
        return setup_cmd.SetupResult()

    monkeypatch.setattr(setup_cmd, "run_setup", fake_run_setup)
    from facelock.cli import main

    assert main(["setup"]) == 0
    assert captured.get("systemd") is False


def test_cli_setup_systemd_flag(monkeypatch):
    captured = {}

    def fake_run_setup(**kwargs):
        captured.update(kwargs)
        return setup_cmd.SetupResult()

    monkeypatch.setattr(setup_cmd, "run_setup", fake_run_setup)
    from facelock.cli import main

    assert main(["setup", "--systemd"]) == 0
    assert captured.get("systemd") is True


def test_cli_setup_reports_setup_error(monkeypatch, capsys):
    def boom(**kwargs):
        raise setup_cmd.SetupError("SHA-256 mismatch for m.onnx")

    monkeypatch.setattr(setup_cmd, "run_setup", boom)
    from facelock.cli import main

    rc = main(["setup"])
    assert rc == 2
    err = capsys.readouterr().err.lower()
    assert "setup failed" in err and "mismatch" in err
