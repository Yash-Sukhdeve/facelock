"""XDG path + secure-file tests (design section 11.1)."""

from __future__ import annotations

import os
import stat

from facelock import paths


def test_xdg_env_honoured(xdg_sandbox):
    assert str(xdg_sandbox) in str(paths.data_home())
    assert paths.config_path().name == "config.toml"
    assert paths.template_path().name == "owner.tmpl"
    assert paths.control_socket_path().name == "control.sock"


def test_ensure_dir_mode(tmp_path):
    d = paths.ensure_dir(tmp_path / "sub" / "dir", 0o700)
    assert d.exists()
    assert stat.S_IMODE(os.stat(d).st_mode) == 0o700


def test_secure_write_bytes_perms_and_content(tmp_path):
    f = tmp_path / "secret.bin"
    paths.secure_write_bytes(f, b"topsecret", 0o600)
    assert f.read_bytes() == b"topsecret"
    assert stat.S_IMODE(os.stat(f).st_mode) == 0o600


def test_secure_write_atomic_overwrite(tmp_path):
    f = tmp_path / "s.bin"
    paths.secure_write_bytes(f, b"one", 0o600)
    paths.secure_write_bytes(f, b"two", 0o600)
    assert f.read_bytes() == b"two"


def test_is_mode(tmp_path):
    f = tmp_path / "m"
    paths.secure_write_bytes(f, b"x", 0o600)
    assert paths.is_mode(f, 0o600)
    assert not paths.is_mode(f, 0o644)
    assert not paths.is_mode(tmp_path / "missing", 0o600)


def test_runtime_dir_fallback(monkeypatch):
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    d = paths.runtime_dir()
    # Fallback is a per-uid /tmp dir so multiple users never collide.
    assert d.name.startswith("facelock-")
    assert str(d).startswith("/tmp/")
