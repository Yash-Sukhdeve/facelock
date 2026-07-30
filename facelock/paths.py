"""XDG base-directory resolution and secure directory creation.

Implements the on-disk layout of design section 11.1. No SQL; storage is a
small set of files under the XDG base directories with strict permissions
(dirs 0700, secret files 0600). The runtime control socket lives on tmpfs and
is never persisted.

All paths honour the XDG_* environment variables so the layout is testable in
an isolated temp HOME without touching the real user profile.
"""

from __future__ import annotations

import os
import secrets
import stat
from pathlib import Path

APP = "facelock"

# Bundled model filenames (provisioned at runtime by scripts/download_models.sh
# into the models dir; see design section 3.1 / 11.1).
YUNET_MODEL = "face_detection_yunet_2023mar.onnx"
SFACE_MODEL = "face_recognition_sface_2021dec.onnx"
PAD_MODEL = "minifasnet_pad.onnx"  # Hardening only


def _base(env: str, default_rel: str) -> Path:
    """Return an XDG base directory, honouring the environment variable."""
    value = os.environ.get(env)
    if value:
        return Path(value)
    return Path.home() / default_rel


def data_home() -> Path:
    """$XDG_DATA_HOME/facelock  (default ~/.local/share/facelock)."""
    return _base("XDG_DATA_HOME", ".local/share") / APP


def config_home() -> Path:
    """$XDG_CONFIG_HOME/facelock  (default ~/.config/facelock)."""
    return _base("XDG_CONFIG_HOME", ".config") / APP


def state_home() -> Path:
    """$XDG_STATE_HOME/facelock  (default ~/.local/state/facelock)."""
    return _base("XDG_STATE_HOME", ".local/state") / APP


def runtime_dir() -> Path:
    """$XDG_RUNTIME_DIR/facelock  (tmpfs; fallback to a 0700 /tmp dir).

    The control socket lives here. If XDG_RUNTIME_DIR is unset (rare on a
    logged-in GNOME session) we fall back to a per-uid /tmp directory so the
    tool still runs; the directory is always created 0700.
    """
    value = os.environ.get("XDG_RUNTIME_DIR")
    if value:
        return Path(value) / APP
    return Path("/tmp") / f"{APP}-{os.getuid()}"


def config_path() -> Path:
    """Absolute path of the TOML config file (design section 12.1)."""
    return config_home() / "config.toml"


def templates_dir() -> Path:
    return data_home() / "templates"


def template_path() -> Path:
    return templates_dir() / "owner.tmpl"


def template_backup_path() -> Path:
    return templates_dir() / "owner.tmpl.bak"


def template_sig_path() -> Path:
    return templates_dir() / "owner.tmpl.sig"


def models_dir() -> Path:
    return data_home() / "models"


def impostor_path() -> Path:
    """Bundled impostor embedding set for tau calibration (embeddings only)."""
    return data_home() / "impostor_embeddings.npz"


def events_log_path() -> Path:
    return state_home() / "events.log"


def audit_log_path() -> Path:
    return state_home() / "audit.log"


def health_path() -> Path:
    return state_home() / "health.json"


def control_socket_path() -> Path:
    return runtime_dir() / "control.sock"


def ensure_dir(path: Path, mode: int = 0o700) -> Path:
    """Create ``path`` (and parents) and enforce ``mode`` (default 0700).

    Idempotent. EVERY directory this call creates gets ``mode`` -- not just the
    leaf -- because ``Path.mkdir(parents=True)`` applies the mode only to the
    final component and leaves intermediates at ``0777 & ~umask`` (typically
    0755). Tightening each created component keeps no biometric-adjacent
    directory world-traversable (REQ-NF-14). Pre-existing directories are left
    at their current mode (we do not loosen or surprise other apps) but are
    checked: a directory that already exists as a symlink, or is not owned by
    the current user, is rejected -- this guards the predictable
    ``/tmp/facelock-<uid>`` runtime fallback against pre-planting by another uid.
    """
    if path.is_symlink():
        raise OSError(f"refusing to use symlinked directory {path}")
    if path.exists():
        _verify_owned_dir(path)
        os.chmod(path, mode)
        return path
    # Create each missing ancestor explicitly so the restrictive mode applies to
    # all of them, not only the leaf.
    missing: list[Path] = []
    cur = path
    while not cur.exists():
        missing.append(cur)
        parent = cur.parent
        if parent == cur:  # reached filesystem root
            break
        cur = parent
    for d in reversed(missing):
        try:
            d.mkdir(mode=mode)
        except FileExistsError:
            pass  # raced with a concurrent create; fall through to enforce mode
        if d.is_symlink():
            raise OSError(f"refusing to use symlinked directory {d}")
        _verify_owned_dir(d)
        os.chmod(d, mode)  # mkdir mode is subject to umask; enforce explicitly
    return path


def _verify_owned_dir(path: Path) -> None:
    """Raise if ``path`` is not a real directory owned by the current user."""
    st = os.lstat(path)
    if not stat.S_ISDIR(st.st_mode):
        raise OSError(f"expected a directory at {path}")
    if st.st_uid != os.getuid():
        raise OSError(f"directory {path} is not owned by the current user")


def secure_write_bytes(path: Path, data: bytes, mode: int = 0o600) -> None:
    """Atomically write ``data`` to ``path`` with owner-only perms (0600).

    Writes to a fresh, uniquely-named temp file in the same directory then
    renames, so a reader never sees a partially written secret. The temp file is
    created with ``O_EXCL | O_NOFOLLOW`` so a pre-planted symlink or temp file
    can never redirect the 0600 write to an attacker-chosen target (L2/L3).
    """
    ensure_dir(path.parent, 0o700)
    # Unique temp name so O_EXCL never trips on a stale leftover, and a same-uid
    # actor cannot pre-create a predictable temp path to interfere.
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(6)}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    fd = os.open(tmp, flags, mode)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
    except BaseException:
        try:
            os.unlink(tmp)
        finally:
            raise
    os.chmod(tmp, mode)
    os.replace(tmp, path)


def is_mode(path: Path, mode: int) -> bool:
    """Return True iff ``path`` exists and its permission bits equal ``mode``."""
    try:
        return stat.S_IMODE(os.stat(path).st_mode) == mode
    except FileNotFoundError:
        return False
