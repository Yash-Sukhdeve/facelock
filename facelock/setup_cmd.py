"""`facelock setup` -- one-shot, offline-verifiable provisioning.

Fresh-machine flow:  ``pipx install facelock`` -> ``facelock setup`` -> ``facelock
enroll``. ``setup`` fetches the SHA-256-pinned YuNet + SFace models into the XDG
models dir, verifying every byte against the pinned hash BEFORE it is written
(fail-closed, R6 / FM-11 -- an unpinned or mismatched model is never trusted);
installs the default ``config.toml`` (0600) if absent; and, with ``--systemd``,
installs the ``--user`` units WITHOUT enabling auto-start (enroll-first safety:
a locked screen with no enrolled face can only be cleared with the OS password).

Design trace: sections 3.1 / 11.1 / 11.4 (REQ-NF-26). The pins here are kept
byte-identical to ``scripts/models.sha256`` (a single traceable record; a test
enforces that they never drift). The HTTP fetch is injected so the whole
command is testable with zero network.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import paths as _paths

# Authoritative OpenCV Zoo location (identical to scripts/download_models.sh).
ZOO_BASE = "https://github.com/opencv/opencv_zoo/raw/main/models"

# filename -> (download url, sha256 hex). The hashes are the values verified
# against the OpenCV Zoo and pinned in scripts/models.sha256 (R6: never guessed;
# test_model_pins_match_scripts_file enforces they stay in sync).
MODELS: dict[str, tuple[str, str]] = {
    "face_detection_yunet_2023mar.onnx": (
        f"{ZOO_BASE}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    "face_recognition_sface_2021dec.onnx": (
        f"{ZOO_BASE}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    ),
}

_UNITS = ("facelockd.service", "facelock-guardian.service")

Opener = Callable[[str], bytes]


class SetupError(RuntimeError):
    """A fatal, fail-closed setup condition (e.g. a SHA-256 mismatch)."""


@dataclass
class SetupResult:
    """What ``run_setup`` did, for callers/tests to assert on."""

    models: dict[str, str] = field(default_factory=dict)  # name -> downloaded|skipped
    config: str = "unknown"  # installed|kept
    systemd: list[str] = field(default_factory=list)  # unit files installed
    systemd_requested: bool = False


# --------------------------------------------------------------------------- #
# Hashing + default network fetch (the only real-I/O piece; always injected in
# tests). Every fetched blob is SHA-256-verified before it is trusted, so even
# a compromised mirror cannot install an unpinned model.
# --------------------------------------------------------------------------- #
def _http_get(url: str, timeout: float = 60.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "facelock-setup"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (sha-verified)
        return resp.read()


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Models.
# --------------------------------------------------------------------------- #
def ensure_model(name: str, url: str, expected_sha: str, dest_dir: Path,
                 opener: Opener) -> str:
    """Provision one SHA-pinned model into ``dest_dir``.

    Returns ``"skipped"`` if it is already present with the correct hash
    (idempotent), else fetches it, VERIFIES the SHA-256, and writes it (0644)
    plus a ``.sha256`` sidecar; returns ``"downloaded"``. Raises
    :class:`SetupError` on a hash mismatch (fail-closed, R6 / FM-11) -- nothing
    is written in that case.
    """
    target = dest_dir / name
    if target.exists() and _sha256_file(target) == expected_sha:
        return "skipped"

    data = opener(url)
    actual = _sha256_hex(data)
    if actual != expected_sha:
        raise SetupError(
            f"SHA-256 mismatch for {name}: expected {expected_sha}, got {actual} "
            "-- refusing to install an unverified model (fail-closed, R6/FM-11)."
        )
    _paths.secure_write_bytes(target, data, 0o644)
    (dest_dir / f"{name}.sha256").write_text(f"{actual}  {name}\n")
    return "downloaded"


# --------------------------------------------------------------------------- #
# Config.
# --------------------------------------------------------------------------- #
def install_config(src: Path, dest: Path) -> str:
    """Install the default config 0600 if ``dest`` is absent; never clobber.

    Returns ``"installed"`` or ``"kept"``.
    """
    if dest.exists():
        return "kept"
    _paths.ensure_dir(dest.parent, 0o700)
    _paths.secure_write_bytes(dest, src.read_bytes(), 0o600)
    return "installed"


# --------------------------------------------------------------------------- #
# systemd --user units (installed, NEVER enabled).
# --------------------------------------------------------------------------- #
def install_systemd_units(src_dir: Path, dest_dir: Path) -> list[str]:
    """Copy the two ``--user`` unit files into ``dest_dir`` (0644).

    Does NOT enable or start anything -- enabling before enrollment would lock
    the screen with no face able to clear it. Returns the installed filenames.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    installed: list[str] = []
    for unit in _UNITS:
        s = src_dir / unit
        if not s.exists():
            raise SetupError(f"packaged systemd unit not found: {s}")
        d = dest_dir / unit
        d.write_bytes(s.read_bytes())
        os.chmod(d, 0o644)
        installed.append(unit)
    return installed


def systemd_user_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "systemd" / "user"


def _daemon_reload() -> None:
    """Best-effort ``systemctl --user daemon-reload`` (no-op if unavailable)."""
    exe = shutil.which("systemctl")
    if not exe:
        return
    try:
        subprocess.run([exe, "--user", "daemon-reload"], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# Packaged-data locators. The default config + unit files ship in the wheel
# (see pyproject [tool.setuptools.data-files]) at <sys.prefix>/config and
# <sys.prefix>/systemd; from a source checkout they live at the repo root.
# --------------------------------------------------------------------------- #
def _candidate_roots() -> list[Path]:
    here = Path(__file__).resolve()
    roots = [
        Path(sys.prefix),          # installed wheel (pipx/venv data-files)
        here.parents[1],           # source checkout / editable install
    ]
    # De-duplicate while preserving order.
    seen: set[str] = set()
    out: list[Path] = []
    for r in roots:
        key = str(r)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def _locate(relpath: str) -> Path:
    for root in _candidate_roots():
        cand = root / relpath
        if cand.exists():
            return cand
    searched = ", ".join(str(r / relpath) for r in _candidate_roots())
    raise SetupError(f"packaged data not found: {relpath} (searched: {searched})")


def packaged_config() -> Path:
    return _locate("config/facelock.toml")


def packaged_systemd_dir() -> Path:
    return _locate(f"systemd/{_UNITS[0]}").parent


# --------------------------------------------------------------------------- #
# Orchestrator.
# --------------------------------------------------------------------------- #
def run_setup(
    *,
    models_dir: Path | None = None,
    config_dest: Path | None = None,
    config_src: Path | None = None,
    systemd: bool = False,
    unit_src_dir: Path | None = None,
    unit_dest_dir: Path | None = None,
    opener: Opener | None = None,
    reloader: Callable[[], None] | None = None,
    out: Callable[[str], None] = print,
) -> SetupResult:
    """Provision models + config (+ optional systemd units). Offline-testable.

    Every path and the network ``opener`` are injectable so this runs fully
    offline in tests. Fail-closed: a model that will not verify aborts setup.
    """
    opener = opener or _http_get
    models_dir = models_dir or _paths.models_dir()
    config_dest = config_dest or _paths.config_path()
    config_src = config_src if config_src is not None else packaged_config()

    result = SetupResult(systemd_requested=systemd)

    out("== facelock setup ==")

    # 1. Models (data_home 0700, models dir 0700; each model SHA-verified).
    _paths.ensure_dir(models_dir.parent, 0o700)
    _paths.ensure_dir(models_dir, 0o700)
    for name, (url, sha) in MODELS.items():
        status = ensure_model(name, url, sha, models_dir, opener)
        result.models[name] = status
        note = "already present, hash OK" if status == "skipped" else "SHA-256 verified"
        out(f">> model {name}: {status} ({note})")

    # 2. Config (0600, never clobber).
    result.config = install_config(config_src, config_dest)
    if result.config == "installed":
        out(f">> config installed (0600): {config_dest}")
    else:
        out(f">> config exists; left untouched: {config_dest}")

    # 3. systemd --user units -- installed but NOT enabled.
    if systemd:
        unit_src_dir = unit_src_dir if unit_src_dir is not None else packaged_systemd_dir()
        unit_dest_dir = unit_dest_dir if unit_dest_dir is not None else systemd_user_dir()
        result.systemd = install_systemd_units(unit_src_dir, unit_dest_dir)
        (reloader or _daemon_reload)()
        out(f">> systemd --user units installed to {unit_dest_dir} (NOT enabled)")
    else:
        out(">> systemd units NOT installed "
            "(add them later with:  facelock setup --systemd)")

    _print_next_steps(out, systemd=systemd)
    return result


def _print_next_steps(out: Callable[[str], None], *, systemd: bool) -> None:
    out("")
    out("Setup complete.")
    out("Next:  enroll your face ->   facelock enroll --name <YourName>")
    out("")
    out("Then enable face-unlock (ONLY after enrolling):")
    out("  systemctl --user enable --now facelock-guardian facelockd")
    if not systemd:
        out("  (first install the units:  facelock setup --systemd)")
    out("")
    out("Do NOT enable the services before enrolling -- a locked screen with no")
    out("enrolled face can only be cleared with your OS password.")
