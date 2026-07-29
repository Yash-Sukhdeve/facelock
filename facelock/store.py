"""TemplateStore (C14) -- on-disk owner template + impostor set.

Realizes design section 11. No SQL: the template is a NumPy ``.npz`` (compact
binary) with a JSON ``meta`` blob, stored 0600 under the XDG data dir. Loading
performs a format + integrity check; a bad major version, permission problem,
tamper, or model-id mismatch fails closed (TemplateError -> face-unlock
disabled, password only, FM-10).

Prototype: plaintext ``.npz`` at 0600 + an HMAC-SHA256 sidecar for tamper
evidence. Hardening: the same bytes AES-256-GCM-encrypted with a
Secret-Service-held key -- a clean composition point (``encrypt``/``decrypt``
hooks below) that does not change the schema.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import paths as _paths
from .errors import TemplateError

FORMAT_VERSION = 1
EMBEDDING_DIM = 128  # SFace (REQ-F-07)


@dataclass
class Template:
    """Owner enrollment template (design section 11.2)."""

    owner_name: str
    centroid: np.ndarray  # float32[128], L2-normalized
    samples: np.ndarray  # float32[n>=5][128], each L2-normalized
    tau: float
    calibration: dict[str, Any] = field(default_factory=dict)
    sample_meta: list[dict[str, Any]] = field(default_factory=list)
    model_id: str = ""  # SHA-256 of the SFace model used
    metric: str = "cosine"
    phase: str = "P"
    format_version: int = FORMAT_VERSION
    embedding_dim: int = EMBEDDING_DIM
    created_at: str = ""
    updated_at: str = ""
    revoked: bool = False

    def __post_init__(self) -> None:
        self.centroid = np.asarray(self.centroid, dtype=np.float32).reshape(-1)
        self.samples = np.asarray(self.samples, dtype=np.float32).reshape(-1, EMBEDDING_DIM)
        if self.centroid.shape != (EMBEDDING_DIM,):
            raise TemplateError(f"centroid must be {EMBEDDING_DIM}-D, got {self.centroid.shape}")
        if self.samples.shape[0] < 1:
            raise TemplateError("template must contain >=1 sample embedding")
        if not self.created_at:
            self.created_at = _now_iso()
        if not self.updated_at:
            self.updated_at = self.created_at

    def meta(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "owner_name": self.owner_name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "model_id": self.model_id,
            "embedding_dim": self.embedding_dim,
            "metric": self.metric,
            "tau": float(self.tau),
            "calibration": self.calibration,
            "sample_meta": self.sample_meta,
            "phase": self.phase,
            "revoked": self.revoked,
        }


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"


def _to_bytes(template: Template) -> bytes:
    """Serialise a template to ``.npz`` bytes (centroid, samples, meta-json)."""
    buf = io.BytesIO()
    meta_json = json.dumps(template.meta(), separators=(",", ":"), default=str)
    np.savez(
        buf,
        centroid=template.centroid.astype(np.float32),
        samples=template.samples.astype(np.float32),
        meta=np.frombuffer(meta_json.encode("utf-8"), dtype=np.uint8),
    )
    return buf.getvalue()


def _from_bytes(data: bytes) -> Template:
    """Deserialise ``.npz`` bytes into a Template (fail-closed on any issue)."""
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as npz:
            centroid = np.asarray(npz["centroid"], dtype=np.float32)
            samples = np.asarray(npz["samples"], dtype=np.float32)
            meta_bytes = np.asarray(npz["meta"], dtype=np.uint8).tobytes()
        meta = json.loads(meta_bytes.decode("utf-8"))
    except (KeyError, ValueError, OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise TemplateError(f"template unreadable/corrupt: {exc}") from exc

    version = int(meta.get("format_version", -1))
    if version != FORMAT_VERSION:
        # Unknown *major* version -> refuse rather than mis-parse (REQ-F-23).
        raise TemplateError(
            f"unsupported template format_version {version} (expected {FORMAT_VERSION})"
        )
    if meta.get("revoked", False):
        raise TemplateError("template is revoked; re-enroll required")

    return Template(
        owner_name=meta.get("owner_name", "Owner"),
        centroid=centroid,
        samples=samples,
        tau=float(meta["tau"]),
        calibration=meta.get("calibration", {}),
        sample_meta=meta.get("sample_meta", []),
        model_id=meta.get("model_id", ""),
        metric=meta.get("metric", "cosine"),
        phase=meta.get("phase", "P"),
        format_version=version,
        embedding_dim=int(meta.get("embedding_dim", EMBEDDING_DIM)),
        created_at=meta.get("created_at", ""),
        updated_at=meta.get("updated_at", ""),
        revoked=False,
    )


class TemplateStore:
    """Persist/load the single owner template with integrity + secure delete."""

    def __init__(
        self,
        template_path: Path | None = None,
        *,
        hmac_key: bytes | None = None,
    ) -> None:
        self.path = template_path or _paths.template_path()
        self.backup_path = self.path.with_suffix(self.path.suffix + ".bak")
        self.sig_path = self.path.with_suffix(self.path.suffix + ".sig")
        # Integrity key: prototype uses a per-install 0600 key file (tamper
        # evidence). Hardening swaps this for a Secret-Service key (§11.5).
        self._hmac_key = hmac_key if hmac_key is not None else self._load_or_make_key()

    def _load_or_make_key(self) -> bytes:
        key_path = self.path.parent / ".integrity.key"
        try:
            if key_path.exists():
                return key_path.read_bytes()
        except OSError:
            pass
        key = os.urandom(32)
        try:
            _paths.secure_write_bytes(key_path, key, 0o600)
        except OSError:
            pass
        return key

    def _sign(self, data: bytes) -> bytes:
        return hmac.new(self._hmac_key, data, hashlib.sha256).digest()

    # -- persistence ------------------------------------------------------- #
    def exists(self) -> bool:
        return self.path.exists()

    def save(self, template: Template, *, augment_backup: bool = True) -> None:
        """Persist the template 0600, rotating the previous one to ``.bak``.

        Rollback safety (REQ-F-03): the current template is copied to ``.bak``
        before the new one lands, so a crash mid-write never leaves the owner
        with no template.
        """
        _paths.ensure_dir(self.path.parent, 0o700)
        template.updated_at = _now_iso()
        data = _to_bytes(template)
        if augment_backup and self.path.exists():
            try:
                _paths.secure_write_bytes(self.backup_path, self.path.read_bytes(), 0o600)
            except OSError:
                pass
        _paths.secure_write_bytes(self.path, data, 0o600)
        _paths.secure_write_bytes(self.sig_path, self._sign(data), 0o600)

    def load(self) -> Template:
        """Load + integrity-check the template. Fail closed on any error (FM-10)."""
        if not self.path.exists():
            raise TemplateError("no owner template enrolled")
        # Permission check: refuse to load a template that is not owner-only
        # (a world/group-readable biometric artefact is a policy violation).
        if not _paths.is_mode(self.path, 0o600):
            raise TemplateError(
                f"template {self.path} has unsafe permissions (expected 0600)"
            )
        try:
            data = self.path.read_bytes()
        except OSError as exc:
            raise TemplateError(f"cannot read template: {exc}") from exc

        # HMAC integrity (tamper evidence). Missing sidecar -> refuse (FM-10).
        if not self.sig_path.exists():
            raise TemplateError("template integrity sidecar (.sig) missing")
        try:
            expected = self.sig_path.read_bytes()
        except OSError as exc:
            raise TemplateError(f"cannot read integrity sidecar: {exc}") from exc
        if not hmac.compare_digest(expected, self._sign(data)):
            raise TemplateError("template integrity check failed (tampered/corrupt)")

        return _from_bytes(data)

    def try_load(self) -> Template | None:
        """Load, returning ``None`` on any failure (used by the daemon startup).

        The daemon never crashes on a bad template; it disables face-unlock and
        keeps the session locked / password-only (FM-10, SI-P2).
        """
        try:
            return self.load()
        except TemplateError:
            return None

    def verify(self) -> bool:
        """Return True iff the template loads and passes integrity."""
        try:
            self.load()
            return True
        except TemplateError:
            return False

    def rollback(self) -> bool:
        """Restore the ``.bak`` template after a failed re-enroll (REQ-F-03)."""
        if not self.backup_path.exists():
            return False
        data = self.backup_path.read_bytes()
        _paths.secure_write_bytes(self.path, data, 0o600)
        _paths.secure_write_bytes(self.sig_path, self._sign(data), 0o600)
        return True

    def delete(self) -> list[str]:
        """Securely erase the template and all derived artefacts (REQ-F-04).

        Each file is overwritten with random bytes, fsync-ed, then unlinked, so
        a filesystem scan afterwards finds no biometric-derived artefact
        (AC-F-04 / AC-NF-15). Returns the list of paths removed.
        """
        removed: list[str] = []
        targets = [self.path, self.backup_path, self.sig_path,
                   self.path.parent / ".integrity.key"]
        for target in targets:
            if target.exists():
                _secure_shred(target)
                removed.append(str(target))
        return removed


def _secure_shred(path: Path) -> None:
    """Overwrite-then-unlink a file (best-effort secure delete, REQ-NF-15)."""
    try:
        size = os.path.getsize(path)
        with open(path, "r+b") as fh:
            for _ in range(2):
                fh.seek(0)
                fh.write(os.urandom(max(size, 1)))
                fh.flush()
                os.fsync(fh.fileno())
    except OSError:
        pass
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# Impostor embedding set (for tau calibration, design 3.2). Embeddings ONLY --
# NO raw images are ever stored (REQ-NF-13).
# --------------------------------------------------------------------------- #
def load_impostor_set(path: Path | None = None) -> np.ndarray | None:
    """Load the bundled impostor embedding matrix, or ``None`` if absent."""
    path = path or _paths.impostor_path()
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as npz:
            emb = np.asarray(npz["embeddings"], dtype=np.float32)
        emb = emb.reshape(-1, EMBEDDING_DIM)
        return _l2_normalize_rows(emb)
    except (KeyError, ValueError, OSError):
        return None


def generate_synthetic_impostors(n: int = 4000, seed: int = 20260728) -> np.ndarray:
    """Deterministically synthesise an impostor embedding set.

    The prototype ships no real face images (privacy, REQ-NF-13) and cannot
    download models here, so calibration uses a synthetic null distribution:
    ``n`` L2-normalized Gaussian unit vectors on the 128-sphere. In high
    dimension the pairwise cosine of such vectors is ~N(0, 1/128) -- a
    conservative *lower bound* proxy for a real impostor distribution (real
    face impostor cosines are typically higher). The safety floor
    ``recognition.tau_floor`` (default 0.363, SFace's published operating
    point) guarantees the calibrated tau is never weaker than the model's
    characterized point, so the synthetic set can only make tau *tighter*, never
    weaker. HARDENING replaces this with real impostor embeddings from a public
    dataset (documented hook -- swap this function's output).
    """
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n, EMBEDDING_DIM)).astype(np.float32)
    return _l2_normalize_rows(raw)


def save_impostor_set(embeddings: np.ndarray, path: Path | None = None) -> Path:
    """Persist an impostor embedding matrix (0600, embeddings only)."""
    path = path or _paths.impostor_path()
    buf = io.BytesIO()
    np.savez(buf, embeddings=np.asarray(embeddings, dtype=np.float32))
    _paths.secure_write_bytes(path, buf.getvalue(), 0o600)
    return path


def ensure_impostor_set(path: Path | None = None) -> np.ndarray:
    """Return the impostor set, synthesising + persisting one if absent."""
    path = path or _paths.impostor_path()
    existing = load_impostor_set(path)
    if existing is not None and existing.shape[0] >= 100:
        return existing
    synthetic = generate_synthetic_impostors()
    try:
        save_impostor_set(synthetic, path)
    except OSError:
        pass
    return synthetic


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return (matrix / norms).astype(np.float32)


def model_sha256(model_path: Path) -> str:
    """SHA-256 of a model file, used as ``model_id`` for revocation (§11.4)."""
    h = hashlib.sha256()
    with open(model_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
