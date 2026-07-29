"""TemplateStore tests (C14, design section 11): round-trip, integrity, delete."""

from __future__ import annotations

import os
import stat

import numpy as np
import pytest

from facelock.errors import TemplateError
from facelock.store import (
    Template,
    TemplateStore,
    ensure_impostor_set,
    generate_synthetic_impostors,
    load_impostor_set,
    model_sha256,
    save_impostor_set,
)
from tests.conftest import owner_cluster, unit_vec


def _template():
    samples = owner_cluster()
    return Template(
        owner_name="Yash",
        centroid=samples.mean(axis=0) / np.linalg.norm(samples.mean(axis=0)),
        samples=samples,
        tau=0.5,
        calibration={"meets_target": True},
        model_id="deadbeef",
    )


def test_roundtrip_and_permissions(tmp_path):
    store = TemplateStore(tmp_path / "owner.tmpl")
    store.save(_template())
    assert store.exists()
    mode = stat.S_IMODE(os.stat(store.path).st_mode)
    assert mode == 0o600                              # REQ-NF-14
    loaded = store.load()
    assert loaded.owner_name == "Yash"
    assert loaded.tau == 0.5
    assert loaded.samples.shape[1] == 128


def test_tamper_detected(tmp_path):
    store = TemplateStore(tmp_path / "owner.tmpl")
    store.save(_template())
    # Flip some bytes in the template file -> integrity check must fail (FM-10).
    data = bytearray(store.path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    store.path.write_bytes(bytes(data))
    with pytest.raises(TemplateError):
        store.load()


def test_missing_sig_fails_closed(tmp_path):
    store = TemplateStore(tmp_path / "owner.tmpl")
    store.save(_template())
    store.sig_path.unlink()
    with pytest.raises(TemplateError):
        store.load()


def test_unsafe_permissions_refused(tmp_path):
    store = TemplateStore(tmp_path / "owner.tmpl")
    store.save(_template())
    os.chmod(store.path, 0o644)                       # world-readable
    with pytest.raises(TemplateError):
        store.load()


def test_try_load_returns_none_on_corruption(tmp_path):
    store = TemplateStore(tmp_path / "owner.tmpl")
    store.path.write_bytes(b"not a real npz")
    store.sig_path.write_bytes(b"x")
    assert store.try_load() is None


def test_secure_delete_removes_all(tmp_path):
    store = TemplateStore(tmp_path / "owner.tmpl")
    store.save(_template())
    store.save(_template())  # creates a .bak
    removed = store.delete()
    assert not store.path.exists()
    assert not store.sig_path.exists()
    assert not store.backup_path.exists()
    assert len(removed) >= 2


def test_rollback_restores_backup(tmp_path):
    store = TemplateStore(tmp_path / "owner.tmpl")
    t1 = _template(); t1.owner_name = "First"
    store.save(t1)
    t2 = _template(); t2.owner_name = "Second"
    store.save(t2)                                    # writes .bak = First
    assert store.load().owner_name == "Second"
    assert store.rollback()
    assert store.load().owner_name == "First"


def test_bad_centroid_shape_rejected():
    with pytest.raises(TemplateError):
        Template(owner_name="x", centroid=np.zeros(64), samples=owner_cluster(), tau=0.5)


def test_format_version_mismatch_refused(tmp_path):
    store = TemplateStore(tmp_path / "owner.tmpl")
    t = _template()
    t.format_version = 999
    store.save(t)
    with pytest.raises(TemplateError):
        store.load()


def test_impostor_set_roundtrip(tmp_path):
    emb = generate_synthetic_impostors(n=500, seed=5)
    path = save_impostor_set(emb, tmp_path / "imp.npz")
    loaded = load_impostor_set(path)
    assert loaded is not None and loaded.shape == (500, 128)
    # rows are unit-norm.
    assert np.allclose(np.linalg.norm(loaded, axis=1), 1.0, atol=1e-4)


def test_ensure_impostor_set_generates(tmp_path):
    emb = ensure_impostor_set(tmp_path / "imp.npz")
    assert emb.shape[0] >= 100 and (tmp_path / "imp.npz").exists()


def test_model_sha256(tmp_path):
    f = tmp_path / "m.bin"
    f.write_bytes(b"hello")
    import hashlib
    assert model_sha256(f) == hashlib.sha256(b"hello").hexdigest()
