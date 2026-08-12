"""EnrollmentTool full-flow offline test (pre-publish fix, REQ-F-15).

``facelock enroll --name Bob`` must:
  (a) build a template owned by "Bob" (already true before this fix), AND
  (b) persist ``owner_name = "Bob"`` into the user's config.toml so the
      RUNTIME greeting (``guardian.py``/``fsm.py``, which read
      ``config.unlock.owner_name``) matches the enrolled owner -- not the
      shipped/author default.

No camera, no display, no network. ``CameraCapture``/``FaceDetector``/
``FaceEmbedder`` are replaced with deterministic fakes; the real ``cv2``
image-quality math (``cvtColor``/``Laplacian``) runs on synthetic in-memory
frames only -- no hardware, no window is ever opened.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import facelock.capture as capture_mod
import facelock.detect as detect_mod
import facelock.embed as embed_mod
from facelock.config import load_config
from facelock.enroll import EnrollmentTool
from facelock.store import TemplateStore
from tests.conftest import owner_cluster

REPO_ROOT = Path(__file__).resolve().parent.parent
SHIPPED_CONFIG = REPO_ROOT / "config" / "facelock.toml"


class _FakeCamera:
    """Duck-types the ``CameraCapture`` subset ``EnrollmentTool.enroll()`` uses."""

    def __init__(self, device, *, width=0, height=0, pixel_format="", fps=0):
        self.device = device

    def open(self):
        return None  # "opens" instantly, never busy

    def read(self):
        rng = np.random.default_rng(1234)
        frame = SimpleNamespace(bgr=rng.integers(60, 200, (240, 320, 3), dtype=np.uint8))
        return frame, None

    def release(self):
        pass


class _FakeDet:
    """A single, well-formed, always-frontal detection."""

    def __init__(self):
        self.bbox = (50.0, 50.0, 200.0, 200.0)  # min dim 200 >= min_face_px(80)
        self.score = 0.99
        self.landmarks = np.zeros((5, 2), dtype=np.float32)  # nose centered -> frontal


class _FakeDetector:
    def __init__(self, model_path, *, confidence_floor=0.0, nms_threshold=0.0, min_face_px=0):
        pass

    def detect(self, bgr):
        return [_FakeDet()]


class _FakeEmbedder:
    """Yields a tight owner cluster so calibration succeeds -- mirrors the
    pattern already proven in tests/test_enroll_core.py's build_template test."""

    def __init__(self, model_path):
        self._samples = owner_cluster(n=10, seed=7)
        self._i = 0

    def embed(self, bgr, det):
        v = self._samples[self._i % len(self._samples)]
        self._i += 1
        return v


@pytest.fixture(autouse=True)
def _fake_hardware(monkeypatch):
    # enroll.py does ``from .capture import CameraCapture`` etc. INSIDE the
    # enroll() method body, so patching the source module's attribute (not a
    # local alias) is what the local import picks up on each call.
    monkeypatch.setattr(capture_mod, "CameraCapture", _FakeCamera)
    monkeypatch.setattr(detect_mod, "FaceDetector", _FakeDetector)
    monkeypatch.setattr(embed_mod, "FaceEmbedder", _FakeEmbedder)


def _write_config(tmp_path: Path) -> Path:
    """A real, on-disk config.toml (copy of the shipped default) with model
    paths pointed at guaranteed-nonexistent files.

    Without this, ``recognition.model_path`` defaults to ``""``, and
    ``Path("").exists()`` is True (it resolves to the CWD) -- enroll() would
    then try to sha256 a *directory* as the model file and crash. The real
    detector/embedder are fully faked above, so the path value itself is
    never opened for real; only its existence is probed.
    """
    text = SHIPPED_CONFIG.read_text(encoding="utf-8")
    text = text.replace('model_path            = ""', 'model_path            = "/nonexistent/model.onnx"')
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(text, encoding="utf-8")
    return cfg_path


def test_enroll_name_bob_persists_to_config_and_template(tmp_path):
    cfg_path = _write_config(tmp_path)
    cfg = load_config(cfg_path)

    tool = EnrollmentTool(cfg)
    rc = tool.enroll(
        "Bob",
        multipose=False,
        gui=False,
        min_samples=2,
        samples_per_pose=1,
        settle_s=0.0,
        capture_interval_s=0.0,
        timeout_s=30.0,
    )

    assert rc == 0

    # (a) the TEMPLATE owner_name is "Bob".
    tmpl = TemplateStore().try_load()
    assert tmpl is not None
    assert tmpl.owner_name == "Bob"

    # (b) config.unlock.owner_name == "Bob" in the file enroll() wrote to.
    reloaded = load_config(cfg_path)
    assert reloaded.unlock.owner_name == "Bob"
    assert reloaded.unlock.owner_name != "Yash"


def test_enroll_with_missing_config_source_still_succeeds(tmp_path):
    # If the config's source file is absent when enrollment finishes (removed,
    # or loaded from a path that no longer exists), persisting owner_name must
    # be skipped gracefully: enrollment still completes (rc == 0) and NO config
    # file is conjured out of thin air.
    #
    # NB: we start from a REAL config (valid model_path) and only repoint
    # source_path at a missing file. A literally-empty config would default
    # model_path to "" -> Path("") -> "." and crash enroll() on an unrelated
    # code path (see _write_config's docstring); that is not what this covers.
    cfg = dataclasses.replace(
        load_config(_write_config(tmp_path)),
        source_path=tmp_path / "not-created.toml",
    )
    missing = cfg.source_path
    assert not missing.exists()

    tool = EnrollmentTool(cfg)
    rc = tool.enroll(
        "Bob",
        multipose=False,
        gui=False,
        min_samples=2,
        samples_per_pose=1,
        settle_s=0.0,
        capture_interval_s=0.0,
        timeout_s=30.0,
    )

    assert rc == 0
    tmpl = TemplateStore().try_load()
    assert tmpl is not None and tmpl.owner_name == "Bob"
    # persist saw a missing target -> skipped; nothing materialized.
    assert not missing.exists()


def test_enroll_config_write_failure_does_not_abort_a_successful_enrollment(
    tmp_path, monkeypatch,
):
    # Fail-safe contract: a config-write hiccup (disk full, permission error,
    # a corrupt config, etc.) must warn and continue, NEVER void an otherwise
    # successful enrollment.
    cfg_path = _write_config(tmp_path)
    cfg = load_config(cfg_path)

    import facelock.enroll as enroll_mod

    def _boom(path, name):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(enroll_mod, "update_owner_name", _boom)

    tool = EnrollmentTool(cfg)
    rc = tool.enroll(
        "Bob",
        multipose=False,
        gui=False,
        min_samples=2,
        samples_per_pose=1,
        settle_s=0.0,
        capture_interval_s=0.0,
        timeout_s=30.0,
    )

    assert rc == 0  # enrollment itself still succeeded
    tmpl = TemplateStore().try_load()
    assert tmpl is not None and tmpl.owner_name == "Bob"
