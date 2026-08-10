"""Dataset-embedding tests (T3): image-dir / LFW -> impostor embedding matrix.

The embedder turns a directory of images (or the LFW dataset) into an anonymous
128-D embedding matrix through the SAME YuNet+SFace pipeline the daemon uses:
per image -> detect (exactly-one-face gate) -> align -> embed -> stack, keeping
ONLY the embeddings (pixels are streamed and discarded, REQ-NF-13). These tests
mock the detector/embedder and feed synthetic frames, so nothing opens a camera,
loads a model, or hits the network. The cv2 / sklearn paths are exercised with
importorskip + a monkeypatched loader (never a real download).
"""

from __future__ import annotations

import numpy as np
import pytest

from facelock.eval import embed_dataset as ED
from facelock.errors import ModelError


# --------------------------------------------------------------------------- #
# Synthetic frames + fake detector/embedder (no cv2, no models).
#
# A frame is a (H, W, 3) uint8 array that encodes its own test behaviour in the
# top-left pixel so the fakes are pure functions of the frame:
#   channel 0 -> number of faces the detector should return
#   channel 1 -> 255 means "embedding fails" (embedder returns None)
#   channel 2 -> a per-frame seed for a deterministic unit embedding
# --------------------------------------------------------------------------- #
def frame(n_faces: int = 1, fail: bool = False, seed: int = 0, h: int = 64, w: int = 64) -> np.ndarray:
    f = np.zeros((h, w, 3), dtype=np.uint8)
    f[0, 0, 0] = int(n_faces)
    f[0, 0, 1] = 255 if fail else 0
    f[0, 0, 2] = int(seed) % 256
    return f


class FakeDet:
    def __init__(self, idx: int) -> None:
        self.idx = idx


class FakeDetector:
    """Returns ``frame[0,0,0]`` detections -- lets a frame script the face count."""

    def __init__(self) -> None:
        self.frames_seen = 0

    def detect(self, bgr: np.ndarray) -> list:
        self.frames_seen += 1
        n = int(np.asarray(bgr)[0, 0, 0])
        return [FakeDet(i) for i in range(n)]


class FakeEmbedder:
    """Deterministic unit embedding; returns None when the frame flags failure."""

    def embed(self, bgr: np.ndarray, det) -> np.ndarray | None:
        a = np.asarray(bgr)
        if int(a[0, 0, 1]) == 255:
            return None
        rng = np.random.default_rng(int(a[0, 0, 2]))
        v = rng.standard_normal(128).astype(np.float32)
        return (v / np.linalg.norm(v)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Core: the exactly-one-face gate + counts + shape.
# --------------------------------------------------------------------------- #
def test_one_face_gate_counts_and_shape():
    frames = [frame(1, seed=1), frame(0), frame(2), frame(1, seed=2), frame(1, seed=3)]
    res = ED.embed_frames(iter(frames), FakeDetector(), FakeEmbedder())
    assert res.embeddings.shape == (3, 128)
    p = res.provenance
    assert p["n_in"] == 5
    assert p["n_valid"] == 3
    assert p["n_skipped_no_face"] == 1
    assert p["n_skipped_multi_face"] == 1
    assert p["n_skipped_embed_fail"] == 0


def test_rows_are_unit_norm_128d():
    frames = [frame(1, seed=s) for s in range(1, 6)]
    res = ED.embed_frames(iter(frames), FakeDetector(), FakeEmbedder())
    assert res.embeddings.dtype == np.float32
    norms = np.linalg.norm(res.embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    assert res.embeddings.shape[1] == 128


def test_embed_failures_are_skipped_and_counted():
    frames = [frame(1, seed=1), frame(1, fail=True, seed=2), frame(1, seed=3)]
    res = ED.embed_frames(iter(frames), FakeDetector(), FakeEmbedder())
    assert res.embeddings.shape == (2, 128)
    assert res.provenance["n_skipped_embed_fail"] == 1
    assert res.provenance["n_valid"] == 2


def test_empty_input_returns_empty_matrix():
    res = ED.embed_frames(iter([]), FakeDetector(), FakeEmbedder())
    assert res.embeddings.shape == (0, 128)
    assert res.provenance["n_in"] == 0
    assert res.provenance["n_valid"] == 0


# --------------------------------------------------------------------------- #
# Privacy: pixels are streamed lazily and NEVER retained (REQ-NF-13).
# --------------------------------------------------------------------------- #
def test_pixels_are_discarded_generator_fully_consumed():
    def gen():
        for s in range(4):
            yield frame(1, seed=s)

    g = gen()
    res = ED.embed_frames(g, FakeDetector(), FakeEmbedder())
    # The generator must be fully drained -> the embedder never buffered frames.
    with pytest.raises(StopIteration):
        next(g)
    # The result object holds ONLY the 128-D embeddings + scalar provenance --
    # no image-shaped array anywhere (that would be a privacy leak).
    assert set(vars(res).keys()) == {"embeddings", "provenance"}
    assert res.embeddings.ndim == 2 and res.embeddings.shape[1] == 128
    for value in res.provenance.values():
        assert not isinstance(value, np.ndarray)


# --------------------------------------------------------------------------- #
# Model-consistency: refuse to mix models (protocol §2b step 4).
# --------------------------------------------------------------------------- #
def test_refuses_model_id_mismatch():
    with pytest.raises(ModelError):
        ED.embed_frames(
            iter([frame(1)]), FakeDetector(), FakeEmbedder(),
            model_id="model-B", expected_model_id="model-A",
        )


def test_matching_model_id_is_recorded():
    res = ED.embed_frames(
        iter([frame(1, seed=7)]), FakeDetector(), FakeEmbedder(),
        model_id="model-A", expected_model_id="model-A",
    )
    assert res.provenance["model_id"] == "model-A"
    assert res.provenance["n_valid"] == 1


# --------------------------------------------------------------------------- #
# npz round-trip (embeddings-only persistence, reused by the report CLI).
# --------------------------------------------------------------------------- #
def test_write_and_load_embeddings_npz_roundtrip(tmp_path):
    res = ED.embed_frames([frame(1, seed=s) for s in range(1, 4)],
                          FakeDetector(), FakeEmbedder())
    path = tmp_path / "imp.npz"
    ED.write_embeddings_npz(res, path)
    emb, meta = ED.load_embeddings_npz(path)
    assert emb.shape == (3, 128)
    assert np.allclose(emb, res.embeddings, atol=1e-6)
    assert meta["n_valid"] == 3
    # 0600 -- an embeddings artefact is owner-only.
    import stat
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


# --------------------------------------------------------------------------- #
# Image-directory path (cv2 only for imread; detector/embedder still mocked).
# --------------------------------------------------------------------------- #
def test_embed_image_dir_streams_through_gate(tmp_path):
    cv2 = pytest.importorskip("cv2")
    d = tmp_path / "imgs"
    d.mkdir()
    cv2.imwrite(str(d / "a.png"), frame(1, seed=1))
    cv2.imwrite(str(d / "b.png"), frame(0))       # no face -> dropped
    cv2.imwrite(str(d / "c.png"), frame(2))       # two faces -> dropped
    cv2.imwrite(str(d / "d.png"), frame(1, seed=2))
    (d / "notes.txt").write_text("not an image")  # ignored (wrong extension)
    res = ED.embed_image_dir(d, FakeDetector(), FakeEmbedder())
    assert res.provenance["n_in"] == 4
    assert res.embeddings.shape == (2, 128)
    assert res.provenance["n_skipped_no_face"] == 1
    assert res.provenance["n_skipped_multi_face"] == 1
    assert res.provenance["dataset"] == "image-dir"


# --------------------------------------------------------------------------- #
# LFW path (sklearn import gated inside; fetch monkeypatched -> NO download).
# --------------------------------------------------------------------------- #
def test_embed_lfw_flows_without_download(monkeypatch):
    pytest.importorskip("sklearn")
    import types

    # RGB float [0,1] frames whose channels 0 and 2 are equal, so the RGB->BGR
    # channel flip inside embed_lfw does not disturb the fake face-count byte.
    def rgb_frame(seed):
        f = np.zeros((40, 40, 3), dtype=np.float32)
        f[0, 0, 0] = 1 / 255.0     # 1 face (survives *255 -> uint8 1)
        f[0, 0, 2] = 1 / 255.0
        return f

    imgs = np.stack([rgb_frame(s) for s in range(3)])
    bunch = types.SimpleNamespace(
        images=imgs,
        target=np.array([0, 1, 1]),
        target_names=np.array(["a", "b"]),
    )
    monkeypatch.setattr("sklearn.datasets.fetch_lfw_people", lambda **k: bunch)

    res = ED.embed_lfw(FakeDetector(), FakeEmbedder(), min_faces_per_person=0)
    assert res.embeddings.shape == (3, 128)
    assert res.provenance["dataset"] == "LFW"
    assert res.provenance["n_valid"] == 3


def test_embed_lfw_one_per_identity(monkeypatch):
    pytest.importorskip("sklearn")
    import types

    def rgb_frame():
        f = np.zeros((40, 40, 3), dtype=np.float32)
        f[0, 0, 0] = 1 / 255.0
        f[0, 0, 2] = 1 / 255.0
        return f

    imgs = np.stack([rgb_frame() for _ in range(5)])
    bunch = types.SimpleNamespace(
        images=imgs,
        target=np.array([0, 0, 1, 1, 2]),   # 3 unique identities
        target_names=np.array(["a", "b", "c"]),
    )
    monkeypatch.setattr("sklearn.datasets.fetch_lfw_people", lambda **k: bunch)

    res = ED.embed_lfw(FakeDetector(), FakeEmbedder(), one_per_identity=True)
    # One image per unique identity -> at most 3 valid embeddings.
    assert res.embeddings.shape == (3, 128)
    assert res.provenance["n_identities"] == 3
