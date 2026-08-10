"""Dataset -> impostor embedding matrix for the eval harness (T3).

Turns a directory of images -- or the LFW dataset [C4] -- into an anonymous
128-D embedding matrix through the **exact** YuNet+SFace pipeline the daemon
uses at runtime (:class:`facelock.detect.FaceDetector` +
:class:`facelock.embed.FaceEmbedder`). For each image:

    detect  -> exactly-ONE-face gate (skip 0 or >1 faces, both counted)
    align   -> YuNet 5-landmark ``alignCrop`` (inside FaceEmbedder)
    embed   -> L2-normalized 128-D SFace vector
    stack   -> (n_valid, 128) float32

Only the **embeddings** are kept. Frames are streamed one at a time and never
retained (privacy, REQ-NF-13): the loaders are generators, so a caller reading
from disk / LFW holds at most one frame in memory and this module holds none.

Import safety: ``cv2`` (image-dir) and ``sklearn`` (LFW) are imported *inside*
the loader functions, never at module import, so the harness core is testable
with no OpenCV, no models, no network. The tests mock the detector/embedder and
feed synthetic frames; the real cv2/sklearn paths are covered with
``importorskip`` + a monkeypatched LFW fetch (never a live download).

Model consistency (protocol §2b step 4): pass ``expected_model_id`` (the owner
template's ``model_id``) to refuse embedding with a different model -- you must
never mix the model that produced the owner template with a different one.

References
----------
[C4] Huang, Ramesh, Berg & Learned-Miller (2007). Labeled Faces in the Wild.
[C6] scikit-learn ``fetch_lfw_people`` -- reproducible, cached LFW loader.
"""

from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import numpy as np

from .. import paths as _paths
from ..errors import ModelError

EMBEDDING_DIM = 128

# Image extensions accepted by the plain image-directory loader.
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".ppm", ".pgm", ".webp", ".tif", ".tiff")

__all__ = [
    "DatasetEmbeddings",
    "embed_frames",
    "embed_image_dir",
    "embed_lfw",
    "build_pipeline",
    "write_embeddings_npz",
    "load_embeddings_npz",
    "file_sha256",
]


@dataclass
class DatasetEmbeddings:
    """The ONLY artefacts kept from a dataset: embeddings + scalar provenance.

    ``embeddings`` is an ``(n_valid, 128)`` float32 matrix (each row L2-norm 1);
    ``provenance`` is a flat dict of scalars (counts, model_id, dataset name,
    timestamp). No image-shaped array is ever stored on this object -- that is a
    hard privacy invariant, asserted by the tests.
    """

    embeddings: np.ndarray
    provenance: dict[str, Any]


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _empty() -> np.ndarray:
    return np.empty((0, EMBEDDING_DIM), dtype=np.float32)


def embed_frames(
    frames: Iterable[np.ndarray],
    detector,
    embedder,
    *,
    dataset: str = "frames",
    dataset_version: str = "",
    model_id: str = "",
    expected_model_id: str | None = None,
    extra_provenance: dict[str, Any] | None = None,
) -> DatasetEmbeddings:
    """Push an iterable of BGR frames through detect -> gate -> embed -> stack.

    ``detector`` must expose ``detect(bgr) -> list`` and ``embedder`` must expose
    ``embed(bgr, detection) -> np.ndarray | None`` (the production
    ``FaceDetector`` / ``FaceEmbedder`` satisfy both; tests inject fakes). The
    exactly-one-face gate mirrors the runtime single-face rule
    (``matcher.py`` ``face_count == 1``): frames with 0 or >1 detections are
    dropped and counted separately, as are embedding failures (``None``).

    ``frames`` is consumed lazily; no frame is retained after it is embedded
    (REQ-NF-13). If ``expected_model_id`` is given and differs from ``model_id``
    the call fails closed with :class:`ModelError` (never mix models).
    """
    if expected_model_id is not None and model_id and expected_model_id != model_id:
        raise ModelError(
            f"model_id mismatch: dataset embedded with {model_id!r} but template "
            f"expects {expected_model_id!r} -- refusing to mix models (protocol §2b)"
        )

    n_in = 0
    n_no_face = 0
    n_multi_face = 0
    n_embed_fail = 0
    rows: list[np.ndarray] = []

    for frame in frames:
        n_in += 1
        if frame is None or getattr(frame, "size", 0) == 0:
            n_no_face += 1
            continue
        dets = detector.detect(frame)
        if not dets:
            n_no_face += 1
            continue
        if len(dets) > 1:
            n_multi_face += 1
            continue
        emb = embedder.embed(frame, dets[0])
        if emb is None:
            n_embed_fail += 1
            continue
        vec = np.asarray(emb, dtype=np.float32).reshape(-1)
        if vec.size != EMBEDDING_DIM or not np.all(np.isfinite(vec)):
            n_embed_fail += 1
            continue
        rows.append(vec)
        # `frame` and `emb` fall out of scope on the next iteration; nothing here
        # references the pixels, so no image data survives the loop (privacy).

    embeddings = np.stack(rows).astype(np.float32) if rows else _empty()
    provenance: dict[str, Any] = {
        "dataset": dataset,
        "dataset_version": dataset_version,
        "n_in": int(n_in),
        "n_valid": int(embeddings.shape[0]),
        "n_skipped_no_face": int(n_no_face),
        "n_skipped_multi_face": int(n_multi_face),
        "n_skipped_embed_fail": int(n_embed_fail),
        "model_id": str(model_id),
        "embedding_dim": EMBEDDING_DIM,
        "generated_at": _now_iso(),
    }
    if extra_provenance:
        provenance.update({k: v for k, v in extra_provenance.items()})
    return DatasetEmbeddings(embeddings=embeddings, provenance=provenance)


# --------------------------------------------------------------------------- #
# Plain image directory.
# --------------------------------------------------------------------------- #
def _iter_dir_frames(paths: list[Path], on_unreadable: Callable[[], None]) -> Iterator[np.ndarray]:
    """Lazily ``cv2.imread`` each path (BGR uint8); skip + count unreadable."""
    import cv2  # gated: only needed for the real image-dir path

    for p in paths:
        bgr = cv2.imread(str(p))
        if bgr is None:
            on_unreadable()
            continue
        yield bgr  # streamed one at a time; caller (embed_frames) discards it


def embed_image_dir(
    image_dir: str | Path,
    detector,
    embedder,
    *,
    dataset: str = "image-dir",
    model_id: str = "",
    expected_model_id: str | None = None,
) -> DatasetEmbeddings:
    """Embed every image file under ``image_dir`` (non-recursive, sorted).

    Unreadable files (``cv2.imread`` -> ``None``) and non-image extensions are
    skipped; the exactly-one-face gate is applied by :func:`embed_frames`. Pixels
    are streamed (:func:`_iter_dir_frames`) and never retained.
    """
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"image directory not found: {image_dir}")
    paths = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    )
    unreadable = {"n": 0}

    def _mark() -> None:
        unreadable["n"] += 1

    result = embed_frames(
        _iter_dir_frames(paths, _mark),
        detector,
        embedder,
        dataset=dataset,
        dataset_version=str(image_dir),
        model_id=model_id,
        expected_model_id=expected_model_id,
    )
    # `n_in` from embed_frames counts frames that reached it (readable images);
    # record unreadable files separately so the accounting is complete.
    result.provenance["n_skipped_unreadable"] = int(unreadable["n"])
    result.provenance["n_files"] = int(len(paths) + unreadable["n"])
    return result


# --------------------------------------------------------------------------- #
# LFW via scikit-learn (import gated; NEVER downloads in tests).
# --------------------------------------------------------------------------- #
def _lfw_to_bgr(img: np.ndarray) -> np.ndarray:
    """Convert one LFW image (RGB float [0,1] or uint8) to BGR uint8 for cv2."""
    a = np.asarray(img)
    if a.dtype.kind == "f":
        mx = float(a.max()) if a.size else 0.0
        if mx <= 1.5:  # sklearn returns [0,1] floats
            a = a * 255.0
        a = np.clip(a, 0.0, 255.0).astype(np.uint8)
    else:
        a = a.astype(np.uint8)
    if a.ndim == 3 and a.shape[2] == 3:
        a = a[:, :, ::-1]  # RGB -> BGR (cv2 convention)
    return np.ascontiguousarray(a)


def _iter_lfw_frames(images: np.ndarray, indices: Iterable[int]) -> Iterator[np.ndarray]:
    for i in indices:
        yield _lfw_to_bgr(images[i])  # one frame at a time; discarded downstream


def embed_lfw(
    detector,
    embedder,
    *,
    funneled: bool = True,
    min_faces_per_person: int = 0,
    one_per_identity: bool = False,
    model_id: str = "",
    expected_model_id: str | None = None,
    fetch: Callable[..., Any] | None = None,
) -> DatasetEmbeddings:
    """Embed the LFW dataset through the deployed pipeline (protocol §2b).

    Loads LFW colour images via :func:`sklearn.datasets.fetch_lfw_people`
    (imported here, not at module load; downloads/caches to
    ``~/scikit_learn_data`` on first real use). ``one_per_identity=True`` keeps a
    single image per unique identity -- the *independent-comparison* primary
    estimate (protocol §2b independence note); the default keeps every valid
    image (the correlated secondary/sensitivity estimate).

    ``fetch`` is an injection seam for tests; production leaves it ``None`` and
    the real loader is used. LFW pixels are streamed and never persisted -- only
    the anonymous embeddings survive (REQ-NF-13, protocol §2b licence note).
    """
    if fetch is None:
        from sklearn.datasets import fetch_lfw_people  # gated: real LFW loader

        fetch = fetch_lfw_people

    sklearn_version = ""
    try:
        import sklearn  # noqa: F811 -- version stamp only

        sklearn_version = getattr(sklearn, "__version__", "")
    except Exception:
        sklearn_version = ""

    data = fetch(color=True, funneled=funneled, min_faces_per_person=min_faces_per_person)
    images = np.asarray(data.images)
    n_images = int(images.shape[0])

    if one_per_identity and getattr(data, "target", None) is not None:
        targets = np.asarray(data.target)
        seen: set[int] = set()
        indices: list[int] = []
        for i, t in enumerate(targets):
            t = int(t)
            if t not in seen:
                seen.add(t)
                indices.append(i)
        n_identities = len(seen)
    else:
        indices = list(range(n_images))
        n_identities = (
            int(np.unique(np.asarray(data.target)).size)
            if getattr(data, "target", None) is not None
            else 0
        )

    result = embed_frames(
        _iter_lfw_frames(images, indices),
        detector,
        embedder,
        dataset="LFW",
        dataset_version=f"sklearn=={sklearn_version}" if sklearn_version else "sklearn",
        model_id=model_id,
        expected_model_id=expected_model_id,
        extra_provenance={
            "n_images_in": n_images,
            "n_identities": int(n_identities),
            "one_per_identity": bool(one_per_identity),
            "min_faces_per_person": int(min_faces_per_person),
        },
    )
    return result


# --------------------------------------------------------------------------- #
# Real-pipeline construction (cv2-backed; gated). Not exercised in unit tests.
# --------------------------------------------------------------------------- #
def build_pipeline(
    yunet_path: str | Path,
    sface_path: str | Path,
    *,
    confidence_floor: float = 0.90,
    nms_threshold: float = 0.30,
    min_face_px: int = 80,
) -> tuple[Any, Any, str]:
    """Construct ``(FaceDetector, FaceEmbedder, sface_model_id)`` from model files.

    All heavy imports (cv2 via detect/embed) happen here, not at module import,
    so the harness stays camera/model-free until a real embedding run is asked
    for. ``sface_model_id`` is the SHA-256 of the SFace model -- the value that
    must equal the owner template's ``model_id`` (fed to ``expected_model_id``).
    """
    from ..detect import FaceDetector
    from ..embed import FaceEmbedder
    from ..store import model_sha256

    detector = FaceDetector(
        yunet_path,
        confidence_floor=confidence_floor,
        nms_threshold=nms_threshold,
        min_face_px=min_face_px,
    )
    embedder = FaceEmbedder(sface_path)
    try:
        sface_id = model_sha256(Path(sface_path))
    except OSError:
        sface_id = ""
    return detector, embedder, sface_id


# --------------------------------------------------------------------------- #
# Embeddings-only persistence (reused by the report CLI). Key = 'embeddings'
# (matches store.load_impostor_set); provenance carried in a 'meta' json blob.
# --------------------------------------------------------------------------- #
def write_embeddings_npz(
    result: "DatasetEmbeddings | np.ndarray",
    path: str | Path,
    *,
    provenance: dict[str, Any] | None = None,
) -> Path:
    """Persist an embedding matrix (0600) with its provenance meta blob."""
    if isinstance(result, DatasetEmbeddings):
        emb = result.embeddings
        meta = dict(result.provenance)
    else:
        emb = np.asarray(result, dtype=np.float32)
        meta = {}
    if provenance:
        meta.update(provenance)
    emb = np.asarray(emb, dtype=np.float32).reshape(-1, EMBEDDING_DIM)

    buf = io.BytesIO()
    meta_json = json.dumps(meta, separators=(",", ":"), default=str)
    np.savez(
        buf,
        embeddings=emb,
        meta=np.frombuffer(meta_json.encode("utf-8"), dtype=np.uint8),
    )
    path = Path(path)
    _paths.secure_write_bytes(path, buf.getvalue(), 0o600)
    return path


def load_embeddings_npz(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load an ``(n, 128)`` embedding matrix + its provenance meta (``{}`` if none)."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as npz:
        if "embeddings" not in npz.files:
            raise ValueError(f"{path}: missing 'embeddings' array")
        emb = np.asarray(npz["embeddings"], dtype=np.float32).reshape(-1, EMBEDDING_DIM)
        meta: dict[str, Any] = {}
        if "meta" in npz.files:
            try:
                raw = np.asarray(npz["meta"], dtype=np.uint8).tobytes().decode("utf-8")
                meta = json.loads(raw)
            except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
                meta = {}
    return emb, meta


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file (provenance/reproducibility stamp for a dataset npz)."""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
