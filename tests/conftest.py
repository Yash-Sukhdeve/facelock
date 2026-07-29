"""Shared pytest fixtures. Every test runs in an isolated XDG sandbox so the
real user profile is never touched, and no test opens a camera or a window."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure the project root (containing the 'facelock' package and 'tests') is on
# sys.path so tests run without an editable install.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@pytest.fixture(autouse=True)
def xdg_sandbox(tmp_path, monkeypatch):
    """Point all XDG base dirs at a temp directory for the duration of a test."""
    for var in ("XDG_DATA_HOME", "XDG_CONFIG_HOME", "XDG_STATE_HOME", "XDG_RUNTIME_DIR"):
        d = tmp_path / var.lower()
        d.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(var, str(d))
    yield tmp_path


def unit_vec(seed: int, dim: int = 128) -> np.ndarray:
    """Deterministic L2-normalized random vector."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


def owner_cluster(n: int = 6, seed: int = 1, jitter: float = 0.05) -> np.ndarray:
    """A tight cluster of owner sample embeddings around one direction."""
    base = unit_vec(seed)
    rng = np.random.default_rng(seed + 1000)
    out = []
    for _ in range(n):
        v = base + jitter * rng.standard_normal(base.shape).astype(np.float32)
        out.append((v / np.linalg.norm(v)).astype(np.float32))
    return np.stack(out)


@pytest.fixture
def impostors():
    from facelock.store import generate_synthetic_impostors
    return generate_synthetic_impostors(n=2000, seed=42)
