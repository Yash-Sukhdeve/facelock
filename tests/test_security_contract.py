"""Security/privacy-contract regression guards (REQ-F-18, NF-12, NF-13, NF-24).

These are SOURCE-SCANNING tests: they walk the installed facelock package and
assert that forbidden patterns are absent from real code lines. They lock the
prototype's stated contract so a future edit (human or subagent) that quietly
adds a network call, a raw-frame dump, or a PAM import fails CI immediately --
no camera, display, or daemon required.

Contract being enforced:
  * REQ-F-18  -- face-unlock is NOT wired into PAM/login/sudo (password path
                untouched). No `import pam` anywhere.
  * REQ-NF-12 -- local-only: no outbound network (no requests/urllib/http.client;
                no INET sockets). Unix-domain sockets for the local control IPC
                are allowed (AF_UNIX only).
  * REQ-NF-13 -- no raw camera frames are ever written to disk (no cv2.imwrite).
"""

from __future__ import annotations

import os

import facelock

PKG_DIR = os.path.dirname(facelock.__file__)


def _py_files() -> list[str]:
    out = []
    for root, _dirs, files in os.walk(PKG_DIR):
        for f in files:
            if f.endswith(".py"):
                out.append(os.path.join(root, f))
    assert out, f"no .py files found under {PKG_DIR}"
    return out


def _code_lines(path: str):
    """Yield (lineno, stripped) for lines that are real code, not blank/comment."""
    with open(path, encoding="utf-8") as fh:
        for i, raw in enumerate(fh, 1):
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            yield i, s


def _scan(predicate) -> list[str]:
    hits = []
    for path in _py_files():
        rel = os.path.relpath(path, PKG_DIR)
        for lineno, line in _code_lines(path):
            if predicate(line):
                hits.append(f"{rel}:{lineno}: {line}")
    return hits


# --- REQ-F-18: no PAM integration ------------------------------------------- #
def test_no_pam_import_anywhere():
    hits = _scan(lambda s: s.startswith("import pam") or s.startswith("from pam"))
    assert not hits, "REQ-F-18 violated -- PAM import found:\n" + "\n".join(hits)


# --- REQ-NF-12: local-only, no outbound network ----------------------------- #
def test_no_network_client_imports():
    forbidden = ("import requests", "from requests", "import urllib", "from urllib",
                 "import http.client", "from http.client", "import httpx", "from httpx",
                 "import aiohttp", "urllib.request", "urlopen(")
    hits = _scan(lambda s: any(tok in s for tok in forbidden))
    assert not hits, "REQ-NF-12 violated -- network client usage found:\n" + "\n".join(hits)


def test_no_inet_sockets_only_unix():
    # AF_UNIX is fine (local control IPC); AF_INET/AF_INET6 would be outbound-capable.
    hits = _scan(lambda s: "AF_INET" in s or "AF_INET6" in s
                 or "socket.create_connection" in s)
    assert not hits, "REQ-NF-12 violated -- INET socket found (only AF_UNIX allowed):\n" + "\n".join(hits)


# --- REQ-NF-13: no raw camera frames written to disk ------------------------ #
def test_no_raw_frame_writes():
    # cv2.imwrite is the canonical "dump a frame to disk" call; it must never appear.
    hits = _scan(lambda s: "cv2.imwrite" in s or "imwrite(" in s)
    assert not hits, "REQ-NF-13 violated -- raw frame write found:\n" + "\n".join(hits)


def test_pillow_image_save_absent():
    # PIL Image.save() would also persist an image; the prototype does not use PIL.
    hits = _scan(lambda s: s.startswith("import PIL") or s.startswith("from PIL"))
    assert not hits, "REQ-NF-13 concern -- PIL imported (image persistence path):\n" + "\n".join(hits)
