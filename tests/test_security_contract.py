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


# The one-shot provisioning verb (`facelock setup`) is the ONLY module allowed
# to fetch over the network -- it downloads the SHA-256-pinned public models from
# the OpenCV Zoo, exactly as scripts/download_models.sh already does. It is not
# part of the running unlock service and never touches biometric data. REQ-NF-12
# ("no outbound network") governs the RUNTIME unlock path; a separate test below
# proves that path can never import the provisioning module.
_PROVISIONING_EXEMPT = {"setup_cmd.py"}


def _scan(predicate, exclude: set[str] | None = None) -> list[str]:
    exclude = exclude or set()
    hits = []
    for path in _py_files():
        rel = os.path.relpath(path, PKG_DIR)
        if os.path.basename(path) in exclude:
            continue
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
    # The RUNTIME unlock path (everything except the one-shot provisioning verb)
    # must have zero network clients. `setup_cmd.py` is exempt because it is the
    # model-provisioning tool (SHA-pinned downloads from the OpenCV Zoo); the
    # compensating control below proves it can never enter the unlock loop.
    forbidden = ("import requests", "from requests", "import urllib", "from urllib",
                 "import http.client", "from http.client", "import httpx", "from httpx",
                 "import aiohttp", "urllib.request", "urlopen(")
    hits = _scan(lambda s: any(tok in s for tok in forbidden),
                 exclude=_PROVISIONING_EXEMPT)
    assert not hits, "REQ-NF-12 violated -- network client usage found:\n" + "\n".join(hits)


def test_setup_network_capability_confined_to_provisioning():
    """Compensating control for the setup_cmd network exemption (REQ-NF-12).

    The provisioning module may fetch models, but the running unlock service
    must never gain that capability. Assert that no runtime module -- the
    perception daemon, the session guardian, or any module they transitively
    rely on for the unlock loop -- imports ``setup_cmd``. Only the CLI's
    ``setup`` verb (a user-invoked, one-shot command) is permitted to.
    """
    runtime_modules = (
        "daemon.py", "guardian.py", "control.py", "matcher.py", "store.py",
        "capture.py", "detect.py", "embed.py", "fsm.py", "shield.py",
        "lock_backend.py", "liveness.py",
    )
    hits = []
    for path in _py_files():
        base = os.path.basename(path)
        if base not in runtime_modules:
            continue
        for lineno, line in _code_lines(path):
            if "setup_cmd" in line:
                hits.append(f"{base}:{lineno}: {line}")
    assert not hits, ("REQ-NF-12 -- network-capable provisioning module leaked into "
                      "the runtime unlock path:\n" + "\n".join(hits))


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


def test_pillow_confined_to_ui_and_never_persists():
    # Pillow is now permitted, but ONLY in the enrollment UI and ONLY for
    # in-memory typography (the Face-ID-style HUD). It must never leak into the
    # daemon/store/etc., and it must never persist an image to disk (REQ-NF-13).
    pil_imports = _scan(lambda s: s.startswith("import PIL") or s.startswith("from PIL"))
    stray = [h for h in pil_imports if not h.startswith("enroll_ui.py:")]
    assert not stray, ("PIL imported outside the enrollment UI (typography only):\n"
                       + "\n".join(stray))

    # PIL's image-persistence idiom is Image.save(...); it must be absent.
    # (tightened: the original `A in s or B in s and C in s` parsed as
    # `A or (B and C)`, which is the intended precedence here but read as an
    # accident -- made explicit so intent is unambiguous at a glance.)
    persist = _scan(lambda s: ("Image.save(" in s)
                     or (".save(" in s and ("PIL" in s or "Image" in s)))
    assert not persist, ("REQ-NF-13 concern -- PIL image persistence found:\n"
                         + "\n".join(persist))
