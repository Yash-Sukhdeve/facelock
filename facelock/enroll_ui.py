"""Dynamic, interactive enrollment UI -- an iPhone/Windows-Hello-style face scan.

The user keeps their face in a bright "spotlight" (the rest of the frame is
dimmed) while a ring of segments around the face lights up as they slowly rotate
their head. It is *interactive*: the HUD pulses the next segment to fill and
draws an arrow toward it, flashes on every capture, shows live status
("Hold still" / "Great!"), and completes once the ring is swept.

Head orientation is estimated from YuNet's 5 landmarks (nose offset within the
face box -- no extra model, REQ-NF-21) and mapped to a ring segment. All drawing
is dependency-light OpenCV on a NumPy frame, so the pure geometry
(:func:`head_offset`, :func:`segment_of`, :func:`nearest_uncovered`,
:func:`progress_fraction`) and :func:`render` are unit-testable headless; only
the window (``cv2.imshow``) needs a display and lives in ``enroll.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# BGR colours (OpenCV order) tuned to the shield's neon palette on near-black.
_CYAN = (255, 229, 0)
_BLUE = (255, 155, 61)
_GREEN = (118, 230, 0)
_RED = (85, 45, 234)
_AMBER = (0, 190, 255)
_WHITE = (232, 234, 237)
_GREY = (96, 92, 88)
_FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX (avoid importing cv2 for the constant)

_REJECT_TEXT = {
    "no_face": "Center your face in the circle",
    "multiple_faces": "Only one face, please",
    "face_too_small": "Move a little closer",
    "too_blurry": "Hold still",
    "bad_brightness": "Adjust your lighting",
}


@dataclass
class RingView:
    """State for one rendered enrollment frame."""

    owner: str
    captured: int
    target: int
    n_segments: int
    covered: frozenset = field(default_factory=frozenset)
    current: int | None = None          # segment the head points at now
    target_segment: int | None = None   # next segment to fill (guidance)
    frontal_done: bool = False
    instruction: str = "Slowly move your head in a circle"
    status: str = ""                     # live coaching line ("Great!", ...)
    phase: str = "capture"              # "capture" | "done"
    bbox: tuple | None = None
    quality_ok: bool = False
    reject: str | None = None
    flash: float = 0.0                   # capture-flash intensity, 0..1 (decays)
    tick: int = 0                        # animation frame counter


def head_offset(landmarks, bbox) -> tuple[float, float]:
    """Normalised nose offset within the face box, each component in [-1, 1].

    (0, 0) is looking straight at the camera; the vector points toward where the
    head is turned. Pure + robust to short/degenerate landmark arrays.
    """
    import numpy as np

    pts = np.asarray(landmarks, dtype=float).reshape(-1, 2)
    if pts.shape[0] < 3:
        return 0.0, 0.0
    nose = pts[2]
    x, y, w, h = (float(v) for v in bbox)
    hw = max(w / 2.0, 1e-6)
    hh = max(h / 2.0, 1e-6)
    nx = (nose[0] - (x + w / 2.0)) / hw
    ny = (nose[1] - (y + h / 2.0)) / hh
    return float(max(-1.0, min(1.0, nx))), float(max(-1.0, min(1.0, ny)))


def segment_of(nx: float, ny: float, n_segments: int, deadzone: float = 0.12) -> int | None:
    """Map a head-offset vector to a ring segment index, or ``None`` if frontal."""
    if n_segments <= 0:
        return None
    if math.hypot(nx, ny) < deadzone:
        return None
    ang = math.degrees(math.atan2(ny, nx)) % 360.0
    return int(ang // (360.0 / n_segments)) % n_segments


def nearest_uncovered(current: int | None, covered, n_segments: int) -> int | None:
    """The uncovered segment nearest ``current`` (circular), for guidance."""
    remaining = [i for i in range(n_segments) if i not in covered]
    if not remaining:
        return None
    if current is None:
        return remaining[0]

    def _dist(i: int) -> int:
        d = abs(i - current) % n_segments
        return min(d, n_segments - d)

    return min(remaining, key=_dist)


def progress_fraction(captured: int, target: int) -> float:
    if target <= 0:
        return 0.0
    return max(0.0, min(1.0, captured / target))


def coverage_fraction(covered, n_segments: int) -> float:
    if n_segments <= 0:
        return 0.0
    return max(0.0, min(1.0, len(covered) / n_segments))


def _dim(colour, f: float = 0.45):
    return tuple(int(c * f) for c in colour)


def _put(cv2, img, text, org, scale, colour, thick=2, *, center=False):
    if center:
        (tw, th), _ = cv2.getTextSize(text, _FONT, scale, thick)
        org = (int(org[0] - tw / 2), int(org[1] + th / 2))
    cv2.putText(img, text, (int(org[0]), int(org[1])), _FONT, scale, colour,
                thick, cv2.LINE_AA)


def _seg_midangle(i: int, n: int) -> float:
    """Screen angle (deg) of segment i's centre, matching the drawn ring."""
    return -90.0 + (i + 0.5) * (360.0 / n)


def render(frame_bgr, view: RingView):
    """Return a copy of ``frame_bgr`` with the interactive scan HUD drawn on it."""
    import cv2
    import numpy as np

    img = np.ascontiguousarray(frame_bgr).copy()
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    spot_r = int(min(w, h) * 0.22)
    ring_r = int(spot_r * 1.42)

    # Spotlight: bright face circle, dimmed everywhere else.
    dark = (img.astype(np.float32) * 0.30).astype(np.uint8)
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (cx, cy), spot_r, 255, -1)
    out = np.ascontiguousarray(np.where(mask[:, :, None].astype(bool), img, dark))

    # Capture flash: a bright expanding ping when a sample was just taken.
    if view.flash > 0.01:
        fr = int(ring_r + 18 * view.flash)
        cv2.circle(out, (cx, cy), fr, _WHITE, max(1, int(3 * view.flash)), cv2.LINE_AA)

    # Face circle + faint glow.
    cv2.circle(out, (cx, cy), spot_r + 4, _dim(_CYAN), 1, cv2.LINE_AA)
    cv2.circle(out, (cx, cy), spot_r, _CYAN, 2, cv2.LINE_AA)

    n = max(1, view.n_segments)
    seg = 360.0 / n
    gap = seg * 0.30
    pulse = 0.5 + 0.5 * math.sin(view.tick * 0.35)  # for the guidance target
    for i in range(n):
        a0 = -90 + i * seg + gap / 2
        a1 = -90 + (i + 1) * seg - gap / 2
        if i in view.covered:
            colour, thick = _GREEN, 8
        elif i == view.target_segment:
            colour, thick = _CYAN, int(5 + 6 * pulse)   # pulsing "go here"
        elif view.current is not None and i == view.current:
            colour, thick = _BLUE, 8
        else:
            colour, thick = _GREY, 4
        cv2.ellipse(out, (cx, cy), (ring_r, ring_r), 0, a0, a1, colour, thick,
                    cv2.LINE_AA)

    # Guidance arrow pointing to the next segment to fill.
    if view.target_segment is not None:
        ang = math.radians(_seg_midangle(view.target_segment, n))
        tip = (int(cx + math.cos(ang) * (ring_r - 10)),
               int(cy + math.sin(ang) * (ring_r - 10)))
        tail = (int(cx + math.cos(ang) * (spot_r + 12)),
                int(cy + math.sin(ang) * (spot_r + 12)))
        cv2.arrowedLine(out, tail, tip, _CYAN, 2, cv2.LINE_AA, tipLength=0.35)

    # Frontal indicator (centre dot).
    cv2.circle(out, (cx, cy), 6, _GREEN if view.frontal_done else _GREY, -1, cv2.LINE_AA)

    # Brand + instruction + live status + percentage.
    _put(cv2, out, "F A C E L O C K", (cx, 46), 0.8, _CYAN, 2, center=True)
    _put(cv2, out, view.instruction, (cx, h - 66), 0.72, _WHITE, 2, center=True)
    if view.status:
        _put(cv2, out, view.status, (cx, cy - ring_r - 26), 0.7, _GREEN, 2, center=True)
    pct = int(coverage_fraction(view.covered, n) * 100)
    _put(cv2, out, f"{pct}%", (cx, cy + ring_r + 44), 0.9, _WHITE, 2, center=True)

    # Guidance when the current frame is rejected.
    if view.reject and not view.quality_ok:
        _put(cv2, out, _REJECT_TEXT.get(view.reject, view.reject.replace("_", " ")),
             (cx, h - 34), 0.62, _AMBER, 2, center=True)

    if view.phase == "done":
        cv2.circle(out, (cx, cy), spot_r, _GREEN, 3, cv2.LINE_AA)
        _put(cv2, out, "COMPLETE", (cx, h - 66), 0.9, _GREEN, 2, center=True)

    return out
