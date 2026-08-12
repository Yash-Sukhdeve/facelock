"""Dynamic, interactive enrollment UI -- an Apple-Face-ID-style face scan.

The user keeps their face in a bright, feathered circular "spotlight" while a
ring of segments around the face lights up (green, with a soft glow) as they
slowly rotate their head. It is *interactive*: the HUD pulses the next segment
to fill for guidance, flashes on every capture, shows live coaching status
("Hold still" / "Great!"), a running percentage, and finishes with a glowing
completion checkmark once the ring is swept.

Head orientation is estimated from YuNet's 5 landmarks (nose offset within the
face box -- no extra model, REQ-NF-21) and mapped to a ring segment. The pure
geometry (:func:`head_offset`, :func:`segment_of`, :func:`nearest_uncovered`,
:func:`progress_fraction`, :func:`coverage_fraction`) and :func:`render` are
unit-testable headless; only the window (``cv2.imshow``) needs a display and
lives in ``enroll.py``.

Typography uses the bundled Inter typeface (SIL Open Font License, shipped under
``facelock/assets/fonts/``) via Pillow, with a graceful fallback chain: bundled
Inter -> system Inter -> DejaVuSans -> OpenCV Hershey. :func:`render` NEVER
crashes when Pillow or the fonts are unavailable -- it degrades to ``cv2.putText``
so the enrollment flow keeps working on a bare machine.
"""

from __future__ import annotations

import importlib.resources as _ir
import io
import math
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# Palette (BGR, OpenCV order). Tuned to the Apple systemGreen success accent on
# a near-black cool background -- high contrast, calm, premium.
# --------------------------------------------------------------------------- #
_GREEN = (88, 209, 48)       # #30D158 systemGreen -- a covered/success segment
_GREEN_HI = (150, 240, 130)  # brighter leading edge + glow core
_BLUE = (255, 176, 79)       # the segment the head currently points at
_TICK_DIM = (54, 50, 58)     # an unfilled segment
_WHITE = (247, 245, 245)     # primary title text
_SUBTLE = (168, 162, 158)    # secondary / coaching text (neutral)
_AMBER = (32, 176, 255)      # #FFB020 -- a corrective/reject hint
_BG = (16, 13, 17)           # near-black cool base

_FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX (avoid importing cv2 for the constant)

_REJECT_TEXT = {
    "no_face": "Center your face in the circle",
    "multiple_faces": "Only one face, please",
    "face_too_small": "Move a little closer",
    "too_blurry": "Hold still",
    "bad_brightness": "Adjust your lighting",
}

# Weight name -> (bundled asset filename, system Inter filename, DejaVu fallback).
_FONT_CHAIN = {
    "regular":  ("Inter-Regular.otf",  "Inter-Regular.otf",  "DejaVuSans.ttf"),
    "medium":   ("Inter-Medium.otf",   "Inter-Medium.otf",   "DejaVuSans.ttf"),
    "semibold": ("Inter-SemiBold.otf", "Inter-SemiBold.otf", "DejaVuSans-Bold.ttf"),
}
_SYS_INTER_DIR = "/usr/share/fonts/opentype/inter/"
_DEJAVU_DIR = "/usr/share/fonts/truetype/dejavu/"

# Small caches so we do not re-read font bytes / re-decode faces every frame.
_bundled_bytes_cache: dict[str, bytes | None] = {}
_font_cache: dict[tuple[str, int], object] = {}


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


# --------------------------------------------------------------------------- #
# Pure geometry -- unchanged, headless-testable, no cv2/PIL required.
# --------------------------------------------------------------------------- #
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


def _seg_midangle(i: int, n: int) -> float:
    """Screen angle (deg) of segment i's centre, matching the drawn ring."""
    return -90.0 + (i + 0.5) * (360.0 / n)


# --------------------------------------------------------------------------- #
# Typography: Pillow + Inter with a graceful fallback to cv2.putText.
# --------------------------------------------------------------------------- #
def _pil():
    """Return ``(Image, ImageDraw, ImageFont)`` or ``None`` if Pillow is absent.

    The import is done here (not at module load) so a monkeypatched/broken PIL
    only downgrades typography to the OpenCV Hershey fallback -- ``render`` still
    returns a valid image. This is the robustness contract for a bare machine.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        return Image, ImageDraw, ImageFont
    except Exception:
        return None


def _bundled_font_bytes(fname: str) -> bytes | None:
    """Read a bundled font from the package (works from a source tree or wheel)."""
    if fname in _bundled_bytes_cache:
        return _bundled_bytes_cache[fname]
    data: bytes | None
    try:
        res = _ir.files("facelock").joinpath("assets", "fonts", fname)
        data = res.read_bytes()
    except Exception:
        data = None
    _bundled_bytes_cache[fname] = data
    return data


def _load_font(ImageFont, weight: str, px: int):
    """Load an Inter ``weight`` at ``px``, trying the full fallback chain.

    Order: bundled Inter -> system Inter -> DejaVuSans. Returns a Pillow font, or
    ``None`` when every candidate fails (caller then uses cv2.putText).
    """
    px = max(1, int(px))
    key = (weight, px)
    if key in _font_cache:
        return _font_cache[key]

    bundled, system, dejavu = _FONT_CHAIN.get(weight, _FONT_CHAIN["regular"])
    font = None

    data = _bundled_font_bytes(bundled)
    if data is not None:
        try:
            font = ImageFont.truetype(io.BytesIO(data), px)
        except Exception:
            font = None

    if font is None:
        for path in (_SYS_INTER_DIR + system, _DEJAVU_DIR + dejavu):
            try:
                font = ImageFont.truetype(path, px)
                break
            except Exception:
                continue

    _font_cache[key] = font
    return font


def _bgr2rgb(bgr):
    return (int(bgr[2]), int(bgr[1]), int(bgr[0]))


def _put(cv2, img, text, org, scale, colour, thick=2, *, center=False):
    """OpenCV Hershey fallback line-drawer (used when Pillow/fonts are absent)."""
    if center:
        (tw, th), _ = cv2.getTextSize(text, _FONT, scale, thick)
        org = (int(org[0] - tw / 2), int(org[1] + th / 2))
    cv2.putText(img, text, (int(org[0]), int(org[1])), _FONT, scale, colour,
                thick, cv2.LINE_AA)


def _draw_texts(cv2, np, canvas_u8, items):
    """Overlay all text items on an 8-bit BGR image.

    ``items`` is a list of dicts: ``text, cx, y (top), px, weight, fill (BGR)``.
    Uses Pillow+Inter for crisp anti-aliased type; on ANY failure (Pillow
    missing, every font in the chain unavailable, a draw error) it falls back to
    ``cv2.putText`` so the frame is always rendered. Returns the drawn image.
    """
    items = [it for it in items if it.get("text")]
    if not items:
        return canvas_u8

    pil = _pil()
    if pil is not None:
        Image, ImageDraw, ImageFont = pil
        try:
            im = Image.fromarray(cv2.cvtColor(canvas_u8, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(im)
            for it in items:
                font = _load_font(ImageFont, it["weight"], it["px"])
                if font is None:
                    raise RuntimeError("no usable font in chain")
                draw.text((int(it["cx"]), int(it["y"])), it["text"], font=font,
                          fill=_bgr2rgb(it["fill"]), anchor="ma")
            return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
        except Exception:
            pass  # fall through to the cv2 Hershey fallback below

    for it in items:
        scale = max(0.32, it["px"] / 32.0)
        thick = max(1, int(round(it["px"] / 18.0)))
        # PIL anchor "ma" tops the text at y; approximate that for cv2 by moving
        # the (centered) origin down half the glyph height.
        _put(cv2, canvas_u8, it["text"], (it["cx"], it["y"] + it["px"] * 0.5),
             scale, it["fill"], thick, center=True)
    return canvas_u8


# --------------------------------------------------------------------------- #
# Resolution fitting: scale a camera frame up to the *display's* resolution
# BEFORE rendering, so the fullscreen HUD is 1:1 with the monitor (no upscaling
# of the text/ring). Aspect is preserved (letterboxed) so the face keeps its true
# proportions; ``render`` then center-crops the fitted frame into the face disc.
# --------------------------------------------------------------------------- #
def letterbox(frame_bgr, target_w: int, target_h: int):
    """Return ``frame_bgr`` resized to exactly ``target_w x target_h``, aspect
    preserved with centered black padding. Pure + headless-testable; never
    mutates the input and never raises on a degenerate (empty) frame.
    """
    import cv2
    import numpy as np

    tw = max(1, int(target_w))
    th = max(1, int(target_h))
    src = np.ascontiguousarray(frame_bgr)
    out = np.zeros((th, tw, 3), np.uint8)
    if src.ndim != 3 or src.shape[0] <= 0 or src.shape[1] <= 0 or src.size == 0:
        return out
    h, w = src.shape[:2]
    scale = min(tw / w, th / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(src, (nw, nh), interpolation=interp)
    ox, oy = (tw - nw) // 2, (th - nh) // 2
    out[oy:oy + nh, ox:ox + nw] = resized
    return out


# --------------------------------------------------------------------------- #
# The renderer.
# --------------------------------------------------------------------------- #
def _lerp(a, b, t):
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


def render(frame_bgr, view: RingView):
    """Render the enrollment ``RingView`` as a polished Face-ID-style HUD.

    Returns a NEW same-size 3-channel uint8 BGR image (the input frame is never
    mutated). Everything scales from the frame height, so it looks right at any
    resolution. Robust to ``bbox=None``/no-face, tiny frames, and a missing
    Pillow/font stack (text then falls back to ``cv2.putText``).
    """
    import cv2
    import numpy as np

    src = np.ascontiguousarray(frame_bgr)
    h, w = src.shape[:2]
    h = max(1, h)
    w = max(1, w)
    K = h / 800.0

    def S(x: float) -> int:
        return max(1, int(round(x * K)))

    cx, cy = w // 2, h // 2
    face_r = max(8, int(min(w, h) * 0.30))
    ring_r = face_r + S(40)

    # --- background: near-black with a soft radial vignette ------------------ #
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    denom = max(1.0, 0.75 * math.hypot(w, h) / 2.0)
    vig = np.clip(1.0 - (dist / denom) ** 2, 0.0, 1.0)
    canvas = (np.array(_BG, np.float32)[None, None, :]
              * (0.80 + 0.35 * vig[:, :, None])).astype(np.float32)

    # --- feathered circular face from the frame (center-cropped) ------------- #
    y0, x0 = cy - face_r, cx - face_r
    roi = canvas[max(0, y0):y0 + 2 * face_r, max(0, x0):x0 + 2 * face_r]
    dh, dw = roi.shape[:2]
    if dh > 0 and dw > 0:
        s = min(h, w)
        sy, sx = (h - s) // 2, (w - s) // 2
        crop = src[sy:sy + s, sx:sx + s]
        if crop.size == 0:
            crop = src
        disc = cv2.resize(crop, (dw, dh), interpolation=cv2.INTER_AREA)
        disc = np.clip(disc.astype(np.float32) * 1.05 + 4.0, 0, 255)
        mask = np.zeros((dh, dw), np.float32)
        cv2.circle(mask, (dw // 2, dh // 2), max(1, min(dw, dh) // 2 - S(6)),
                   1.0, -1, cv2.LINE_AA)
        mask = cv2.GaussianBlur(mask, (0, 0), max(1.0, S(6)))[:, :, None]
        ry0, rx0 = max(0, y0), max(0, x0)
        canvas[ry0:ry0 + dh, rx0:rx0 + dw] = roi * (1 - mask) + disc * mask
    cv2.circle(canvas, (cx, cy), face_r + S(5), (70, 66, 74), max(1, S(2)),
               cv2.LINE_AA)

    # --- ring of segments: green + glow as coverage fills -------------------- #
    n = max(1, view.n_segments)
    seg = 360.0 / n
    gap = seg * 0.28
    pulse = 0.5 + 0.5 * math.sin(view.tick * 0.35)  # guidance breathing
    glow = np.zeros((h, w, 3), np.float32)
    axes = (ring_r, ring_r)
    for i in range(n):
        a0 = -90 + i * seg + gap / 2
        a1 = -90 + (i + 1) * seg - gap / 2
        if i in view.covered:
            cv2.ellipse(canvas, (cx, cy), axes, 0, a0, a1, _GREEN_HI, S(6),
                        cv2.LINE_AA)
            cv2.ellipse(glow, (cx, cy), axes, 0, a0, a1, _GREEN, S(11),
                        cv2.LINE_AA)
        elif i == view.target_segment:
            col = _lerp(_TICK_DIM, _WHITE, pulse)
            cv2.ellipse(canvas, (cx, cy), axes, 0, a0, a1, col,
                        max(1, S(3 + 3 * pulse)), cv2.LINE_AA)
            cv2.ellipse(glow, (cx, cy), axes, 0, a0, a1,
                        _lerp((0, 0, 0), _BLUE, pulse), S(7), cv2.LINE_AA)
        elif view.current is not None and i == view.current:
            cv2.ellipse(canvas, (cx, cy), axes, 0, a0, a1, _BLUE, S(5),
                        cv2.LINE_AA)
            cv2.ellipse(glow, (cx, cy), axes, 0, a0, a1, _BLUE, S(8),
                        cv2.LINE_AA)
        else:
            cv2.ellipse(canvas, (cx, cy), axes, 0, a0, a1, _TICK_DIM, S(3),
                        cv2.LINE_AA)
    canvas = np.clip(canvas + cv2.GaussianBlur(glow, (0, 0), max(1.0, S(9))) * 0.8,
                     0, 255)

    # --- pulsing guidance marker toward the next segment to fill ------------- #
    if view.target_segment is not None and view.phase != "done":
        ang = math.radians(_seg_midangle(view.target_segment, n))
        tip = (int(cx + math.cos(ang) * (ring_r - S(3))),
               int(cy + math.sin(ang) * (ring_r - S(3))))
        tail = (int(cx + math.cos(ang) * (face_r + S(10))),
                int(cy + math.sin(ang) * (face_r + S(10))))
        col = _lerp(_TICK_DIM, _WHITE, pulse)
        cv2.arrowedLine(canvas, tail, tip, col, max(1, S(2)), cv2.LINE_AA,
                        tipLength=0.35)

    # --- frontal indicator (centre dot) -------------------------------------- #
    cv2.circle(canvas, (cx, cy), max(2, S(5)),
               _GREEN if view.frontal_done else _TICK_DIM, -1, cv2.LINE_AA)

    # --- capture flash: a bright expanding ping ------------------------------ #
    if view.flash > 0.01:
        ov = canvas.copy()
        cv2.circle(ov, (cx, cy),
                   int(ring_r + S(12) + (1.0 - view.flash) * S(34)),
                   _GREEN_HI, max(1, S(3)), cv2.LINE_AA)
        f = float(max(0.0, min(1.0, view.flash))) * 0.5
        canvas = cv2.addWeighted(ov, f, canvas, 1 - f, 0)

    # --- completion checkmark ------------------------------------------------ #
    if view.phase == "done":
        sc = 1.0 - (1.0 - min(1.0, max(0.0, view.tick / 12.0))) ** 3  # eased 0..1
        sc = max(0.35, sc)
        cg = np.zeros((h, w, 3), np.float32)
        a = (cx - int(S(46) * sc), cy - int(S(2) * sc))
        b = (cx - int(S(12) * sc), cy + int(S(30) * sc))
        c = (cx + int(S(52) * sc), cy - int(S(40) * sc))
        cv2.line(canvas, a, b, _GREEN_HI, max(2, S(9)), cv2.LINE_AA)
        cv2.line(canvas, b, c, _GREEN_HI, max(2, S(9)), cv2.LINE_AA)
        cv2.line(cg, a, b, _GREEN, max(2, S(15)), cv2.LINE_AA)
        cv2.line(cg, b, c, _GREEN, max(2, S(15)), cv2.LINE_AA)
        canvas = np.clip(canvas + cv2.GaussianBlur(cg, (0, 0), max(1.0, S(10))) * 0.8,
                         0, 255)

    out = np.clip(canvas, 0, 255).astype(np.uint8)

    # --- typography (Inter via Pillow, else cv2.putText) --------------------- #
    items: list[dict] = []
    title = view.instruction or ("All set" if view.phase == "done" else "")
    items.append(dict(text=title, cx=cx, y=S(40), px=S(30),
                      weight="semibold", fill=_WHITE))

    if view.phase == "done":
        msg = view.status or "Your face is enrolled"
        items.append(dict(text=msg, cx=cx, y=cy + ring_r + S(30), px=S(20),
                          weight="medium", fill=_GREEN_HI))
    else:
        # Coaching line: a reject hint (amber) takes priority, else the live
        # status (green when the frame is good, neutral otherwise).
        if view.reject and not view.quality_ok:
            coach = _REJECT_TEXT.get(view.reject, view.reject.replace("_", " "))
            items.append(dict(text=coach, cx=cx, y=cy + ring_r + S(30), px=S(19),
                              weight="medium", fill=_AMBER))
        elif view.status:
            good = view.quality_ok or view.status.strip().endswith("!")
            items.append(dict(text=view.status, cx=cx, y=cy + ring_r + S(30),
                              px=S(19), weight="medium",
                              fill=_GREEN_HI if good else _SUBTLE))
        pct = int(round(progress_fraction(view.captured, view.target) * 100))
        items.append(dict(text=f"{pct}%", cx=cx, y=cy + ring_r + S(60), px=S(17),
                          weight="medium", fill=_SUBTLE))

    return _draw_texts(cv2, np, out, items)
