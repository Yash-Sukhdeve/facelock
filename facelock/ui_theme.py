"""ui_theme -- the futuristic shield UI's PURE animation + colour engine.

This module has ZERO GUI dependencies (no tkinter, no display). It is the
computation behind the animated lock screen: per-phase neon colour palettes,
rotating scan-ring geometry, pulse/blink envelopes, colour interpolation for
glows/fades, and enrollment progress arcs. ``shield.py`` feeds these numbers
into a tkinter ``Canvas``; keeping the maths here means the whole look-and-feel
is unit-testable headless (per REQ-NF-21: no extra dependency, and tests never
open a window).

All angles are in degrees, matching tkinter ``Canvas.create_arc`` conventions
(counter-clockwise, 0 deg at 3 o'clock). All fractions are clamped to their
documented ranges so a renderer can trust the output without re-validating.
"""

from __future__ import annotations

from dataclasses import dataclass

# Shield phases the daemon/guardian drive (mirrors shield.set_* methods).
LOCKED = "locked"
RECOGNIZING = "recognizing"
DENIED = "denied"
WELCOME = "welcome"
ENROLLING = "enrolling"

BACKGROUND = "#05070d"  # near-black cool base for the whole shield


@dataclass(frozen=True)
class PhaseTheme:
    """The neon palette + motion parameters for one phase."""

    ring: str        # primary scan-ring colour
    glow: str        # dim halo colour (concentric rings / gradient)
    text: str        # caption colour
    accent: str      # secondary highlight (reticle / ticks)
    spin_dps: float  # scan-ring rotation speed, degrees per animation tick
    pulse_period: int  # frames per pulse cycle (breathing)


# Futuristic palette: cyan idle, electric-blue scan, magenta-red alert,
# aurora-green welcome, violet enrollment. Chosen for high contrast on BACKGROUND.
PHASE_THEMES: dict[str, PhaseTheme] = {
    LOCKED:      PhaseTheme("#00e5ff", "#0b3b45", "#cfefff", "#1de9ff", 2.0, 46),
    RECOGNIZING: PhaseTheme("#3d9bff", "#10305e", "#dbeafe", "#00d4ff", 9.0, 20),
    DENIED:      PhaseTheme("#ff2d55", "#4a0f1c", "#ffd6dd", "#ff6b81", 6.0, 10),
    WELCOME:     PhaseTheme("#00e676", "#0c3d24", "#d7ffe8", "#69f0ae", 3.0, 30),
    ENROLLING:   PhaseTheme("#a855f7", "#2e1150", "#ede9fe", "#c98bff", 5.0, 24),
}


def theme_for(phase: str) -> PhaseTheme:
    """Return the palette for a phase, defaulting to LOCKED for unknowns."""
    return PHASE_THEMES.get(phase, PHASE_THEMES[LOCKED])


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def triangle(tick: int, period: int) -> float:
    """A 0->1->0 triangle wave over ``period`` frames (breathing envelope)."""
    if period <= 0:
        return 0.0
    phase = (tick % period) / period
    return 1.0 - abs(2.0 * phase - 1.0)


def pulse(tick: int, period: int, lo: float = 0.0, hi: float = 1.0) -> float:
    """Triangle envelope scaled into ``[lo, hi]``."""
    return lo + (hi - lo) * triangle(tick, period)


def blink_on(tick: int, period: int = 8) -> bool:
    """Square blink: True for ``period`` frames, then False for ``period``."""
    if period <= 0:
        return True
    return (tick // period) % 2 == 0


def ring_sweep(tick: int, spin_dps: float, extent: float = 100.0) -> tuple[float, float]:
    """Rotating scan-arc (start_deg, extent_deg) for ``create_arc``.

    ``start`` advances by ``spin_dps`` each tick and wraps at 360; ``extent`` is
    the arc length (clamped to a sane 5..355 so it never degenerates to a full
    circle or an invisible sliver).
    """
    start = (tick * spin_dps) % 360.0
    return start, _clamp(extent, 5.0, 355.0)


def progress_extent(done: int, total: int) -> float:
    """Enrollment progress as an arc extent in degrees (0..360)."""
    if total <= 0:
        return 0.0
    return _clamp(done / total, 0.0, 1.0) * 360.0


def scanline_frac(tick: int, period: int = 44) -> float:
    """Vertical scan-line position as a fraction 0..1 (sweeps down then up)."""
    return triangle(tick, period)


def _parse_hex(colour: str) -> tuple[int, int, int]:
    c = colour.lstrip("#")
    if len(c) != 6:
        raise ValueError(f"expected #rrggbb, got {colour!r}")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def hex_lerp(a: str, b: str, t: float) -> str:
    """Linearly interpolate two ``#rrggbb`` colours (t in [0,1]) -> ``#rrggbb``.

    Used for glow gradients and the welcome fade. ``t`` is clamped.
    """
    t = _clamp(t, 0.0, 1.0)
    ar, ag, ab = _parse_hex(a)
    br, bg, bb = _parse_hex(b)
    r = round(ar + (br - ar) * t)
    g = round(ag + (bg - ag) * t)
    bl = round(ab + (bb - ab) * t)
    return f"#{r:02x}{g:02x}{bl:02x}"


def glow_ramp(base: str, tick: int, period: int, steps: int = 4) -> list[str]:
    """Concentric halo colours from ``base`` fading toward BACKGROUND.

    Returns ``steps`` colours (outer->inner-ish) whose blend point breathes with
    the pulse, giving the rings a living glow. Deterministic and testable.
    """
    steps = max(1, steps)
    breathe = 0.35 + 0.45 * triangle(tick, period)  # 0.35..0.80
    out: list[str] = []
    for i in range(steps):
        frac = breathe * (i + 1) / steps
        out.append(hex_lerp(base, BACKGROUND, frac))
    return out


def phase_caption(phase: str, owner_name: str = "") -> str:
    """The big status word for a phase (uppercase, futuristic)."""
    name = (owner_name or "").strip().upper()
    return {
        LOCKED: "LOCKED",
        RECOGNIZING: "CHECKING AUTHORIZATION",
        DENIED: "UNAUTHORIZED",
        WELCOME: f"AUTHORIZED  -  WELCOME BACK, {name}" if name else "AUTHORIZED",
        ENROLLING: "ENROLLING",
    }.get(phase, "LOCKED")


def phase_subcaption(phase: str) -> str:
    """The small helper line under the caption."""
    return {
        LOCKED: "Face the camera to unlock  -  Esc for password",
        RECOGNIZING: "Verifying your identity",
        DENIED: "Access denied",
        WELCOME: "Access granted",
        ENROLLING: "Slowly turn your head to map every angle",
    }.get(phase, "")
