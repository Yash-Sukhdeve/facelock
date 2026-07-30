"""console -- the JARVIS-style terminal HUD engine (pure, testable, no deps).

Every ``facelock`` CLI verb renders through this module so the terminal feels
like the console of a very high-tech machine: neon-framed panels, corner
brackets, telemetry rows, status reticles and boot lines -- all in the same
cyan/blue/magenta/green palette as the animated lock shield (``ui_theme``).

Design rules mirrored from the rest of the codebase:
  * **Zero dependencies.** Pure stdlib string maths so the whole look is
    unit-testable headless; a renderer just prints the strings.
  * **Degrades gracefully.** Colour is emitted only to a real TTY that is not
    ``NO_COLOR`` / ``TERM=dumb``; piped or redirected output is clean text, so
    ``facelock status --json | jq`` and log scraping keep working.
  * **Never the source of truth.** This is presentation only. It formats what
    the guardian reports; it holds no authority and never changes a decision.

All colours are the ``#rrggbb`` neon values from :mod:`facelock.ui_theme`,
emitted as 24-bit ("truecolor") ANSI. On terminals without truecolor the
sequences are still valid SGR and render as a near colour or plain text.
"""

from __future__ import annotations

import os
import sys
from typing import Iterable

from . import ui_theme

# -- palette (shared with the shield UI) ----------------------------------- #
CYAN = "#00e5ff"      # idle / primary accent
BLUE = "#3d9bff"      # scanning / info
RED = "#ff2d55"       # alert / locked / denied
GREEN = "#00e676"     # authorized / ok
VIOLET = "#a855f7"    # enrollment
AMBER = "#ffb300"     # warning / caution
TEXT = "#cfefff"      # bright caption text
DIM = "#6b7a8f"       # muted labels / rules
WHITE = "#e8eaed"

_RESET = "\x1b[0m"
_BOLD = "\x1b[1m"

# Box-drawing set for the HUD frame + Iron-Man corner reticles.
_H = "─"
_V = "│"
_TL, _TR, _BL, _BR = "╭", "╮", "╰", "╯"

DEFAULT_WIDTH = 66


# --------------------------------------------------------------------------- #
# Colour capability + low-level colouring.
# --------------------------------------------------------------------------- #
def supports_color(stream: object | None = None) -> bool:
    """True iff colour should be emitted to ``stream`` (default stdout).

    Honours ``NO_COLOR`` (any value disables), ``FORCE_COLOR`` (any value
    enables), ``TERM=dumb`` and non-TTY streams. This keeps piped/redirected
    output clean so machine consumers are never fed escape codes.
    """
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("FORCE_COLOR") is not None:
        return True
    if os.environ.get("TERM", "") == "dumb":
        return False
    stream = sys.stdout if stream is None else stream
    try:
        return bool(stream.isatty())  # type: ignore[attr-defined]
    except Exception:
        return False


def _rgb(hex_colour: str) -> tuple[int, int, int]:
    c = hex_colour.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


class Console:
    """Stateful renderer bound to one colour decision + width.

    Construct once per command (``Console.auto()``) and call the pure helpers.
    When colour is off every method returns plain text of the SAME visible
    width, so layout is identical piped or on a terminal.
    """

    def __init__(self, *, color: bool = True, width: int = DEFAULT_WIDTH) -> None:
        self.color = bool(color)
        self.width = max(24, int(width))

    @classmethod
    def auto(cls, *, width: int = DEFAULT_WIDTH, stream: object | None = None) -> "Console":
        return cls(color=supports_color(stream), width=width)

    # -- primitives -------------------------------------------------------- #
    def paint(self, text: str, hex_colour: str, *, bold: bool = False) -> str:
        """Wrap ``text`` in a truecolor SGR (no-op when colour is disabled)."""
        if not self.color or not text:
            return text
        r, g, b = _rgb(hex_colour)
        prefix = f"\x1b[38;2;{r};{g};{b}m" + (_BOLD if bold else "")
        return f"{prefix}{text}{_RESET}"

    def dim(self, text: str) -> str:
        return self.paint(text, DIM)

    # -- banner ------------------------------------------------------------ #
    def banner(self, subtitle: str = "") -> str:
        """The boot wordmark with a JARVIS subtitle underneath (neon gradient)."""
        grad = (CYAN, "#22ccff", BLUE)
        out = [self.paint(row, grad[i % len(grad)], bold=True)
               for i, row in enumerate(_WORDMARK)]
        block = "\n".join(out)
        if subtitle:
            block += "\n" + self.paint(subtitle.center(len(_WORDMARK[-1])), TEXT)
        return block

    # -- rules + framed panels --------------------------------------------- #
    def rule(self, label: str = "", colour: str = CYAN) -> str:
        """A horizontal neon rule, optionally with an inline label."""
        if not label:
            return self.paint(_H * self.width, colour)
        tag = f" {label} "
        right = max(0, self.width - _vis_len(tag) - 3)
        return (self.paint(_H * 3, colour)
                + self.paint(tag, TEXT, bold=True)
                + self.paint(_H * right, colour))

    def panel(self, title: str, rows: Iterable[str], *, colour: str = CYAN,
              footer: str = "") -> str:
        """Box ``rows`` in a neon frame with a titled header.

        ``rows`` are pre-rendered content lines (may contain colour codes); each
        is padded to the interior width using the VISIBLE length so the right
        border always lines up regardless of embedded SGR codes.
        """
        inner = self.width - 2
        title_txt = f" {title.upper()} "
        head_fill = max(0, inner - _vis_len(title_txt) - 2)
        top = (self.paint(_TL + _H, colour)
               + self.paint(title_txt, colour, bold=True)
               + self.paint(_H * head_fill + _H + _TR, colour))
        lines = [top]
        for row in rows:
            lines.append(self._body_line(row, colour))
        if footer:
            lines.append(self._body_line("", colour))
            for seg in self.wrap(footer, DIM):
                lines.append(self._body_line(seg, colour))
        bottom = self.paint(_BL + _H * inner + _BR, colour)
        return "\n".join(lines) + "\n" + bottom

    def wrap(self, text: str, colour: str = TEXT) -> list[str]:
        """Word-wrap plain ``text`` to the panel interior, returning painted rows.

        Use for any content that might exceed one line (disclosures, footers) so
        it never runs past the neon frame. Input must be uncoloured text.
        """
        width = max(8, self.width - 3)
        out: list[str] = []
        for para in text.split("\n"):
            if not para:
                out.append("")
                continue
            line = ""
            for word in para.split(" "):
                if line and len(line) + 1 + len(word) > width:
                    out.append(self.paint(line, colour))
                    line = word
                else:
                    line = f"{line} {word}" if line else word
            out.append(self.paint(line, colour))
        return out

    def _body_line(self, content: str, colour: str) -> str:
        inner = self.width - 2
        pad = max(0, inner - 1 - _vis_len(content))
        return (self.paint(_V, colour) + " " + content + " " * pad
                + self.paint(_V, colour))

    # -- rows -------------------------------------------------------------- #
    def kv(self, key: str, value: str, *, value_colour: str = WHITE,
           key_width: int = 22) -> str:
        """A ``label            value`` telemetry row (dim label, bright value)."""
        label = key[:key_width].ljust(key_width)
        return f"{self.paint(label, DIM)}{self.paint(str(value), value_colour, bold=True)}"

    def badge(self, text: str, colour: str) -> str:
        """A bracketed status token, e.g. ``[ LOCKED ]``."""
        return self.paint(f"[ {text} ]", colour, bold=True)

    def bar(self, frac: float, *, width: int = 24, colour: str = CYAN) -> str:
        """A segmented telemetry bar in ``▰▰▰▱▱ 60%`` style."""
        frac = 0.0 if frac < 0 else 1.0 if frac > 1 else float(frac)
        filled = int(round(frac * width))
        on = self.paint("▰" * filled, colour)
        off = self.dim("▱" * (width - filled))
        return f"{on}{off} {self.paint(f'{int(frac * 100)}%', colour, bold=True)}"


# --------------------------------------------------------------------------- #
# Visible-length helper (strip SGR when measuring for alignment).
# --------------------------------------------------------------------------- #
def _vis_len(text: str) -> int:
    """Length of ``text`` ignoring ANSI SGR escapes (for column alignment)."""
    out = 0
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "\x1b":
            j = text.find("m", i)
            if j == -1:
                break
            i = j + 1
            continue
        out += 1
        i += 1
    return out


# --------------------------------------------------------------------------- #
# The wordmark (compact block letters for FACELOCK).
# --------------------------------------------------------------------------- #
_WORDMARK = (
    "┏━╸┏━┓┏━╸┏━╸╻  ┏━┓┏━╸╻┏ ",
    "┣╸ ┣━┫┃  ┣╸ ┃  ┃ ┃┃  ┣┻┓",
    "╹  ╹ ╹┗━╸┗━╸┗━╸┗━┛┗━╸╹ ╹",
)


def phase_colour(phase: str) -> str:
    """Map a shield phase to its console accent colour."""
    return {
        ui_theme.LOCKED: CYAN,
        ui_theme.RECOGNIZING: BLUE,
        ui_theme.DENIED: RED,
        ui_theme.WELCOME: GREEN,
        ui_theme.ENROLLING: VIOLET,
    }.get(phase, CYAN)
