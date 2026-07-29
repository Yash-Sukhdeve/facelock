"""ShieldWindow (C11) + Greeter (C12) -- the guardian's user-facing surfaces.

Realizes REQ-F-14 (the face-dismissible convenience lock) and REQ-F-15 (the
"Welcome back, <name>" greeting). Both import tkinter LAZILY inside methods so
importing this module never requires a display -- tests and headless self-checks
can import ``facelock.shield`` without opening a window.

ShieldWindow is an X11 full-screen, override-redirect, always-on-top window that
grabs pointer + keyboard input (the prototype convenience lock). It is only as
strong as its owning process, which is exactly why it lives in the *guardian*
(separate from perception) and why the guardian escalates to the real OS lock on
stranger/panic/heartbeat-miss (SI-P4, ADR-3).

Safety: the shield NEVER traps the user. A dedicated key (Escape) invokes an
``on_password_escape`` callback so the guardian can drop the shield and engage
the real OS lock screen (password), preserving SI (the password path is always
reachable).

Wayland note (OQ-10): Wayland has no global input grab / override-redirect; a
Wayland shield would use ``ext-session-lock-v1`` behind this same interface.
When no display is available the shield reports failure and the guardian relies
on the OS-lock backend as the barrier (still fail-closed).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Callable


def has_display() -> bool:
    """True if an X11 or Wayland display is present."""
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


class ShieldWindow:
    """X11 input-grabbing full-screen shield (prototype convenience lock)."""

    BACKGROUND = "#05070d"  # matches ui_theme.BACKGROUND (fallback label UI)
    # Status text colours per phase (feedback so the user/intruder sees state).
    _NEUTRAL = "#e8eaed"   # plain "Locked" states
    _BLUE = "#8ab4f8"      # recognizing the owner
    _RED = "#ea4335"       # unauthorized / denied
    _RED_DIM = "#7a1c15"   # denied blink off-phase
    _GREEN = "#34a853"     # welcome (unlock granted)

    def __init__(
        self,
        owner_name: str = "Yash",
        *,
        on_password_escape: Callable[[], None] | None = None,
    ) -> None:
        self.owner_name = owner_name
        self.on_password_escape = on_password_escape
        self._root = None  # tkinter.Tk, created lazily
        self._status_var = None
        self._status_label = None  # the status Label (fallback UI only)
        self._canvas = None        # tkinter.Canvas (futuristic UI)
        self._mode: str | None = None  # "canvas" | "label"
        self._up = False
        # Animated-status state (driven by pump()): phase in
        # {locked, recognizing, denied, welcome}; the Canvas renders it, the
        # label fallback cycles dots / blinks.
        self._anim_kind: str | None = None
        self._anim_tick = 0
        self._base_text = "Locked"
        self._phase = "locked"
        self._caption_name = owner_name
        self._progress = 0.0            # k-of-n verification progress, 0..1
        self._votes = (0, 0)           # (votes_so_far, votes_needed)
        self._w = 0
        self._h = 0

    @property
    def is_up(self) -> bool:
        return self._up

    @staticmethod
    def _dots(tick: int) -> str:
        """Cycle '', '.', '..', '...' about every 0.3 s at a ~0.1 s pump."""
        return "." * ((tick // 3) % 4)

    def raise_shield(self, status: str = "Locked") -> bool:
        """Show the shield, grabbing input. Returns True on success.

        The Tk root is created ONCE and then reused across lock/unlock cycles
        (dismiss hides it; raise re-shows it). Creating a fresh ``tk.Tk()`` on
        every lock -- as the previous implementation did -- hits the well-known
        Tkinter multiple-root gotcha (a second ``Tk()`` after the first is
        destroyed loses the default root and can fail to map/grab), which is the
        root cause of "locked the 1st time but not the 2nd". Reusing one root
        makes re-locking reliable indefinitely.

        On any failure (no display, tkinter missing, grab denied) returns False
        so the guardian can fall back to the OS-lock backend (fail-closed).
        """
        if self._up:
            self.set_status(status)
            return True
        if not has_display():
            return False
        try:
            import tkinter as tk
        except Exception:
            return False
        try:
            if self._root is None:
                self._build_root(tk, status)
            else:
                # Reuse the existing (withdrawn) root.
                self.set_status(status)
                self._root.deiconify()
                self._root.attributes("-fullscreen", True)
                self._root.attributes("-topmost", True)
            self._root.update_idletasks()
            self._root.update()
            self._grab()                # grab input AFTER the window is mapped
            self._root.focus_force()
            self._up = True
            return True
        except Exception:
            self._safe_destroy()
            return False

    def _build_root(self, tk: "object", status: str) -> None:
        """Create the Tk root once, preferring the futuristic Canvas UI.

        Falls back to a plain-label layout if the Canvas cannot be built, so the
        shield (and its input grab) is never lost to a rendering problem.
        """
        from . import ui_theme

        root = tk.Tk()  # type: ignore[attr-defined]
        root.title("facelock")
        root.configure(bg=ui_theme.BACKGROUND)
        root.attributes("-fullscreen", True)
        root.attributes("-topmost", True)
        try:
            root.overrideredirect(True)  # no window manager decorations
        except Exception:
            pass
        self._root = root
        self._base_text = status
        try:
            self._w = int(root.winfo_screenwidth())
            self._h = int(root.winfo_screenheight())
        except Exception:
            self._w, self._h = 1920, 1080
        try:
            self._build_canvas(tk, ui_theme)
            self._mode = "canvas"
        except Exception:
            self._build_labels(tk, status)
            self._mode = "label"

        # Consume every key so shortcuts cannot escape the shield, except the
        # deliberate password-escape (Escape). Bindings survive reuse.
        root.bind_all("<Key>", lambda _e: "break")
        root.bind_all("<Escape>", self._on_escape)
        root.protocol("WM_DELETE_WINDOW", lambda: "break")

    def _build_canvas(self, tk: "object", ui_theme: "object") -> None:
        """Full-window Canvas for the animated futuristic lock screen."""
        canvas = tk.Canvas(  # type: ignore[attr-defined]
            self._root, bg=ui_theme.BACKGROUND, highlightthickness=0,
            width=self._w, height=self._h, bd=0,
        )
        canvas.pack(fill="both", expand=True)
        self._canvas = canvas
        self._render()

    def _build_labels(self, tk: "object", status: str) -> None:
        """Minimal, robust fallback UI (used only if the Canvas fails)."""
        self._status_var = tk.StringVar(value=status)  # type: ignore[attr-defined]
        tk.Label(  # type: ignore[attr-defined]
            self._root, text="facelock", fg="#8ab4f8", bg=self.BACKGROUND,
            font=("DejaVu Sans", 40, "bold"),
        ).pack(expand=True)
        self._status_label = tk.Label(  # type: ignore[attr-defined]
            self._root, textvariable=self._status_var, fg=self._NEUTRAL,
            bg=self.BACKGROUND, font=("DejaVu Sans", 22),
        )
        self._status_label.pack()
        tk.Label(  # type: ignore[attr-defined]
            self._root,
            text=("Face the camera to unlock.\n"
                  "Press Esc for the OS password lock screen."),
            fg="#9aa0a6", bg=self.BACKGROUND, font=("DejaVu Sans", 14),
        ).pack(pady=20)

    # -- futuristic Canvas renderer (animated by pump()) ------------------ #
    def _render(self) -> None:
        """Draw one animated frame of the lock scene. No-op unless Canvas mode."""
        if self._canvas is None:
            return
        from . import ui_theme

        c = self._canvas
        tick = self._anim_tick
        phase = self._phase
        th = ui_theme.theme_for(phase)
        cx, cy = self._w / 2.0, self._h * 0.42
        base_r = max(80.0, min(self._w, self._h) * 0.16)

        try:
            c.delete("all")
            # Concentric breathing glow halos (outer -> in).
            halos = ui_theme.glow_ramp(th.ring, tick, th.pulse_period, steps=4)
            for i, colour in enumerate(halos):
                rr = base_r * (1.9 - 0.28 * i)
                c.create_oval(cx - rr, cy - rr, cx + rr, cy + rr,
                              outline=colour, width=2)

            # Welcome: the ring expands outward with the pulse.
            grow = 1.0 + (0.18 * ui_theme.triangle(tick, th.pulse_period)
                          if phase == ui_theme.WELCOME else 0.0)
            r = base_r * grow

            # Denied: blink the main ring between full and dim.
            ring_colour = th.ring
            if phase == ui_theme.DENIED and not ui_theme.blink_on(tick, 6):
                ring_colour = ui_theme.hex_lerp(th.ring, ui_theme.BACKGROUND, 0.55)
            c.create_oval(cx - r, cy - r, cx + r, cy + r,
                          outline=ring_colour, width=4)

            # Rotating scan arc (the "it's working" motion).
            start, extent = ui_theme.ring_sweep(tick, th.spin_dps)
            ar = r * 1.14
            c.create_arc(cx - ar, cy - ar, cx + ar, cy + ar,
                         start=start, extent=extent, style="arc",
                         outline=th.accent, width=5)
            c.create_arc(cx - ar, cy - ar, cx + ar, cy + ar,
                         start=start + 180, extent=extent * 0.6, style="arc",
                         outline=th.accent, width=3)

            # Cardinal tick marks.
            for ang in (0, 90, 180, 270):
                import math
                a = math.radians(ang)
                x0 = cx + math.cos(a) * (r * 1.28)
                y0 = cy - math.sin(a) * (r * 1.28)
                x1 = cx + math.cos(a) * (r * 1.4)
                y1 = cy - math.sin(a) * (r * 1.4)
                c.create_line(x0, y0, x1, y1, fill=th.accent, width=2)

            # Face reticle + a downward scan line while recognizing.
            rr = r * 0.55
            c.create_oval(cx - rr, cy - rr * 1.25, cx + rr, cy + rr * 1.25,
                          outline=ui_theme.hex_lerp(th.glow, th.ring, 0.6), width=2)
            if phase == ui_theme.RECOGNIZING:
                sy = cy - r + (2 * r) * ui_theme.scanline_frac(tick, 34)
                c.create_line(cx - r * 0.8, sy, cx + r * 0.8, sy,
                              fill=th.accent, width=2)

            # Brand + caption + sub-caption.
            c.create_text(cx, cy - r * 1.75, text="f a c e l o c k",
                          fill="#8ab4f8", font=("DejaVu Sans", 26, "bold"))
            caption = ui_theme.phase_caption(phase, self._caption_name)
            if phase == ui_theme.RECOGNIZING:
                caption = caption + self._dots(tick)
            c.create_text(cx, cy + r + 70, text=caption, fill=th.text,
                          font=("DejaVu Sans", 40, "bold"))
            c.create_text(cx, cy + r + 118, text=ui_theme.phase_subcaption(phase),
                          fill="#9aa0a6", font=("DejaVu Sans", 15))

            # CHECKING AUTHORIZATION: a real k-of-n verification progress bar.
            if phase == ui_theme.RECOGNIZING:
                bw = self._w * 0.34
                bh = 16
                bx = cx - bw / 2
                by = cy + r + 150
                frac = max(0.0, min(1.0, self._progress))
                c.create_rectangle(bx - 2, by - 2, bx + bw + 2, by + bh + 2,
                                   outline=th.accent, width=2)
                if frac > 0:
                    c.create_rectangle(bx, by, bx + bw * frac, by + bh,
                                       fill=th.ring, outline="")
                vk, vn = self._votes
                label = f"{int(frac * 100)}%"
                if vn:
                    label += f"   {vk}/{vn} checks"
                c.create_text(cx, by + bh + 22, text=label, fill=th.text,
                              font=("DejaVu Sans", 16))
        except Exception:
            # A rendering hiccup must never crash the guardian loop.
            pass

    def _grab(self) -> None:
        """Grab pointer + keyboard globally, falling back to a local grab."""
        if self._root is None:
            return
        try:
            self._root.grab_set_global()
        except Exception:
            # Local grab is weaker but still consumes app-directed input.
            try:
                self._root.grab_set()
            except Exception:
                pass

    def _on_escape(self, _event: object = None) -> str:
        if self.on_password_escape is not None:
            try:
                self.on_password_escape()
            except Exception:
                pass
        return "break"

    # -- status surfaces (feedback the user/intruder sees) ---------------- #
    def _apply(self, text: str, colour: str) -> None:
        """Set the status text + colour (no-op safe when headless)."""
        self._base_text = text
        if self._status_var is not None:
            try:
                self._status_var.set(text)
            except Exception:
                pass
        if self._status_label is not None:
            try:
                self._status_label.config(fg=colour)
            except Exception:
                pass

    def _set_phase(self, phase: str, text: str, colour: str, anim: str | None) -> None:
        """Update the current phase + drive whichever UI is active."""
        self._phase = phase
        self._anim_kind = anim
        self._anim_tick = 0
        self._apply(text, colour)      # label fallback text/colour (no-op headless)
        if self._mode == "canvas" and self._up:
            self._render()             # immediate repaint on phase change

    def set_status(self, status: str) -> None:
        """Plain locked status (neutral); clears any animation."""
        self._set_phase("locked", status, self._NEUTRAL, None)

    def set_recognizing(self, progress: float = 0.0, votes_k: int = 0,
                        votes_need: int = 0) -> None:
        """CHECKING AUTHORIZATION: show a real k-of-n progress bar (REQ-F-12).

        ``progress`` (0..1) is the matcher's actual verification progress; the
        vote counts annotate the bar. Display-only -- the grant still requires the
        genuine decision.
        """
        self._progress = max(0.0, min(1.0, float(progress)))
        self._votes = (int(votes_k), int(votes_need))
        pct = int(self._progress * 100)
        self._set_phase("recognizing", f"Checking authorization  {pct}%",
                        self._BLUE, "recognizing")

    def set_denied(self, text: str = "Unauthorized user") -> None:
        """A non-owner is at the camera: blinking red warning (REQ-F-11)."""
        self._set_phase("denied", text, self._RED, "denied")

    def set_welcome(self, name: str) -> None:
        """AUTHORIZED verdict: green 'Welcome back, <name>' splash (REQ-F-15)."""
        self._caption_name = name
        self._set_phase("welcome", f"AUTHORIZED - Welcome back, {name}",
                        self._GREEN, None)

    def pump(self) -> None:
        """Process pending tk events + advance the animation one frame.

        Called every guardian loop (~0.1 s). Only touches widgets while the
        shield is up; a no-op headless.
        """
        if self._root is None or not self._up:
            return
        try:
            self._anim_tick += 1
            if self._mode == "canvas":
                self._render()
            elif self._anim_kind == "recognizing":
                if self._status_var is not None:
                    self._status_var.set(self._base_text + self._dots(self._anim_tick))
            elif self._anim_kind == "denied" and self._status_label is not None:
                on = (self._anim_tick // 4) % 2 == 0
                self._status_label.config(fg=self._RED if on else self._RED_DIM)
            self._root.update()
        except Exception:
            pass

    def dismiss(self) -> None:
        """Release the grab and HIDE the window, keeping the root for reuse.

        Idempotent. The root is withdrawn (not destroyed) so the next
        ``raise_shield`` re-shows the same window -- see the class/raise docs for
        why recreating the root each cycle is unreliable. The desktop is fully
        usable after dismiss: the input grab is released and the window hidden.
        """
        if self._root is not None:
            try:
                self._root.grab_release()
            except Exception:
                pass
            try:
                self._root.withdraw()
            except Exception:
                # If hiding fails, fall back to a hard destroy so we never leave
                # a grabbing window on screen (fail-safe for usability).
                self._safe_destroy()
        self._up = False

    def _safe_destroy(self) -> None:
        if self._root is not None:
            try:
                self._root.grab_release()
            except Exception:
                pass
            try:
                self._root.destroy()
            except Exception:
                pass
        self._root = None
        self._status_var = None
        self._status_label = None
        self._canvas = None
        self._mode = None


class Greeter:
    """Transient "Welcome back, <name>" notification (REQ-F-15, name-only)."""

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled

    def show(self, name: str, ttl_s: int = 3) -> None:
        """Show the greeting; non-fatal on failure (never blocks unlock)."""
        if not self.enabled:
            return
        message = f"Welcome back, {name}"
        if shutil.which("notify-send") is not None:
            try:
                subprocess.run(
                    ["notify-send", "-t", str(int(ttl_s * 1000)), "facelock", message],
                    check=False, timeout=2.0,
                )
                return
            except (subprocess.TimeoutExpired, OSError):
                pass
        # Fallback: a stdout line. Notification failure must never affect
        # the unlock/lock path (REQ-F-15, I-10).
        try:
            print(f"[facelock] {message}", flush=True)
        except Exception:
            pass
