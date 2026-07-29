"""ConfigLoader (C13) -- TOML config with typed, range-validated settings.

Realizes design section 12. Parsing uses the Python 3.11 stdlib ``tomllib``
(zero extra dependency, REQ-NF-21 / ADR-5). Every setting has a type, a
default, and a validated range/enum, each traced to a REQ-ID.

Fail-closed rules (REQ-F-23, design section 12.1):
  * On an out-of-range / unparsable value the loader either REFUSES to start
    (default) or substitutes the documented default -- selected by
    ``config.on_invalid`` (``refuse`` | ``default``).
  * Security-critical keys (tau, fmr_target, stranger.policy, liveness.mode,
    security.phase, security.template_encryption) are ALWAYS refuse-on-invalid,
    regardless of ``config.on_invalid`` -- a bad security value never silently
    defaults.

The result is a frozen, attribute-accessible :class:`Config` tree so callers
read e.g. ``cfg.recognition.tau`` / ``cfg.stranger.policy``.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .errors import ConfigError

# Phase normalisation: the design table uses P/H; the task brief uses the
# words "prototype"/"hardening"/"screensaver-only". Accept all, normalise to P/H.
_PHASE_ALIASES = {
    "p": "P",
    "prototype": "P",
    "screensaver-only": "P",
    "screensaver_only": "P",
    "h": "H",
    "hardening": "H",
    "hardened": "H",
}


# --------------------------------------------------------------------------- #
# Validators. Each returns the coerced value or raises ValueError(message).
# --------------------------------------------------------------------------- #
def v_int(lo: int, hi: int) -> Callable[[Any], int]:
    def _v(x: Any) -> int:
        if isinstance(x, bool) or not isinstance(x, int):
            raise ValueError(f"expected integer in [{lo},{hi}]")
        if not (lo <= x <= hi):
            raise ValueError(f"out of range [{lo},{hi}]")
        return x

    return _v


def v_float(lo: float, hi: float) -> Callable[[Any], float]:
    def _v(x: Any) -> float:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise ValueError(f"expected number in [{lo},{hi}]")
        x = float(x)
        if not (lo <= x <= hi):
            raise ValueError(f"out of range [{lo},{hi}]")
        return x

    return _v


def v_bool(x: Any) -> bool:
    if not isinstance(x, bool):
        raise ValueError("expected boolean")
    return x


def v_enum(*choices: str) -> Callable[[Any], str]:
    def _v(x: Any) -> str:
        if not isinstance(x, str) or x not in choices:
            raise ValueError(f"expected one of {choices!r}")
        return x

    return _v


def v_nonempty_str(x: Any) -> str:
    if not isinstance(x, str) or not x.strip():
        raise ValueError("expected non-empty string")
    return x


def v_str(x: Any) -> str:
    if not isinstance(x, str):
        raise ValueError("expected string")
    return x


def v_resolution(x: Any) -> list[int]:
    if (
        not isinstance(x, (list, tuple))
        or len(x) != 2
        or any(isinstance(i, bool) or not isinstance(i, int) or i <= 0 for i in x)
    ):
        raise ValueError("expected [width, height] positive integers")
    return [int(x[0]), int(x[1])]


def v_phase(x: Any) -> str:
    if not isinstance(x, str):
        raise ValueError("expected string P|H")
    norm = _PHASE_ALIASES.get(x.strip().lower())
    if norm is None:
        raise ValueError("expected one of P|H (or prototype|hardening)")
    return norm


def v_reason_subset(x: Any) -> list[str]:
    allowed = {
        "away",
        "stranger",
        "panic",
        "camera_loss",
        "suspend",
        "shutdown",
        "cooldown",
        "heartbeat_miss",
    }
    if not isinstance(x, list) or any(i not in allowed for i in x):
        raise ValueError(f"expected subset of {sorted(allowed)}")
    return list(x)


def v_persist_frames_false(x: Any) -> bool:
    # REQ-NF-13: raw frames must never be persisted in production.
    if not isinstance(x, bool):
        raise ValueError("expected boolean")
    if x is True:
        raise ValueError("privacy.persist_frames MUST be false (REQ-NF-13)")
    return x


# --------------------------------------------------------------------------- #
# Field specification. Defaults are given per phase where the design differs.
# ``security`` marks always-refuse-on-invalid keys (design section 12.2).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Field:
    section: str
    key: str
    validator: Callable[[Any], Any]
    default_p: Any
    default_h: Any = None  # falls back to default_p when None-sentinel unused
    security: bool = False
    req: str = ""

    def default_for(self, phase: str) -> Any:
        if self.default_h is _UNSET:
            return self.default_p
        return self.default_h if phase == "H" else self.default_p


_UNSET = object()

# Model-path defaults are resolved lazily (see Config.resolve_model_paths) so the
# config validates even before models are downloaded (existence is a load-time
# FM-11 check, not a parse-time check).
SCHEMA: tuple[Field, ...] = (
    # camera
    Field("camera", "device", v_str, "/dev/video0", _UNSET, req="REQ-F-05"),
    Field("camera", "resolution", v_resolution, [640, 480], _UNSET, req="REQ-F-05,ASM-08"),
    Field("camera", "pixel_format", v_enum("YUYV", "MJPG"), "YUYV", _UNSET, req="ASM-08"),
    Field("camera", "fps_active", v_int(1, 30), 5, _UNSET, req="REQ-NF-01"),
    Field("camera", "fps_idle", v_int(0, 5), 1, _UNSET, req="REQ-NF-06,FM-07"),
    Field("camera", "long_absence_release_s", v_int(30, 86400), 120, _UNSET, req="FM-07"),
    Field("camera", "loss_grace_s", v_int(0, 30), 5, _UNSET, req="FM-01"),
    # detection
    Field("detection", "model_path", v_str, "", _UNSET, req="REQ-F-06,FM-11"),
    Field("detection", "confidence_floor", v_float(0.5, 0.99), 0.90, _UNSET, req="REQ-F-06,FM-05"),
    Field("detection", "min_face_px", v_int(40, 320), 80, _UNSET, req="REQ-F-02,FM-05"),
    Field("detection", "nms_threshold", v_float(0.1, 0.9), 0.30, _UNSET, req="REQ-F-06"),
    # recognition
    Field("recognition", "model_path", v_str, "", _UNSET, req="REQ-F-07,FM-11"),
    Field("recognition", "metric", v_enum("cosine", "l2"), "cosine", _UNSET, req="DES-3.2"),
    # tau: 0 means "use the template's calibrated tau" (the normal path).
    Field("recognition", "tau", v_float(0.0, 1.0), 0.0, _UNSET, security=True, req="REQ-NF-10"),
    # tau_seed / tau_floor: SFace published cosine operating point (design 3.2 step 1).
    Field("recognition", "tau_seed", v_float(0.0, 1.0), 0.363, _UNSET, req="REQ-NF-10"),
    Field("recognition", "tau_floor", v_float(0.0, 1.0), 0.363, _UNSET, security=True, req="REQ-NF-10,REQ-NF-22"),
    Field("recognition", "fmr_target", v_float(1e-6, 0.1), 0.01, 0.001, security=True, req="REQ-NF-10,ASM-05"),
    Field("recognition", "fnmr_target", v_float(1e-6, 0.5), 0.05, 0.03, req="REQ-NF-10"),
    Field("recognition", "probe_frames", v_int(3, 15), 5, _UNSET, req="REQ-NF-02,FM-03"),
    Field("recognition", "match_votes", v_int(1, 15), 3, _UNSET, req="FM-02,FM-03"),
    # Multi-pose matching: score each probe against the best of several enrolled
    # pose sub-templates so off-angle faces authenticate easily (RGB-only, no 3D).
    Field("recognition", "pose_templates", v_bool, True, _UNSET, req="REQ-F-07,ASM-03"),
    Field("recognition", "pose_max", v_int(1, 15), 5, _UNSET, req="REQ-F-07"),
    # presence
    Field("presence", "away_dwell_s", v_int(5, 600), 30, _UNSET, req="REQ-F-10,ASM-02,OQ-1"),
    Field("presence", "poll_s", v_float(0.2, 5.0), 1.0, _UNSET, req="REQ-F-09"),
    Field("presence", "grace_frames", v_int(1, 10), 2, _UNSET, req="REQ-F-09"),
    # stranger
    Field("stranger", "policy", v_enum("lenient", "strict"), "lenient", _UNSET, security=True, req="REQ-F-11,ASM-03,OQ-2"),
    Field("stranger", "dwell_s", v_int(0, 30), 3, _UNSET, req="REQ-F-11,NF-04"),
    # liveness
    Field("liveness", "mode", v_enum("off", "blink", "turn", "passive", "full"), "off", "full", security=True, req="REQ-F-19"),
    Field("liveness", "challenge_timeout_s", v_int(1, 15), 4, _UNSET, req="REQ-F-19,FM-04"),
    Field("liveness", "pad_model_path", v_str, "", _UNSET, req="REQ-NF-11"),
    Field("liveness", "pad_threshold", v_float(0.0, 1.0), 0.5, _UNSET, req="REQ-NF-11"),
    Field("liveness", "turn_yaw_deg", v_float(5.0, 60.0), 15.0, _UNSET, req="REQ-F-19"),
    # lock
    Field("lock", "backend", v_enum("auto", "gnome_dbus", "loginctl", "xdg"), "auto", _UNSET, req="REQ-F-13,NF-19"),
    Field("lock", "shield", v_bool, True, _UNSET, req="REQ-F-14"),
    # Prototype default: a stranger/away raises the (face-dismissable) SHIELD, not
    # the OS password lock, so the returning owner always face-unlocks. Only
    # genuine fail-closed events escalate to the password path (SI-P4, FM-08/13).
    Field("lock", "escalate_os_lock_on", v_reason_subset,
          ["panic", "heartbeat_miss", "suspend"], _UNSET, req="SI-P4,FM-08,FM-13"),
    # Turn the physical monitor off (DPMS) while the shield is up; on to unlock.
    Field("lock", "screen_off", v_bool, True, _UNSET, req="REQ-F-14"),
    Field("lock", "verify_engaged_ms", v_int(100, 3000), 500, _UNSET, req="SI-P5,FM-16"),
    # unlock
    Field("unlock", "max_fail_attempts", v_int(1, 20), 5, _UNSET, req="REQ-F-25,ASM-11"),
    Field("unlock", "cooldown_s", v_int(5, 600), 30, _UNSET, req="REQ-F-25,FM-15"),
    Field("unlock", "owner_name", v_nonempty_str, "Yash", _UNSET, req="REQ-F-15,ASM-01"),
    Field("unlock", "greeting", v_bool, True, _UNSET, req="REQ-F-15"),
    # On unlock, hold a "Welcome back" splash on the shield this long before it
    # dismisses (0 = dismiss instantly). Small by default to keep unlock snappy.
    Field("unlock", "welcome_hold_s", v_float(0.0, 3.0), 0.6, _UNSET, req="REQ-F-15"),
    # privacy
    Field("privacy", "persist_frames", v_persist_frames_false, False, _UNSET, security=True, req="REQ-NF-13"),
    Field("privacy", "camera_indicator", v_bool, True, _UNSET, req="REQ-F-27,NF-16"),
    # logging
    Field("logging", "level", v_enum("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"), "INFO", _UNSET, req="REQ-F-22"),
    Field("logging", "max_size_mb", v_int(1, 100), 10, _UNSET, req="REQ-NF-09,FM-12"),
    Field("logging", "rotate_count", v_int(1, 20), 5, _UNSET, req="REQ-NF-09"),
    # service
    Field("service", "autostart", v_enum("systemd", "xdg"), "systemd", _UNSET, req="ASM-10,OQ-6"),
    Field("service", "heartbeat_sec", v_int(1, 10), 2, _UNSET, req="FM-08,NF-23"),
    Field("service", "restart", v_enum("always", "on-failure", "no"), "always", _UNSET, req="REQ-F-26"),
    # security
    Field("security", "template_encryption", v_enum("none", "keyring", "keyfile"), "none", "keyring", security=True, req="REQ-NF-14"),
    Field("security", "audit", v_bool, False, True, req="REQ-NF-25"),
    Field("security", "phase", v_phase, "P", _UNSET, security=True, req="DES-13"),
    # config meta
    Field("config", "on_invalid", v_enum("refuse", "default"), "refuse", _UNSET, req="REQ-F-23"),
    # runtime
    Field("runtime", "threads", v_int(1, 24), 4, _UNSET, req="REQ-NF-06/07"),
)


@dataclass(frozen=True)
class Section:
    """Attribute-accessible view over a validated section dict."""

    _values: dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - trivial
        try:
            return object.__getattribute__(self, "_values")[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def as_dict(self) -> dict[str, Any]:
        return dict(self._values)


@dataclass(frozen=True)
class Config:
    """Validated configuration tree (design section 12)."""

    sections: dict[str, Section]
    warnings: tuple[str, ...]
    source_path: Path | None
    phase: str

    def __getattr__(self, name: str) -> Section:
        try:
            return object.__getattribute__(self, "sections")[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def get(self, section: str, key: str) -> Any:
        return self.sections[section]._values[key]

    def resolve_model_paths(self, models_dir: Path) -> "Config":
        """Return a copy with empty model paths filled from ``models_dir``.

        Empty (unset) model paths default to the bundled filenames under the
        data dir; explicit paths in the config are left untouched.
        """
        from . import paths as _paths

        mapping = {
            ("detection", "model_path"): _paths.YUNET_MODEL,
            ("recognition", "model_path"): _paths.SFACE_MODEL,
            ("liveness", "pad_model_path"): _paths.PAD_MODEL,
        }
        new_sections = {name: Section(dict(sec._values)) for name, sec in self.sections.items()}
        for (sec_name, key), filename in mapping.items():
            if not new_sections[sec_name]._values.get(key):
                new_sections[sec_name]._values[key] = str(models_dir / filename)
        return Config(new_sections, self.warnings, self.source_path, self.phase)

    def as_dict(self) -> dict[str, dict[str, Any]]:
        return {name: sec.as_dict() for name, sec in self.sections.items()}


def _determine_phase(raw: dict[str, Any]) -> str:
    """Resolve the phase first, since defaults depend on it."""
    try:
        value = raw.get("security", {}).get("phase", "P")
    except AttributeError:
        raise ConfigError("security section must be a table", ["security: not a table"])
    try:
        return v_phase(value)
    except ValueError as exc:
        # phase is security-critical -> always refuse.
        raise ConfigError(
            f"security.phase invalid: {exc}", [f"security.phase: {exc}"]
        ) from exc


def load_config(source: Path | str | None = None, *, raw: dict[str, Any] | None = None) -> Config:
    """Load and validate configuration.

    Parameters
    ----------
    source:
        Path to a TOML file. If ``None`` and ``raw`` is ``None`` the default
        config path is used; a missing file yields the all-default config.
    raw:
        Pre-parsed table (used by tests and by ``reload_config``). Takes
        precedence over ``source`` when given.

    Raises
    ------
    ConfigError:
        If any security-critical key is invalid, or if ``config.on_invalid``
        resolves to ``refuse`` and any key is invalid, or if the file cannot be
        parsed.
    """
    source_path: Path | None = None
    if raw is None:
        if source is None:
            from . import paths as _paths

            source = _paths.config_path()
        source_path = Path(source)
        if not source_path.exists():
            raw = {}
        else:
            try:
                with open(source_path, "rb") as fh:
                    raw = tomllib.load(fh)
            except (tomllib.TOMLDecodeError, OSError) as exc:
                raise ConfigError(f"cannot parse config {source_path}: {exc}",
                                  [f"parse: {exc}"]) from exc

    if not isinstance(raw, dict):
        raise ConfigError("config root must be a table", ["root: not a table"])

    phase = _determine_phase(raw)

    # Determine on_invalid policy early (it governs non-security keys).
    on_invalid_raw = raw.get("config", {}).get("on_invalid", "refuse")
    try:
        on_invalid = v_enum("refuse", "default")(on_invalid_raw)
    except ValueError:
        # config.on_invalid itself invalid -> safest interpretation is refuse.
        raise ConfigError(
            "config.on_invalid must be 'refuse' or 'default'",
            [f"config.on_invalid: {on_invalid_raw!r}"],
        )

    values: dict[str, dict[str, Any]] = {}
    hard_errors: list[str] = []
    warnings: list[str] = []

    for spec in SCHEMA:
        sec = raw.get(spec.section, {})
        if not isinstance(sec, dict):
            hard_errors.append(f"{spec.section}: not a table")
            continue
        present = spec.key in sec
        default = spec.default_for(phase)
        if not present:
            values.setdefault(spec.section, {})[spec.key] = default
            continue
        try:
            coerced = spec.validator(sec[spec.key])
            values.setdefault(spec.section, {})[spec.key] = coerced
        except ValueError as exc:
            fqk = f"{spec.section}.{spec.key}"
            if spec.security or on_invalid == "refuse":
                hard_errors.append(f"{fqk}: {exc}")
                # still record default so downstream cross-checks do not KeyError
                values.setdefault(spec.section, {})[spec.key] = default
            else:
                warnings.append(f"{fqk}: {exc} -> using default {default!r}")
                values.setdefault(spec.section, {})[spec.key] = default

    # Cross-field checks.
    rec = values["recognition"]
    if rec["match_votes"] > rec["probe_frames"]:
        msg = (
            f"recognition.match_votes ({rec['match_votes']}) must be "
            f"<= recognition.probe_frames ({rec['probe_frames']})"
        )
        if on_invalid == "refuse":
            hard_errors.append(msg)
        else:
            warnings.append(msg + " -> clamping match_votes")
            rec["match_votes"] = rec["probe_frames"]

    # liveness.mode 'off' is only permitted in the Prototype phase (design I-6).
    if values["liveness"]["mode"] == "off" and phase == "H":
        hard_errors.append(
            "liveness.mode='off' is forbidden when security.phase=H "
            "(liveness is mandatory in Hardening, REQ-F-19)"
        )
    # Non-'off' liveness that requires a model but has none is caught at load
    # time (fail-closed); config parse allows it so 'test' can report clearly.

    if hard_errors:
        raise ConfigError(
            "configuration is invalid; refusing to start (fail-closed, REQ-F-23)",
            hard_errors,
        )

    sections = {name: Section(vals) for name, vals in values.items()}
    return Config(
        sections=sections,
        warnings=tuple(warnings),
        source_path=source_path,
        phase=phase,
    )
