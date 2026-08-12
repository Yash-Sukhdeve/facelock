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

import re
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
    # Passive-PAD temporal quorum (k): frames in a burst that must INDEPENDENTLY
    # clear pad_threshold before a live verdict (k-of-n, mirrors match_votes).
    # Replaces max-across-frames aggregation (I2, fail-closed on the time axis).
    Field("liveness", "pad_min_live_frames", v_int(1, 15), 3, _UNSET, req="REQ-NF-11,FM-03"),
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
    Field("unlock", "owner_name", v_nonempty_str, "User", _UNSET, req="REQ-F-15,ASM-01"),
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
    # SAFE test mode: log OS-lock escalations but NEVER actuate the OS lock
    # (no loginctl/gdbus/xdg). security=True -> a bad value refuses to start,
    # exactly like tau/phase. Default false in BOTH phases (phase-independent).
    # The recommended surface is the ephemeral --dry-run CLI flag; this config
    # key is for CI/interactive use and is systemd-hard-gated (resolve_dry_run).
    Field("security", "dry_run", v_bool, False, _UNSET, security=True, req="DES-DRYRUN"),
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


# --------------------------------------------------------------------------- #
# SAFE dry-run resolution + systemd hard-gate (DES-DRYRUN, design section 4.2).
# --------------------------------------------------------------------------- #
class DryRunUnderSystemdError(RuntimeError):
    """Raised when dry-run is requested via config ONLY while under systemd.

    Fail-closed refusal (design section 4.2): a ``security.dry_run=true`` that
    arrives from config on a systemd-managed service is refused so a shipped
    unit can never silently disable OS-lock protection. An explicit ``--dry-run``
    on the command line is always honoured (the deliberate developer escape
    hatch). The caller (each ``main()``) prints this message and exits non-zero.
    """


def under_systemd() -> bool:
    """True when running as a ``Type=notify`` systemd service.

    Both ``INVOCATION_ID`` and ``NOTIFY_SOCKET`` are set by the units
    (``systemd/*.service``); ``NOTIFY_SOCKET`` is already consumed by the
    processes' ``_sd_notify``. Detection is env-only (no subprocess).
    """
    import os

    return ("INVOCATION_ID" in os.environ) or ("NOTIFY_SOCKET" in os.environ)


def resolve_dry_run(cli_flag: bool, cfg: "Config") -> bool:
    """Resolve the effective dry-run flag with the systemd hard-gate.

    Effective value is ``cli_flag OR cfg.security.dry_run``. If dry-run was
    requested ONLY via config (not the CLI) while under a systemd invocation,
    this refuses to start by raising :class:`DryRunUnderSystemdError` -- closing
    the "someone committed ``dry_run=true`` and shipped it" hole. An explicit
    ``--dry-run`` is always honoured.
    """
    want = bool(cli_flag) or bool(cfg.security.dry_run)
    if want and under_systemd() and not cli_flag:
        raise DryRunUnderSystemdError(
            "refusing to start: security.dry_run=true under systemd; dry-run is "
            "for interactive/CI use only -- pass --dry-run explicitly to override"
        )
    return want


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


# --------------------------------------------------------------------------- #
# Targeted config write-back: enroll --name -> config.toml (REQ-F-15).
# --------------------------------------------------------------------------- #
# Matches the ``owner_name = "<value>"`` assignment (any leading indent), up to
# but NOT including any trailing whitespace/inline comment, so those survive
# untouched. Scoped to a section body slice by the caller -- never matches
# outside ``[unlock]`` (e.g. a same-named key in a different table).
_UNLOCK_OWNER_NAME_RE = re.compile(r'(?m)^(?P<prefix>[ \t]*owner_name[ \t]*=[ \t]*)"(?:[^"\\]|\\.)*"')
_SECTION_HEADER_RE = re.compile(r'(?m)^\[(?P<name>[^\]]+)\]\s*$')


def update_owner_name(path: Path, name: str) -> tuple[bool, str]:
    """Rewrite ``[unlock] owner_name`` in an EXISTING ``config.toml`` in place.

    Persists ``facelock enroll --name <name>`` so the runtime "Welcome back"
    greeting (``guardian.py`` / ``fsm.py``, which read
    ``config.unlock.owner_name``) matches the enrolled owner instead of the
    shipped/code default (REQ-F-15). Only the ``owner_name`` value inside the
    ``[unlock]`` table is touched -- every other key, comment, and blank line
    in the file is preserved byte-for-byte. The file is rewritten atomically
    at mode 0600 via :func:`paths.secure_write_bytes`.

    This function NEVER raises. It returns ``(success, message)`` so callers
    -- notably :class:`~facelock.enroll.EnrollmentTool` -- can warn and
    continue rather than aborting an otherwise-successful enrollment over a
    config-write hiccup. This is a deliberately FAIL-SAFE (not fail-closed)
    path: a stale greeting name is a cosmetic issue, not a security one (the
    security-critical config keys are unaffected and still fail-closed via
    :func:`load_config`).
    """
    from . import paths as _paths

    try:
        if not path.exists():
            return False, f"config not found at {path}; owner_name not persisted"
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return False, f"could not read config {path}: {exc}"

    headers = list(_SECTION_HEADER_RE.finditer(text))
    body_start: int | None = None
    body_end = len(text)
    for i, m in enumerate(headers):
        if m.group("name").strip() == "unlock":
            body_start = m.end()
            body_end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
            break
    if body_start is None:
        return False, f"config {path} has no [unlock] section; owner_name not updated"

    body = text[body_start:body_end]
    escaped = name.replace("\\", "\\\\").replace('"', '\\"')
    new_body, count = _UNLOCK_OWNER_NAME_RE.subn(
        lambda m: f'{m.group("prefix")}"{escaped}"', body, count=1,
    )
    if count == 0:
        return False, f"config {path} [unlock] section has no owner_name key; not updated"

    new_text = text[:body_start] + new_body + text[body_end:]
    try:
        _paths.secure_write_bytes(path, new_text.encode("utf-8"), 0o600)
    except OSError as exc:
        return False, f"could not write config {path}: {exc}"
    return True, f"owner_name updated to {name!r} in {path}"
