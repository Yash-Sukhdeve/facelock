"""facelock CLI (design section 10.4) -- thin control client + offline tools.

Verbs:
  enroll   -- guided enrollment / re-enroll (offline, C16)      REQ-F-01/02/03
  delete   -- secure-delete the template + artefacts            REQ-F-04
  calibrate-- re-run tau calibration on the current template    REQ-NF-10
  lock     -- immediate panic lock                              REQ-F-25
  disable  -- turn face-unlock off (password still works)       REQ-F-25
  enable   -- turn face-unlock back on                          REQ-F-25
  status   -- show guardian state / health (no images)          REQ-NF-24
  test     -- detection/verify/PAD self-test (fps, latency, score)  REQ-NF-01/02/10
  disclosure -- print the prototype spoof-limitation notice     REQ-F-17
  config-check -- validate the config file (fail-closed report) REQ-F-23

Error contract: every verb returns a non-zero exit code + a message on failure.
Control verbs talk to the guardian over the owner-only Unix socket; if the
guardian is not running they report it and exit non-zero (no unlock side effect).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from . import PROTOTYPE_SPOOF_DISCLOSURE, __version__
from . import console as _con
from . import paths as _paths
from .config import Config, load_config
from .control import send_command
from .errors import ConfigError

# One console for the whole invocation; --no-color / NO_COLOR force plain text.
_C = _con.Console.auto()


def _apply_color_pref(no_color: bool) -> None:
    """Honour a global ``--no-color`` flag (env vars are handled in Console)."""
    global _C
    if no_color:
        _C = _con.Console(color=False, width=_C.width)


def _load_cfg(config_path: Path | None) -> Config:
    return load_config(config_path).resolve_model_paths(_paths.models_dir())


def _print_config_errors(exc: Exception) -> None:
    rows = [_C.paint(f"config error: {exc}", _con.RED, bold=True)]
    for err in getattr(exc, "errors", []) or []:
        rows.append(_C.paint("  ✕ ", _con.RED) + str(err))
    print(_C.panel("CONFIGURATION REJECTED", rows, colour=_con.RED), file=sys.stderr)


def _maybe_show_disclosure() -> None:
    marker = _paths.state_home() / ".disclosed"
    if marker.exists():
        return
    _print_disclosure_panel()
    try:
        _paths.ensure_dir(marker.parent, 0o700)
        marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    except OSError:
        pass


def _print_disclosure_panel() -> None:
    rows = [_C.paint("⚠  PROTOTYPE — CONVENIENCE-LEVEL SECURITY ONLY", _con.AMBER, bold=True), ""]
    # Collapse the disclosure's mid-sentence newlines, then reflow to the frame.
    body = " ".join(PROTOTYPE_SPOOF_DISCLOSURE.split())
    rows.extend(_C.wrap(body, _con.TEXT))
    print(_C.panel("SECURITY DISCLOSURE · REQ-F-17", rows, colour=_con.AMBER))


def _control(cmd: dict[str, Any]) -> tuple[dict[str, Any], int]:
    sock = _paths.control_socket_path()
    if not sock.exists():
        print(_C.paint("⚠  guardian offline", _con.AMBER, bold=True)
              + " — no control socket. Start it with: "
              + _C.paint("systemctl --user start facelock-guardian", _con.CYAN),
              file=sys.stderr)
        return {"ok": False, "reason": "no_guardian"}, 3
    resp = send_command(sock, cmd)
    if not resp.get("ok"):
        print(_C.paint(f"✕ command failed: {resp.get('reason', 'unknown')}",
                       _con.RED, bold=True), file=sys.stderr)
        return resp, 1
    return resp, 0


def _ok(text: str) -> None:
    print(_C.paint("✓ ", _con.GREEN, bold=True) + _C.paint(text, _con.GREEN))


# --------------------------------------------------------------------------- #
# Verb implementations.
# --------------------------------------------------------------------------- #
def cmd_enroll(args: argparse.Namespace) -> int:
    from .enroll import EnrollmentTool

    _maybe_show_disclosure()
    try:
        cfg = _load_cfg(args.config)
    except ConfigError as exc:
        _print_config_errors(exc)
        return 2
    return EnrollmentTool(cfg).enroll(
        args.name,
        augment=args.augment,
        samples_per_pose=args.samples_per_pose,
        multipose=not args.single_pose,
        gui=not args.no_gui,
        settle_s=args.settle_seconds,
        capture_interval_s=args.interval_seconds,
    )


def cmd_delete(args: argparse.Namespace) -> int:
    from .enroll import EnrollmentTool

    try:
        cfg = _load_cfg(args.config)
    except ConfigError as exc:
        _print_config_errors(exc)
        return 2
    if not args.yes:
        reply = input("Delete your enrolled face template permanently? [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            print("Aborted.")
            return 0
    return EnrollmentTool(cfg).delete()


def cmd_calibrate(args: argparse.Namespace) -> int:
    from .enroll import EnrollmentTool

    try:
        cfg = _load_cfg(args.config)
    except ConfigError as exc:
        _print_config_errors(exc)
        return 2
    return EnrollmentTool(cfg).recalibrate()


def cmd_lock(args: argparse.Namespace) -> int:
    _resp, code = _control({"cmd": "lock", "reason": "panic"})
    if code == 0:
        print(_C.badge("LOCKED", _con.RED) + _C.paint("  session sealed — panic lock engaged", _con.TEXT))
    return code


def cmd_disable(args: argparse.Namespace) -> int:
    _resp, code = _control({"cmd": "disable"})
    if code == 0:
        print(_C.badge("DISARMED", _con.AMBER)
              + _C.paint("  face-unlock off — the OS password still works", _con.TEXT))
    return code


def cmd_enable(args: argparse.Namespace) -> int:
    _resp, code = _control({"cmd": "enable"})
    if code == 0:
        print(_C.badge("ARMED", _con.GREEN) + _C.paint("  face-unlock re-armed", _con.TEXT))
    return code


def _render_status(resp: dict[str, Any]) -> str:
    """Format a guardian status reply as a JARVIS-style telemetry panel."""
    locked = bool(resp.get("locked"))
    face = bool(resp.get("face_unlock"))
    watchdog = bool(resp.get("watchdog_tripped"))
    hb_age = resp.get("last_heartbeat_age_s")
    daemon_state = str(resp.get("daemon_state", "?"))

    state_txt, state_col = (("LOCKED", _con.RED) if locked else ("UNLOCKED", _con.GREEN))
    face_txt, face_col = (("ARMED", _con.GREEN) if face else ("DISARMED", _con.AMBER))
    # Heartbeat health: fresh <6s green, stale amber, missing red.
    if watchdog:
        link_txt, link_col = "WATCHDOG TRIPPED", _con.RED
    elif isinstance(hb_age, (int, float)):
        fresh = hb_age < 6.0
        link_txt = f"HEARTBEAT {hb_age:.1f}s AGO"
        link_col = _con.GREEN if fresh else _con.AMBER
    else:
        link_txt, link_col = "NO SIGNAL", _con.RED

    rows = [
        _C.kv("SESSION", state_txt, value_colour=state_col),
        _C.kv("FACE UNLOCK", face_txt, value_colour=face_col),
        _C.kv("PERCEPTION LINK", link_txt, value_colour=link_col),
        _C.kv("DAEMON STATE", daemon_state,
              value_colour=_con.GREEN if daemon_state in ("ACTIVE", "PRESENT", "SCANNING") else _con.BLUE),
        _C.kv("OS LOCK BACKEND", "ENGAGED" if resp.get("os_locked") else "standby",
              value_colour=_con.CYAN),
        "",
        _C.kv("SHIELD", "UP" if resp.get("shield_up") else "down",
              value_colour=_con.CYAN if resp.get("shield_up") else _con.DIM),
        _C.kv("LOCK EPOCH", str(resp.get("lock_epoch", "?")), value_colour=_con.WHITE),
        _C.kv("AUDIT TRAIL", "ON" if resp.get("audit_enabled") else "off",
              value_colour=_con.GREEN if resp.get("audit_enabled") else _con.DIM),
        _C.kv("PERCEPTION", "PAUSED (enrolling)" if resp.get("perception_paused") else "live",
              value_colour=_con.AMBER if resp.get("perception_paused") else _con.GREEN),
    ]
    footer = f"facelock v{__version__} · guardian online · {'password path always available' if locked else 'owner verified'}"
    return _C.panel("GUARDIAN TELEMETRY", rows, colour=state_col, footer=footer)


def cmd_status(args: argparse.Namespace) -> int:
    resp, code = _control({"cmd": "status"})
    if code != 0:
        return code
    if getattr(args, "json", False):
        print(json.dumps(resp, indent=2))
        return 0
    print(_C.banner("LOCAL FACE AUTHENTICATION"))
    print()
    print(_render_status(resp))
    return 0


def cmd_disclosure(args: argparse.Namespace) -> int:
    _print_disclosure_panel()
    return 0


def cmd_config_check(args: argparse.Namespace) -> int:
    try:
        cfg = _load_cfg(args.config)
    except ConfigError as exc:
        _print_config_errors(exc)
        return 2
    phase_name = "PROTOTYPE (P)" if cfg.phase == "P" else "HARDENING (H)"
    rows = [
        _C.kv("VALIDATION", "PASSED", value_colour=_con.GREEN),
        _C.kv("SECURITY PHASE", phase_name, value_colour=_con.CYAN),
        _C.kv("SOURCE", str(cfg.source_path or "built-in defaults"), value_colour=_con.WHITE),
    ]
    if cfg.warnings:
        rows.append("")
        for warn in cfg.warnings:
            rows.append(_C.paint("⚠ ", _con.AMBER) + _C.paint(str(warn), _con.TEXT))
    print(_C.panel("CONFIG CHECK", rows, colour=_con.GREEN))
    return 0


def cmd_test(args: argparse.Namespace) -> int:
    """Camera + pipeline self-test: fps, per-frame latency, score vs tau."""
    try:
        cfg = _load_cfg(args.config)
    except ConfigError as exc:
        _print_config_errors(exc)
        return 2
    try:
        import cv2  # noqa: F401
        from .capture import CameraCapture
        from .detect import FaceDetector
        from .embed import FaceEmbedder
    except Exception as exc:  # pragma: no cover
        print(f"test: perception unavailable: {exc}", file=sys.stderr)
        return 2
    from .errors import CameraError, ModelError
    from .matcher import Matcher
    from .store import TemplateStore

    try:
        detector = FaceDetector(
            cfg.detection.model_path,
            confidence_floor=cfg.detection.confidence_floor,
            nms_threshold=cfg.detection.nms_threshold,
            min_face_px=cfg.detection.min_face_px,
        )
        embedder = FaceEmbedder(cfg.recognition.model_path)
    except ModelError as exc:
        print(f"test: {exc}\nRun scripts/download_models.sh first.", file=sys.stderr)
        return 2

    tmpl = TemplateStore().try_load()
    matcher = None
    if tmpl is not None:
        matcher = Matcher(tmpl.centroid, tmpl.tau, k=cfg.recognition.match_votes,
                          n=cfg.recognition.probe_frames, metric=cfg.recognition.metric)
        print(_C.panel("BIOMETRIC PROFILE", [
            _C.kv("OWNER", tmpl.owner_name, value_colour=_con.CYAN),
            _C.kv("THRESHOLD τ", f"{tmpl.tau:.4f}", value_colour=_con.WHITE),
            _C.kv("VOTING", f"{cfg.recognition.match_votes}-of-{cfg.recognition.probe_frames}",
                  value_colour=_con.WHITE),
        ], colour=_con.VIOLET))
    else:
        print(_C.badge("NO PROFILE", _con.AMBER)
              + _C.paint("  no owner enrolled — the verify test will be skipped", _con.TEXT))
    print(_C.paint(f"⟳ probing camera {cfg.camera.device} for {args.seconds:g}s …", _con.BLUE))

    cam = CameraCapture(cfg.camera.device, width=cfg.camera.resolution[0],
                        height=cfg.camera.resolution[1],
                        pixel_format=cfg.camera.pixel_format, fps=cfg.camera.fps_active)
    try:
        cam.open()
    except CameraError as exc:
        print(f"test: cannot open camera: {exc}", file=sys.stderr)
        return 2

    frames = 0
    faces_seen = 0
    latencies: list[float] = []
    best_score = -1.0
    t_end = time.monotonic() + args.seconds
    try:
        while time.monotonic() < t_end:
            frame, err = cam.read()
            if frame is None:
                continue
            t0 = time.perf_counter()
            dets = detector.detect(frame.bgr)
            if len(dets) == 1:
                emb = embedder.embed(frame.bgr, dets[0])
                if emb is not None and matcher is not None:
                    best_score = max(best_score, matcher.score_only(emb))
            latencies.append((time.perf_counter() - t0) * 1000.0)
            frames += 1
            faces_seen += len(dets)
    finally:
        cam.release()

    if frames == 0:
        print(_C.paint("✕ no frames captured — check the camera device.", _con.RED, bold=True),
              file=sys.stderr)
        return 1
    fps = frames / args.seconds
    latencies.sort()
    p95 = latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))]
    mean_ms = sum(latencies) / len(latencies)
    fps_ok = fps >= 5
    p95_ok = p95 <= 200

    rows = [
        _C.kv("FRAMES CAPTURED", str(frames), value_colour=_con.WHITE),
        _C.kv("THROUGHPUT", f"{fps:.1f} fps",
              value_colour=_con.GREEN if fps_ok else _con.AMBER),
        _C.kv("FACES / FRAME", f"{faces_seen / frames:.2f}", value_colour=_con.WHITE),
        _C.kv("LATENCY (mean)", f"{mean_ms:.1f} ms", value_colour=_con.WHITE),
        _C.kv("LATENCY (p95)", f"{p95:.1f} ms",
              value_colour=_con.GREEN if p95_ok else _con.AMBER),
    ]
    if matcher is not None:
        is_owner = matcher.passes(best_score)
        rows += [
            "",
            _C.kv("BEST MATCH SCORE", f"{best_score:.4f}  (τ {matcher.tau:.4f})",
                  value_colour=_con.WHITE),
            _C.kv("VERDICT", "OWNER — AUTHORIZED" if is_owner else "NOT RECOGNIZED",
                  value_colour=_con.GREEN if is_owner else _con.RED),
        ]
    rows += [
        "",
        _C.kv("TARGET fps ≥ 5", "OK" if fps_ok else "LOW",
              value_colour=_con.GREEN if fps_ok else _con.RED),
        _C.kv("TARGET p95 ≤ 200ms", "OK" if p95_ok else "HIGH",
              value_colour=_con.GREEN if p95_ok else _con.RED),
    ]
    all_ok = fps_ok and p95_ok
    print(_C.panel("PIPELINE SELF-TEST", rows,
                   colour=_con.GREEN if all_ok else _con.AMBER,
                   footer="detect + embed budget ≤ 200 ms · REQ-NF-02"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="facelock",
                                     description="facelock -- screensaver-only face-unlock (prototype)")
    parser.add_argument("--version", action="version", version=f"facelock {__version__}")
    parser.add_argument("--config", type=Path, default=None, help="path to config.toml")
    parser.add_argument("--no-color", action="store_true",
                        help="disable the coloured HUD (also honours NO_COLOR)")
    sub = parser.add_subparsers(dest="verb", required=True)

    p_enroll = sub.add_parser("enroll", help="enroll or re-enroll the owner face")
    p_enroll.add_argument("--name", default="Yash", help="owner display name (greeting)")
    p_enroll.add_argument("--augment", action="store_true", help="add to the existing template")
    p_enroll.add_argument("--samples-per-pose", type=int, default=3,
                          help="frames captured per guided head position (default 3)")
    p_enroll.add_argument("--single-pose", action="store_true",
                          help="legacy single-position capture (no guided multi-pose)")
    p_enroll.add_argument("--no-gui", action="store_true",
                          help="disable the graphical preview (text prompts only)")
    p_enroll.add_argument("--settle-seconds", type=float, default=2.5,
                          help="get-ready countdown before each pose (default 2.5)")
    p_enroll.add_argument("--interval-seconds", type=float, default=0.7,
                          help="seconds between captured frames within a pose (default 0.7)")
    p_enroll.set_defaults(func=cmd_enroll)

    p_delete = sub.add_parser("delete", help="securely delete the template")
    p_delete.add_argument("--yes", action="store_true", help="skip confirmation")
    p_delete.set_defaults(func=cmd_delete)

    sub.add_parser("calibrate", help="re-run tau calibration").set_defaults(func=cmd_calibrate)
    sub.add_parser("lock", help="immediate panic lock").set_defaults(func=cmd_lock)
    sub.add_parser("disable", help="disable face-unlock (password still works)").set_defaults(func=cmd_disable)
    sub.add_parser("enable", help="enable face-unlock").set_defaults(func=cmd_enable)
    p_status = sub.add_parser("status", help="show guardian state/health")
    p_status.add_argument("--json", action="store_true",
                          help="emit raw JSON instead of the HUD (for scripts)")
    p_status.set_defaults(func=cmd_status)
    sub.add_parser("disclosure", help="print the prototype spoof-limitation notice").set_defaults(func=cmd_disclosure)
    sub.add_parser("config-check", help="validate the config file").set_defaults(func=cmd_config_check)

    p_test = sub.add_parser("test", help="camera + pipeline self-test")
    p_test.add_argument("--seconds", type=float, default=5.0, help="test duration")
    p_test.add_argument("--pad", action="store_true", help="include PAD self-test (Hardening)")
    p_test.set_defaults(func=cmd_test)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _apply_color_pref(bool(getattr(args, "no_color", False)))
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print(_C.paint("\n⏹ interrupted.", _con.AMBER), file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
