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
from . import paths as _paths
from .config import Config, load_config
from .control import send_command
from .errors import ConfigError


def _load_cfg(config_path: Path | None) -> Config:
    return load_config(config_path).resolve_model_paths(_paths.models_dir())


def _print_config_errors(exc: Exception) -> None:
    print(f"config error: {exc}", file=sys.stderr)
    for err in getattr(exc, "errors", []) or []:
        print(f"  - {err}", file=sys.stderr)


def _maybe_show_disclosure() -> None:
    marker = _paths.state_home() / ".disclosed"
    if marker.exists():
        return
    print("=" * 70)
    print(PROTOTYPE_SPOOF_DISCLOSURE)
    print("=" * 70)
    try:
        _paths.ensure_dir(marker.parent, 0o700)
        marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    except OSError:
        pass


def _control(cmd: dict[str, Any]) -> tuple[dict[str, Any], int]:
    sock = _paths.control_socket_path()
    if not sock.exists():
        print("facelock guardian is not running "
              "(no control socket). Start facelock-guardian first.", file=sys.stderr)
        return {"ok": False, "reason": "no_guardian"}, 3
    resp = send_command(sock, cmd)
    if not resp.get("ok"):
        print(f"command failed: {resp.get('reason', 'unknown')}", file=sys.stderr)
        return resp, 1
    return resp, 0


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
        screen=args.screen,
        windowed=args.windowed,
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
        print("Locked.")
    return code


def cmd_disable(args: argparse.Namespace) -> int:
    _resp, code = _control({"cmd": "disable"})
    if code == 0:
        print("Face-unlock disabled. The OS password still works.")
    return code


def cmd_enable(args: argparse.Namespace) -> int:
    _resp, code = _control({"cmd": "enable"})
    if code == 0:
        print("Face-unlock enabled.")
    return code


def cmd_pause(args: argparse.Namespace) -> int:
    """Pause perception (release the camera) for video-conferencing etc.

    With --minutes N the guardian auto-resumes after N minutes; otherwise it
    stays paused until `facelock resume`. Face-unlock stays fail-closed while
    paused: the current lock state is held, only the camera is released.
    """
    cmd: dict[str, Any] = {"cmd": "pause_perception"}
    minutes = getattr(args, "minutes", None)
    if minutes is not None:
        cmd["minutes"] = minutes
    _resp, code = _control(cmd)
    if code == 0:
        if minutes is not None:
            print(f"Perception paused; camera released. Auto-resumes in {minutes} min "
                  f"(or run: facelock resume).")
        else:
            print("Perception paused; camera released. Resume with: facelock resume")
    return code


def cmd_resume(args: argparse.Namespace) -> int:
    _resp, code = _control({"cmd": "resume_perception"})
    if code == 0:
        print("Perception resumed; camera reacquired.")
    return code


def cmd_status(args: argparse.Namespace) -> int:
    resp, code = _control({"cmd": "status"})
    if code == 0:
        print(json.dumps(resp, indent=2))
    return code


def cmd_disclosure(args: argparse.Namespace) -> int:
    print(PROTOTYPE_SPOOF_DISCLOSURE)
    return 0


def cmd_config_check(args: argparse.Namespace) -> int:
    try:
        cfg = _load_cfg(args.config)
    except ConfigError as exc:
        _print_config_errors(exc)
        return 2
    print(f"config OK (phase={cfg.phase}, source={cfg.source_path or 'defaults'})")
    for warn in cfg.warnings:
        print(f"  warning: {warn}")
    # Loud fail-safe: a persisted dry-run config does NOT protect the session
    # (OS-lock actuation disabled). install.sh must never write this (DES-DRYRUN).
    if cfg.security.dry_run:
        print("WARNING: security.dry_run=true -- this config does NOT protect the "
              "session; OS-lock actuation is DISABLED (no loginctl/gdbus/xdg).")
    return 0


def cmd_setup(args: argparse.Namespace) -> int:
    """Provision models (SHA-pinned) + config (+ optional systemd units).

    Fresh-machine step between install and enroll. Fail-closed on a model hash
    mismatch. Never enables auto-start (enroll first). See ``setup_cmd``.
    """
    from . import setup_cmd

    try:
        setup_cmd.run_setup(systemd=bool(args.systemd))
    except setup_cmd.SetupError as exc:
        print(f"setup failed: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"setup failed: {exc}", file=sys.stderr)
        return 2
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
        print(f"template: owner='{tmpl.owner_name}' tau={tmpl.tau:.4f}")
    else:
        print("template: none enrolled (verify test will be skipped)")

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
        print("test: no frames captured.", file=sys.stderr)
        return 1
    fps = frames / args.seconds
    latencies.sort()
    p95 = latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))]
    print(f"frames: {frames}  fps: {fps:.1f}  faces/frame: {faces_seen / frames:.2f}")
    print(f"per-frame detect+embed: mean {sum(latencies) / len(latencies):.1f} ms, "
          f"p95 {p95:.1f} ms  (budget <=200 ms, REQ-NF-02)")
    if matcher is not None:
        verdict = "OWNER" if matcher.passes(best_score) else "not-owner"
        print(f"best score: {best_score:.4f}  vs tau {matcher.tau:.4f}  -> {verdict}")
    print(f"targets: fps>=5 {'OK' if fps >= 5 else 'LOW'}, "
          f"p95<=200ms {'OK' if p95 <= 200 else 'HIGH'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="facelock",
                                     description="facelock -- screensaver-only face-unlock (prototype)")
    parser.add_argument("--version", action="version", version=f"facelock {__version__}")
    parser.add_argument("--config", type=Path, default=None, help="path to config.toml")
    sub = parser.add_subparsers(dest="verb", required=True)

    p_setup = sub.add_parser(
        "setup", help="download SHA-pinned models + install config (run once after install)")
    p_setup.add_argument("--systemd", action="store_true",
                         help="also install the --user systemd units (does NOT enable auto-start)")
    p_setup.set_defaults(func=cmd_setup)

    p_enroll = sub.add_parser("enroll", help="enroll or re-enroll the owner face")
    p_enroll.add_argument("--name", default="User", help="owner display name (greeting)")
    p_enroll.add_argument("--augment", action="store_true", help="add to the existing template")
    p_enroll.add_argument("--samples-per-pose", type=int, default=3,
                          help="frames captured per guided head position (default 3)")
    p_enroll.add_argument("--single-pose", action="store_true",
                          help="legacy single-position capture (no guided multi-pose)")
    p_enroll.add_argument("--no-gui", action="store_true",
                          help="disable the graphical preview (text prompts only)")
    p_enroll.add_argument("--screen", type=int, default=0,
                          help="monitor index for the fullscreen preview "
                               "(0-based; see 'xrandr --listmonitors', default 0)")
    p_enroll.add_argument("--windowed", action="store_true",
                          help="show the preview in a window instead of fullscreen")
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
    p_pause = sub.add_parser(
        "pause", help="pause perception / release the camera (e.g. video calls)")
    p_pause.add_argument(
        "--minutes", type=float, default=None,
        help="auto-resume after N minutes (default: stay paused until 'resume')")
    p_pause.set_defaults(func=cmd_pause)
    sub.add_parser("resume", help="resume perception after a pause").set_defaults(func=cmd_resume)
    sub.add_parser("status", help="show guardian state/health").set_defaults(func=cmd_status)
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
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
