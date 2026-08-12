"""EnrollmentTool (C16) -- guided capture, quality gate, template + tau calib.

Realizes REQ-F-01/02/03/04 and design section 6.4. The interactive capture loop
requires a camera; the pure pieces (quality gate, template building, tau
calibration) are separated into module functions so they are unit-testable with
synthetic embeddings/images and no camera.

Privacy (REQ-NF-13): no raw frame is ever written to disk; only embeddings +
non-image sample metadata enter the template.
"""

from __future__ import annotations

import math
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from . import paths as _paths
from .calibrate import calibrate, centroid_of
from .config import Config
from .enroll_ui import letterbox
from .errors import CalibrationError, CameraError, ModelError
from .logging_setup import event, get_logger
from .store import (
    Template,
    TemplateStore,
    ensure_impostor_set,
    model_sha256,
)

EMBEDDING_DIM = 128

# Shown when the camera cannot be opened -- almost always because the running
# perception daemon holds the single-access UVC device. Stopping BOTH services is
# required: stopping only facelockd would let the guardian's watchdog trip
# (heartbeat miss) mid-enrollment. The shutdown path no longer forces the OS
# password lock, so stopping is safe and won't drop you to the login screen.
CAMERA_BUSY_HINT = (
    "  The camera is most likely held by the running facelock daemon.\n"
    "  Stop the services, enroll, then start them again:\n"
    "    systemctl --user stop facelockd.service facelock-guardian.service\n"
    "    facelock enroll --name <name>\n"
    "    systemctl --user start facelockd.service facelock-guardian.service"
)

# Guided head positions for multi-pose enrollment. Capturing a spread of poses
# populates the matcher's pose sub-templates so an off-angle face authenticates
# easily (the RGB-only, dependency-light path to iPhone-like convenience).
POSES: tuple[tuple[str, str], ...] = (
    ("center", "Look STRAIGHT at the camera"),
    ("left", "Slowly turn your head a little to the LEFT"),
    ("right", "Now turn a little to the RIGHT"),
    ("up", "Lift your chin UP slightly"),
    ("down", "Lower your chin DOWN slightly"),
)


# Enrollment capture resolution/format. Independent of the daemon's runtime
# camera config (which stays at its low-power YUYV/640x480): a 720p MJPG grab
# gives a crisp face in the circle for a better template crop. SFace aligns to
# 112x112 internally, so a higher source res only sharpens the aligned crop and
# never changes the pipeline. We do NOT touch cfg.camera here.
ENROLL_CAM_W = 1280
ENROLL_CAM_H = 720
ENROLL_CAM_FMT = "MJPG"


class Monitor(NamedTuple):
    """One connected display: name + pixel geometry (w, h) at offset (x, y)."""

    name: str
    w: int
    h: int
    x: int
    y: int


# Safe fallback when xrandr is missing / unparseable (headless X, minimal WM):
# a single 1080p monitor at the origin. Enrollment then fullscreens on :0 @ 0,0.
FALLBACK_MONITOR = Monitor(name="default", w=1920, h=1080, x=0, y=0)

# `xrandr --listmonitors` line, e.g.
#   " 0: +*eDP-1 1920/344x1080/193+0+0  eDP-1"
# and the mm-less variant " 0: +*DP-1 3440x1440+0+0  DP-1".
_MON_LINE = re.compile(r"^\s*(\d+):\s+(\S+)\s+(\S+)")
_MON_GEOM = re.compile(
    r"^(\d+)(?:/\d+)?x(\d+)(?:/\d+)?\+(-?\d+)\+(-?\d+)$"
)


def parse_xrandr_monitors(text: str) -> list[Monitor]:
    """Parse ``xrandr --listmonitors`` output into a list of :class:`Monitor`.

    Pure + robust: ignores the ``Monitors: N`` header and any line that is not a
    well-formed monitor row, so garbage yields ``[]`` (the caller then falls back
    to :data:`FALLBACK_MONITOR`). Never raises.
    """
    monitors: list[Monitor] = []
    for line in (text or "").splitlines():
        m = _MON_LINE.match(line)
        if not m:
            continue
        g = _MON_GEOM.match(m.group(3))
        if not g:
            continue
        name = m.group(2).lstrip("+*")
        w, h, x, y = (int(v) for v in g.groups())
        monitors.append(Monitor(name=name, w=w, h=h, x=x, y=y))
    return monitors


def list_monitors() -> list[Monitor]:
    """Enumerate connected monitors via ``xrandr --listmonitors``.

    Falls back to ``[FALLBACK_MONITOR]`` when xrandr is absent, times out, exits
    non-zero, or emits nothing parseable. Never raises (fail-safe for the picker).
    """
    try:
        proc = subprocess.run(
            ["xrandr", "--listmonitors"],
            capture_output=True, text=True, timeout=3.0,
        )
    except Exception:
        return [FALLBACK_MONITOR]
    if proc.returncode != 0:
        return [FALLBACK_MONITOR]
    monitors = parse_xrandr_monitors(proc.stdout)
    return monitors or [FALLBACK_MONITOR]


def pick_monitor(monitors: list[Monitor], index: int) -> Monitor:
    """Return ``monitors[index]`` with the index clamped into range.

    An empty list yields :data:`FALLBACK_MONITOR` (never an ``IndexError``), so a
    bad ``--screen N`` can never crash enrollment.
    """
    if not monitors:
        return FALLBACK_MONITOR
    i = max(0, min(int(index), len(monitors) - 1))
    return monitors[i]


def pose_plan(
    min_samples: int = 5,
    samples_per_pose: int = 3,
    *,
    multipose: bool = True,
    poses: tuple[tuple[str, str], ...] = POSES,
) -> list[tuple[str, str, int]]:
    """Return the capture plan: a list of ``(pose_hint, instruction, count)``.

    Pure + deterministic (unit-testable). Guarantees the total capture count is
    at least ``min_samples`` (topping up the first pose if needed). With
    ``multipose=False`` it degrades to a single 'center' pose (legacy behaviour).
    """
    spp = max(1, int(samples_per_pose))
    used = poses if multipose else (("center", "Look STRAIGHT at the camera"),)
    plan = [(hint, instr, spp) for hint, instr in used]
    total = sum(n for _, _, n in plan)
    i = 0
    while total < max(1, int(min_samples)):
        hint, instr, n = plan[i % len(plan)]
        plan[i % len(plan)] = (hint, instr, n + 1)
        total += 1
        i += 1
    return plan


@dataclass
class QualityResult:
    ok: bool
    reason: str
    face_px: float = 0.0
    sharpness: float = 0.0
    brightness: float = 0.0


def assess_quality(
    detections: list,
    *,
    sharpness: float,
    brightness: float,
    min_face_px: int = 80,
    sharpness_floor: float = 40.0,
    brightness_range: tuple[float, float] = (40.0, 225.0),
) -> QualityResult:
    """Enrollment quality gate (REQ-F-02).

    Rejects frames with zero or >1 faces, a too-small face, a blurry frame
    (variance of Laplacian below floor), or out-of-range brightness. Sharpness
    and brightness are passed in (computed by the caller from the frame) so this
    function is pure and testable without OpenCV.
    """
    if len(detections) == 0:
        return QualityResult(False, "no_face")
    if len(detections) > 1:
        return QualityResult(False, "multiple_faces")
    det = detections[0]
    face_px = float(min(det.bbox[2], det.bbox[3]))
    if face_px < min_face_px:
        return QualityResult(False, "face_too_small", face_px, sharpness, brightness)
    if sharpness < sharpness_floor:
        return QualityResult(False, "too_blurry", face_px, sharpness, brightness)
    lo, hi = brightness_range
    if not (lo <= brightness <= hi):
        return QualityResult(False, "bad_brightness", face_px, sharpness, brightness)
    return QualityResult(True, "ok", face_px, sharpness, brightness)


def build_template(
    name: str,
    sample_embeddings: np.ndarray,
    sample_meta: list[dict[str, Any]],
    impostors: np.ndarray,
    *,
    model_id: str,
    phase: str,
    metric: str,
    fmr_target: float,
    fnmr_target: float,
    tau_floor: float,
) -> Template:
    """Build + calibrate an owner template from accepted sample embeddings.

    Pure and testable: no camera, no disk. Raises :class:`CalibrationError`
    only if there are too few samples for leave-one-out calibration.
    """
    samples = np.asarray(sample_embeddings, dtype=np.float32).reshape(-1, EMBEDDING_DIM)
    if samples.shape[0] < 2:
        raise CalibrationError("need >= 2 accepted samples to calibrate")
    centroid = centroid_of(samples)
    result = calibrate(
        samples, impostors,
        fmr_target=fmr_target, fnmr_target=fnmr_target,
        tau_floor=tau_floor, metric=metric,
    )
    return Template(
        owner_name=name,
        centroid=centroid,
        samples=samples,
        tau=result.tau,
        calibration=result.as_meta(),
        sample_meta=sample_meta,
        model_id=model_id,
        metric=metric,
        phase=phase,
    )


class EnrollmentTool:
    """Interactive enrollment (camera required)."""

    def __init__(self, config: Config, *, logger: Any = None) -> None:
        self.cfg = config
        self.log = logger or get_logger("facelock.enroll", level=config.logging.level)

    def enroll(
        self,
        name: str,
        *,
        augment: bool = False,
        min_samples: int = 5,
        samples_per_pose: int = 3,
        multipose: bool = True,
        gui: bool = True,
        settle_s: float = 2.5,
        capture_interval_s: float = 0.7,
        timeout_s: float = 240.0,
        screen: int = 0,
        windowed: bool = False,
    ) -> int:
        """Run the guided enrollment flow. Returns a process exit code.

        ``screen`` selects the monitor (0-based) for the fullscreen preview and
        ``windowed`` shows it in a movable window instead. The preview renders at
        the chosen display's native resolution (1:1, no fullscreen upscaling).
        """
        from .capture import CameraCapture
        from .detect import FaceDetector
        from .embed import FaceEmbedder

        try:
            import cv2
        except Exception as exc:  # pragma: no cover
            print(f"enroll: OpenCV unavailable: {exc}")
            return 2

        try:
            detector = FaceDetector(
                self.cfg.detection.model_path,
                confidence_floor=self.cfg.detection.confidence_floor,
                nms_threshold=self.cfg.detection.nms_threshold,
                min_face_px=self.cfg.detection.min_face_px,
            )
            embedder = FaceEmbedder(self.cfg.recognition.model_path)
        except ModelError as exc:
            print(f"enroll: {exc}\nRun scripts/download_models.sh first.")
            return 2

        store = TemplateStore()
        existing_samples: list[np.ndarray] = []
        existing_meta: list[dict[str, Any]] = []
        if augment and store.exists():
            tmpl = store.try_load()
            if tmpl is not None:
                existing_samples = list(np.asarray(tmpl.samples, dtype=np.float32))
                existing_meta = list(tmpl.sample_meta)
                print(f"Augmenting existing template with {len(existing_samples)} samples.")

        print(f"Enrolling '{name}'. A scan window will open: keep your face in the "
              "circle and slowly move your head to fill the ring.")
        # Grab at 720p MJPG for a crisp face in the circle. This is an
        # enrollment-only override -- the daemon's low-power runtime config
        # (cfg.camera) is left untouched.
        camera = CameraCapture(
            self.cfg.camera.device,
            width=ENROLL_CAM_W,
            height=ENROLL_CAM_H,
            pixel_format=ENROLL_CAM_FMT,
            fps=self.cfg.camera.fps_active,
        )
        # Free the camera -- auto-pause a running facelock daemon if it holds it,
        # and resume it (with the new template) when we are done.
        opened, paused_by_us = self._acquire_camera(camera)
        if not opened:
            return 2

        from .enroll_ui import (RingView, head_offset, nearest_uncovered,
                                render, segment_of)
        from .shield import has_display

        N_SEG = 16
        DEADZONE = 0.12
        FRONTAL_NEED = 2
        if multipose:
            coverage_goal = min(N_SEG, max(8, samples_per_pose * 3))
            target_new = coverage_goal + FRONTAL_NEED
            instruction = "Slowly move your head in a circle"
        else:
            coverage_goal = 0
            target_new = max(min_samples, samples_per_pose * 5)
            instruction = "Look at the camera"

        accepted_emb: list[np.ndarray] = list(existing_samples)
        accepted_meta: list[dict[str, Any]] = list(existing_meta)
        base_n = len(existing_samples)
        target_total = base_n + target_new
        covered: set[int] = set()
        frontal = 0
        overall_deadline = time.monotonic() + timeout_s
        last_accept = -1e9
        flash = 0.0
        tick = 0
        wname = "facelock enrollment"
        monitors = list_monitors()
        gui, disp_w, disp_h = self._init_preview(
            cv2, gui and has_display(), wname,
            monitors=monitors, screen=screen, windowed=windowed)
        if not gui:
            print(f"(no GUI) {instruction} -- capturing {target_new} samples...")

        def _fit(frame_bgr: Any) -> Any:
            """Scale the camera frame up to the display resolution before the HUD
            is drawn, so the fullscreen output is 1:1 with the monitor (the ring
            and text are never upscaled). ``render`` returns a same-size image."""
            return letterbox(frame_bgr, disp_w, disp_h)

        def _done() -> bool:
            if multipose:
                return len(covered) >= coverage_goal and frontal >= FRONTAL_NEED
            return (len(accepted_emb) - base_n) >= target_new

        try:
            # Brief "get ready" so the user can settle before the scan starts.
            if gui and settle_s > 0:
                ready_until = time.monotonic() + settle_s
                while time.monotonic() < ready_until:
                    frame, _ = camera.read()
                    if frame is None:
                        time.sleep(0.03)
                        continue
                    tick += 1
                    left = ready_until - time.monotonic()
                    view = RingView(owner=name, captured=base_n, target=target_total,
                                    n_segments=N_SEG, instruction="Get ready...",
                                    status=str(int(math.ceil(left))), tick=tick)
                    if self._show(cv2, wname, render(_fit(frame.bgr), view)) == 27:
                        print("\nenroll: cancelled by user.")
                        return 1

            while not _done():
                now = time.monotonic()
                if now > overall_deadline:
                    print("\nenroll: timeout -- proceeding with what was captured.")
                    break
                frame, err = camera.read()
                if frame is None:
                    time.sleep(0.03)
                    continue
                tick += 1
                flash *= 0.82
                gray = cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2GRAY)
                sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                brightness = float(gray.mean())
                dets = detector.detect(frame.bgr)
                q = assess_quality(dets, sharpness=sharpness, brightness=brightness,
                                   min_face_px=self.cfg.detection.min_face_px)
                seg = None
                bbox = None
                if dets:
                    bbox = tuple(dets[0].bbox)
                    nx, ny = head_offset(dets[0].landmarks, dets[0].bbox)
                    seg = segment_of(nx, ny, N_SEG, DEADZONE)

                # Interactive capture: only accept a NEW direction, so the user
                # must actually sweep their head to progress (multipose); in
                # single-pose we accept any spaced quality frontal frame.
                take = False
                if q.ok and (now - last_accept) >= capture_interval_s:
                    if not multipose:
                        take = True
                    elif seg is None:
                        take = frontal < FRONTAL_NEED
                    else:
                        take = seg not in covered
                if take:
                    emb = embedder.embed(frame.bgr, dets[0])
                    if emb is not None and accepted_emb and \
                            float(np.dot(emb, accepted_emb[-1])) > 0.999:
                        emb = None  # near-duplicate
                    if emb is not None:
                        accepted_emb.append(np.asarray(emb, dtype=np.float32))
                        accepted_meta.append({
                            "pose_hint": "frontal" if seg is None else f"seg{seg}",
                            "sharpness": round(q.sharpness, 1),
                            "brightness": round(q.brightness, 1),
                            "det_score": round(float(dets[0].score), 3),
                        })
                        last_accept = now
                        flash = 1.0
                        if seg is None:
                            frontal += 1
                        else:
                            covered.add(seg)
                    else:
                        take = False

                if not q.ok:
                    status = ""
                elif take:
                    status = "Great!"
                elif multipose and frontal < FRONTAL_NEED:
                    status = "Look straight ahead"
                elif multipose:
                    status = "Keep turning to fill the ring"
                else:
                    status = "Hold steady"

                target_seg = nearest_uncovered(seg, covered, N_SEG) if multipose else None
                if gui:
                    view = RingView(
                        owner=name, captured=len(accepted_emb), target=target_total,
                        n_segments=N_SEG, covered=frozenset(covered), current=seg,
                        target_segment=target_seg, frontal_done=frontal >= FRONTAL_NEED,
                        instruction=instruction, status=status, phase="capture",
                        bbox=bbox, quality_ok=q.ok,
                        reject=(None if q.ok else q.reason), flash=flash, tick=tick)
                    if self._show(cv2, wname, render(_fit(frame.bgr), view)) == 27:
                        print("\nenroll: cancelled by user.")
                        return 1
                else:
                    cov = f" ring {len(covered)}/{N_SEG}" if multipose else ""
                    print(f"  captured {len(accepted_emb) - base_n}/{target_new}{cov}"
                          f"          ", end="\r")

            if gui:  # completion celebration
                frame, _ = camera.read()
                if frame is not None:
                    view = RingView(
                        owner=name, captured=len(accepted_emb), target=target_total,
                        n_segments=N_SEG, covered=frozenset(range(N_SEG)),
                        current=None, target_segment=None, frontal_done=True,
                        instruction="Enrollment complete", status="All set!",
                        phase="done", tick=tick)
                    self._show(cv2, wname, render(_fit(frame.bgr), view))
                    cv2.waitKey(1000)

            if len(accepted_emb) < 2:
                print("enroll: not enough quality samples; try better lighting.")
                return 1

            model_id = ""
            sface_path = Path(self.cfg.recognition.model_path)
            if sface_path.exists():
                model_id = model_sha256(sface_path)
            impostors = ensure_impostor_set()

            try:
                template = build_template(
                    name, np.stack(accepted_emb), accepted_meta, impostors,
                    model_id=model_id, phase=self.cfg.phase,
                    metric=self.cfg.recognition.metric,
                    fmr_target=self.cfg.recognition.fmr_target,
                    fnmr_target=self.cfg.recognition.fnmr_target,
                    tau_floor=self.cfg.recognition.tau_floor,
                )
            except CalibrationError as exc:
                print(f"enroll: calibration failed: {exc}")
                return 1

            store.save(template)
            calib = template.calibration
            event(self.log, "enroll", owner=name, samples=len(accepted_emb),
                  tau=round(template.tau, 4), meets_target=calib.get("meets_target"))
            print(f"\nEnrolled '{name}' with {len(accepted_emb)} samples.")
            print(f"  tau = {template.tau:.4f}  "
                  f"(FMR~{calib.get('fmr_measured'):.4f}, "
                  f"FNMR~{calib.get('fnmr_measured'):.4f})")
            if not calib.get("meets_target", False):
                print("  WARNING: accuracy target not fully met; tau was NOT relaxed.")
                for warn in calib.get("warnings", []) if isinstance(calib, dict) else []:
                    print(f"    - {warn}")
            print(f"  template: {store.path} (mode 0600)")
            return 0
        finally:
            if gui:
                self._close_preview(cv2, wname)
            camera.release()
            if paused_by_us:
                self._resume_daemon()

    # -- enrollment preview window (futuristic HUD) ----------------------- #
    def _init_preview(
        self,
        cv2: Any,
        want: bool,
        wname: str,
        *,
        monitors: list[Monitor] | None = None,
        screen: int = 0,
        windowed: bool = False,
    ) -> tuple[bool, int, int]:
        """Create the preview window on the chosen monitor.

        Returns ``(gui_ok, render_w, render_h)`` -- the render size is the
        display's native resolution so the caller can scale the camera frame to
        it (1:1 fullscreen, no upscaling). A headless / no-highgui OpenCV build
        raises here; enrollment then runs with terminal prompts only, returning
        ``(False, 0, 0)`` and never crashing.
        """
        if not want:
            return False, 0, 0
        mon = pick_monitor(monitors if monitors else [FALLBACK_MONITOR], screen)
        try:
            cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
            if windowed:
                # Movable window on the chosen monitor; render at a lighter size
                # (still 1:1 with the window, so the HUD stays crisp).
                w = min(int(mon.w), 1280)
                h = min(int(mon.h), 720)
                cv2.resizeWindow(wname, w, h)
                cv2.moveWindow(wname, int(mon.x) + 40, int(mon.y) + 40)
                return True, w, h
            # Fullscreen on the chosen monitor. Place -> flip fullscreen ->
            # re-place: some WMs recenter the window when the property changes
            # (mirrors the vetted capture_pro placement sequence).
            cv2.moveWindow(wname, int(mon.x), int(mon.y))
            cv2.resizeWindow(wname, int(mon.w), int(mon.h))
            cv2.setWindowProperty(wname, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)
            cv2.moveWindow(wname, int(mon.x), int(mon.y))
            return True, int(mon.w), int(mon.h)
        except Exception:
            return False, 0, 0

    def _show(self, cv2: Any, wname: str, img: Any) -> int:
        """imshow + pump events; returns the pressed key (or -1). Never raises."""
        try:
            cv2.imshow(wname, img)
            return int(cv2.waitKey(1) & 0xFF)
        except Exception:
            return -1

    def _close_preview(self, cv2: Any, wname: str) -> None:
        try:
            cv2.destroyWindow(wname)
            cv2.waitKey(1)
        except Exception:
            pass

    # -- camera coordination with a running daemon ------------------------ #
    def _acquire_camera(self, camera: Any, *, wait_s: float | None = None) -> tuple[bool, bool]:
        """Open the camera, auto-pausing a running facelock daemon if it's busy.

        Returns ``(opened, paused_by_us)``. If the device is busy we ask the
        guardian (over the control socket) to pause perception, wait for the
        daemon to release the camera (one heartbeat cycle), then retry. If no
        guardian answers, we print the manual hint and give up (fail-closed).
        """
        from . import paths as _paths
        from .control import send_command
        from .errors import CameraError as _CameraError

        try:
            camera.open()
            return True, False
        except _CameraError:
            pass

        sock = _paths.control_socket_path()
        resp = send_command(sock, {"cmd": "pause_perception"})
        if not resp.get("ok"):
            print(f"enroll: cannot open camera {self.cfg.camera.device} (FM-01)")
            print(CAMERA_BUSY_HINT)
            return False, False

        hb = float(resp.get("heartbeat_sec", 2) or 2)
        print("Pausing facelock to free the camera for enrollment...")
        budget = wait_s if wait_s is not None else max(8.0, hb * 4)
        deadline = time.monotonic() + budget
        while time.monotonic() < deadline:
            time.sleep(0.25)
            try:
                camera.open()
                return True, True
            except _CameraError:
                continue
        print("enroll: camera still busy after pausing facelock -- is another app "
              "(browser, Zoom, Cheese) using it? Check: fuser /dev/video0")
        self._resume_daemon()  # undo our pause; we never got the camera
        return False, True

    def _resume_daemon(self) -> None:
        """Tell the guardian to resume perception (reacquire + reload template)."""
        from . import paths as _paths
        from .control import send_command

        try:
            send_command(_paths.control_socket_path(), {"cmd": "resume_perception"})
            print("Resumed facelock (new template is now live).")
        except Exception:
            pass

    def delete(self) -> int:
        """Securely delete the template + derived artefacts (REQ-F-04)."""
        store = TemplateStore()
        if not store.exists():
            print("No template to delete. Face-unlock is already inert.")
            return 0
        removed = store.delete()
        event(self.log, "delete_template", removed=len(removed))
        print("Deleted biometric artefacts (secure-shredded):")
        for path in removed:
            print(f"  - {path}")
        print("Face-unlock is now inert; the OS password path is unchanged.")
        return 0

    def recalibrate(self) -> int:
        """Re-run tau calibration on the current template (CLI 'calibrate')."""
        store = TemplateStore()
        tmpl = store.try_load()
        if tmpl is None:
            print("calibrate: no valid template. Enroll first.")
            return 1
        impostors = ensure_impostor_set()
        result = calibrate(
            np.asarray(tmpl.samples, dtype=np.float32), impostors,
            fmr_target=self.cfg.recognition.fmr_target,
            fnmr_target=self.cfg.recognition.fnmr_target,
            tau_floor=self.cfg.recognition.tau_floor,
            metric=self.cfg.recognition.metric,
        )
        tmpl.tau = result.tau
        tmpl.calibration = result.as_meta()
        store.save(tmpl, augment_backup=True)
        print(f"Recalibrated: tau = {result.tau:.4f} "
              f"(FMR~{result.fmr_measured:.4f}, FNMR~{result.fnmr_measured:.4f}, "
              f"meets_target={result.meets_target})")
        return 0
