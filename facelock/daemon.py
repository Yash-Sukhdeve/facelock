"""facelockd -- the Perception Daemon process (C1..C7).

Wires CameraCapture (C1) -> FaceDetector (C2) -> FaceEmbedder (C3) ->
Matcher (C4) -> LivenessEngine (C5) -> PresenceStateMachine (C6) ->
DecisionEmitter + HeartbeatSender (C7). It can ONLY *request* actions from the
guardian; it holds no lock authority (SI-P1). On any error it fails closed
(stays locked) and never emits a grant.

Camera lifecycle (FM-07): the loop throttles to ``fps_idle`` while locked/idle
and releases the device (+ UVC LED, REQ-F-27) after ``long_absence_release_s``,
duty-cycling a re-acquire to detect the owner's return.

Suspend/resume (FM-13): detected by comparing CLOCK_BOOTTIME vs CLOCK_MONOTONIC
gaps (monotonic does not advance across suspend on Linux); on resume the camera
is re-initialised and a fresh verification is required (default LOCKED).
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path
from typing import Any

from . import paths as _paths
from .config import (
    Config,
    DryRunUnderSystemdError,
    load_config,
    resolve_dry_run,
)
from .control import DecisionEmitter
from .errors import CameraError, ModelError, TemplateError
from .fsm import FSMConfig, Observation, PresenceStateMachine, State
from .liveness import LivenessEngine, LivenessObservation, build_liveness_observation
from .logging_setup import emit_dry_run_banner, event, get_logger
from .matcher import Matcher
from .store import TemplateStore

MAX_FACES = 5  # cap per-frame embeddings (bounds strict-policy CPU, OQ-2)
_ACTIVE_STATES = frozenset({
    State.VERIFYING, State.LIVENESS_CHALLENGE, State.UNLOCKED_PRESENT,
    State.UNLOCKED_GRACE, State.LOCKED_STRANGER,
})


def _sd_notify(state: str) -> None:
    import os
    import socket as _socket

    addr = os.environ.get("NOTIFY_SOCKET")
    if not addr:
        return
    if addr.startswith("@"):
        addr = "\0" + addr[1:]
    try:
        with _socket.socket(_socket.AF_UNIX, _socket.SOCK_DGRAM) as sock:
            sock.connect(addr)
            sock.sendall(state.encode("utf-8"))
    except OSError:
        pass


class PerceptionDaemon:
    """The perception loop. Camera/detector/embedder are injectable for tests."""

    def __init__(
        self,
        config: Config,
        *,
        logger: Any = None,
        emitter: DecisionEmitter | None = None,
        install_signals: bool = True,
        dry_run: bool | None = None,
    ) -> None:
        self.cfg = config
        # SAFE dry-run mode (DES-DRYRUN): surfaced in the banner + heartbeat
        # health. The daemon holds NO lock authority (SI-P1) -- the guardian is
        # the sole actuator -- so the safety property is enforced there; the
        # daemon's flag exists to declare the mode and (future tier-2) to swap
        # the camera. Effective value = CLI override or the config key.
        self._dry_run = bool(config.security.dry_run if dry_run is None else dry_run)
        self.log = logger or get_logger(
            "facelock.daemon",
            level=config.logging.level,
            max_size_mb=config.logging.max_size_mb,
            rotate_count=config.logging.rotate_count,
        )
        self.emitter = emitter or DecisionEmitter()
        self._stop = False
        self._install_signals = install_signals

        # Perception components (built lazily / defensively).
        self.camera = None
        self.detector = None
        self.embedder = None
        self.matcher: Matcher | None = None
        self.liveness = LivenessEngine(
            mode=config.liveness.mode,
            phase=config.phase,
            challenge_timeout_s=config.liveness.challenge_timeout_s,
            turn_yaw_deg=config.liveness.turn_yaw_deg,
            pad_model_path=config.liveness.pad_model_path,
            pad_threshold=config.liveness.pad_threshold,
            pad_min_live_frames=config.liveness.pad_min_live_frames,
        )
        self.fsm = PresenceStateMachine(FSMConfig(
            away_dwell_s=config.presence.away_dwell_s,
            stranger_policy=config.stranger.policy,
            stranger_dwell_s=config.stranger.dwell_s,
            max_fail_attempts=config.unlock.max_fail_attempts,
            cooldown_s=config.unlock.cooldown_s,
            loss_grace_s=config.camera.loss_grace_s,
            liveness_requires_frames=self.liveness.requires_frames,
            challenge_timeout_s=config.liveness.challenge_timeout_s,
            probe_frames=config.recognition.probe_frames,
            verify_timeout_s=max(3.0, config.recognition.probe_frames
                                 * (1.0 / max(config.camera.fps_active, 1)) * 3.0),
            owner_name=config.unlock.owner_name,
        ))

        self.perception_ok = False
        self.template_ok = False
        self._hb_seq = 0
        self._last_hb = 0.0
        self._absent_since: float | None = None
        self._camera_released = False
        self._paused = False  # enrollment pause: camera released, perception idle
        self._next_recheck = 0.0
        self._challenge = None
        self._challenge_obs: list[LivenessObservation] = []
        self._prev_boot: float | None = None
        self._prev_mono: float | None = None
        # On-shield feedback phase last pushed to the guardian (debounced).
        self._last_shield_phase: str | None = None
        self._last_phase_push = float("-inf")  # first push is never rate-limited
        self._phase_min_interval = 0.25
        self._last_progress = 0.0
        self._last_fail = 0  # fsm.fail_count last seen (detect a fresh rejection)

    # -- setup ------------------------------------------------------------ #
    def _build_perception(self) -> None:
        """Load models + template. Degrades (not crashes) on failure (FM-10/11)."""
        from .capture import CameraCapture  # lazy: importing cv2
        from .detect import FaceDetector
        from .embed import FaceEmbedder

        try:
            self.detector = FaceDetector(
                self.cfg.detection.model_path,
                confidence_floor=self.cfg.detection.confidence_floor,
                nms_threshold=self.cfg.detection.nms_threshold,
                min_face_px=self.cfg.detection.min_face_px,
            )
            self.embedder = FaceEmbedder(self.cfg.recognition.model_path)
            self.perception_ok = True
        except ModelError as exc:
            event(self.log, "model_error", error=str(exc))
            self.perception_ok = False

        self._load_template_and_matcher()

        # Camera object (opened on demand).
        try:
            self.camera = CameraCapture(
                self.cfg.camera.device,
                width=self.cfg.camera.resolution[0],
                height=self.cfg.camera.resolution[1],
                pixel_format=self.cfg.camera.pixel_format,
                fps=self.cfg.camera.fps_active,
            )
        except CameraError as exc:
            event(self.log, "camera_init_error", error=str(exc))
            self.camera = None

    def _load_template_and_matcher(self) -> None:
        """Load the owner template + build the (multi-pose) matcher.

        Extracted so it can be re-run after enrollment (see :meth:`_exit_pause`)
        to pick up a freshly enrolled template WITHOUT restarting the daemon.
        """
        store = TemplateStore()
        template = store.try_load()
        if template is not None:
            tau = template.tau
            if self.cfg.recognition.tau > 0.0:
                tau = self.cfg.recognition.tau  # explicit override (config)
            # Multi-pose bank: feed the enrolled per-pose samples so an off-angle
            # face matches the nearest pose (easy auth; REQ-F-07). tau unchanged.
            pose_extras = (template.samples
                           if self.cfg.recognition.pose_templates else None)
            self.matcher = Matcher(
                template.centroid, tau,
                k=self.cfg.recognition.match_votes,
                n=self.cfg.recognition.probe_frames,
                metric=self.cfg.recognition.metric,
                extra_templates=pose_extras,
                pose_max=self.cfg.recognition.pose_max,
            )
            self.template_ok = self.matcher.loaded
            event(self.log, "template_loaded", owner=template.owner_name,
                  tau=round(tau, 4), poses=self.matcher.pose_count,
                  calibration=template.calibration.get("meets_target"))
        else:
            # No/corrupt template -> face-unlock disabled, password only (FM-10).
            self.matcher = Matcher(None, self.cfg.recognition.tau_floor,
                                   k=self.cfg.recognition.match_votes,
                                   n=self.cfg.recognition.probe_frames,
                                   metric=self.cfg.recognition.metric)
            self.template_ok = False
            event(self.log, "no_template", detail="face-unlock disabled; password only")

    # -- enrollment pause (release the camera for `facelock enroll`) ------- #
    def _enter_pause(self) -> None:
        """Release the camera so the enrollment tool can open it."""
        if self.camera is not None:
            self.camera.release()
        self._camera_released = False   # not a long-absence release
        self._paused = True
        event(self.log, "perception_paused", detail="camera released for enrollment")

    def _exit_pause(self) -> None:
        """Resume perception + reload the (possibly re-enrolled) template.

        The FSM state is left as-is: enrollment happens while the owner is present
        and already-authenticated (session unlocked), so we must NOT force a
        re-lock -- that would needlessly blank the screen and, against the
        already-unlocked guardian, spin an unlock-reject loop. The new template
        takes effect immediately for the next natural lock/verify cycle.
        """
        self._paused = False
        try:
            self._load_template_and_matcher()  # pick up the new template live
        except Exception as exc:
            event(self.log, "template_reload_error", error=str(exc))
        if self.matcher is not None:
            self.matcher.reset()   # drop stale votes from the old template
        event(self.log, "perception_resumed",
              poses=(self.matcher.pose_count if self.matcher else 0))

    # -- suspend detection ------------------------------------------------ #
    def _detect_suspend(self) -> bool:
        try:
            boot = time.clock_gettime(time.CLOCK_BOOTTIME)
        except (AttributeError, OSError):
            return False
        mono = time.monotonic()
        suspended = False
        if self._prev_boot is not None and self._prev_mono is not None:
            d_boot = boot - self._prev_boot
            d_mono = mono - self._prev_mono
            if (d_boot - d_mono) > 2.0:  # >2s of wall time with frozen monotonic
                suspended = True
        self._prev_boot = boot
        self._prev_mono = mono
        return suspended

    # -- camera management ------------------------------------------------ #
    def _target_fps(self) -> int:
        return (self.cfg.camera.fps_active if self.fsm.state in _ACTIVE_STATES
                else self.cfg.camera.fps_idle)

    def _manage_camera(self, now: float) -> None:
        """Open/idle/release the camera per state + absence (FM-07, REQ-NF-06)."""
        if self.camera is None:
            return
        idle_locked = self.fsm.state in (State.LOCKED_ABSENT, State.COOLDOWN)
        if idle_locked:
            if self._absent_since is None:
                self._absent_since = now
            long_gone = (now - self._absent_since) >= self.cfg.camera.long_absence_release_s
            if long_gone and not self._camera_released:
                self.camera.release()
                self._camera_released = True
                self._next_recheck = now + max(3.0, 1.0 / max(self.cfg.camera.fps_idle, 1))
                event(self.log, "camera_released", reason="long_absence")
        else:
            self._absent_since = None

        if self._camera_released:
            # Duty-cycle: only re-acquire briefly to check for a returning face.
            if now >= self._next_recheck:
                if self.camera.reacquire():
                    self._camera_released = False
                    # Edge-trigger the exit-from-released on an ACTUAL owner
                    # return, not on stale absence. This probe reacquire does
                    # not mean the owner is back (the FSM is still LOCKED_ABSENT
                    # until _observe sees a face), so restart the settle/absence
                    # clock: a re-release cannot fire again until a fresh full
                    # `long_absence_release_s` has elapsed. Without this reset,
                    # `_absent_since` stayed stale, `long_gone` was permanently
                    # true, and the device re-released on the very next tick --
                    # the unbounded camera flap (FM-07 regression).
                    self._absent_since = now
                    event(self.log, "camera_reacquired")
                else:
                    self._next_recheck = now + max(3.0, 1.0 / max(self.cfg.camera.fps_idle, 1))
            return

        if not self.camera.is_open:
            self.camera.reacquire()
        self.camera.set_rate(self._target_fps())

    # -- per-frame perception --------------------------------------------- #
    def _observe(self, now: float, suspend: bool) -> Observation:
        obs = Observation(now=now, suspend=suspend)
        if not self.perception_ok or self.camera is None:
            obs.camera_error = True
            return obs
        if self._camera_released:
            # Camera intentionally released and no re-acquire due yet: treat as
            # "no face" absence (not a camera error) so we stay locked calmly.
            if self.matcher is not None:
                self.matcher.verify(None, 0)
            return obs
        if not self.camera.is_open:
            obs.camera_error = True
            return obs

        frame, err = self.camera.read()
        if err is not None or frame is None:
            obs.camera_error = True
            return obs
        if frame.is_dark:
            # Shutter closed / lens covered (FM-09): treat as no face.
            if self.matcher is not None:
                self.matcher.verify(None, 0)
            return obs

        detections = self.detector.detect(frame.bgr) if self.detector else []
        face_count = len(detections)
        obs.face_count = face_count

        decision_emb = None
        owner_visible = False
        stranger_visible = False
        if face_count >= 1 and self.embedder is not None and self.matcher is not None:
            for det in detections[:MAX_FACES]:
                emb = self.embedder.embed(frame.bgr, det)
                if emb is None:
                    stranger_visible = stranger_visible or True
                    continue
                score = self.matcher.score_only(emb)
                if self.matcher.loaded and self.matcher.passes(score):
                    owner_visible = True
                else:
                    stranger_visible = True
                if face_count == 1:
                    decision_emb = emb
            self._maybe_collect_challenge(frame, detections)

        if self.matcher is not None:
            obs.match = self.matcher.verify(decision_emb, face_count)
        obs.owner_visible = owner_visible
        obs.stranger_visible = stranger_visible

        # Liveness result once a challenge concludes (turn/passive/full).
        obs.liveness = self._evaluate_challenge(now)
        return obs

    def _maybe_collect_challenge(self, frame: Any, detections: list) -> None:
        if self.fsm.state != State.LIVENESS_CHALLENGE or not detections:
            return
        det = detections[0]
        # PAD consumes a bbox-context crop (design 2.2, correction A1), NOT the
        # SFace recognition warp; build it purely from the detector bbox so
        # embedder.align stays recognition-only.
        self._challenge_obs.append(
            build_liveness_observation(frame.bgr, frame.ts_monotonic, det, self.liveness.mode)
        )

    def _evaluate_challenge(self, now: float) -> Any:
        if self.fsm.state != State.LIVENESS_CHALLENGE:
            self._challenge_obs = []
            self._challenge = None
            return None
        if self._challenge is None:
            self._challenge = self.liveness.new_challenge()
            self._challenge_obs = []
            return None
        if len(self._challenge_obs) < 2:
            return None
        result = self.liveness.check(self._challenge_obs, self._challenge)
        return result if result.passed else None  # None => keep trying until timeout

    # -- emit handling ---------------------------------------------------- #
    def _handle_emits(self, emits: list) -> None:
        for sig in emits:
            if sig.kind == "LOCK":
                reason = sig.payload.get("reason", "manual")
                # No enrolled template -> face-unlock is inert. Don't request
                # automatic (presence/stranger) locks; the guardian ignores them
                # anyway (no-owner passive), so this just avoids the churn. Explicit
                # user actions (panic/disable) still pass through.
                if not self.template_ok and reason not in ("panic", "disable"):
                    event(self.log, "lock_suppressed_no_template", reason=reason)
                    continue
                resp = self.emitter.request_lock(reason)
                event(self.log, "emit_lock", reason=reason,
                      ok=resp.get("ok"), escalated=resp.get("escalated"))
                if self.matcher is not None:
                    self.matcher.reset()
            elif sig.kind == "UNLOCK_GRANT":
                resp = self.emitter.request_unlock(
                    sig.payload.get("score", 0.0),
                    sig.payload.get("tau", 0.0),
                    bool(sig.payload.get("live", False)),
                )
                if resp.get("ok"):
                    event(self.log, "emit_unlock", score=sig.payload.get("score"),
                          tau=sig.payload.get("tau"), live=sig.payload.get("live"))
                    if self.matcher is not None:
                        self.matcher.reset()
                else:
                    # Guardian rejected (stale/disabled/transport): realign to
                    # LOCKED and retry verification (fail-closed, SI-P1).
                    event(self.log, "unlock_grant_rejected", reason=resp.get("reason"))
                    self.fsm.state = State.LOCKED_ABSENT
                    if self.matcher is not None:
                        self.matcher.reset()
            # GREET is handled by the guardian on a successful unlock_grant.

    # -- on-shield feedback (cosmetic; NO lock authority) ----------------- #
    @staticmethod
    def _shield_phase(state: State, obs: Observation) -> str | None:
        """Map the current (state, observation) to a shield feedback phase.

        While a face is being verified we always report ``recognizing`` (the
        "CHECKING AUTHORIZATION" phase) -- the accept/reject verdict is signalled
        separately on the actual matcher decision, never guessed mid-check.
        Returns ``locked`` for a plain locked wait, ``denied`` for a settled
        stranger lock, or ``None`` when the shield is down (guardian owns it).
        """
        if state in (State.VERIFYING, State.LIVENESS_CHALLENGE):
            return "recognizing"
        if state == State.LOCKED_STRANGER:
            return "denied"
        if state in (State.LOCKED_ABSENT, State.COOLDOWN):
            return "locked"
        return None  # UNLOCKED_*/GRANT/etc: no feedback push

    def _push_shield_feedback(self, now: float, state: State, obs: Observation) -> None:
        """Push shield feedback (checking-progress / verdict) to the guardian.

        Progress is the REAL k-of-n verification progress read from the matcher's
        vote counters -- not a timer. A fresh rejection (fail counter bumped) is
        pushed as the UNAUTHORIZED verdict; the guardian holds it briefly.
        """
        fail_bump = self.fsm.fail_count > self._last_fail
        self._last_fail = self.fsm.fail_count

        phase = self._shield_phase(state, obs)
        progress = votes_k = votes_need = frames = frames_need = None
        if fail_bump:
            phase = "denied"  # UNAUTHORIZED verdict (a verify just failed)
        elif phase == "recognizing" and obs.match is not None and self.matcher is not None:
            from .matcher import verification_progress
            m = obs.match
            votes_k, votes_need = int(m.votes_k), int(self.matcher.k)
            frames, frames_need = int(m.votes_n), int(self.matcher.n)
            progress = verification_progress(m.votes_k, m.votes_n,
                                             self.matcher.k, self.matcher.n)

        if phase is None:
            return
        phase_changed = phase != self._last_shield_phase
        prog_changed = (phase == "recognizing" and progress is not None
                        and abs(progress - self._last_progress) >= 0.12)
        # Always emit a phase change or a verdict promptly; rate-limit mere
        # progress ticks so the bar updates without flooding the socket.
        if not (phase_changed or fail_bump):
            if not prog_changed or (now - self._last_phase_push) < self._phase_min_interval:
                return
        self._last_shield_phase = phase
        self._last_progress = progress if progress is not None else 0.0
        self._last_phase_push = now
        try:
            self.emitter.shield_status(
                phase, self.fsm.last_reason, progress=progress,
                votes_k=votes_k, votes_need=votes_need,
                frames=frames, frames_need=frames_need)
        except Exception as exc:  # feedback is cosmetic; never disrupt perception
            event(self.log, "shield_status_error", error=str(exc))

    # -- heartbeat -------------------------------------------------------- #
    def _heartbeat(self, now: float) -> None:
        if (now - self._last_hb) < self.cfg.service.heartbeat_sec:
            return
        self._last_hb = now
        self._hb_seq += 1
        health = {
            "healthy": self.perception_ok,
            "perception_ok": self.perception_ok,
            "template": self.template_ok,
            "camera_open": bool(self.camera and self.camera.is_open),
            "camera_released": self._camera_released,
            "fail_count": self.fsm.fail_count,
            "phase": self.cfg.phase,
            "liveness_mode": self.liveness.mode,
            "dry_run": self._dry_run,
        }
        resp = self.emitter.heartbeat(self._hb_seq, self.fsm.state.value, health)
        _sd_notify("WATCHDOG=1")
        if not resp.get("ok"):
            return
        # Learn the guardian's camera-pause state (for enrollment).
        if "pause_camera" in resp:
            want_pause = bool(resp["pause_camera"])
            if want_pause and not self._paused:
                self._enter_pause()
            elif not want_pause and self._paused:
                self._exit_pause()
        # Learn the guardian's disable state and feed it to the FSM.
        if "face_unlock" in resp:
            face_unlock = bool(resp["face_unlock"])
            if not face_unlock and self.fsm.state != State.DISABLED:
                self.fsm.step(Observation(now=now, disable=True))
            elif face_unlock and self.fsm.state == State.DISABLED:
                self.fsm.step(Observation(now=now, enable=True))

    # -- main loop -------------------------------------------------------- #
    def run(self) -> int:
        if self._install_signals:
            signal.signal(signal.SIGTERM, self._on_signal)
            signal.signal(signal.SIGINT, self._on_signal)
        self._build_perception()
        if self._dry_run:
            emit_dry_run_banner(self.log, "daemon")
        event(self.log, "daemon_started", perception_ok=self.perception_ok,
              template=self.template_ok, phase=self.cfg.phase,
              liveness=self.liveness.mode, dry_run=self._dry_run)
        _sd_notify("READY=1")

        # Prime the FSM out of INIT (fail-closed LOCKED); the guardian is the
        # source of truth for the shield, so we just request an initial lock.
        init = self.fsm.step(Observation(now=time.monotonic()))
        self._handle_emits(init.emits)

        try:
            while not self._stop:
                loop_start = time.monotonic()
                # Paused for enrollment: the camera is released and perception is
                # idle, but keep heartbeating so the guardian's watchdog is happy
                # and we notice the resume signal.
                if self._paused:
                    self._heartbeat(loop_start)
                    time.sleep(0.2)
                    continue
                suspend = self._detect_suspend()
                if suspend:
                    event(self.log, "resume_detected", detail="re-init camera, fresh verify")
                    if self.camera is not None:
                        self.camera.release()
                        self._camera_released = False
                    self._prev_boot = None  # avoid double-trigger next iter

                self._manage_camera(loop_start)
                obs = self._observe(loop_start, suspend)
                transition = self.fsm.step(obs)
                if transition.emits:
                    self._handle_emits(transition.emits)
                self._push_shield_feedback(loop_start, self.fsm.state, obs)
                self._heartbeat(loop_start)

                # Pace the loop to the target fps.
                period = 1.0 / max(self._target_fps(), 1)
                elapsed = time.monotonic() - loop_start
                if elapsed < period:
                    time.sleep(period - elapsed)
        finally:
            self._shutdown()
        return 0

    def _on_signal(self, signum: int, _frame: Any) -> None:
        event(self.log, "signal", signum=signum)
        self._stop = True

    def _shutdown(self) -> None:
        # Graceful shutdown: release the camera (LED off) and request a final
        # lock so the guardian keeps the session locked (SI-P2).
        if self.camera is not None:
            self.camera.release()
        try:
            self.emitter.request_lock("shutdown")
        except Exception:
            pass
        event(self.log, "daemon_stopped")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="facelockd",
                                     description="facelock perception daemon")
    parser.add_argument("--config", type=Path, default=None, help="path to config.toml")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="SAFE test mode: declare no-OS-lock dry-run (the guardian is the "
             "actuator; this surfaces the mode in the banner + health)")
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config).resolve_model_paths(_paths.models_dir())
    except Exception as exc:
        print(f"facelockd: config error: {exc}", file=sys.stderr)
        for err in getattr(exc, "errors", []) or []:
            print(f"  - {err}", file=sys.stderr)
        return 2
    try:
        dry_run = resolve_dry_run(args.dry_run, cfg)
    except DryRunUnderSystemdError as exc:
        # systemd hard-gate (fail-closed refuse; exit 2).
        print(f"facelockd: {exc}", file=sys.stderr)
        return 2
    return PerceptionDaemon(cfg, dry_run=dry_run).run()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
