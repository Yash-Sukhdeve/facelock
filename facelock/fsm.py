"""PresenceStateMachine (C6) -- the deterministic presence/liveness FSM.

Realizes design section 5 (REQ-F-09/10/11/12). It is a pure, deterministic
state machine: :meth:`PresenceStateMachine.step` takes an :class:`Observation`
and returns a :class:`Transition` (new state + emitted signals). All timing is
injected via ``observation.now`` (monotonic seconds) so the machine is fully
testable without a camera or wall clock.

Fail-closed encodings (SI-P2):
  * ``INIT`` and every error/edge boundary resolve to a LOCKED state.
  * ``GRANT`` is the ONLY state that emits ``UNLOCK_GRANT``, and it is reachable
    only through the guarded owner + single-face (+ liveness) path.
  * Any unexpected exception inside :meth:`step` forces a LOCKED transition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .liveness import LivenessResult
from .matcher import MatchResult


class State(str, Enum):
    INIT = "INIT"
    UNLOCKED_PRESENT = "UNLOCKED_PRESENT"
    UNLOCKED_GRACE = "UNLOCKED_GRACE"
    LOCKED_ABSENT = "LOCKED_ABSENT"
    LOCKED_STRANGER = "LOCKED_STRANGER"
    VERIFYING = "VERIFYING"
    LIVENESS_CHALLENGE = "LIVENESS_CHALLENGE"
    GRANT = "GRANT"
    COOLDOWN = "COOLDOWN"
    CAMERA_DOWN = "CAMERA_DOWN"
    DISABLED = "DISABLED"


# States where the shield is DOWN (the desktop is usable).
SHIELD_DOWN = frozenset({State.UNLOCKED_PRESENT, State.UNLOCKED_GRACE})


@dataclass(frozen=True)
class Signal:
    kind: str  # "LOCK" | "UNLOCK_GRANT" | "GREET"
    payload: dict[str, Any] = field(default_factory=dict)


def lock(reason: str) -> Signal:
    return Signal("LOCK", {"reason": reason})


def unlock_grant(score: float, tau: float, live: bool) -> Signal:
    return Signal("UNLOCK_GRANT", {"score": score, "tau": tau, "live": live})


def greet(name: str) -> Signal:
    return Signal("GREET", {"name": name})


@dataclass
class Observation:
    """Everything the FSM needs to make one transition decision."""

    now: float
    face_count: int = 0
    owner_visible: bool = False  # owner face present this frame (single/among faces)
    stranger_visible: bool = False  # a non-owner face present this frame
    match: MatchResult | None = None  # k-of-n verdict for the dominant face
    liveness: LivenessResult | None = None  # result once a challenge concludes
    # Edge events (mutually compatible; precedence handled in step()):
    camera_error: bool = False
    suspend: bool = False
    disable: bool = False
    enable: bool = False
    panic: bool = False


@dataclass
class Transition:
    prev_state: State
    new_state: State
    emits: list[Signal] = field(default_factory=list)


@dataclass
class FSMConfig:
    away_dwell_s: float = 30.0
    stranger_policy: str = "lenient"  # lenient | strict
    stranger_dwell_s: float = 3.0
    max_fail_attempts: int = 5
    cooldown_s: float = 30.0
    loss_grace_s: float = 5.0
    liveness_requires_frames: bool = False  # mode != "off"
    challenge_timeout_s: float = 4.0
    probe_frames: int = 5  # k-of-n window size (n); reject only after a full window
    verify_timeout_s: float = 3.0  # bound VERIFYING so flicker cannot hang it
    owner_name: str = "Yash"


class PresenceStateMachine:
    """Deterministic presence/liveness FSM (design section 5)."""

    def __init__(self, config: FSMConfig, *, initial: State = State.INIT) -> None:
        self.cfg = config
        self.state = initial
        self.fail_count = 0
        # Timers (monotonic seconds); None = inactive.
        self._absent_since: float | None = None
        self._stranger_since: float | None = None
        self._cooldown_until: float | None = None
        self._camera_lost_since: float | None = None
        self._challenge_start: float | None = None
        self._verify_start: float | None = None
        self.last_reason: str | None = None

    # -- public API -------------------------------------------------------- #
    def reduced_state(self) -> str:
        """The guardian's coarse mirror: LOCKED unless the shield is down."""
        return "UNLOCKED" if self.state in SHIELD_DOWN else "LOCKED"

    def step(self, obs: Observation) -> Transition:
        prev = self.state
        try:
            new_state, emits = self._decide(obs)
        except Exception:
            # Any logic error -> forced LOCKED transition (SI-P2).
            new_state, emits = State.LOCKED_ABSENT, [lock("error")]
            self._reset_timers()
        self.state = new_state
        return Transition(prev_state=prev, new_state=new_state, emits=list(emits))

    # -- internals --------------------------------------------------------- #
    def _reset_timers(self) -> None:
        self._absent_since = None
        self._stranger_since = None
        self._camera_lost_since = None
        self._challenge_start = None
        self._verify_start = None

    def _enter_locked(self, reason: str, target: State = State.LOCKED_ABSENT) -> tuple[State, list[Signal]]:
        """Enter a locked state, emitting LOCK only when leaving shield-down.

        Reasons that MUST be actuated regardless of prior state (suspend,
        panic, disable, camera_loss, shutdown, cooldown) always emit LOCK so the
        guardian can escalate to the OS lock (SI-P4/P5).
        """
        always = {"suspend", "panic", "disable", "camera_loss", "shutdown", "startup"}
        self._reset_timers()
        self.last_reason = reason
        if self.state in SHIELD_DOWN or reason in always:
            return target, [lock(reason)]
        return target, []

    def _count_fail(self, obs: Observation, *, stranger: bool) -> tuple[State, list[Signal]]:
        """Increment the fail counter; enter COOLDOWN at the threshold (FM-15)."""
        self.fail_count += 1
        target = State.LOCKED_STRANGER if stranger else State.LOCKED_ABSENT
        self._verify_start = None
        self._challenge_start = None
        if self.fail_count >= self.cfg.max_fail_attempts:
            self._cooldown_until = obs.now + self.cfg.cooldown_s
            # Already locked while VERIFYING; no need to re-raise the shield.
            return State.COOLDOWN, []
        return target, []

    def _stranger_triggered(self, obs: Observation) -> bool:
        """Evaluate the stranger policy (design 6.2)."""
        if not obs.stranger_visible:
            self._stranger_since = None
            return False
        if self._stranger_since is None:
            self._stranger_since = obs.now
        dwell_ok = (obs.now - self._stranger_since) >= self.cfg.stranger_dwell_s
        if self.cfg.stranger_policy == "strict":
            # Lock on any non-owner face, even with the owner co-present.
            return dwell_ok
        # lenient: only when the owner is NOT co-present (REQ-NF-18).
        return dwell_ok and not obs.owner_visible

    def _decide(self, obs: Observation) -> tuple[State, list[Signal]]:
        # --- highest-precedence edge events (any state) ------------------- #
        if obs.disable:
            self.fail_count = 0
            return self._enter_locked("disable", State.DISABLED)
        if self.state == State.DISABLED:
            if obs.enable:
                self.fail_count = 0
                return self._enter_locked("startup", State.LOCKED_ABSENT)
            return State.DISABLED, []  # password/OS lock governs; no face-unlock
        if obs.panic:
            self.fail_count = 0
            return self._enter_locked("panic", State.LOCKED_ABSENT)
        if obs.suspend:
            self.fail_count = 0
            return self._enter_locked("suspend", State.LOCKED_ABSENT)

        # --- camera error handling (FM-01/09/13) ------------------------- #
        if obs.camera_error:
            if self.state in SHIELD_DOWN:
                if self._camera_lost_since is None:
                    self._camera_lost_since = obs.now
                if (obs.now - self._camera_lost_since) >= self.cfg.loss_grace_s:
                    return self._enter_locked("camera_loss", State.CAMERA_DOWN)
                # Within grace: stay unlocked but blind (do not run on stale data).
                return self.state, []
            return State.CAMERA_DOWN, []
        self._camera_lost_since = None

        # --- per-state logic --------------------------------------------- #
        handler = {
            State.INIT: self._on_init,
            State.CAMERA_DOWN: self._on_camera_down,
            State.COOLDOWN: self._on_cooldown,
            State.LOCKED_ABSENT: self._on_locked,
            State.LOCKED_STRANGER: self._on_locked,
            State.VERIFYING: self._on_verifying,
            State.LIVENESS_CHALLENGE: self._on_challenge,
            State.GRANT: self._on_grant,
            State.UNLOCKED_PRESENT: self._on_unlocked,
            State.UNLOCKED_GRACE: self._on_unlocked_grace,
        }[self.state]
        return handler(obs)

    def _on_init(self, obs: Observation) -> tuple[State, list[Signal]]:
        # Default LOCKED on startup (SI-P2).
        return self._enter_locked("startup", State.LOCKED_ABSENT)

    def _on_camera_down(self, obs: Observation) -> tuple[State, list[Signal]]:
        # Camera recovered (no camera_error reached here) -> require fresh verify.
        return State.LOCKED_ABSENT, []

    def _on_cooldown(self, obs: Observation) -> tuple[State, list[Signal]]:
        if self._cooldown_until is not None and obs.now >= self._cooldown_until:
            self._cooldown_until = None
            self.fail_count = 0
            return State.LOCKED_ABSENT, []
        return State.COOLDOWN, []  # grants suppressed

    def _on_locked(self, obs: Observation) -> tuple[State, list[Signal]]:
        if obs.face_count >= 1:
            self._verify_start = obs.now
            return State.VERIFYING, []
        return self.state, []

    def _on_verifying(self, obs: Observation) -> tuple[State, list[Signal]]:
        if obs.face_count == 0:
            self._verify_start = None
            return State.LOCKED_ABSENT, []
        if obs.face_count > 1:
            # >1 face during grant eval never unlocks (REQ-F-08, FM-06).
            return self._count_fail(obs, stranger=True)

        match = obs.match
        if match is not None and match.is_owner:
            if self.cfg.liveness_requires_frames:
                self._challenge_start = obs.now
                return State.LIVENESS_CHALLENGE, []
            return self._to_grant(match, live=False)

        # Not owner. Only count a genuine reject once the k-of-n window is full
        # (or the verify budget elapses), otherwise keep accumulating probes so
        # a single blurry frame does not cost the owner an attempt (FM-03).
        window_full = match is not None and match.votes_n >= self.cfg.probe_frames
        timed_out = (self._verify_start is not None and
                     (obs.now - self._verify_start) >= self.cfg.verify_timeout_s)
        if window_full or timed_out:
            return self._count_fail(obs, stranger=obs.stranger_visible)
        return State.VERIFYING, []

    def _on_challenge(self, obs: Observation) -> tuple[State, list[Signal]]:
        if obs.face_count != 1:
            return self._count_fail(obs, stranger=obs.stranger_visible or obs.face_count > 1)
        if obs.liveness is None:
            if self._challenge_start is not None and \
                    (obs.now - self._challenge_start) >= self.cfg.challenge_timeout_s:
                return self._count_fail(obs, stranger=False)  # timeout (FM-04)
            return State.LIVENESS_CHALLENGE, []
        if obs.liveness.passed and obs.match is not None and obs.match.is_owner:
            return self._to_grant(obs.match, live=True)
        return self._count_fail(obs, stranger=False)

    def _to_grant(self, match: MatchResult, *, live: bool) -> tuple[State, list[Signal]]:
        self.fail_count = 0
        self._reset_timers()
        self.last_reason = None
        return State.GRANT, [
            unlock_grant(match.score, match.tau, live),
            greet(self.cfg.owner_name),
        ]

    def _on_grant(self, obs: Observation) -> tuple[State, list[Signal]]:
        # GRANT is momentary; settle into UNLOCKED_PRESENT on the next step.
        self._absent_since = None
        return State.UNLOCKED_PRESENT, []

    def _on_unlocked(self, obs: Observation) -> tuple[State, list[Signal]]:
        if self._stranger_triggered(obs):
            return self._enter_locked("stranger", State.LOCKED_STRANGER)
        if not obs.owner_visible:
            self._absent_since = obs.now
            return State.UNLOCKED_GRACE, []
        self._absent_since = None
        return State.UNLOCKED_PRESENT, []

    def _on_unlocked_grace(self, obs: Observation) -> tuple[State, list[Signal]]:
        if self._stranger_triggered(obs):
            return self._enter_locked("stranger", State.LOCKED_STRANGER)
        if obs.owner_visible:
            self._absent_since = None
            return State.UNLOCKED_PRESENT, []
        if self._absent_since is None:
            self._absent_since = obs.now
        if (obs.now - self._absent_since) >= self.cfg.away_dwell_s:
            return self._enter_locked("away", State.LOCKED_ABSENT)
        return State.UNLOCKED_GRACE, []
