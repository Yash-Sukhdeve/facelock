"""PresenceStateMachine tests (C6, design section 5). Pure, no hardware."""

from __future__ import annotations

from facelock.fsm import (
    FSMConfig,
    Observation,
    PresenceStateMachine,
    State,
)
from facelock.liveness import LivenessResult
from facelock.matcher import MatchResult


def owner_match(is_owner=True, votes_k=3, votes_n=5, score=0.9, tau=0.5, face_count=1):
    return MatchResult(is_owner, score, tau, votes_k, votes_n, face_count)


def make(**kw):
    cfg = FSMConfig(**kw)
    return PresenceStateMachine(cfg, initial=State.INIT)


def kinds(transition):
    return [s.kind for s in transition.emits]


def test_init_defaults_locked():
    m = make()
    t = m.step(Observation(now=0.0))
    assert t.new_state == State.LOCKED_ABSENT
    assert "LOCK" in kinds(t)
    assert m.reduced_state() == "LOCKED"


def test_away_locks_after_dwell():
    m = make(away_dwell_s=30)
    m.state = State.UNLOCKED_PRESENT
    # Owner disappears -> grace.
    t1 = m.step(Observation(now=0.0, face_count=0, owner_visible=False))
    assert t1.new_state == State.UNLOCKED_GRACE
    # Still within dwell.
    t2 = m.step(Observation(now=10.0, owner_visible=False))
    assert t2.new_state == State.UNLOCKED_GRACE
    # Dwell elapsed -> locked with LOCK{away}.
    t3 = m.step(Observation(now=31.0, owner_visible=False))
    assert t3.new_state == State.LOCKED_ABSENT
    assert t3.emits[0].payload["reason"] == "away"


def test_owner_returns_in_grace():
    m = make()
    m.state = State.UNLOCKED_GRACE
    m._absent_since = 0.0
    t = m.step(Observation(now=5.0, face_count=1, owner_visible=True))
    assert t.new_state == State.UNLOCKED_PRESENT


def test_stranger_lenient_no_lock_when_owner_present():
    m = make(stranger_policy="lenient", stranger_dwell_s=3)
    m.state = State.UNLOCKED_PRESENT
    # Owner co-present + stranger -> lenient must NOT lock (REQ-NF-18).
    for t in (0.0, 4.0, 8.0):
        tr = m.step(Observation(now=t, face_count=2, owner_visible=True, stranger_visible=True))
    assert tr.new_state == State.UNLOCKED_PRESENT


def test_stranger_lenient_locks_when_owner_absent():
    m = make(stranger_policy="lenient", stranger_dwell_s=3)
    m.state = State.UNLOCKED_PRESENT
    m.step(Observation(now=0.0, face_count=1, owner_visible=False, stranger_visible=True))
    tr = m.step(Observation(now=4.0, face_count=1, owner_visible=False, stranger_visible=True))
    assert tr.new_state == State.LOCKED_STRANGER
    assert tr.emits[0].payload["reason"] == "stranger"


def test_stranger_strict_locks_even_with_owner_present():
    m = make(stranger_policy="strict", stranger_dwell_s=0)
    m.state = State.UNLOCKED_PRESENT
    tr = m.step(Observation(now=0.0, face_count=2, owner_visible=True, stranger_visible=True))
    assert tr.new_state == State.LOCKED_STRANGER


def test_locked_face_begins_verifying():
    m = make()
    m.state = State.LOCKED_ABSENT
    tr = m.step(Observation(now=0.0, face_count=1))
    assert tr.new_state == State.VERIFYING


def test_verify_owner_grants_when_liveness_off():
    m = make(liveness_requires_frames=False)
    m.state = State.VERIFYING
    m._verify_start = 0.0
    tr = m.step(Observation(now=0.5, face_count=1, owner_visible=True,
                            match=owner_match(is_owner=True)))
    assert tr.new_state == State.GRANT
    assert set(kinds(tr)) == {"UNLOCK_GRANT", "GREET"}
    # GRANT settles into UNLOCKED_PRESENT next step.
    tr2 = m.step(Observation(now=0.6, face_count=1, owner_visible=True))
    assert tr2.new_state == State.UNLOCKED_PRESENT


def test_multiface_never_grants():
    m = make()
    m.state = State.VERIFYING
    m._verify_start = 0.0
    tr = m.step(Observation(now=0.5, face_count=2, stranger_visible=True,
                            match=owner_match(is_owner=False, face_count=2)))
    assert tr.new_state == State.LOCKED_STRANGER
    assert m.fail_count == 1


def test_reject_then_cooldown_then_recover():
    m = make(max_fail_attempts=2, cooldown_s=5, probe_frames=3)
    # First reject (full window, not owner).
    m.state = State.VERIFYING; m._verify_start = 0.0
    t1 = m.step(Observation(now=0.5, face_count=1,
                            match=owner_match(is_owner=False, votes_k=0, votes_n=3)))
    assert t1.new_state == State.LOCKED_ABSENT and m.fail_count == 1
    # Re-enter verifying, second reject -> cooldown.
    m.step(Observation(now=1.0, face_count=1))            # -> VERIFYING
    t2 = m.step(Observation(now=1.5, face_count=1,
                            match=owner_match(is_owner=False, votes_k=0, votes_n=3)))
    assert t2.new_state == State.COOLDOWN and m.fail_count == 2
    # Cooldown suppresses grants until it expires.
    t3 = m.step(Observation(now=2.0, face_count=1))
    assert t3.new_state == State.COOLDOWN
    t4 = m.step(Observation(now=100.0))
    assert t4.new_state == State.LOCKED_ABSENT and m.fail_count == 0


def test_camera_error_grace_then_lock():
    m = make(loss_grace_s=5)
    m.state = State.UNLOCKED_PRESENT
    t1 = m.step(Observation(now=0.0, camera_error=True))
    assert t1.new_state == State.UNLOCKED_PRESENT       # within grace
    t2 = m.step(Observation(now=6.0, camera_error=True))
    assert t2.new_state == State.CAMERA_DOWN
    assert t2.emits[0].payload["reason"] == "camera_loss"
    # Recovery requires a fresh verify.
    t3 = m.step(Observation(now=7.0, face_count=0))
    assert t3.new_state == State.LOCKED_ABSENT


def test_suspend_locks():
    m = make()
    m.state = State.UNLOCKED_PRESENT
    tr = m.step(Observation(now=0.0, suspend=True))
    assert tr.new_state == State.LOCKED_ABSENT
    assert tr.emits[0].payload["reason"] == "suspend"


def test_disable_then_enable():
    m = make()
    m.state = State.UNLOCKED_PRESENT
    t1 = m.step(Observation(now=0.0, disable=True))
    assert t1.new_state == State.DISABLED
    assert t1.emits[0].payload["reason"] == "disable"
    # While disabled, a present owner does NOT unlock.
    t2 = m.step(Observation(now=1.0, face_count=1, owner_visible=True,
                            match=owner_match(is_owner=True)))
    assert t2.new_state == State.DISABLED
    t3 = m.step(Observation(now=2.0, enable=True))
    assert t3.new_state == State.LOCKED_ABSENT


def test_panic_locks():
    m = make()
    m.state = State.UNLOCKED_PRESENT
    tr = m.step(Observation(now=0.0, panic=True))
    assert tr.new_state == State.LOCKED_ABSENT
    assert tr.emits[0].payload["reason"] == "panic"


def test_liveness_challenge_pass_grants():
    m = make(liveness_requires_frames=True, challenge_timeout_s=4)
    m.state = State.VERIFYING; m._verify_start = 0.0
    t1 = m.step(Observation(now=0.5, face_count=1, owner_visible=True,
                            match=owner_match(is_owner=True)))
    assert t1.new_state == State.LIVENESS_CHALLENGE
    t2 = m.step(Observation(now=1.0, face_count=1, owner_visible=True,
                            match=owner_match(is_owner=True),
                            liveness=LivenessResult(True, "turn", 0.9)))
    assert t2.new_state == State.GRANT


def test_liveness_challenge_timeout_fails():
    m = make(liveness_requires_frames=True, challenge_timeout_s=4)
    m.state = State.LIVENESS_CHALLENGE
    m._challenge_start = 0.0
    tr = m.step(Observation(now=5.0, face_count=1, match=owner_match(is_owner=True),
                            liveness=None))
    assert tr.new_state in (State.LOCKED_ABSENT, State.LOCKED_STRANGER)
    assert m.fail_count == 1


def _run_away_cycle(m, t0, away_dwell):
    """Owner arrives->verifies->unlocks, then leaves and auto-locks.

    Returns (lock_transition, next_t). Asserts a LOCK{away} is emitted.
    """
    # Owner face appears while locked -> VERIFYING -> GRANT -> UNLOCKED_PRESENT.
    m.step(Observation(now=t0, face_count=1))
    m._verify_start = t0
    g = m.step(Observation(now=t0 + 0.2, face_count=1, owner_visible=True,
                           match=owner_match(is_owner=True)))
    assert g.new_state == State.GRANT
    s = m.step(Observation(now=t0 + 0.4, face_count=1, owner_visible=True))
    assert s.new_state == State.UNLOCKED_PRESENT
    # Owner leaves -> grace -> away lock after dwell.
    m.step(Observation(now=t0 + 1.0, face_count=0, owner_visible=False))
    lock = m.step(Observation(now=t0 + 1.0 + away_dwell + 1.0,
                              face_count=0, owner_visible=False))
    return lock, t0 + 1.0 + away_dwell + 2.0


def test_relock_rearms_across_multiple_away_cycles():
    """Regression: re-locking after unlock must work indefinitely, not just once.

    Drives THREE full lock->unlock->lock cycles and asserts every cycle both
    reaches UNLOCKED_PRESENT and re-locks with a fresh LOCK{away}.
    """
    away = 5
    m = make(away_dwell_s=away, stranger_dwell_s=1)
    m.state = State.LOCKED_ABSENT
    t = 0.0
    for cycle in range(3):
        lock, t = _run_away_cycle(m, t, away)
        assert lock.new_state == State.LOCKED_ABSENT, f"cycle {cycle} did not re-lock"
        assert "LOCK" in kinds(lock), f"cycle {cycle} emitted no LOCK signal"
        assert lock.emits[0].payload["reason"] == "away"


def test_relock_rearms_across_stranger_cycles():
    """A stranger event must re-lock every cycle after prior unlocks (lenient)."""
    m = make(away_dwell_s=999, stranger_dwell_s=2, stranger_policy="lenient")
    m.state = State.UNLOCKED_PRESENT
    t = 0.0
    for cycle in range(3):
        # Stranger present, owner absent, past dwell -> LOCKED_STRANGER.
        m.step(Observation(now=t, face_count=1, owner_visible=False, stranger_visible=True))
        lock = m.step(Observation(now=t + 3.0, face_count=1,
                                  owner_visible=False, stranger_visible=True))
        assert lock.new_state == State.LOCKED_STRANGER, f"cycle {cycle} did not re-lock"
        assert lock.emits[0].payload["reason"] == "stranger"
        # Owner returns and unlocks.
        m.step(Observation(now=t + 3.2, face_count=1))
        m._verify_start = t + 3.2
        m.step(Observation(now=t + 3.4, face_count=1, owner_visible=True,
                           match=owner_match(is_owner=True)))
        s = m.step(Observation(now=t + 3.6, face_count=1, owner_visible=True))
        assert s.new_state == State.UNLOCKED_PRESENT
        t += 10.0


def test_step_exception_forces_locked():
    m = make()
    m.state = State.UNLOCKED_GRACE
    m._absent_since = 0.0
    m.cfg.away_dwell_s = None  # comparison will raise -> forced LOCKED (SI-P2)
    tr = m.step(Observation(now=5.0, owner_visible=False))
    assert tr.new_state == State.LOCKED_ABSENT
    assert tr.emits[0].payload["reason"] == "error"


def test_fsmconfig_owner_name_default_is_neutral_not_a_persons_name():
    # Pre-publish fix: FSMConfig's shipped default must not be the author's
    # name ("Yash") even though daemon.py always overrides it explicitly --
    # a bare FSMConfig() must not carry someone else's name either.
    cfg = FSMConfig()
    assert cfg.owner_name == "User"
    assert cfg.owner_name != "Yash"
