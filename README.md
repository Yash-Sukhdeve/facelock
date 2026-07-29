# facelock — screensaver-only face-unlock (prototype)

A lightweight, **local**, privacy-preserving face-unlock utility for a single
Linux workstation. It locks the screen when you walk away or an unknown face
appears, and on your return it face-verifies you and dismisses its own lock with
a personalized greeting — **"Welcome back, Yash"**. Everyone else is blocked and
must use the normal OS password, which is never removed or weakened.

This is the **Prototype (P)**: a user-space convenience layer over the screen
lock. See the safety notice below.

- Target (verified): Ubuntu 24.04, Python 3.11.13, OpenCV 4.12.0
  (`cv2.FaceDetectorYN` + `cv2.FaceRecognizerSF`), X11 / GNOME, CPU-only,
  webcam at `/dev/video0`.
- Design: [`docs/design.md`](docs/design.md) (16 components, two-process
  fail-closed architecture, YuNet + SFace, per-owner τ + k-of-n voting).
- Requirements: [`docs/requirements.md`](docs/requirements.md).
- Biometric spec (matcher, models, operating points): [`docs/BIOMETRICS.md`](docs/BIOMETRICS.md).

---

## ⚠️ Safety notice (read first — REQ-F-17)

> **facelock PROTOTYPE — convenience-level security only.**
> This build controls **only the screensaver/shield** of an already-logged-in
> session. It is **NOT a password replacement** and is **NOT wired into PAM,
> login, or sudo** (REQ-F-18). With anti-spoofing disabled (the prototype
> default), it **can be fooled by a printed photo or a phone/monitor video of
> the owner** — the same limitation Howdy documents. Your **OS password lock is
> never removed or weakened and always works.** For presentation-attack
> resistance (liveness/PAD to ISO/IEC 30107-3 targets) and optional OS-auth
> integration, use the Hardening phase (`security.phase = H`).

Print it any time with `facelock disclosure`.

---

## How it works (fail-closed by construction)

Two cooperating **user-space** processes (no root):

- **`facelockd`** — perception daemon. Camera → YuNet detect → SFace embed →
  cosine match (per-owner calibrated τ, 3-of-5 voting) → presence state machine.
  It can **only *request*** a nonce-bound unlock grant; it holds no lock
  authority (SI-P1).
- **`facelock-guardian`** — session guardian. The **sole** holder of lock
  authority: owns the input-grabbing shield and the OS lock backend, runs the
  watchdog, and dismisses the shield **only** on a valid, fresh, nonce+epoch
  bound grant. On any error, crash, or missed heartbeat it keeps/engages the
  lock (SI-P2..P5).

**LOCKED is the default on every boundary.** The OS password path is never
touched. If the daemon crashes, the guardian keeps the shield up and escalates
to the real OS lock.

---

## Install (one command, user-space)

```bash
cd apps/face-unlock
./scripts/install.sh
```

This creates an isolated venv, installs pinned deps + the `facelock` package,
links `facelock` / `facelockd` / `facelock-guardian` into `~/.local/bin`,
installs the config at `~/.config/facelock/config.toml` (0600), **downloads the
SHA-pinned models**, and installs + enables the `systemd --user` units.

Prerequisites (system packages, not pip):

```bash
sudo apt-get install python3-venv python3-tk   # tkinter is needed for the shield
```

Manual install (equivalent):

```bash
python3 -m venv ~/.local/share/facelock/venv
~/.local/share/facelock/venv/bin/pip install -r requirements.txt
~/.local/share/facelock/venv/bin/pip install .
```

## (a) Download the models

Done automatically by `install.sh`; to run it by itself:

```bash
./scripts/download_models.sh
# fetches YuNet + SFace from the OpenCV Zoo into ~/.local/share/facelock/models,
# records SHA-256s in scripts/models.sha256 (verify then keep them pinned — R6).
```

## (b) Enroll your face

```bash
facelock enroll --name Yash          # look at the camera, vary pose slightly
# re-enroll / add samples later:
facelock enroll --name Yash --augment
```

Enrollment captures ≥5 quality-gated samples, builds a 128-D template, and
**calibrates τ** (never below the 0.363 safety floor). The template is stored
0600 at `~/.local/share/facelock/templates/owner.tmpl`; **no raw images are ever
written** (REQ-NF-13).

## (c) Run / test lock–unlock

```bash
# start the two services (guardian first)
systemctl --user start facelock-guardian facelockd

# check state / health
facelock status

# camera + pipeline self-test (fps, per-frame latency, your score vs τ)
facelock test --seconds 5

# manual controls
facelock lock         # panic lock now
facelock disable      # turn face-unlock off (password still works)
facelock enable
```

Then: walk away → the screen locks after `presence.away_dwell_s` (default 30 s);
return → face-verify → **"Welcome back, Yash"** and the shield drops. An
unrecognized face triggers a lock per `stranger.policy` (default `lenient`).

## Uninstall

```bash
./scripts/uninstall.sh            # keeps your template + config
./scripts/uninstall.sh --purge    # also securely deletes the template
```

Delete only your biometric data (keep the tool):

```bash
facelock delete
```

---

## Configuration

`~/.config/facelock/config.toml` (TOML, `tomllib`). Every setting is typed and
range-validated; a bad value fails closed (`config.on_invalid = refuse` by
default). **Security-critical keys** (`recognition.tau*`, `fmr_target`,
`stranger.policy`, `liveness.mode`, `security.phase`,
`security.template_encryption`, `privacy.persist_frames`) **always** refuse an
invalid value. Validate a config without starting anything:

```bash
facelock config-check
```

Key switches:

| Key | Default (P) | Meaning |
|---|---|---|
| `security.phase` | `P` | `P` = screensaver-only prototype; `H` = hardening. |
| `stranger.policy` | `lenient` | lock only when a stranger persists **and** the owner is absent; `strict` locks on any non-owner face. |
| `liveness.mode` | `off` | `off` (photo-spoofable, documented), `turn` (head-turn geometry), `passive`/`full` (Hardening PAD). |
| `presence.away_dwell_s` | `30` | seconds of absence before auto-lock. |
| `lock.backend` | `auto` | `loginctl` → GNOME D-Bus → `xdg-screensaver`, verified-engaged. |

See `config/facelock.toml` for the full annotated table with REQ traces.

---

## What is fully functional vs. a documented hook

**Fully functional (Prototype):** camera lifecycle, YuNet detection, SFace
embedding, cosine matcher + k-of-n voting, per-owner τ calibration, template
store (0600 + HMAC integrity + secure delete), config system, presence/lock
state machine, nonce-bound control IPC, the guardian (shield + watchdog +
verified OS-lock escalation + greeting), enrollment/delete/calibrate, CLI, the
`turn` active-liveness geometry, and the systemd units.

**Documented hooks (Hardening, interface in place, not stubs):**

- **Passive PAD (`liveness.mode = passive|full`)** — MiniFASNet via `cv2.dnn`.
  Runs *if* a compatible model is provisioned at `liveness.pad_model_path`; if
  absent it **fails closed** (deny). Swap in the model-specific output decoding
  when the model is pinned.
- **`blink` liveness** — requires a tiny eye-state model (YuNet gives only eye
  *centre* points, so EAR-blink is impossible without one). Fails closed until
  a model is provided.
- **Template encryption (`security.template_encryption = keyring`)** — the
  Hardening design binds the integrity/encryption key to the Secret Service;
  the prototype uses a 0600 key file for HMAC integrity. Same schema.
- **PAM / OS-auth integration** — **intentionally absent** in the prototype
  (REQ-F-18). The Hardening design (§13) adds an opt-in, liveness-gated PAM
  module that always falls through to the password.
- **Impostor calibration set** — the prototype ships a *synthetic* impostor
  embedding set (privacy-safe; no images). The `tau_floor` guarantees τ is
  never weaker than SFace's characterized operating point. Hardening replaces
  it with real impostor embeddings from a public dataset.

---

## Requirement traceability (highlights)

REQ-F-01..04 enrollment/delete (`enroll.py`, `store.py`) · REQ-F-05..08
perception (`capture/detect/embed/matcher`) · REQ-F-09..12 presence
(`fsm.py`) · REQ-F-13/14 lock/shield (`lock_backend.py`, `shield.py`,
`guardian.py`) · REQ-F-15 greeting (`shield.Greeter`) · REQ-F-16/18 password
untouched / no PAM (architecture) · REQ-F-17 this notice · REQ-F-19 liveness
(`liveness.py`) · REQ-F-22/23 logging/config (`logging_setup.py`, `config.py`) ·
REQ-F-25 panic/disable/cooldown · REQ-F-26 watchdog (`guardian.py`) ·
REQ-NF-10 τ calibration (`calibrate.py`) · REQ-NF-12/13 local-only, no frames ·
REQ-NF-14 0600 template · REQ-NF-19/22 backend abstraction + fail-closed ·
REQ-NF-21/26 pinned deps + reproducible build.

## Tests

Hardware-independent unit tests (no camera, no daemon, no display):

```bash
python3 -m pytest        # 114 tests
```

They cover matcher math, k-of-n voting, τ calibration + floor, config
validation (incl. security-critical refuse), template store round-trip +
integrity + secure delete, the FSM (away/stranger/verify/grant/cooldown/
camera-loss/suspend/disable + fail-closed forcing), the nonce-bound grant
authority + control socket, liveness geometry + fail-closed hooks, the lock
controller fallback/verify, and the guardian dispatch.

## License

MIT.
