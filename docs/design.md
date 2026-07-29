# System Design — Pilot Face-Unlock ("Welcome back, Yash")

- **Project**: Lightweight Linux Face-Unlock (screensaver auto-lock + face-verified unlock)
- **Document type**: SDLC Design Specification (planning artifact)
- **Phase**: sdlc / design (UWS phase_1_planning)
- **Author role**: architect (Principal System Architect persona)
- **Status**: Draft for implementer hand-off and human review gate
- **Version**: 1.0
- **Date**: 2026-07-28
- **Upstream artifact**: `docs/pilot-face-unlock/requirements.md` v1.0 (REQ-F-01..27, REQ-NF-01..27, FM-01..16, SI, OQ-1..10)
- **Target repo path on approval**: `docs/pilot-face-unlock/design.md`

---

## 0. How to read this document

This design realizes every requirement in the approved requirements specification. Every design decision is traced to one or more `REQ-F-NN` / `REQ-NF-NN` / `FM-NN` / `CST-N` / `ASM-N` IDs. Nothing is asserted without a source ID.

The document is organized around the **four design-phase exit-criteria deliverables**, interpreted for a **local desktop daemon** (there is no web server or SQL database):

| Generic SDLC deliverable | This project's interpretation | Section |
|---|---|---|
| Architecture document with component diagram | Daemon + guardian processes, component catalog, ASCII C4-L2 diagram, control-flow traces | §2–§8 |
| API specification with all endpoints | Internal module contracts + local control-socket IPC protocol + CLI (no REST; justified §10.0) | §10 |
| Database schema documented | On-disk enrollment/template store + model store, file layout, permissions, versioning/revocation, encryption | §11 |
| Config system defined | Config file location/format, every setting with type + default + REQ trace, validation, stranger-policy/thresholds | §12 |

Two hard rules thread the whole design and are stated once:

> **Safety Invariant (SI):** Face-unlock is an *additive convenience path*. It MUST NEVER remove, weaken, or bypass the existing OS password / lock authentication path, and on ANY error, uncertainty, or component failure the system MUST fail **closed** (remain locked). The structural realization of SI is specified in **§6**.

> **Prototype/Hardening boundary (P vs H):** The Prototype (P) is a user-space, non-root convenience layer over the screen lock with a bounded blast radius (CST-3). Hardening (H) adds software presentation-attack detection to ISO/IEC 30107-3 targets and an *opt-in*, liveness-gated PAM path. The clean boundary is specified in **§13**.

---

## 1. Design goals and driving constraints (recap from measured platform)

The design is dimensioned to the confirmed machine (requirements §1.4), not to assumptions:

| Driver | Value | Design consequence |
|---|---|---|
| Session | Ubuntu 24.04.3, X11, GNOME | Prototype targets X11 lock backends; core stays desktop-agnostic (REQ-NF-19/20). |
| Camera | Logitech Brio 500 UVC, `/dev/video0`, YUYV 640×480@30, **no IR** (ASM-06) | RGB-only pipeline; hardware liveness out of scope; software PAD in Hardening (REQ-F-19). UVC LED = free camera-in-use indicator (REQ-F-27). |
| Compute | Ryzen 9 5900X (24 threads); RTX 4080 **unusable** (NVML mismatch, ASM-07) | **CPU-only** execution provider; bounded thread pool (REQ-NF-01/02/06/07). GPU optional accelerator only (REQ-NF-05). |
| Libraries present | OpenCV 4.12.0, ONNX Runtime 1.21.0, NumPy, Pillow, Python 3.11 | Stack = OpenCV (V4L2 capture + YuNet/SFace) + ONNX Runtime (PAD). `tomllib` stdlib for config. **Verified on target**: `cv2.FaceDetectorYN` and `cv2.FaceRecognizerSF` present; ORT providers = `['AzureExecutionProvider','CPUExecutionProvider']` → CPU is the only usable provider (REQ-NF-21). |
| Libraries absent | dlib, face_recognition, mediapipe, insightface | Design MUST NOT depend on them → no 68-landmark EAR blink; active liveness uses YuNet's 5 landmarks (REQ-NF-21). |
| Auth | Full PAM present | Prototype does NOT touch PAM (CST-3, REQ-F-18); Hardening MAY add an opt-in module (REQ-F-20, OQ-5). |

**Design budget headroom (to be confirmed by the benchmark harness, not claimed here — R1).** YuNet detection at 640×480 and SFace 128-D embedding are both small CPU ONNX graphs; the per-frame `detect + align + embed` path is expected to fit inside the **≤200 ms/frame** budget (REQ-NF-02) and sustain **≥5 fps** (REQ-NF-01) with a bounded thread pool on the 5900X. These are *design targets*; verification owns measurement (AC-NF-01/02).

---

## 2. Architecture overview

### 2.1 Process model (two cooperating user-space processes)

The system is split into **two processes** under one systemd **user** unit set. The split is deliberate and is the structural backbone of the fail-closed guarantee (§6): the process that makes *perception decisions* is separated from the process that holds *lock authority*. A crash, hang, or compromise of the perception pipeline can therefore never, by construction, leave a stranger in front of an unlocked desktop.

| Process | Privilege | Responsibility | Never does |
|---|---|---|---|
| **`facelockd`** — Perception Daemon | Unprivileged user (CST-3, REQ-NF-27) | Camera lifecycle, detect → align → embed → match, liveness, presence/absence state machine. Emits *decisions* (`LOCK`, `UNLOCK_GRANT`, `HEARTBEAT`) over the control socket. | Hold any lock authority; touch PAM/sudo (REQ-F-18); persist frames (REQ-NF-13). |
| **`facelock-guardian`** — Session Guardian | Unprivileged user | Owns the lock **shield** (input-grabbing overlay) and the abstracted lock backend; is the *only* component that can raise/dismiss the shield or actuate the OS lock; runs the watchdog; renders the greeting. | Unlock on its own perception; unlock without a signed `UNLOCK_GRANT`; ever stay passive when `facelockd`'s heartbeat is missed. |

Rationale for two processes rather than two threads: fault isolation. If `facelockd` segfaults (FM-08, FM-11), the guardian survives, keeps the shield up, and escalates to the real OS lock. In one process a segfault would tear down the shield with it. The Prototype MAY co-locate them only if the guardian's shield-ownership and OS-lock escalation run in a separate supervised subprocess; the default and reference design is two processes.

### 2.2 ASCII component diagram (C4 Level 2)

```
                        ┌─────────────────────────────── User graphical session (X11/GNOME) ───────────────────────────────┐
                        │                                                                                                   │
  /dev/video0  ───────► │  ┌──────────────────────────── facelockd  (Perception Daemon, non-root) ─────────────────────┐   │
  Brio 500 (UVC, RGB)   │  │                                                                                            │   │
  UVC activity LED ◄────┼──┤ (C1) CameraCapture ──► (C2) FaceDetector ──► (C3) FaceEmbedder ──► (C4) Matcher           │   │
   (REQ-F-27)           │  │      V4L2/OpenCV          YuNet ONNX            SFace ONNX (128-D)     cosine, τ, k-of-n    │   │
                        │  │      640×480 @5/1fps      bbox+5 landmarks      align+embed            (REQ-F-07,NF-10)     │   │
                        │  │        │                     │                     │                     │                  │   │
                        │  │        │                     └──► (C5) LivenessEngine ◄──┘  (H: PAD; P: optional blink/turn)│   │
                        │  │        │                              MiniFASNet ONNX + head-turn (REQ-F-19,NF-11)          │   │
                        │  │        ▼                                                      ▼                             │   │
                        │  │ (C6) PresenceStateMachine  ◄───────── decisions ────────────►  (C7) DecisionEmitter        │   │
                        │  │  ABSENT/PRESENT/STRANGER/                                        + HeartbeatSender          │   │
                        │  │  VERIFYING/GRANT (REQ-F-09..12)                                  (REQ-F-26, FM-08)          │   │
                        │  └───────────────┬───────────────────────────────────────────────────────┬──────────────────┘   │
                        │                  │                                                         │                      │
                        │        control socket (Unix domain, 0600)  $XDG_RUNTIME_DIR/facelock/control.sock  (§10.3)       │
                        │                  │  LOCK{reason} | UNLOCK_GRANT{id,score,live} | HEARTBEAT{seq} | GREET{name}     │
                        │                  ▼                                                         ▼                      │
                        │  ┌──────────────────────── facelock-guardian (Session Guardian, non-root) ──────────────────┐    │
                        │  │  (C8) ControlServer  ─►  (C9) LockController  ─►  (C10) LockBackend (abstracted)          │    │
                        │  │       + Watchdog          shield raise/dismiss     ┌─ GnomeDbusBackend (org.gnome.*)      │    │
                        │  │       (FM-08,16)          OS-lock escalate         ├─ LoginctlBackend (loginctl lock)     │    │
                        │  │         │                 (REQ-F-13,NF-22)         └─ XdgScreensaverBackend (fallback)     │    │
                        │  │         ▼                        │                    (REQ-NF-19/20)                       │    │
                        │  │  (C11) ShieldWindow (X11 override-redirect + KB/pointer grab)   (C12) Greeter (notify)    │    │
                        │  │        the face-dismissible convenience lock (REQ-F-14)         "Welcome back, Yash"       │    │
                        │  └──────────────────────────────────────────────────────────────── (REQ-F-15) ─────────────┘    │
                        │                                                                                                   │
                        └───────────────────────────────────────────────────────────────────────────────────────────────┘
        Cross-cutting (both processes): (C13) ConfigLoader (tomllib, §12)  (C14) TemplateStore (§11)  (C15) Logger/Audit (REQ-F-22,NF-24/25)
        Offline tool: (C16) EnrollmentTool `facelock enroll` (REQ-F-01/02/03/04)   Control CLI: `facelock <verb>` → ControlServer (§10.4)
        Backstop OUTSIDE the tool (always present, never weakened): GNOME lock screen + PAM password path (SI, REQ-F-16)
```

### 2.3 Component catalog

Each component lists: responsibility, primary REQ trace, and its owned failure modes (detailed handling in §14/§15).

| ID | Component | Responsibility | Primary REQ | Owned FMs |
|---|---|---|---|---|
| C1 | **CameraCapture** | Open/close V4L2 device, deliver frames at active/idle fps, release on long absence | REQ-F-05, NF-01/06 | FM-01, FM-05, FM-07, FM-09, FM-13 |
| C2 | **FaceDetector** (YuNet) | Detect 0/1/N faces, bbox + score + 5 landmarks | REQ-F-06 | FM-05, FM-06, FM-11 |
| C3 | **FaceEmbedder** (SFace) | Align via landmarks, produce L2-normalized 128-D embedding | REQ-F-07 | FM-05, FM-11 |
| C4 | **Matcher** | Cosine score vs owner template, threshold τ, k-of-n voting | REQ-F-07/08, NF-10 | FM-02, FM-03, FM-06, FM-10 |
| C5 | **LivenessEngine** | P: optional blink/head-turn challenge; H: passive PAD + active challenge | REQ-F-19, NF-11 | FM-04 |
| C6 | **PresenceStateMachine** | Own the session presence state; decide lock/verify/grant transitions | REQ-F-09/10/11/12 | FM-06, FM-07, FM-15 |
| C7 | **DecisionEmitter + HeartbeatSender** | Emit signed decisions + periodic heartbeat to guardian | REQ-F-26 | FM-08 |
| C8 | **ControlServer + Watchdog** | Receive decisions/CLI, verify heartbeat, escalate on miss | REQ-F-25/26, NF-23 | FM-08, FM-16 |
| C9 | **LockController** | Orchestrate shield raise/dismiss + OS-lock escalation policy | REQ-F-13, NF-22 | FM-08, FM-13, FM-16 |
| C10 | **LockBackend** (abstracted) | Concrete lock actuation via GNOME D-Bus / loginctl / xdg | REQ-F-13, NF-19/20 | FM-16 |
| C11 | **ShieldWindow** | X11 override-redirect full-screen input-grabbing overlay (Prototype convenience lock) | REQ-F-14 | FM-08, FM-16 |
| C12 | **Greeter** | Transient "Welcome back, <name>" notification ≤3 s | REQ-F-15, NF-16 | — |
| C13 | **ConfigLoader** | Load + validate TOML config; fail-closed on bad values | REQ-F-23 | FM-11(config) |
| C14 | **TemplateStore** | Persist/load owner template, integrity check, secure delete | REQ-F-01/04, NF-13/14/15 | FM-10 |
| C15 | **Logger/Audit** | Structured, image-free event log; H tamper-evident audit | REQ-F-22, NF-24/25 | FM-12 |
| C16 | **EnrollmentTool** | Guided capture, quality gate, template build, τ calibration | REQ-F-01/02/03/04 | FM-05, FM-14 |

---

## 3. Face-recognition stack decision and threshold strategy

This is the load-bearing technology choice (ADR-1/ADR-2, §17). It is fixed here so the implementer does not have to decide.

### 3.1 Chosen models (concrete, dependency-light, CPU-only)

| Stage | Model | Format / runtime | Size | Output | Rationale (REQ trace) |
|---|---|---|---|---|---|
| Detection | **YuNet** (`face_detection_yunet_2023mar.onnx`, OpenCV Zoo) via `cv2.FaceDetectorYN` | ONNX via OpenCV DNN, CPU | ~0.34 MB | bbox, det score, 5 landmarks (2 eyes, nose, 2 mouth corners) | Ships with the *already-present* OpenCV 4.12 objdetect module; millisecond-class on CPU; landmarks feed both alignment and the active-liveness head-turn check. No new dependency (REQ-NF-21, CST-8). |
| Embedding | **SFace** (`face_recognition_sface_2021dec.onnx`, OpenCV Zoo) via `cv2.FaceRecognizerSF` | ONNX via OpenCV DNN, CPU | ~37 MB | 128-D float32 embedding + built-in 5-point alignment (`alignCrop`) | Native `FaceRecognizerSF`; provides both alignment and embedding; published operating thresholds (cosine 0.363 / L2 1.128) give a calibrated starting point; small enough for REQ-NF-08 (≤500 MB RSS) and REQ-NF-09 (≤300 MB install). |
| Passive PAD (H) | **MiniFASNet / Silent-Face-Anti-Spoofing** (Minivision), ONNX | ONNX via **ONNX Runtime** CPU | ~2 MB (2 scales) | bona-fide vs spoof score | Small RGB texture/moire model runnable on CPU; the only passive PAD path available without IR hardware (ASM-06); target APCER≤5%@BPCER≤5% (REQ-NF-11). |
| Active challenge (P opt / H) | **Head-turn / nod** from YuNet 5 landmarks (yaw/pitch geometry); optional blink via a tiny eye-state ONNX classifier | pure geometry (no model) + optional ORT | ~0 / ~0.2 MB | challenge-passed bool | 68-landmark EAR blink is impossible (dlib/mediapipe absent, REQ-NF-21); a randomized head-turn is computable from YuNet's 5 points and defeats a *static* photo (AC-F-19). |

Total model footprint ≈ 40 MB, comfortably inside REQ-NF-08/09. Models are provisioned at build time from OpenCV Zoo and **pinned by SHA-256** (REQ-NF-26, §11.4); a missing/corrupt model → FM-11 fail-closed.

**Why not dlib/face_recognition/insightface?** They are absent on the target (requirements §1.4) and would break the one-command reproducible build (CST-8, REQ-NF-21/26). YuNet+SFace are already importable on this machine (verified §1). Trade-off: SFace is weaker than server-grade ArcFace/NIST algorithms — explicitly accepted for the pilot (ASM-05, REQ-NF-10) and disclosed (REQ-F-17).

### 3.2 Matching threshold strategy (τ)

The Matcher (C4) uses **cosine similarity** on L2-normalized SFace embeddings (equivalently OpenCV `FR_COSINE`). The threshold is **not** hard-coded to the model's stock value; it is **calibrated per-owner at enrollment** to hit the phase accuracy target:

1. **Seed.** Start from SFace's published cosine operating point (~0.363) as the prior.
2. **Genuine distribution.** From the ≥5 accepted enrollment samples (REQ-F-01), compute all genuine pair cosines (leave-one-out against the centroid) → genuine score distribution.
3. **Impostor distribution.** Score the enrollment centroid against a bundled, offline impostor embedding set (public/synthetic faces shipped with the tool, *not* raw images — embeddings only, REQ-NF-13) → impostor score distribution.
4. **Pick τ** as the smallest threshold with **FMR ≤ 1e-2** on the impostor set (ASM-05, REQ-NF-10-P), then verify **FNMR ≤ 5%** on genuine. If both cannot be met, enrollment warns and stores the achieved operating point with confidence intervals; it never silently ships a weak τ (R1).
5. **Persist** τ and its calibration metadata in the template (§11.2). τ is **never auto-relaxed at runtime** to overcome bad light or drift (FM-05/FM-14) — the design forbids any code path that lowers τ on failure (REQ-NF-22).
6. **k-of-n voting.** A single frame never decides. The Matcher requires **k matching frames out of the last n probes** (default 3-of-5, config `recognition.match_votes` / `probe_frames`) each ≥ τ, all with exactly one detected face, before emitting `UNLOCK_GRANT`. This suppresses single-frame false accepts (FM-02) and single-frame false rejects (FM-03) and bounds latency to the REQ-NF-02 budget.
7. **Hardening tightening.** In H, τ is recalibrated to **FMR ≤ 1e-3 / FNMR ≤ 3%** (REQ-NF-10-H) *and* every grant is additionally gated on LivenessEngine pass (REQ-F-19, §13).

---

## 4. Fail-closed structural realization (Safety Invariant) — dedicated

SI is not a runtime check bolted on; it is realized **structurally** by five design properties. This section is the answer to "how is fail-closed guaranteed."

**SI-P1 — Authority separation (unlock is a grant, not a decision).**
`facelockd` cannot unlock anything. It can only *emit a grant request*. Only `facelock-guardian` holds shield/lock authority, and it dismisses the shield **only** on receipt of a well-formed `UNLOCK_GRANT` that (a) carries a fresh nonce the guardian issued, (b) references the current lock epoch, and (c) arrives within the challenge window. Any malformed/absent/stale grant → shield stays up. There is no default-unlock branch anywhere in the guardian (REQ-NF-22, AC-NF-22).

**SI-P2 — Default state is LOCKED on every transition boundary.**
On process start, restart (FM-08), config error (FM-11/§12), suspend/resume (FM-13), template/model load failure (FM-10/FM-11), camera loss while unlocked past grace (FM-01/FM-09), and heartbeat miss — the state machine's initial/failure state is `LOCKED`. Reaching `UNLOCKED` requires an explicit positive, live (H), single-face, k-of-n owner verification. No error transitions to `UNLOCKED`.

**SI-P3 — The security backstop is the OS lock, which the tool never weakens.**
Face-unlock only ever *dismisses the tool's own shield* (Prototype) or *satisfies PAM which still offers password* (Hardening). It never removes, disables, or reconfigures the GNOME lock screen or the PAM password factor (REQ-F-16/18/24, CST-4/6). The password field remains present and functional at all times (AC-F-16). The Prototype installs **zero** PAM/login/sudo hooks (REQ-F-18, AC-F-18).

**SI-P4 — Independent watchdog escalation (a dead monitor can't fail open).**
The guardian's Watchdog (C8) expects a `HEARTBEAT` from `facelockd` every `service.heartbeat_sec`. On miss (crash, hang, kill — FM-08), the guardian **escalates to the real OS lock** (`loginctl lock-session`) and keeps the shield up, moving to password-only. systemd `Restart=always` + `WatchdogSec` restarts `facelockd`, which comes up `LOCKED`. Because the guardian owns the shield in a *separate process*, killing `facelockd` cannot drop the shield (SI-P1 + process isolation). RTO ≤ `heartbeat_sec + 1 s`.

**SI-P5 — Lock actuation is verified, not assumed.**
LockController (C9) calls the backend, then **confirms** the lock actually engaged (`org.gnome.ScreenSaver.GetActive` / logind `LockedHint`), and if not, retries the next backend (FM-16). If no backend can be confirmed, it holds the shield and raises a critical alert — it never reports "locked" optimistically (REQ-NF-19/22).

**Consequence:** there is no single component whose failure yields an unlock. Camera failure, model corruption, template tamper, perception crash, config error, backend failure, and suspend/resume each map to a `LOCKED` outcome (fault-injection matrix, §15, AC-NF-22).

---

## 5. Presence / liveness state machine

The PresenceStateMachine (C6) is the heart of the perception daemon. It is a deterministic FSM; the guardian mirrors a reduced `{LOCKED, UNLOCKED}` view driven only by guardian-authored transitions.

### 5.1 States

| State | Meaning | Camera fps | Lock reality |
|---|---|---|---|
| `INIT` | Startup, loading models/template/config | active | **LOCKED** (SI-P2) |
| `UNLOCKED_PRESENT` | Owner verified & continuously present | active→throttle | shield down |
| `UNLOCKED_GRACE` | Owner momentarily not detected (< away_dwell) | active | shield down, timer running |
| `LOCKED_ABSENT` | Owner away ≥ away_dwell; auto-locked | idle (1 fps) | shield up (+ OS-lock per policy) |
| `LOCKED_STRANGER` | Non-owner triggered lock per policy | active | shield up |
| `VERIFYING` | Face present while locked; running k-of-n (+liveness H) | active | shield up |
| `LIVENESS_CHALLENGE` | (H / P-opt) awaiting blink/head-turn | active | shield up |
| `COOLDOWN` | N consecutive fails → password-only fall-through | idle | shield up, grants suppressed |
| `CAMERA_DOWN` | Device lost/busy/blocked | retry/backoff | **LOCKED** (fail-closed) |
| `DISABLED` | Owner toggled face-unlock off | idle | password-only (SI) |

### 5.2 Transition table (normative)

| From | Event | Guard | To | Side effect |
|---|---|---|---|---|
| INIT | models+template+config OK | — | LOCKED_ABSENT | if session already locked; else UNLOCKED_PRESENT after first owner verify |
| INIT | any load error | — | CAMERA_DOWN / LOCKED | disable face-unlock, log FM-10/11, password only |
| UNLOCKED_PRESENT | owner not detected | 1 frame | UNLOCKED_GRACE | start away timer |
| UNLOCKED_GRACE | owner re-detected | — | UNLOCKED_PRESENT | cancel timer |
| UNLOCKED_GRACE | timer ≥ away_dwell (ASM-02) | — | LOCKED_ABSENT | `LOCK{reason=away}` → shield (REQ-F-10) |
| UNLOCKED_PRESENT/GRACE | non-owner face per policy (ASM-03) | policy trigger | LOCKED_STRANGER | `LOCK{reason=stranger}` → shield ≤2 s (REQ-F-11/NF-04) |
| UNLOCKED_* | >1 face during any grant eval | — | (stay) no grant | fail-closed (REQ-F-08/FM-06) |
| LOCKED_* | any face detected | — | VERIFYING | begin k-of-n (REQ-F-12) |
| VERIFYING | k-of-n owner, 1 face, (H:live) | — | LIVENESS_CHALLENGE (H/P-opt) or GRANT | — |
| LIVENESS_CHALLENGE | challenge passed | within timeout | GRANT | — |
| LIVENESS_CHALLENGE | timeout/fail | — | LOCKED_* | count fail (FM-04) |
| VERIFYING | reject / low-conf / >1 face | — | LOCKED_* | count fail (FM-03/06) |
| GRANT | — | — | UNLOCKED_PRESENT | emit `UNLOCK_GRANT` → guardian dismisses shield → `GREET` (REQ-F-14/15) |
| VERIFYING/LIVENESS | fail_count ≥ max (REQ-F-25) | — | COOLDOWN | suppress grants for cooldown_s (ASM-11) |
| any | camera error | — | CAMERA_DOWN | backoff; if was UNLOCKED, lock after camera_loss_grace (FM-01) |
| any | suspend/resume signal | — | LOCKED_ABSENT | invalidate frames, re-init camera, require fresh verify (FM-13) |
| any | owner `disable` (REQ-F-25) | — | DISABLED | password-only, shield/OS-lock still work |

`GRANT` is the *only* state that emits `UNLOCK_GRANT`, and it is reachable only through the guarded owner+single-face+(liveness) path — a direct encoding of SI-P1/P2.

---

## 6. End-to-end control-flow traces (every user feature)

Each trace marks failure points `[Fx]` and the fail-closed outcome. Traces cover all user stories US-1..US-12.

### 6.1 Primary flow: away → lock → return → verify → unlock (US-2, US-4)

```
1. Owner present, UNLOCKED_PRESENT. C1 captures @≥5fps [F: camera busy→CAMERA_DOWN, FM-01].
2. Owner leaves. C2 returns 0 faces. C6: UNLOCKED_PRESENT→UNLOCKED_GRACE, start away timer.
3. away_dwell (30s ASM-02) elapses with no owner → LOCKED_ABSENT.
   C7 emits LOCK{away} → guardian C9 raises shield C11 + (policy) OS-lock, verifies engaged [F: backend fails→FM-16 retry next backend]. (REQ-F-10, NF-03: ≤dwell+2s)
4. C1 throttles to 1fps to save power; after long_absence releases camera + LED off (FM-07, REQ-NF-06).
5. Owner returns → any face detected → C1 re-acquires camera → C6: LOCKED_ABSENT→VERIFYING.
6. C2 detects exactly 1 face [F: >1 face→no grant, FM-06]. C3 aligns+embeds. C4 cosine≥τ on k-of-n
   [F: <τ→count fail, FM-03; bad light→low conf, no accept, FM-05].
7. (H / P-opt) C5 issues randomized head-turn challenge [F: static photo can't comply→deny, FM-04].
8. C6→GRANT. C7 emits UNLOCK_GRANT{nonce,epoch,score,live=true}.
9. Guardian C8 validates nonce+epoch+window [F: stale/forged→ignore, shield stays, SI-P1].
10. C9 dismisses shield C11; C12 Greeter shows "Welcome back, Yash" ≤3s (REQ-F-15). UNLOCKED_PRESENT.
    P95 face→unlock ≤2.0s (REQ-NF-02).
```

### 6.2 Stranger-present flow (US-3, US-5)

```
1. UNLOCKED_* , owner may or may not be co-present.
2. C2 detects a non-owner face (C4 cosine < τ for a face that is not the owner).
3. C6 applies stranger policy (ASM-03 / config, §12):
   - lenient (default, OQ-2): trigger only if non-owner persists ≥stranger_dwell AND owner not co-present.
   - strict (OQ-2 alt): trigger on ANY non-owner face, even with owner present.
4. Trigger met → LOCKED_STRANGER, LOCK{stranger} → shield ≤2s (REQ-F-11, NF-04).
5. Stranger's face in VERIFYING never reaches GRANT (cosine<τ) → stays locked; password required (REQ-F-16, FM-02).
```

### 6.3 Non-owner at lock screen / spoof attempt (US-5, US-7)

```
Locked (shield up). Non-owner or a printed photo of owner appears.
- Non-owner: C4 cosine<τ → deny, no grant. OS password field untouched & usable (AC-F-16).
- Photo (Prototype, liveness off): may score ≥τ → KNOWN LIMITATION (FM-04, REQ-F-17); blast radius = dismiss
  shield of an already-logged-in session only (CST-3, REQ-F-18); never PAM/sudo. Disclosed on first run.
- Photo (Prototype blink/turn on): fails head-turn challenge → deny (AC-F-19).
- Photo/replay (Hardening): C5 passive PAD + challenge → APCER≤5%@BPCER≤5% deny (REQ-NF-11).
```

### 6.4 Enrollment flow (US-1)

```
`facelock enroll --name Yash`  (C16, offline, camera required)
1. Open camera [F: busy→FM-01 abort with message].
2. Loop capturing frames; per frame C2 detects; quality gate (REQ-F-02): exactly 1 face, size≥min_face_px,
   sharpness≥floor, brightness in range [F: 0 or >1 face / too small / dark → reject with reason, not counted].
3. Collect ≥5 accepted samples spanning small pose/expression variation; C3 embeds each.
4. Build template: centroid + per-sample embeddings; calibrate τ (§3.2) against bundled impostor embeddings.
5. Persist to TemplateStore C14 (0600, §11); log enroll event (no image, REQ-NF-13). ≤2 min (AC-F-01).
Re-enroll (REQ-F-03): same flow; supersede or augment per config; write template.bak; log.
Delete (REQ-F-04): `facelock delete` → secure-shred template + derived artefacts; face-unlock inert; password remains.
```

### 6.5 Failure-injection traces (fail-closed proof, AC-NF-22)

```
Kill facelockd while UNLOCKED: heartbeat miss → guardian escalates OS-lock, shield up → password only (SI-P4).
Kill facelockd while LOCKED: shield already owned by guardian → stays locked; systemd restarts→INIT→LOCKED.
Kill guardian: systemd Restart=always → guardian restarts; on start it re-locks (SI-P2). Meanwhile facelockd
   grants are dropped (no ControlServer) → cannot unlock. Fail-closed.
Corrupt template (FM-10): C14 integrity check fails on load → face-unlock disabled, password only, alert.
Corrupt/missing model (FM-11): ONNX load error → CAMERA_DOWN/disabled, service unhealthy, password unaffected.
Corrupt config (FM-11/§12): ConfigLoader validation fails → refuse start OR safe defaults per policy (fail-closed).
Suspend/resume (FM-13): resume → LOCKED_ABSENT, camera re-init, fresh verify required.
Disk full (FM-12): Logger degrades (rotate/drop) but locking logic unaffected (SI).
```

---

## 7. Deployment and service lifecycle

- **Packaging.** Single Python package + pinned `requirements.txt` (OpenCV, ONNX Runtime, NumPy, Pillow already present; no absent libs) + models fetched & SHA-pinned at build. One-command build/install (`make install` / `pipx`), reproducible from a clean env (REQ-NF-21/26, CST-8, AC-NF-26).
- **Autostart (ASM-10, OQ-6).** Default = **systemd user units**: `facelockd.service`, `facelock-guardian.service` (Guardian `After=graphical-session.target`, both `PartOf=graphical-session.target`), `Restart=always`, `WatchdogSec=<heartbeat*2>`. XDG autostart `.desktop` is a configured alternative (`service.autostart="xdg"`).
- **Health/watchdog.** `facelockd` pings systemd `sd_notify(WATCHDOG=1)` and the guardian heartbeat; either miss triggers restart + guardian escalation (REQ-F-26, NF-23, FM-08).
- **Graceful shutdown.** On `SIGTERM`, `facelockd` releases the camera (LED off) and emits a final `LOCK{shutdown}`; guardian keeps the session locked and escalates OS-lock (fail-closed on stop, SI-P2).
- **Resource caps.** systemd `CPUQuota`, bounded ORT/OpenCV thread pool (`OMP_NUM_THREADS`, `cv2.setNumThreads`) to honor REQ-NF-06/07 (idle ≤15% one core; active ≤~1 core), `MemoryMax` guard for REQ-NF-08 (≤500 MB).

---

## 8. (reserved — see §6 for flows)

---

## 9. (reserved — see §3 for stack)

---

## 10. Internal module / IPC interfaces ("API specification")

### 10.0 Why not REST

There is **no network surface** (CST-2, REQ-NF-12: zero outbound connections). All interfaces are in-process Python module contracts plus **one local Unix-domain control socket** (`SOCK_STREAM`, mode 0600, in `$XDG_RUNTIME_DIR/facelock/`, dir 0700) carrying newline-delimited JSON. A REST/HTTP server would add an attack surface, a dependency, and a listening port for no benefit on a single-user local tool. The control socket is the "endpoint" set; the CLI is a thin client of it.

### 10.1 In-process module contracts (perception pipeline)

Contracts are given as `signature → return`, plus error behavior. All are **synchronous, no-network, fail-closed**.

| # | Contract | Input | Output | Error behavior |
|---|---|---|---|---|
| I-1 | `CameraCapture.read()` | — | `Frame{ndarray BGR HxWx3, ts_monotonic, seq, w, h}` or `None` | On EBUSY/timeout/open-fail → `None` + `CameraError{code}`; caller (C6) → CAMERA_DOWN, backoff (FM-01). Never raises past the loop. |
| I-2 | `CameraCapture.set_rate(fps)` / `.release()` / `.reacquire()` | fps:int | ok:bool | `.release()` idempotent; frees device + LED off (FM-07). Reacquire retries with backoff. |
| I-3 | `FaceDetector.detect(frame)` | `Frame` | `List[Detection{bbox, score, landmarks5}]` | On model error → `[]` + health flag (FM-11); empty list ≠ unlock (SI). |
| I-4 | `FaceEmbedder.embed(frame, detection)` | `Frame`, `Detection` | `Embedding{float32[128], l2norm=1}` | Align/inference error → `None`; caller treats as no-match (fail-closed, FM-05/11). |
| I-5 | `Matcher.verify(embedding)` | `Embedding` | `MatchResult{is_owner:bool, score, tau, votes_k, votes_n, face_count}` | Template unloaded/corrupt → `is_owner=False` always (FM-10). Never returns owner on error. |
| I-6 | `LivenessEngine.check(frames, challenge)` | frames, `Challenge{type,params,nonce}` | `LivenessResult{passed:bool, method, score}` | Timeout/insufficient motion → `passed=False` (FM-04). Off in P → `passed=True` only if `liveness.mode=="off"` AND phase==P (documented weakness). |
| I-7 | `PresenceStateMachine.step(observations)` | detections, match, liveness, timers | `Transition{new_state, emits:[Signal]}` | Any exception → forced `LOCKED` transition (SI-P2). |
| I-8 | `TemplateStore.load()/save(t)/delete()/verify()` | `Template` | `Template`/bool | Integrity/permission failure on load → raise → C6 disables face-unlock, password only (FM-10, REQ-NF-14). |
| I-9 | `LockBackend.lock()/is_locked()/name` | — | bool | Backend failure → `False`; LockController tries next (FM-16, SI-P5). |
| I-10 | `Greeter.show(name, ttl_s)` | str, int | — | Notification failure is non-fatal; never blocks unlock/lock (REQ-F-15). |

### 10.2 Signals emitted by the state machine (C7 → guardian)

| Signal | Fields | Meaning |
|---|---|---|
| `LOCK` | `reason∈{away,stranger,panic,camera_loss,suspend,shutdown,cooldown}`, `ts` | Guardian raises shield + escalation policy (REQ-F-10/11/25). |
| `UNLOCK_GRANT` | `grant_nonce`, `lock_epoch`, `score`, `tau`, `live:bool`, `ts` | Guardian dismisses shield **iff** nonce+epoch+window valid (SI-P1). |
| `HEARTBEAT` | `seq`, `ts`, `state`, `health` | Watchdog liveness (FM-08). |
| `GREET` | `name` | Request greeting (REQ-F-15). |
| `STATUS` | full state snapshot | For `facelock status` CLI. |

### 10.3 Control-socket protocol (guardian's ControlServer)

- **Transport:** `AF_UNIX SOCK_STREAM`, path `$XDG_RUNTIME_DIR/facelock/control.sock`, socket mode **0600**, parent dir **0700**, owner-only (peer-cred checked via `SO_PEERCRED`; reject any uid ≠ owner). No TCP, ever (REQ-NF-12).
- **Framing:** one JSON object per line (`\n`-delimited), UTF-8.
- **Nonce discipline:** the guardian issues a fresh `grant_nonce` bound to the current `lock_epoch` each time it raises the shield; a grant is honored only within `liveness.challenge_timeout_s` of issuance. This prevents replay of a stale grant (SI-P1).

| Message (client→guardian) | Response | Auth/guard | Error |
|---|---|---|---|
| `{"cmd":"lock"}` (panic, REQ-F-25) | `{"ok":true,"state":"LOCKED"}` | peer=owner | actuation fail → try backends (FM-16) |
| `{"cmd":"unlock_grant", ...}` (from C7) | `{"ok":true}` / `{"ok":false,"reason":"stale_nonce"}` | nonce+epoch+window | invalid → `ok:false`, shield stays (SI-P1) |
| `{"cmd":"disable"}` / `{"cmd":"enable"}` (REQ-F-25) | `{"ok":true,"face_unlock":false}` | peer=owner | — |
| `{"cmd":"status"}` | `{state, backend, camera, template, fail_count, cooldown, healthy}` | peer=owner | — |
| `{"cmd":"reload_config"}` | `{"ok":true}` / `{"ok":false,"errors":[...]}` | peer=owner | invalid config → keep old, report (REQ-F-23) |
| `{"cmd":"heartbeat", ...}` (from C7) | (none; server-side liveness update) | — | miss → escalate (FM-08) |

### 10.4 CLI (`facelock <verb>`) — thin client of the control socket + enrollment tool

| Verb | Action | REQ |
|---|---|---|
| `enroll --name <n> [--augment]` | Guided enrollment / re-enroll (offline, C16) | REQ-F-01/02/03 |
| `delete` | Secure-delete template + artefacts | REQ-F-04, NF-15 |
| `lock` | Immediate panic lock | REQ-F-25 |
| `disable` / `enable` | Toggle face-unlock (password still works) | REQ-F-25 |
| `status` | Show state/health/backend/last decision (no images) | REQ-NF-24 |
| `test [--pad]` | Run detection/verify/PAD self-test & report score, τ, fps | REQ-NF-01/02/10 |
| `calibrate` | Re-run τ calibration on the current template | REQ-NF-10 |

**Error contract for the CLI:** every verb returns a non-zero exit code + structured stderr on failure; no verb can produce an unlock as a side effect except a *successful, verified, live* enrollment→runtime path. `enroll`/`delete` require the socket peer to be the owner uid.

---

## 11. Enrollment / template store schema ("Database schema")

No SQL. Storage is a small set of files under XDG base directories with strict permissions. This section is the authoritative on-disk contract.

### 11.1 File layout and permissions

```
$XDG_DATA_HOME/facelock/                         (dir 0700)          # default ~/.local/share/facelock
├── templates/
│   ├── owner.tmpl            (file 0600)   active owner template (schema §11.2)
│   ├── owner.tmpl.bak        (file 0600)   previous template (re-enroll rollback, REQ-F-03)
│   └── owner.tmpl.sig        (file 0600)   HMAC-SHA256 integrity tag (REQ-NF-14; H)
├── models/                                        (dir 0700)
│   ├── face_detection_yunet_2023mar.onnx  (0644)  + .sha256
│   ├── face_recognition_sface_2021dec.onnx(0644)  + .sha256
│   └── minifasnet_pad.onnx  (0644, H)             + .sha256
├── impostor_embeddings.npz  (0600)         bundled impostor set for τ calibration (embeddings only, NO images, REQ-NF-13)
└── keyring-ref              (0600, H)       Secret-Service item id for the template encryption key

$XDG_CONFIG_HOME/facelock/config.toml   (0600)     # §12; default ~/.config/facelock
$XDG_STATE_HOME/facelock/                (dir 0700) # default ~/.local/state/facelock
├── events.log              (0600)   structured, image-free event log (REQ-F-22, rotated FM-12)
├── audit.log              (0600, H)  append-only, HMAC-chained audit (REQ-NF-25)
└── health.json            (0600)     last heartbeat/health snapshot
$XDG_RUNTIME_DIR/facelock/control.sock  (0600, dir 0700)  # §10.3 (tmpfs, not persisted)
```

Design rule (REQ-NF-13, AC-NF-13): **no directory ever contains a raw frame.** Only embeddings/templates/logs. Enforced by making the capture buffer volatile (never handed to any writer) and by a build-time lint that forbids `imwrite` outside the (disabled-by-default) debug path.

### 11.2 Template schema (`owner.tmpl`)

Format: **NumPy `.npz`** (binary, compact) with a JSON `meta` blob (Prototype: plaintext `.npz` at 0600; Hardening: the same bytes AES-256-GCM-encrypted with a Secret-Service-held key, §11.5). Fields:

| Field | Type | Description | REQ |
|---|---|---|---|
| `format_version` | int | Schema version (start=1); load rejects unknown major | REQ-F-23 |
| `owner_name` | str | Display name for greeting (default "Yash", ASM-01) | REQ-F-15 |
| `created_at`, `updated_at` | ISO-8601 str | Enroll / last re-enroll time | REQ-F-03 |
| `model_id` | str | SHA-256 of the SFace model used | revocation §11.4 |
| `embedding_dim` | int | 128 (SFace) | REQ-F-07 |
| `metric` | str | `"cosine"` | §3.2 |
| `centroid` | float32[128] | L2-normalized mean embedding | REQ-F-07 |
| `samples` | float32[n≥5][128] | Per-sample L2-normalized embeddings (for re-scoring/augment) | REQ-F-01 |
| `sample_meta` | list[obj] | Per-sample `{pose_hint, sharpness, brightness, det_score}` — **no image** | REQ-F-02, NF-13 |
| `tau` | float32 | Calibrated threshold (§3.2) | REQ-NF-10 |
| `calibration` | obj | `{fmr_target, fmr_measured, fnmr_measured, ci, impostor_n, calibrated_at}` | REQ-NF-10, R1 |
| `phase` | str | `"P"` or `"H"` (affects τ target + liveness gating) | §13 |
| `revoked` | bool | Tombstone flag (set by delete/model-change) | REQ-F-04, §11.4 |

### 11.3 Permissions and privacy (REQ-NF-14/13/15)

- Prototype: `owner.tmpl` mode **0600**, dir **0700**, owner-only. No encryption but no raw frames and no network (REQ-NF-12/13).
- Hardening: encrypted-at-rest (AES-256-GCM) + integrity (the GCM tag / `owner.tmpl.sig` HMAC). **Tamper or bad key → load fails → fail closed** (FM-10, AC-NF-14).
- Data minimization: only template + config + capped logs exist (REQ-NF-15). `facelock delete` overwrites-then-unlinks all biometric artefacts (`owner.tmpl*`, `.sig`, keyring item) and sets `revoked` before removal (REQ-F-04, AC-F-04/AC-NF-15). After delete, filesystem scan finds no biometric-derived file.

### 11.4 Versioning and revocation

- **Schema versioning:** `format_version`; a load with an unknown *major* version refuses (fail-closed, REQ-F-23) rather than mis-parse.
- **Model-bound revocation:** `model_id` pins the template to a specific SFace model. If the embedding model is upgraded (different SHA), the old template's embeddings are no longer comparable → template is auto-**revoked** and the owner is prompted to re-enroll (prevents silent accuracy regression, REQ-NF-10). Detector model changes do not revoke (detection landmarks feed alignment, not identity).
- **Rollback:** re-enroll writes `owner.tmpl.bak`; a failed re-enroll restores it (never leaves the owner with no template mid-operation).
- **Explicit revocation:** `facelock delete` (REQ-F-04) and template-tamper (FM-10) both set `revoked=True` semantics → face-unlock inert, password only.

### 11.5 Key management (Hardening)

The template encryption key is stored in the **GNOME Keyring via the Secret Service API** (`org.freedesktop.secrets`), referenced by `keyring-ref`. Rationale: user-space, unlocked with the login session, no passphrase re-entry, no key-on-disk. Fallback if Secret Service is unavailable: a `0600` key file with a loud warning (degrades to Prototype-grade protection, logged). Losing the key = losing the template = re-enroll (acceptable; template is re-derivable, not precious data).

---

## 12. Configuration system

### 12.1 Location and format

- **Path:** `$XDG_CONFIG_HOME/facelock/config.toml` (default `~/.config/facelock/config.toml`), mode **0600**.
- **Format:** **TOML**, parsed with Python 3.11 **stdlib `tomllib`** (verified present on target §1) — **zero extra dependency** (REQ-NF-21, CST-8). Chosen over YAML (needs PyYAML) and JSON (no comments) — ADR-5.
- **Validation (REQ-F-23, AC-F-23):** ConfigLoader (C13) validates types + ranges on load and on `reload_config`. On an out-of-range or unparyable value it **refuses to start** (default, fail-closed) or, if `config.on_invalid="default"`, substitutes the documented default and logs a warning. It never boots with a silently-wrong security value (τ, stranger policy, liveness mode always refuse-on-invalid).

### 12.2 Full settings table (type · default · REQ trace)

| Key | Type | Default | Validated range | Satisfies |
|---|---|---|---|---|
| `camera.device` | str | `/dev/video0` | existing V4L2 node | REQ-F-05 |
| `camera.resolution` | [int,int] | `[640,480]` | enumerated by device | REQ-F-05, ASM-08 |
| `camera.pixel_format` | str | `YUYV` | `YUYV`\|`MJPG` | ASM-08, OQ-7 |
| `camera.fps_active` | int | `5` | 1–30 | REQ-NF-01 |
| `camera.fps_idle` | int | `1` | 0–5 | REQ-NF-06, FM-07 |
| `camera.long_absence_release_s` | int | `120` | ≥30 | FM-07, REQ-NF-06 |
| `camera.loss_grace_s` | int | `5` | 0–30 | FM-01 |
| `detection.model_path` | str | bundled YuNet | file exists+sha ok | REQ-F-06, FM-11 |
| `detection.confidence_floor` | float | `0.90` | 0.5–0.99 | REQ-F-06, FM-05 |
| `detection.min_face_px` | int | `80` | 40–320 | REQ-F-02, FM-05 |
| `detection.nms_threshold` | float | `0.30` | 0.1–0.9 | REQ-F-06 |
| `recognition.model_path` | str | bundled SFace | file exists+sha ok | REQ-F-07, FM-11 |
| `recognition.metric` | str | `cosine` | `cosine`\|`l2` | §3.2 |
| `recognition.tau` | float | *calibrated* | 0–1 (cosine) | REQ-NF-10 |
| `recognition.fmr_target` | float | `0.01` (P)/`0.001` (H) | ≤0.1 | REQ-NF-10, ASM-05 |
| `recognition.probe_frames` (n) | int | `5` | 3–15 | REQ-NF-02, FM-03 |
| `recognition.match_votes` (k) | int | `3` | ≤ probe_frames | FM-02/03 |
| `presence.away_dwell_s` | int | `30` | 5–600 | REQ-F-10, ASM-02, OQ-1 |
| `presence.poll_s` | float | `1.0` | 0.2–5 | REQ-F-09 |
| `presence.grace_frames` | int | `2` | 1–10 | REQ-F-09 (hysteresis) |
| `stranger.policy` | str | `lenient` | `lenient`\|`strict` | REQ-F-11, ASM-03, **OQ-2** |
| `stranger.dwell_s` | int | `3` | 0–30 | REQ-F-11, NF-04 |
| `liveness.mode` | str | `off` (P) / `full` (H) | `off`\|`blink`\|`turn`\|`passive`\|`full` | REQ-F-19, **OQ (P/H)** |
| `liveness.challenge_timeout_s` | int | `4` | 1–15 | REQ-F-19, FM-04 |
| `liveness.pad_model_path` | str | bundled MiniFASNet (H) | file exists+sha | REQ-NF-11 |
| `liveness.pad_threshold` | float | `0.5` (calibrated H) | 0–1 | REQ-NF-11 |
| `lock.backend` | str | `auto` | `auto`\|`gnome_dbus`\|`loginctl`\|`xdg` | REQ-F-13, NF-19 |
| `lock.shield` | bool | `true` (P) | — | REQ-F-14 |
| `lock.escalate_os_lock_on` | list[str] | `[stranger,panic,heartbeat_miss,suspend,shutdown]` | subset of reasons | SI-P4, FM-08/13 |
| `lock.verify_engaged_ms` | int | `500` | 100–3000 | SI-P5, FM-16 |
| `unlock.max_fail_attempts` (N) | int | `5` | 1–20 | REQ-F-25, ASM-11 |
| `unlock.cooldown_s` | int | `30` | 5–600 | REQ-F-25, FM-15 |
| `unlock.owner_name` | str | `Yash` | non-empty | REQ-F-15, ASM-01 |
| `unlock.greeting` | bool | `true` | — | REQ-F-15 |
| `privacy.persist_frames` | bool | `false` | must be false in prod | REQ-NF-13 |
| `privacy.camera_indicator` | bool | `true` | — | REQ-F-27, NF-16 |
| `logging.level` | str | `INFO` | standard levels | REQ-F-22 |
| `logging.max_size_mb` | int | `10` | 1–100 | REQ-NF-09, FM-12 |
| `logging.rotate_count` | int | `5` | 1–20 | REQ-NF-09 |
| `service.autostart` | str | `systemd` | `systemd`\|`xdg` | ASM-10, **OQ-6** |
| `service.heartbeat_sec` | int | `2` | 1–10 | FM-08, NF-23 |
| `service.restart` | str | `always` | systemd policy | REQ-F-26 |
| `security.template_encryption` | str | `none` (P) / `keyring` (H) | `none`\|`keyring`\|`keyfile` | REQ-NF-14 |
| `security.audit` | bool | `false` (P) / `true` (H) | — | REQ-NF-25 |
| `security.phase` | str | `P` | `P`\|`H` | §13 |
| `config.on_invalid` | str | `refuse` | `refuse`\|`default` | REQ-F-23 |
| `runtime.threads` | int | `4` | 1–24 | REQ-NF-06/07 |

Security-critical keys (`recognition.tau`, `recognition.fmr_target`, `stranger.policy`, `liveness.mode`, `security.phase`, `security.template_encryption`) are **always** refuse-on-invalid regardless of `config.on_invalid` — a bad security value never silently defaults.

---

## 13. Prototype ↔ Hardening boundary (clean separation)

The boundary is enforced by a single switch, `security.phase` (`P`/`H`), plus the model/liveness/lock config it gates. No perception or state-machine code differs across phases except through these injected components — the boundary is a **composition boundary, not a fork**.

| Concern | Prototype (P) | Hardening (H) | Boundary mechanism |
|---|---|---|---|
| Unlock actuation | Guardian dismisses **daemon-owned shield** (C11) on `UNLOCK_GRANT`, no password (REQ-F-14) | Face satisfies **PAM face factor** gated on liveness; guardian no longer owns a shield — the real GNOME lock is dismissed via PAM; password fallback preserved (REQ-F-24, SI-P3) | `lock.shield` on (P) vs PAM module installed (H). Supersede is clean: same `GRANT` state, different actuator. |
| Anti-spoofing | `liveness.mode∈{off,blink,turn}`; **off = documented photo-spoofable** (REQ-F-17, ASM-04); blink/turn = weak gate | `liveness.mode∈{passive,full}`: MiniFASNet passive PAD + randomized head-turn challenge; APCER≤5%@BPCER≤5% (REQ-NF-11) | LivenessEngine (C5) is a strategy object; P injects no-op/geometry, H injects PAD model. |
| Accuracy | FMR≤1e-2 / FNMR≤5% (ASM-05) | FMR≤1e-3 / FNMR≤3% + liveness gate (REQ-NF-10) | `recognition.fmr_target` + re-calibration (§3.2). |
| OS-auth scope | **None** — zero PAM/login/sudo hooks (REQ-F-18, AC-F-18) | **Opt-in** PAM module + **minimal privileged helper** (REQ-F-20/21); camera/ML stays non-root (REQ-NF-27) | See privileged-helper design below; gated by OQ-5. |
| Template protection | 0600 plaintext (REQ-NF-14) | AES-256-GCM + keyring + HMAC integrity (REQ-NF-14) | `security.template_encryption`. |
| Audit | Basic event log (REQ-F-22) | Append-only HMAC-chained audit of every accept/deny/spoof/PAM event (REQ-NF-25) | `security.audit`. |

**Hardening anti-spoofing approach (software-first, no IR — ASM-06).**
1. *Passive PAD:* MiniFASNet RGB texture/moire/reflectance classifier on the aligned face crop; score-thresholded (`liveness.pad_threshold`, calibrated to APCER≤5%@BPCER≤5% per PAI species, REQ-NF-11).
2. *Active challenge-response:* a randomized head-turn (or nod) computed from YuNet's 5 landmarks within `challenge_timeout_s`; defeats static-photo replay (AC-F-19). Optional blink via a tiny eye-state ONNX classifier if added later.
3. *Both must pass* in `full` mode before `GRANT`. Fail either → deny, count fail (FM-04).
4. *Hardware option (OQ-8):* if an IR/depth camera is added, an IR-liveness backend augments (not replaces) the software PAD.

**Optional PAM integration design (Hardening, opt-in, OQ-5).**
- A Howdy-style PAM module offers *face as a factor* to a configured PAM service (e.g., `sudo`), **only** when `liveness.mode∈{passive,full}` and the live check passes, and **always** falls through to the password factor on any failure, camera-unavailable, or low confidence — **no lockout** (REQ-F-20, AC-F-20, CST-6, SI-P3).
- **Privileged surface is minimal (REQ-F-21):** the camera + ML pipeline runs **unprivileged**; the PAM module talks to `facelockd` over the control socket and receives only a boolean `live_owner_verified` (with a fresh PAM nonce), never raw frames or embeddings. The only privileged code is the thin PAM `.so` doing input validation + the boolean check (AC-F-21). This keeps a false accept in H bounded and auditable (FM-02, risk row "PAM false accept").

---

## 14. Failure-mode design handling (FM-01 .. FM-16)

For each mandated failure mode: **detecting component**, **design-level action**, **fail-closed?**, **phase**. All obey SI (uncertainty → stay locked).

| FM | Detected by | Design-level handling | Fail-closed | Phase |
|---|---|---|---|---|
| **FM-01** Camera busy/unavailable | C1 (V4L2 EBUSY/timeout) | Exponential backoff reacquire; notify "camera unavailable"; if LOCKED → face-unlock simply unavailable, **password fallback** (SI-P3); if UNLOCKED and lost > `loss_grace_s` → C6 locks rather than run blind. Never crash. | Yes | P+H |
| **FM-02** False accept | C4 (post-hoc via audit) | Bounded to shield dismissal only (CST-3, REQ-F-18); k-of-n voting + single-face guard; τ@FMR≤1e-2 (P); H adds liveness gate + τ@FMR≤1e-3 + audit alert. | Bounded blast radius | P+H |
| **FM-03** False reject | C4 | k-of-n across `probe_frames` before giving up; **password always available** (SI); never lower τ (§3.2); chronic → re-enroll prompt (REQ-F-03); COOLDOWN prevents self-DoS (FM-15). | Yes (denies) | P+H |
| **FM-04** Presentation/spoof | C5 (P: minimal; H: PAD+challenge) | **P: KNOWN LIMITATION, disclosed** (REQ-F-17), mitigated by SI scope + optional blink/turn; **H: PAD APCER≤5%@BPCER≤5%** + challenge-response (REQ-NF-11); IR if hardware added (OQ-8). | Yes (deny on fail) | P disclose / H mitigate |
| **FM-05** Poor lighting | C1/C2/C3 | Auto exposure/gain, histogram normalization; below detection/quality floor → **do not accept** (fail closed); never lower τ in bad light; optional 720p (ASM-08, OQ-7). | Yes | P+H |
| **FM-06** Multiple faces | C2 (>1 box ≥ floor) | GRANT requires **exactly one** face → >1 face never unlocks (REQ-F-08); while unlocked, apply stranger policy → lock (ASM-03). | Yes | P+H |
| **FM-07** Long absence | C6/C1 | Auto-lock (FM), throttle to `fps_idle`, release camera + LED off after `long_absence_release_s`; reacquire on wake/activity; bound CPU (REQ-NF-06). | Yes (stays locked) | P+H |
| **FM-08** Crash / monitor absent | C8 Watchdog (heartbeat miss) | Guardian keeps shield + **escalates OS-lock**; systemd `Restart=always` restarts `facelockd`→INIT→LOCKED; never auto-unlock on restart (SI-P2/P4). | Yes | P+H |
| **FM-09** Shutter closed/covered | C1 (uniform dark, no face) | Treat as owner-absent/no-face → stay locked; notify "camera blocked"; password available. | Yes | P+H |
| **FM-10** Template corrupt/tamper | C14 (integrity/format check) | Disable face-unlock, require password, alert; H HMAC/GCM makes tamper evident (REQ-NF-14). | Yes | P+H |
| **FM-11** Model missing/corrupt | C2/C3/C5 (ONNX load / SHA mismatch) | Fail closed; service reports unhealthy; **password unaffected**; reproducible build re-provisions (REQ-NF-26). | Yes | P+H |
| **FM-12** Disk full (logs) | C15 (write error / capacity) | Rotation with hard cap (REQ-NF-09); on write failure degrade logging but **keep locking correct** (SI). | Yes (lock unaffected) | P+H |
| **FM-13** Suspend/resume/DPMS | C6 (logind session/power signals) | On resume: re-init camera, invalidate stale frames, require fresh verify; **default LOCKED on resume** (SI-P2). | Yes | P+H |
| **FM-14** Appearance drift | C4/C16 (rising FNMR) | Multi-sample template + re-enroll (REQ-F-01/03); never auto-relax τ; password fallback. | Yes (denies) | P+H |
| **FM-15** Repeated fails / brute force | C6 (fail counter) | COOLDOWN + fall-through to password (REQ-F-25, ASM-11); H logs to audit (REQ-NF-25); **never lock out password** (SI). | Yes | P+H |
| **FM-16** Lock-backend fail / conflict | C9/C10 (call error / not-engaged) | Abstracted backend with fallbacks; **verify lock engaged** and retry alternate backend; if none succeed, hold shield + critical alert, safe state (SI-P5). | Yes | P+H |

---

## 15. Per-component failure-mode analysis (detection · impact · recovery · RTO)

Minimum three modes per major component (persona Step 3). Only modes not already exhausted by §14 are added; RTO = time to safe state.

| Component | Failure mode | Detection | Impact | Recovery | RTO |
|---|---|---|---|---|---|
| C1 Camera | device busy (video call) | EBUSY | no perception | backoff, password fallback | immediate (stays locked) |
| C1 Camera | frame stall / driver hang | frame timeout | stale decisions | drop frames, reacquire, lock if unlocked | ≤ loss_grace_s |
| C1 Camera | permission revoked | open EACCES | no perception | alert, disable face-unlock, password | immediate |
| C4 Matcher | template unloaded | load flag false | cannot verify | `is_owner=False` always, password | immediate |
| C4 Matcher | NaN/degenerate embedding | norm check | bad score | reject frame, no accept | immediate |
| C4 Matcher | τ uncalibrated | meta missing | undefined boundary | refuse grants, prompt calibrate | immediate |
| C6 StateMachine | stuck timer/logic error | invariant assert | wrong state | forced LOCKED transition, restart | ≤ heartbeat_sec |
| C7/C8 Heartbeat | missed beats | watchdog counter | monitor may be dead | escalate OS-lock, restart daemon | ≤ heartbeat_sec+1s |
| C9/C10 LockBackend | GNOME D-Bus name gone | call exception | lock may not engage | try loginctl→xdg, verify engaged | ≤ verify_engaged_ms×backends |
| C11 Shield | X grab denied by another grab | grab status | shield not modal | escalate OS-lock (SI-P4) | immediate |
| C14 TemplateStore | keyring locked (H) | Secret-Service error | cannot decrypt | disable face-unlock, password | immediate |
| C15 Logger | disk full | write error | no logs | rotate/drop, keep locking | n/a (lock unaffected) |
| C16 Enrollment | insufficient quality samples | quality gate | weak/failed template | abort with reason, no ship | n/a |

---

## 16. Cross-cutting concerns checklist

- [x] **Security.**
  - *AuthN/AuthZ:* face is an *additive* factor; OS PAM password is the authority (SI-P3). Control socket is 0600 + `SO_PEERCRED` owner-only (§10.3). Prototype non-root (REQ-NF-27); H privileged surface = thin PAM `.so` only (REQ-F-21).
  - *Input validation:* config (§12), control-socket messages (schema + nonce/epoch), model SHA-pinning, template integrity (§11). All validated; invalid → fail-closed.
  - *Secrets:* no passwords stored/transmitted (REQ-F-14); H template key in Secret Service (§11.5); no raw frames on disk (REQ-NF-13); no network (REQ-NF-12).
  - *OWASP-style:* no network/injection surface; local IPC is length-framed JSON with strict schema; replay defeated by nonce/epoch; DoS self-inflicted case handled by COOLDOWN (FM-15).
- [x] **Observability.**
  - *Logging:* structured, image-free events (lock/unlock/deny/multi-face/camera-error/liveness-fail) with score, τ, face_count, liveness_result — enough to reconstruct any decision (REQ-F-22, NF-24, AC-NF-24). Rotated + capped (REQ-NF-09).
  - *Metrics/SLIs:* active fps, per-frame latency, P95 face→decision, idle/active CPU, RSS, false-reject count — surfaced by `facelock status`/`test` (REQ-NF-01/02/06/07/08).
  - *Health:* systemd watchdog + `health.json` + heartbeat (REQ-F-26, NF-23).
  - *Audit (H):* append-only HMAC-chained (REQ-NF-25).
- [x] **Configuration.** Every setting typed, defaulted, range-validated, REQ-traced (§12). No hardcoded thresholds; security-critical keys refuse-on-invalid.
- [x] **Deployment.** One-command reproducible build from pinned deps + SHA-pinned models (REQ-NF-21/26); systemd user units w/ Restart+Watchdog + resource caps (§7); graceful shutdown locks (SI-P2).
- [x] **Data.** Template schema + versioning + model-bound revocation + rollback (§11.2/11.4); secure erasure (REQ-F-04); encryption/integrity in H (REQ-NF-14); consistency = single owner, single active template (CST-1); no raw-frame persistence (REQ-NF-13).

---

## 17. Architecture Decision Records (trade-off analysis)

- **ADR-1 — Detector+Embedder = YuNet + SFace via OpenCV.** *Chosen* over dlib/face_recognition (absent), insightface (absent), MTCNN+ArcFace-via-ORT (heavier, extra deps). *Trade-off:* SFace accuracy < server-grade ArcFace; accepted for pilot (ASM-05) and disclosed (REQ-F-17). *Wins:* zero new dependency (verified importable §1), CPU-fast, small, native alignment. (REQ-NF-21/26)
- **ADR-2 — Cosine similarity + per-owner calibrated τ + k-of-n voting.** *Chosen* over fixed stock threshold and over single-frame decisions. *Trade-off:* calibration needs a bundled impostor embedding set (embeddings only, privacy-safe). *Wins:* meets phase FMR/FNMR targets with CIs (REQ-NF-10), suppresses single-frame FA/FR (FM-02/03), never auto-relaxes (REQ-NF-22).
- **ADR-3 — Prototype unlock = daemon-owned input-grabbing shield, NOT PAM.** *Rationale:* GNOME's real lock screen cannot be dismissed in user-space without PAM; the only way to have a stranger-blocking, face-dismissible, no-password, user-space lock is a daemon overlay (CST-3, REQ-F-14). *Trade-off:* a shield is only as strong as its owning process — mitigated by putting shield ownership in the **separate guardian** process + watchdog escalation to the real OS lock (SI-P4). *Alternative rejected:* use GNOME "screensaver active (blank, no lock)" + SetActive(false) — rejected because it blocks nobody (no barrier to a stranger). (§4, §6)
- **ADR-4 — Two-process split (perception vs guardian).** *Chosen* over single process for **fault isolation**: a perception crash cannot drop the shield (SI-P4). *Trade-off:* IPC complexity (one Unix socket) — acceptable, no network. (REQ-NF-22/23)
- **ADR-5 — TOML via stdlib `tomllib`.** *Chosen* over YAML (needs PyYAML dep) and JSON (no comments). *Wins:* zero dependency, human-editable, comments for operators. (REQ-NF-21)
- **ADR-6 — Template = `.npz` + JSON meta + HMAC/GCM.** *Chosen* over SQLite (overkill, no query need) and pickle (unsafe deserialization). *Wins:* compact, integrity-checkable, tamper→fail-closed (FM-10, REQ-NF-14).
- **ADR-7 — Hardening liveness = MiniFASNet passive PAD + 5-landmark head-turn challenge.** *Chosen* over EAR-blink (needs 68 landmarks from absent dlib/mediapipe) and over IR (no sensor, ASM-06). *Trade-off:* software PAD < hardware liveness; accepted, targets ISO/IEC 30107-3 APCER/BPCER (REQ-NF-11). (REQ-NF-21)
- **ADR-8 — Lock backend abstraction, loginctl primary.** *Chosen:* `loginctl lock-session` (session-manager-agnostic) primary, GNOME D-Bus + `xdg-screensaver` fallbacks, runtime-selectable + engagement-verified (REQ-NF-19/20, SI-P5, FM-16). *Wins:* Wayland backend addable without touching perception (REQ-NF-20).
- **ADR-9 — Template protection: 0600 (P) → keyring-encrypted (H).** *Trade-off:* user-space key management has no perfect answer without a passphrase; Secret Service is the pragmatic choice (session-bound, no re-entry). Template is re-derivable, so key loss = re-enroll, not data loss. (REQ-NF-14)

---

## 18. Decisions Pending (OQ-2, OQ-5, and the rest)

Autonomous dispatch: the architect cannot ask the user. The design ships with the requirements' documented **default assumptions**; below is exactly how the architecture changes under each alternative, for the human review gate to decide.

- **OQ-2 — Stranger policy default (`stranger.policy`).**
  - *Default shipped:* `lenient` (ASM-03) — lock only when a non-owner persists ≥`stranger.dwell_s` **and** the owner is not co-present.
  - *If `strict` is chosen:* the PresenceStateMachine (C6) must run recognition on **every** detected face each cycle (not just the dominant one) and trigger `LOCK{stranger}` on **any** non-owner face even with the owner present. Structural changes: (a) Matcher processes N faces/frame → higher active CPU (revisit REQ-NF-07 budget/thread pool); (b) higher nuisance-lock rate (colleague behind owner) → REQ-NF-18 relaxed; (c) presence loop can no longer early-exit on owner-found. No change to fail-closed logic. Cost is CPU + nuisance, not safety. Config flip only — both modes are implemented; strict just changes the trigger predicate.
- **OQ-5 — Hardening OS-auth scope (PAM vs screensaver-only).**
  - *Default shipped:* opt-in, liveness-gated PAM module, password fallback preserved (REQ-F-20/24, CST-6). **Prototype installs nothing** in PAM (REQ-F-18).
  - *If Hardening stays screensaver-only (strongest blast-radius bound):* drop C-privileged PAM module + helper entirely; H differs from P only by liveness strength + template encryption + audit; unlock stays the guardian-shield actuator (no REQ-F-24 supersede). *Wins:* zero privileged surface ever (REQ-F-21 moot). *Costs:* face never satisfies `sudo`/login (US-12 unmet).
  - *If Hardening does integrate PAM:* add the thin PAM `.so` + control-socket boolean protocol (§13); expands threat model (ST-5 review, REQ-NF-25 audit mandatory); guardian's shield is replaced by the real GNOME lock + PAM face factor for the unlock path (REQ-F-24). Everything is gated on `security.phase=="H"` + liveness targets met (CST-6).
- **OQ-1 — Away dwell (`presence.away_dwell_s`).** Shipped 30 s. Pure config; range 5–600 s validated. Shorter = more secure/more nuisance; no structural change.
- **OQ-3 — Accuracy operating point.** Shipped P: FMR≤1e-2/FNMR≤5%. Tightening only re-runs τ calibration (`recognition.fmr_target`); no structural change.
- **OQ-4 — Greeting form (`unlock.greeting`).** Shipped transient `notify-send` toast, name-only ≤3 s (ASM-09, REQ-NF-16). Alternative full-screen overlay = swap C12 Greeter renderer; if name should be hidden on observed screens, set generic text — config-level.
- **OQ-6 — Autostart (`service.autostart`).** Shipped systemd user units; XDG autostart is a config alternative. No structural change (§7).
- **OQ-7 — Capture resolution (`camera.resolution`/`pixel_format`).** Shipped 640×480 YUYV; 1280×720 MJPG selectable → better low-light accuracy (FM-05) at higher CPU — revisit REQ-NF-01/07 budget. Config-level.
- **OQ-8 — IR/depth hardware.** Shipped none (ASM-06). If added: add an IR-liveness backend to C5 (augments MiniFASNet); revisit REQ-F-19/NF-11 with hardware PAD targets. Pluggable, no core change.
- **OQ-9 — Single owner.** Shipped 1:1 (CST-1). Multi-user → TemplateStore holds N templates, Matcher becomes 1:N identification, FM-06/accuracy targets change. Would be a schema + Matcher change (out of pilot scope).
- **OQ-10 — Wayland.** Shipped X11 shield + backends; core is desktop-agnostic (REQ-NF-20). Wayland requires a new lock/shield backend (Wayland has no global input grab / override-redirect — likely uses `ext-session-lock-v1`); pluggable behind LockBackend, no perception change.

---

## 19. Requirement → component traceability (every REQ mapped)

| REQ | Component(s) | Realized in |
|---|---|---|
| REQ-F-01/02/03 | C16, C14, C2/C3 | §6.4, §11 |
| REQ-F-04 | C16, C14 | §6.4, §11.3/11.4 |
| REQ-F-05 | C1 | §2.3, §14 FM-01 |
| REQ-F-06 | C2 (YuNet) | §3, §10 I-3 |
| REQ-F-07 | C3/C4 (SFace, cosine τ) | §3, §10 I-4/5 |
| REQ-F-08 | C4/C6 (single-face guard) | §5.2, §14 FM-06 |
| REQ-F-09/10/11/12 | C6 | §5, §6.1/6.2 |
| REQ-F-13 | C9/C10 | §2.3, §17 ADR-8 |
| REQ-F-14 | C11 shield / C9 | §4, §6.1, ADR-3 |
| REQ-F-15 | C12 Greeter | §6.1, §10 I-10 |
| REQ-F-16 | SI-P3 (OS password path) | §4 |
| REQ-F-17 | docs + Greeter first-run + §13 | §13, FM-04 |
| REQ-F-18 | process model (no PAM in P) | §2.1, §13 |
| REQ-F-19 | C5 LivenessEngine | §13, FM-04 |
| REQ-F-20/21 | PAM `.so` + control boolean (H) | §13 |
| REQ-F-22 | C15 Logger | §16 |
| REQ-F-23 | C13 ConfigLoader | §12 |
| REQ-F-24 | PAM actuator supersede (H) | §13 |
| REQ-F-25 | C6 COOLDOWN + CLI lock/disable | §5, §10.4 |
| REQ-F-26 | C7/C8 heartbeat + systemd | §7, SI-P4 |
| REQ-F-27 | C1 UVC LED + tray | §1, §12 |
| REQ-NF-01/02 | C1..C4 pipeline, thread pool | §1, §3, §7 |
| REQ-NF-03 (away-lock ≤dwell+2 s) | C6 timers + C9 | §5.2, §6.1 |
| REQ-NF-04 (stranger-lock ≤2 s) | C6 timers + C9 | §5.2, §6.2 |
| REQ-NF-05 | ORT provider (CPU default) | §1, §3 |
| REQ-NF-06/07/08/09 | C1 throttle/release, caps, models | §7, §11 |
| REQ-NF-10 | C4 τ calibration | §3.2 |
| REQ-NF-11 | C5 PAD (H) | §13 |
| REQ-NF-12/13 | no-network, no-frame-persist | §10.0, §11.1 |
| REQ-NF-14/15 | C14 perms/encrypt/erase | §11.3/11.5 |
| REQ-NF-16 | C1 indicator, C12 name-only | §12 |
| REQ-NF-17/18 | enroll UX, lenient policy | §6.4, §18 OQ-2 |
| REQ-NF-19/20 | C10 backend abstraction | ADR-8, §18 OQ-10 |
| REQ-NF-21/26 | deps present + pinned build | §1, §7 |
| REQ-NF-22 | SI-P1..P5 | §4, §6.5, §15 |
| REQ-NF-23 | watchdog + restart | §7, SI-P4 |
| REQ-NF-24/25 | C15 log/audit | §16 |
| REQ-NF-27 | non-root P / minimal H helper | §2.1, §13 |
| FM-01..16 | per-component | §14 |

Coverage check: every REQ-F-01..27 and REQ-NF-01..27 maps to ≥1 component; every FM-01..16 has a design-level handling with a detecting component and a fail-closed classification; every user story US-1..US-12 is traced through §6.

---

## 20. Architect Quality Gate (self-check before hand-off)

- [x] Every REQ-ID (F-01..27, NF-01..27) maps to ≥1 component (§19).
- [x] Every component (C1..C16) has responsibility, REQ trace, and failure-mode analysis (§2.3, §14, §15).
- [x] Every integration/actuation point has timeout, retry, degraded mode (lock backend §17 ADR-8 / FM-16; camera FM-01; control socket §10.3).
- [x] Background/lifecycle components have trigger, failure behavior, monitoring (guardian watchdog, systemd, §7, SI-P4).
- [x] End-to-end flows traced for ALL user features — away/return, stranger, non-owner, enrollment, failure-injection (§6).
- [x] Cross-cutting concerns checklist complete — no empty boxes (§16).
- [x] No subsystem mentioned-but-not-designed: camera, detect/embed/match, liveness, presence FSM, lock controller, shield, enrollment, config, logging/audit, template store, IPC, PAM(H) all specified.
- [x] Data/template schema + migration (versioning, model-bound revocation, rollback, erasure) documented (§11).
- [x] Technology decisions have written trade-off analysis (§17 ADR-1..9).
- [x] Fail-closed Safety Invariant realized **structurally** (SI-P1..P5, §4) and proved against a fault-injection matrix (§6.5, §15).
- [x] Prototype vs Hardening a clean composition boundary, not a fork (§13); anti-spoofing + optional PAM specified.
- [x] Open decisions surfaced with per-alternative architectural impact (§18: OQ-2, OQ-5, + OQ-1/3/4/6/7/8/9/10).
- [ ] Human review gate to confirm OQ-2 (stranger policy) and OQ-5 (PAM scope) and remaining OQs — owned by orchestrator + human, NOT the architect.

*Per the Output Contract, the architect does NOT advance the workflow, create checkpoints, or mark deliverables complete. The final unchecked item is intentionally left for the human review gate.*
