# Requirements Specification — Pilot Face-Unlock ("Welcome back, Yash")

- **Project**: Lightweight Linux Face-Unlock (screensaver auto-lock + face-verified unlock)
- **Document type**: SDLC Requirements Specification (planning artifact)
- **Phase**: sdlc / requirements (UWS phase_1_planning)
- **Author role**: researcher (Principal Research Scientist persona)
- **Status**: Draft for architect hand-off and human review gate
- **Version**: 1.0
- **Date**: 2026-07-28
- **Target repo path on approval**: `docs/pilot-face-unlock/requirements.md`

---

## 0. How to read this document

Every requirement carries a unique, stable ID:

- `REQ-F-NN` — functional requirement (a behaviour the system performs).
- `REQ-NF-NN` — non-functional requirement (a quality attribute: performance, resource, privacy, usability, portability, security target, reliability, maintainability).

Each requirement is tagged with a **Phase** attribute that satisfies the "prototype-then-harden" security bar mandated by the task brief:

- **P** — required for the **Prototype** (convenience-first, user-space only, bounded blast radius, NOT a password replacement).
- **H** — required for the **Hardening** phase (liveness/anti-spoofing to ISO/IEC 30107-3 targets, optional OS/PAM integration, audit).
- **P+H** — required in both phases.

Every requirement has at least one testable acceptance criterion (Section 8, consolidated in the acceptance-criteria table in Section 10). Failure modes are inventoried per feature in Section 11. Unresolved specification gaps are surfaced in Section 15 (Open Questions / Assumptions) with a concrete working assumption so that no downstream work is blocked.

A guiding safety invariant threads the whole specification and is stated here once:

> **Safety Invariant (SI):** Face-unlock is an *additive convenience path*. It MUST NEVER remove, weaken, or bypass the existing operating-system password / lock authentication path, and on ANY error, uncertainty, or component failure the system MUST fail **closed** (remain locked). This invariant is grounded in the documented reality that consumer face-unlock layers such as Howdy are "in no way as secure as a password ... a more quick and convenient way of logging in, not a more secure one" [R2][R3].

---

## 1. Purpose and scope

### 1.1 Purpose
Specify a lightweight, local, privacy-preserving face-unlock utility for a single Linux workstation. The utility:
1. Automatically engages the screen lock / screensaver when the owner walks away **or** when an unknown face is present (deterrence / shoulder-surfing protection).
2. On the owner's return, verifies the owner's face and unlocks the session, displaying a personalized greeting ("Welcome back, Yash").
3. Blocks everyone else from unlocking via face; non-owners fall back to the standard OS password, which is never removed.

### 1.2 In scope
- Owner enrollment, face detection, face verification (1:1), presence/absence monitoring, stranger-present detection, auto-lock, face-verified unlock, personalized greeting, password fallback, configuration, logging, camera lifecycle, service lifecycle, privacy controls, and a phased anti-spoofing / OS-auth strategy.

### 1.3 Out of scope (see Section 16 for the full list)
- Multi-user households / multiple enrolled identities (single owner in the pilot).
- Remote / cloud face recognition (100% local processing).
- Replacing the display-manager login (GDM) or `sudo` authentication in the Prototype.
- Wayland support in the Prototype (target is the confirmed X11 + GNOME session; portability is a design constraint, see REQ-NF-19..21).

### 1.4 Target platform (measured, not assumed)
The requirements are grounded in the confirmed state of the target machine:

| Attribute | Measured value | Source |
|---|---|---|
| OS | Ubuntu 24.04.3 LTS (Noble) | `/etc/os-release` |
| Kernel | 6.8.0-136-generic x86_64 | `uname -a` |
| Session / display server | X11 | `XDG_SESSION_TYPE=x11` |
| Desktop environment | GNOME (`ubuntu:GNOME`) | `XDG_CURRENT_DESKTOP` |
| Camera | Logitech Brio 500 (UVC), `/dev/video0` (+ `/dev/video1`, `/dev/media0`) | `v4l2-ctl --list-devices` |
| Camera pixel format / resolution observed | YUYV 4:2:2, discrete up to 640x480 @ 30 fps (plus lower) | `v4l2-ctl --list-formats-ext` |
| Camera IR / depth | None observed; Brio 500 is an RGB webcam with no IR emitter | device enumeration (ASM-06) |
| CPU | AMD Ryzen 9 5900X, 24 threads | `/proc/cpuinfo` |
| RAM | ~64 GB | `/proc/meminfo` |
| GPU | NVIDIA RTX 4080 present but driver/NVML mismatch: `nvidia-smi` fails | `nvidia-smi -L` |
| Screen lock backends available | `loginctl`, `gnome-screensaver-command`, `xdg-screensaver`, `light-locker`, `xset`, `gdbus`/`dbus-send` | `which` probes |
| Auth stack | Full PAM present (`common-auth`, `gdm-password`, `gdm-fingerprint`, ...) | `ls /etc/pam.d/` |
| Python | 3.11.13 | `python3 --version` |
| ML libraries present | OpenCV 4.12.0 (`cv2`), NumPy 1.25.2, ONNX Runtime 1.21.0, Pillow 12.2.0 | import probe |
| ML libraries absent | `dlib`, `face_recognition`, `mediapipe`, `insightface` | import probe |

**Feasibility implication (drives NFRs):** the natural, dependency-light implementation stack is **OpenCV (V4L2 capture) + ONNX Runtime (CPU execution provider)** with a small ONNX face detector and a small ONNX face-embedding model. The RTX 4080 cannot be relied upon (driver mismatch), so all performance targets in Section 7 are specified for **CPU-only** execution; GPU is treated as an optional accelerator (REQ-NF-05). The absence of an IR/depth sensor (ASM-06) means hardware-based liveness (Windows-Hello style) is unavailable without new hardware, so the Hardening anti-spoofing strategy is software/behavioural first (REQ-F-19, REQ-NF-11).

---

## 2. Stakeholders and personas

| ID | Persona | Description | Primary concern |
|---|---|---|---|
| ST-1 | **Owner (Yash)** | Sole enrolled user; technical Linux user; runs this on his own workstation. | Convenience: hands-free lock/unlock without weakening security. |
| ST-2 | **Bystander / colleague** | A non-malicious other person who may walk into camera view. | Must not be able to unlock; must not cause constant nuisance locks. |
| ST-3 | **Opportunistic intruder** | Someone who tries to access the unattended machine, possibly with a printed photo or phone video of the owner. | The system must resist casual walk-up access and (in Hardening) presentation attacks. |
| ST-4 | **Maintainer / developer** | Builds, configures, tests, and operates the tool. | Reproducible build, clear config, logs, debuggability. |
| ST-5 | **Security reviewer (Hardening)** | Reviews the tool before any OS-auth integration. | Threat model, PAD evaluation to ISO/IEC 30107-3, secure template storage, audit trail. |

---

## 3. Definitions and terminology

| Term | Definition |
|---|---|
| **Owner** | The single enrolled, authorized user (Yash). |
| **Template / embedding** | A fixed-length numeric vector derived from face images by the embedding model; used for 1:1 comparison. Not a stored photograph. |
| **Enrollment** | One-time process of capturing owner samples and building the reference template(s). |
| **Verification (1:1)** | Comparing a live probe embedding against the owner template and thresholding a similarity/distance score. |
| **Match threshold (τ)** | Decision boundary on the similarity score above which a probe is accepted as the owner. |
| **FMR** (False Match Rate) | Fraction of impostor comparisons wrongly accepted as the owner (a.k.a. false accept) [R4]. |
| **FNMR** (False Non-Match Rate) | Fraction of genuine (owner) comparisons wrongly rejected (a.k.a. false reject) [R4]. |
| **PAI** (Presentation Attack Instrument) | An artefact used to subvert the system: printed photo, screen/video replay, 2D/3D mask [R1]. |
| **PAD** (Presentation Attack Detection) | Liveness / anti-spoofing: automatic detection of a presentation attack [R1]. |
| **APCER** | Attack Presentation Classification Error Rate: fraction of attack presentations of a given PAI species wrongly classified as bona fide [R1]. |
| **BPCER** | Bona fide Presentation Classification Error Rate: fraction of genuine presentations wrongly classified as attacks [R1]. |
| **Fail closed / fail safe** | On error or uncertainty, the system remains **locked** (denies access), never unlocks. |
| **Auto-lock** | System-initiated engagement of the screensaver/lock. |
| **Away / absence** | No owner face detected for a configured dwell time. |
| **Stranger-present** | A detected face that does not match the owner. |
| **Prototype (P)** | First deliverable: convenience layer over the screensaver only; not wired to PAM/login/sudo. |
| **Hardening (H)** | Second deliverable: liveness to ISO/IEC 30107-3 targets, optional PAM integration, audit, secure storage. |

---

## 4. Assumptions and constraints

Constraints are non-negotiable design boundaries; assumptions are working decisions taken because this is an autonomous dispatch (no user available to ask). Every assumption is flagged for confirmation in Section 15.

### 4.1 Constraints (CST)
- **CST-1** Single physical workstation; single owner in the pilot.
- **CST-2** All face processing is 100% local; no network, no cloud, no telemetry (privacy, REQ-NF-12..15).
- **CST-3** Prototype runs entirely in the owner's user session (no root); it controls only the screensaver/lock via user-space D-Bus / `loginctl` / `xdg-screensaver`. It is NOT integrated with GDM login, PAM, or `sudo`. This bounds the blast radius of a false accept to "unlocking an already-logged-in session," never privilege escalation.
- **CST-4** The existing OS password lock remains fully functional at all times (Safety Invariant SI).
- **CST-5** Target session is X11 + GNOME as measured (Section 1.4); the Prototype targets this configuration.
- **CST-6** Hardening-phase OS/PAM integration, if adopted, MUST gate on liveness meeting REQ-NF-11 targets and MUST preserve the password fallback (SI, REQ-F-24).
- **CST-7** No hardware IR/depth sensor is present (ASM-06); hardware-based liveness is out of scope unless hardware is added.
- **CST-8** Reproducible, one-command build with a pinned Python environment (user rule R5).

### 4.2 Assumptions (ASM) — see Section 15 for confirmation items
- **ASM-01** Owner name for the greeting is "Yash"; it is configurable and defaults to the enrollment-provided name.
- **ASM-02** Away-lock dwell time default = **30 seconds** of continuous owner-absence (aligns with common screensaver grace and balances security vs nuisance).
- **ASM-03** Stranger-present lock policy default = **"lenient"**: lock when a non-owner face persists for >= 3 s AND the owner is not co-present. A **"strict"** mode (lock on ANY non-owner face regardless of owner presence) is configurable. Rationale: strict mode would nuisance-lock whenever a colleague appears behind the owner.
- **ASM-04** Prototype anti-spoofing target = **best-effort only**; the Prototype is explicitly, documentedly vulnerable to photo/replay (as Howdy is [R2][R3]), mitigated by SI (screensaver-only scope) and an optional blink challenge (REQ-F-19).
- **ASM-05** Prototype accuracy operating point: match threshold tuned for **FMR <= 1e-2 (1%)** with **FNMR <= 5%**, measured on a small local owner/impostor set. These are realistic for a lightweight local model and are far weaker than server-grade NIST algorithms (FNMR ~0.1-1% at FMR 1e-5 [R4]); the gap is accepted for the Prototype and tightened in Hardening (REQ-NF-10).
- **ASM-06** The Logitech Brio 500 on this machine is RGB-only (no IR emitter observed in device enumeration). If an IR/depth camera is later added, hardware liveness becomes an option (revisits REQ-F-19/REQ-NF-11).
- **ASM-07** GPU (RTX 4080) is treated as unavailable due to the measured driver/NVML mismatch; CPU-only performance targets govern (REQ-NF-01..05).
- **ASM-08** Capture resolution default = **640x480** (confirmed available, sufficient for face recognition); optionally 1280x720 via MJPG if enumerated at runtime.
- **ASM-09** On unlock, the greeting is shown as a transient desktop notification / overlay for <= 3 s; it must not itself leak that face-unlock is enabled to an onlooker beyond the owner's name (privacy nuance, REQ-NF-16).
- **ASM-10** The tool auto-starts with the user's graphical session (systemd user unit / XDG autostart) and does not require the owner to launch it manually.
- **ASM-11** Repeated failed face attempts do NOT lock out the password path (SI); they only trigger a cool-down / fall-through to password (REQ-F-25) to avoid a self-inflicted denial of service.

---

## 5. Phase model (prototype-then-harden)

| Aspect | Prototype (P) | Hardening (H) |
|---|---|---|
| Blast radius | Screensaver / session lock only (CST-3) | May extend to PAM/login IF liveness targets met (CST-6) |
| Anti-spoofing | Best-effort; optional blink challenge; documented vulnerable (ASM-04) | PAD to ISO/IEC 30107-3, APCER/BPCER targets (REQ-NF-11) [R1] |
| OS auth integration | None | Optional Howdy-style PAM module, password fallback preserved [R2] |
| Template storage | Local file, perms 0600, no raw frames (REQ-NF-13) | Encrypted at rest, integrity-checked, tamper-evident (REQ-NF-14) |
| Accuracy target | FMR<=1e-2, FNMR<=5% (ASM-05) | FMR<=1e-3 with liveness, FNMR<=3% (REQ-NF-10) |
| Audit | Basic local log (REQ-F-22) | Tamper-evident audit trail of all accept/deny/spoof events (REQ-NF-17) |
| Liveness hardware | RGB software only (ASM-06) | Software PAD; IR/depth if hardware added (REQ-F-19) |

The requirements below are individually tagged P / H / P+H so the review gate can see exactly which are Prototype vs Hardening.

---

## 6. Functional requirements (REQ-F)

Each requirement gives: user story, the normative requirement, phase tag, priority (MUST / SHOULD / MAY per RFC 2119 sense), and at least one acceptance criterion (also consolidated in Section 10).

### 6.1 Enrollment and template management

**REQ-F-01 — Owner enrollment** · Phase P+H · MUST
*User story:* As the owner (ST-1), I want to enroll my face once so the system can later recognize me.
*Requirement:* The system SHALL provide an enrollment flow that captures multiple owner face samples (>= 5 frames spanning small pose/expression variation), computes embeddings, and stores a reference template plus the owner display name.
*Acceptance (AC-F-01):* Given the camera is available, when the owner completes enrollment, then a template file exists with perms 0600 and >= 5 contributing samples, and enrollment completes in <= 2 minutes (ties to REQ-NF-17-usability).

**REQ-F-02 — Enrollment quality gate** · Phase P+H · MUST
*User story:* As the owner, I want enrollment to reject bad samples so recognition is reliable.
*Requirement:* The system SHALL reject enrollment frames with no detected face, multiple faces, or below a minimum face size / sharpness / brightness threshold, and SHALL require the minimum sample count from accepted frames only.
*Acceptance (AC-F-02):* Given a frame with zero or >1 faces, when enrollment ingests it, then it is rejected with a reason and does not count toward the minimum.

**REQ-F-03 — Re-enroll / update template** · Phase P+H · SHOULD
*User story:* As the owner, when my appearance changes (glasses, beard, haircut) I want to update my template.
*Requirement:* The system SHALL allow re-enrollment that replaces or augments the template, and SHALL keep at most one active owner identity in the pilot (CST-1).
*Acceptance (AC-F-03):* When the owner re-enrolls, then the previous template is superseded (or augmented per config) and the change is logged.

**REQ-F-04 — Delete template / factory reset** · Phase P+H · MUST
*User story:* As the owner, I want to delete my biometric data completely.
*Requirement:* The system SHALL provide a command to securely delete the template and any derived artefacts, after which face-unlock is disabled and only password unlock remains (SI).
*Acceptance (AC-F-04):* After delete, no template file remains, face-unlock is inert, and the session still locks/unlocks via password.

### 6.2 Perception pipeline

**REQ-F-05 — Camera acquisition** · Phase P+H · MUST
*User story:* As the system, I need camera frames to detect and verify faces.
*Requirement:* The system SHALL open the configured V4L2 device (default `/dev/video0`), capture at the configured resolution (ASM-08), and SHALL detect and handle a busy/unavailable device per FM-01.
*Acceptance (AC-F-05):* Given `/dev/video0` is free, when the service starts, then it captures frames at >= the configured FPS; given the device is busy, then it retries with backoff and never crashes (FM-01).

**REQ-F-06 — Face detection** · Phase P+H · MUST
*User story:* As the system, I need to locate faces in each analysed frame.
*Requirement:* The system SHALL detect zero, one, or many faces per frame with bounding boxes and a detection confidence, using a local model (no network).
*Acceptance (AC-F-06):* Given a frame containing one clear frontal face, when detection runs, then exactly one face is returned with confidence above the configured floor.

**REQ-F-07 — Face verification (1:1)** · Phase P+H · MUST
*User story:* As the owner, I want the system to recognize specifically me, not just "a face."
*Requirement:* The system SHALL compute a probe embedding for the dominant detected face and compare it to the owner template, producing a similarity score and an accept/reject decision at threshold τ (REQ-NF-10).
*Acceptance (AC-F-07):* Given the owner's live face, when verification runs under the accuracy target (ASM-05), then the decision is "owner" with score >= τ; given an impostor face, the decision is "not owner."

**REQ-F-08 — Multiple-face policy (fail closed)** · Phase P+H · MUST
*User story:* As the owner, I do not want the system to unlock when a stranger is also in view.
*Requirement:* When more than one face is detected above the detection floor during an unlock attempt, the system SHALL NOT unlock (fail closed) and SHALL treat the situation per the stranger-present policy (FM-06, ASM-03).
*Acceptance (AC-F-08):* Given owner + a second face in frame, when an unlock is attempted, then the system does not unlock and logs "multiple faces — denied."

### 6.3 Presence, absence, and stranger detection

**REQ-F-09 — Owner presence monitoring** · Phase P+H · MUST
*User story:* As the owner, I want the machine to stay unlocked while I am sitting there.
*Requirement:* While the session is unlocked, the system SHALL periodically confirm owner presence and SHALL NOT lock while the owner is continuously present.
*Acceptance (AC-F-09):* Given the owner remains in view, when presence monitoring runs, then no auto-lock is triggered.

**REQ-F-10 — Away / absence detection and auto-lock** · Phase P+H · MUST
*User story:* As the owner, when I walk away I want the screen to lock automatically.
*Requirement:* When no owner face is detected for the away dwell time (ASM-02, configurable), the system SHALL engage the screen lock (REQ-F-13).
*Acceptance (AC-F-10):* Given the owner leaves the frame, when the dwell time elapses, then the screen locks within dwell + 2 s (REQ-NF-03).

**REQ-F-11 — Stranger-present detection and lock** · Phase P+H · MUST
*User story:* As the owner, if a stranger appears at my unattended screen, I want it to lock.
*Requirement:* While unlocked, if a non-owner face is detected per the active policy (ASM-03: lenient default, strict optional), the system SHALL engage the screen lock and log the event.
*Acceptance (AC-F-11):* Given the active policy's trigger condition is met by a non-owner face, when it persists past the policy dwell, then the screen locks within 2 s (REQ-NF-04).

**REQ-F-12 — Owner-return re-acquire** · Phase P+H · MUST
*User story:* As the owner, when I come back I want the system to notice me quickly.
*Requirement:* While locked, the system SHALL monitor for a returning owner face and initiate verification (REQ-F-07) upon detecting any face, subject to camera lifecycle (FM-01) and privacy (REQ-NF-12).
*Acceptance (AC-F-12):* Given the owner returns, when a face is detected, then verification starts and (on match) unlock completes within the latency budget (REQ-NF-02).

### 6.4 Lock / unlock actuation and greeting

**REQ-F-13 — Engage screen lock** · Phase P+H · MUST
*User story:* As the system, I need to actually lock the screen.
*Requirement:* The system SHALL lock the session using an abstracted lock backend (default: GNOME/`loginctl lock-session` or the `org.gnome.ScreenSaver` D-Bus interface; fallbacks: `xdg-screensaver lock`), selected at runtime from the available backends (Section 1.4).
*Acceptance (AC-F-13):* When a lock trigger fires (REQ-F-10/11), then the active session shows the OS lock screen and requires authentication to proceed.

**REQ-F-14 — Face-verified unlock** · Phase P · MUST (P); superseded by REQ-F-24 in H
*User story:* As the owner, I want my face to unlock the screen without typing a password.
*Requirement:* In the Prototype, upon a positive owner verification (REQ-F-07) with any active liveness check passed (REQ-F-19), the system SHALL release the screensaver/lock for the current session **without** invoking the OS password. This uses the user-space unlock path only (CST-3) and MUST NOT expose or transmit the password.
*Acceptance (AC-F-14):* Given a locked session and a verified live owner, when unlock runs, then the desktop is presented within the latency budget (REQ-NF-02) and no password was entered.

**REQ-F-15 — Personalized welcome message** · Phase P+H · MUST
*User story:* As the owner, I want a friendly "Welcome back, Yash" when I return.
*Requirement:* On a successful face-unlock, the system SHALL display a transient greeting "Welcome back, <owner_name>" (default "Yash", ASM-01) for <= 3 s (ASM-09).
*Acceptance (AC-F-15):* When face-unlock succeeds, then the greeting containing the configured owner name is shown once and auto-dismisses within 3 s.

**REQ-F-16 — Non-owner rejection with password fallback** · Phase P+H · MUST
*User story:* As the owner, anyone who is not me must not get in via face, but should still be able to use a password (their own credentials at the OS layer).
*Requirement:* On a negative verification, low confidence, or liveness failure, the system SHALL NOT unlock and SHALL leave the standard OS password lock screen fully operational (SI).
*Acceptance (AC-F-16):* Given a non-owner (or a spoof) at the lock screen, when face verification fails, then face-unlock is denied and the OS password field remains available and functional.

### 6.5 Anti-spoofing / liveness

**REQ-F-17 — Prototype spoof-limitation disclosure** · Phase P · MUST
*User story:* As the owner, I want to be honestly told what the Prototype does and does not protect against.
*Requirement:* The Prototype SHALL document, in its user-facing README and on first run, that it provides convenience-level security only, is bypassable by a photo/video of the owner (as Howdy is [R2][R3]), controls only the screensaver (CST-3), and is not a password replacement (SI).
*Acceptance (AC-F-17):* The README and first-run notice contain the explicit limitation statement and reference the Hardening phase for stronger guarantees.

**REQ-F-18 — Bounded, non-escalating scope in Prototype** · Phase P · MUST
*User story:* As a security-conscious owner, I want a false accept to cost me only "an unlocked screen," never root.
*Requirement:* The Prototype SHALL NOT be wired into PAM, `sudo`, GDM login, or any privilege boundary; a false accept SHALL be limited to releasing the screensaver of an already-authenticated session (CST-3).
*Acceptance (AC-F-18):* Static inspection confirms no PAM/login/sudo hooks are installed by the Prototype.

**REQ-F-19 — Liveness / anti-spoofing capability** · Phase P (optional blink) / H (full) · MUST(H) / MAY(P)
*User story:* As the owner, I want the hardened system to reject a printed photo or a phone-screen video of me.
*Requirement:*
- **P (MAY):** Optionally require a randomized active blink/turn challenge before unlock as a weak liveness gate (ASM-04) [R5].
- **H (MUST):** Provide software PAD combining passive cues (texture/moire/reflectance, micro-motion) and active challenge-response (blink/turn), meeting the APCER/BPCER targets in REQ-NF-11 against the PAI species in FM-04 [R1][R5]. If IR/depth hardware is added later (ASM-06), hardware liveness MAY augment this.
*Acceptance (AC-F-19):* In Hardening, when the defined print, replay, and 2D-mask PAIs are presented, then unlock is denied at the APCER/BPCER target (REQ-NF-11); in Prototype with blink challenge enabled, a static photo without the challenge motion is denied.

### 6.6 OS-auth integration (hardening)

**REQ-F-20 — Optional PAM integration (hardening)** · Phase H · SHOULD
*User story:* As the owner, in the hardened build I may want face to also satisfy `sudo` / login, like Howdy.
*Requirement:* The Hardening build MAY provide an opt-in PAM module (Howdy-style) that offers face as an authentication factor, ONLY when liveness meets REQ-NF-11, and SHALL always leave the password factor available as fallback (SI, CST-6) [R2].
*Acceptance (AC-F-20):* When PAM integration is enabled and liveness passes, then face satisfies the configured PAM service; when liveness fails or the camera is unavailable, then PAM falls through to password with no lockout.

**REQ-F-21 — Secure privileged helper (hardening)** · Phase H · MUST(if REQ-F-20)
*User story:* As a security reviewer, I need any privileged component to be minimal and auditable.
*Requirement:* If PAM integration (REQ-F-20) is adopted, the privileged operations SHALL be isolated in a minimal, least-privilege helper with validated inputs; the camera/ML pipeline SHALL NOT run as root more than strictly necessary.
*Acceptance (AC-F-21):* Code review confirms the privileged surface is minimal, inputs are validated, and the ML pipeline does not require root for capture/inference.

### 6.7 Configuration, logging, control, lifecycle

**REQ-F-22 — Event logging** · Phase P+H · MUST
*User story:* As the owner/maintainer, I want a local record of locks, unlocks, denials, and spoof detections.
*Requirement:* The system SHALL write structured local logs for lifecycle and decision events (lock, unlock, deny, multiple-faces, camera-error, liveness-fail) with timestamps, WITHOUT storing raw face images (REQ-NF-13).
*Acceptance (AC-F-22):* After a lock/unlock/deny sequence, the log contains one structured entry per event and contains no image data.

**REQ-F-23 — Configuration** · Phase P+H · MUST
*User story:* As the maintainer, I want to tune thresholds and timers without editing code.
*Requirement:* The system SHALL read a config file (camera device, resolution, FPS, match threshold τ, away dwell, stranger policy mode, liveness mode, owner name, log level) with sane defaults (Section 4.2) and SHALL validate values on load.
*Acceptance (AC-F-23):* Given a config with an out-of-range value, when the service loads it, then it rejects the value with a clear error and uses the documented default or refuses to start (fail closed).

**REQ-F-24 — OS-auth unlock path (hardening supersede)** · Phase H · MUST(H)
*User story:* As a security-conscious owner, in the hardened build I want unlock to go through the OS auth stack rather than a user-space screensaver release.
*Requirement:* In Hardening, face-unlock SHALL be performed via the OS authentication mechanism (REQ-F-20) gated on liveness (REQ-F-19), superseding the user-space release of REQ-F-14; the password fallback remains (SI).
*Acceptance (AC-F-24):* In the hardened build, a successful live owner verification results in an OS-authenticated unlock event, and the password path still works independently.

**REQ-F-25 — Manual override, panic-lock, and disable** · Phase P+H · MUST
*User story:* As the owner, I want to force-lock instantly, and to disable face-unlock on demand (e.g., before handing my laptop to someone).
*Requirement:* The system SHALL provide (a) an immediate manual lock, (b) a "disable face-unlock" toggle that leaves password unlock working, and (c) after N consecutive failed face attempts (configurable, default 5) a cool-down that falls through to password only (ASM-11) to prevent self-inflicted denial of service.
*Acceptance (AC-F-25):* Given the disable toggle is on, when the owner appears, then no face-unlock occurs and password is required; given N failed attempts, then face-unlock cools down and password remains available.

**REQ-F-26 — Service lifecycle and crash recovery** · Phase P+H · MUST
*User story:* As the owner, I want the tool to start with my session and to fail safe if it dies.
*Requirement:* The system SHALL run as a user-session service that auto-starts (ASM-10), exposes a health/watchdog signal, and on crash SHALL be restarted; a crashed or absent monitor SHALL NEVER result in an auto-unlock (fail closed, FM-08).
*Acceptance (AC-F-26):* Given the process is killed while the session is locked, when the watchdog restarts it, then the session remains locked until a live owner verification (or password) succeeds.

**REQ-F-27 — Camera in-use indicator** · Phase P+H · SHOULD
*User story:* As the owner, I want to see when the camera is being used by the unlock tool.
*Requirement:* The system SHALL surface a visible indicator (tray icon or the hardware LED via camera activation) whenever it holds the camera open (privacy, REQ-NF-16).
*Acceptance (AC-F-27):* When the tool is capturing, then the indicator/LED reflects camera activity; when idle, the camera is released and the indicator clears.

---

## 7. Non-functional requirements (REQ-NF)

All performance targets assume **CPU-only** execution on the measured Ryzen 9 5900X (ASM-07); GPU is an optional accelerator (REQ-NF-05).

### 7.1 Performance and latency

**REQ-NF-01 — Detection loop throughput** · P+H · MUST
The active perception loop SHALL sustain >= 5 analysed frames/second during unlock attempts and MAY throttle to <= 1 fps during long owner-absence to save power (FM-07).
*AC-NF-01:* Measured active-loop rate >= 5 fps on the target CPU.

**REQ-NF-02 — Unlock decision latency** · P+H · MUST
From the moment a returning owner's face enters the frame to the unlock decision, P95 latency SHALL be <= 2.0 s; per-frame end-to-end inference (detect + embed + compare) SHALL be <= 200 ms on CPU.
*AC-NF-02:* Bench harness reports P95 face-to-decision <= 2.0 s and per-frame <= 200 ms.

**REQ-NF-03 — Away-lock latency** · P+H · MUST
The screen SHALL lock within (away dwell + 2 s) of the owner leaving the frame.
*AC-NF-03:* With dwell=30 s, measured lock occurs within 32 s of departure.

**REQ-NF-04 — Stranger-lock latency** · P+H · MUST
When the active stranger policy trigger is met, the screen SHALL lock within 2 s.
*AC-NF-04:* Measured stranger-trigger-to-lock <= 2 s.

**REQ-NF-05 — Optional GPU acceleration** · P+H · MAY
If a working CUDA/onnxruntime-GPU stack becomes available (currently blocked by driver mismatch, ASM-07), the system MAY use it, but MUST remain fully functional on CPU alone.
*AC-NF-05:* Disabling GPU still meets REQ-NF-01/02.

### 7.2 Resource use

**REQ-NF-06 — Idle CPU** · P+H · MUST
While monitoring an idle/absent scene at reduced fps, average CPU SHALL be <= 15% of one core.
*AC-NF-06:* `top`/`pidstat` mean over 5 min idle <= 15% single-core.

**REQ-NF-07 — Active CPU** · P+H · SHOULD
During an unlock burst, CPU SHALL NOT exceed ~1 full core sustained (bounded thread count) to keep the desktop responsive.
*AC-NF-07:* Peak sustained CPU <= 100% of one core equivalent during a 10 s unlock burst.

**REQ-NF-08 — Memory footprint** · P+H · MUST
Resident memory SHALL be <= 500 MB including loaded models.
*AC-NF-08:* RSS peak <= 500 MB.

**REQ-NF-09 — Storage footprint** · P+H · SHOULD
Installed footprint (models + code) SHALL be <= 300 MB; logs SHALL be size-capped/rotated (FM disk-full).
*AC-NF-09:* Install <= 300 MB; log rotation configured with a hard cap.

### 7.3 Recognition accuracy (security-relevant)

**REQ-NF-10 — Verification accuracy operating point** · P+H · MUST
- **P:** τ tuned for **FMR <= 1e-2** with **FNMR <= 5%** on a local eval set (ASM-05).
- **H:** τ (with liveness) tuned for **FMR <= 1e-3** with **FNMR <= 3%**.
Targets are documented relative to NIST FRTE/FRVT server-grade references (FNMR ~0.1-1% at FMR 1e-5 [R4]); the local lightweight model is expected to be weaker, and this gap is an accepted, documented trade-off.
*AC-NF-10:* On the eval protocol, measured FMR/FNMR meet the phase target; results reported with confidence intervals.

**REQ-NF-11 — Presentation-attack detection targets (hardening)** · H · MUST
Against the PAI species in FM-04 (print, screen/video replay, 2D mask), the Hardening PAD SHALL achieve **APCER <= 5% at BPCER <= 5%** per ISO/IEC 30107-3 evaluation methodology [R1]. The Prototype makes NO PAD guarantee (ASM-04) and MUST disclose this (REQ-F-17).
*AC-NF-11:* An ISO/IEC 30107-3-style PAD test reports APCER <= 5% @ BPCER <= 5% for each listed PAI species.

### 7.4 Privacy of camera data

**REQ-NF-12 — Local-only processing** · P+H · MUST
No face image, embedding, or event SHALL leave the machine; the tool SHALL make no network connections (CST-2).
*AC-NF-12:* Network monitoring during operation shows zero outbound connections from the tool.

**REQ-NF-13 — No raw-frame persistence** · P+H · MUST
Raw camera frames SHALL be held only in volatile memory for the minimum time needed and SHALL NEVER be written to disk (contrast with Howdy's snapshot storage weakness [R3]). Only non-image templates/embeddings are persisted.
*AC-NF-13:* Filesystem inspection after operation finds no stored frames; only the template file exists.

**REQ-NF-14 — Template protection at rest** · P (0600) / H (encrypted) · MUST
Templates SHALL be stored with owner-only permissions (0600) in the Prototype and encrypted-at-rest with an integrity check in Hardening.
*AC-NF-14:* Prototype template is mode 0600; Hardening template is encrypted and integrity-verified on load (tamper -> fail closed).

**REQ-NF-15 — Data minimization and erasure** · P+H · MUST
The system SHALL store only what is required (template, config, capped logs) and SHALL support complete erasure (REQ-F-04).
*AC-NF-15:* After erasure, no biometric-derived artefact remains.

**REQ-NF-16 — Camera-use transparency** · P+H · SHOULD
Whenever the camera is active, the owner SHALL be able to tell (LED / indicator, REQ-F-27); the greeting SHALL NOT reveal sensitive info beyond the owner name (ASM-09).
*AC-NF-16:* Indicator reflects camera state; greeting content limited to the configured name.

### 7.5 Usability

**REQ-NF-17 — Usability targets** · P+H · SHOULD
Enrollment SHALL complete in <= 2 min (REQ-F-01); the greeting SHALL appear within <= 1 s of unlock; a false reject SHALL be recoverable immediately via the always-present password (SI); default behaviour SHALL be safe without any configuration.
*AC-NF-17:* Time-to-enroll <= 2 min; greeting latency <= 1 s; password recovery works on first false reject.

**REQ-NF-18 — Nuisance-lock bound** · P+H · SHOULD
Under normal single-owner use, spurious auto-locks (owner present but locked) SHALL be rare; the stranger policy default (ASM-03) SHALL avoid locking merely because a colleague is co-present with the owner.
*AC-NF-18:* In a scripted co-presence scenario under lenient mode, no lock occurs while the owner remains present.

### 7.6 Portability

**REQ-NF-19 — Abstracted lock backend** · P+H · MUST
The lock/unlock actuator SHALL be behind an interface with multiple backends (GNOME D-Bus / `loginctl` / `xdg-screensaver`), selectable at runtime, so new desktops can be added without touching the perception pipeline.
*AC-NF-19:* Swapping the backend via config changes the lock mechanism with no code change to the pipeline.

**REQ-NF-20 — Wayland-ready design (deferred implementation)** · P (design) / H (support) · SHOULD
The Prototype targets X11 (CST-5) but SHALL NOT hard-code X11-only assumptions in the core; a Wayland lock backend SHALL be addable in Hardening.
*AC-NF-20:* Architecture review confirms no X11-only coupling in the core; a Wayland backend is a pluggable addition.

**REQ-NF-21 — Standard dependency stack** · P+H · MUST
The system SHALL rely on the confirmed available stack (Python 3.11, OpenCV, ONNX Runtime, V4L2) and SHALL NOT require the absent libraries (dlib/face_recognition/mediapipe) unless explicitly bundled and pinned (CST-8, R5).
*AC-NF-21:* Clean-environment `requirements.txt` install reproduces a working system with pinned versions.

### 7.7 Reliability, security posture, maintainability

**REQ-NF-22 — Fail-closed guarantee** · P+H · MUST
On any error (camera loss, model error, config error, crash, timeout, low confidence, ambiguous scene), the system SHALL remain locked / deny access (SI). There SHALL be no code path where a failure yields an unlock.
*AC-NF-22:* Fault-injection tests (kill camera, corrupt model, corrupt template, kill process) each result in "stays locked / password required," never auto-unlock.

**REQ-NF-23 — Availability and watchdog** · P+H · MUST
The monitor SHALL auto-restart on failure (REQ-F-26) with a target of unattended operation across a normal work session without manual intervention.
*AC-NF-23:* Injected crashes are auto-recovered; the lock state is preserved across restarts.

**REQ-NF-24 — Observability** · P+H · SHOULD
The system SHALL expose health status and structured decision logs (REQ-F-22) sufficient to diagnose false accept/reject and camera issues, without leaking images.
*AC-NF-24:* Logs allow reconstruction of any accept/deny decision (score, threshold, liveness result, face count) without raw images.

**REQ-NF-25 — Auditability (hardening)** · H · MUST
The Hardening build SHALL keep a tamper-evident audit trail of all accept/deny/spoof/PAM events for security review (ST-5).
*AC-NF-25:* Audit entries are append-only / integrity-protected and cover every security-relevant decision.

**REQ-NF-26 — Reproducible build** · P+H · MUST
A one-command build SHALL produce a working install from a pinned environment (R5, CST-8).
*AC-NF-26:* A single documented command builds and runs the tool from pinned dependencies on a clean machine.

**REQ-NF-27 — Least privilege** · P+H · MUST
The Prototype SHALL run without root (CST-3); any Hardening privileged component SHALL be minimal and isolated (REQ-F-21).
*AC-NF-27:* Prototype runs as the unprivileged user; Hardening privileged surface is minimized and reviewed.

---

## 8. User stories summary (traceable to REQ-IDs)

| US | As a... | I want... | so that... | Realized by |
|---|---|---|---|---|
| US-1 | Owner | to enroll my face once | the system knows me | REQ-F-01/02/03 |
| US-2 | Owner | the screen to lock when I walk away | my session is protected | REQ-F-09/10, REQ-NF-03 |
| US-3 | Owner | the screen to lock if a stranger appears | no one browses my open screen | REQ-F-11, REQ-NF-04/18 |
| US-4 | Owner | my face to unlock with "Welcome back, Yash" | I return hands-free | REQ-F-12/14/15, REQ-NF-02 |
| US-5 | Owner | non-owners blocked without a password | only I get in by face | REQ-F-08/16, SI |
| US-6 | Owner | the password to always work | I am never locked out | REQ-F-16/25, REQ-NF-22, SI |
| US-7 | Security reviewer | resistance to photo/video/mask | spoofing is hard | REQ-F-17/19, REQ-NF-11 |
| US-8 | Owner | my camera data to stay private and local | no leaks | REQ-NF-12/13/14/15/16 |
| US-9 | Owner | a fast, low-resource, non-nuisance tool | it does not slow me down | REQ-NF-01/02/06/07/08/18 |
| US-10 | Maintainer | reproducible build, config, logs | I can operate it | REQ-F-22/23/26, REQ-NF-21/24/26 |
| US-11 | Owner | to force-lock / disable / recover from failure | I stay in control and safe | REQ-F-04/25/26, REQ-NF-22/23 |
| US-12 | Owner (hardened) | optional OS-auth face factor with fallback | convenience without lockout | REQ-F-20/21/24, REQ-NF-25/27 |

---

## 9. Gap analysis

Rows marked Partial/No are resolved by an assumption (Section 4.2) and flagged as Open Questions (Section 15). No row is left unresolved.

| REQ area | Specified? | Failure mode ref | Edge cases | Dependencies | Risk | Resolution |
|---|---|---|---|---|---|---|
| Owner-away dwell time | Partial | FM-07 | very short trips vs security | ASM-02 | nuisance vs exposure | ASM-02 default 30 s, configurable; OQ-1 |
| Stranger-present policy | Partial | FM-06 | colleague co-present | ASM-03 | nuisance locks | ASM-03 lenient default; OQ-2 |
| Prototype spoof stance | Yes | FM-04 | photo/replay | ASM-04, REQ-F-17 | breach if misused as password | SI + disclosure; H targets |
| Accuracy operating point | Partial | FM-02, FM-03 | lighting, appearance drift | ASM-05, REQ-NF-10 | FAR/FRR trade-off | ASM-05 targets; OQ-3 |
| GPU availability | Yes | n/a | driver mismatch | ASM-07 | perf on CPU | CPU-only targets; GPU optional |
| IR/depth liveness | Yes | FM-04 | no IR hardware | ASM-06 | weaker liveness | software PAD; hardware later |
| Greeting UX / content | Partial | n/a | onlooker sees name | ASM-09, REQ-NF-16 | minor privacy | transient, name-only; OQ-4 |
| PAM integration decision | Yes (deferred to H) | FM-04, FM-08 | lockout risk | REQ-F-20/24, CST-6 | escalation risk | opt-in, liveness-gated, fallback; OQ-5 |
| Autostart mechanism | Partial | FM-08 | systemd vs XDG | ASM-10 | reliability | systemd user unit default; OQ-6 |
| Failed-attempt lockout | Yes | FM-03 | self-DoS | ASM-11, REQ-F-25 | denial of service | cool-down + password; no lockout |
| Capture resolution | Partial | FM-05 | low light needs | ASM-08 | accuracy | 640x480 default; 720p optional; OQ-7 |

---

## 10. Acceptance-criteria table (consolidated, verifiable)

| REQ-ID | Phase | Priority | Acceptance criterion (summary) | Verification method |
|---|---|---|---|---|
| REQ-F-01 | P+H | MUST | Enroll builds 0600 template, >=5 samples, <=2 min | Functional test + file check |
| REQ-F-02 | P+H | MUST | Bad frames (0 / >1 face, too small/dark) rejected | Unit + fixture frames |
| REQ-F-03 | P+H | SHOULD | Re-enroll supersedes/augments, logged | Functional test |
| REQ-F-04 | P+H | MUST | Delete removes all biometric artefacts; password still works | Functional + FS scan |
| REQ-F-05 | P+H | MUST | Captures at target fps; busy device retried, no crash | Integration + fault inject |
| REQ-F-06 | P+H | MUST | One frontal face detected above floor | Unit test |
| REQ-F-07 | P+H | MUST | Owner accepted >=τ, impostor rejected | Eval protocol |
| REQ-F-08 | P+H | MUST | 2 faces -> no unlock, logged | Integration test |
| REQ-F-09 | P+H | MUST | No lock while owner continuously present | Scenario test |
| REQ-F-10 | P+H | MUST | Lock within dwell+2 s after departure | Timed test |
| REQ-F-11 | P+H | MUST | Stranger trigger -> lock within 2 s | Timed test |
| REQ-F-12 | P+H | MUST | Owner return triggers verification, unlock in budget | Timed test |
| REQ-F-13 | P+H | MUST | Lock trigger shows OS lock screen | Integration test |
| REQ-F-14 | P | MUST | Verified live owner unlock without password | Functional test |
| REQ-F-15 | P+H | MUST | "Welcome back, <name>" shown once, <=3 s | UI test |
| REQ-F-16 | P+H | MUST | Non-owner/spoof denied; password field works | Security test |
| REQ-F-17 | P | MUST | README + first-run limitation disclosure present | Doc/inspection |
| REQ-F-18 | P | MUST | No PAM/login/sudo hooks in Prototype | Static inspection |
| REQ-F-19 | P(MAY)/H(MUST) | MUST(H) | PAIs denied at APCER/BPCER target; blink defeats static photo | PAD test |
| REQ-F-20 | H | SHOULD | Face satisfies PAM when live; falls through to password | PAM test |
| REQ-F-21 | H | MUST(if F20) | Privileged surface minimal, inputs validated | Code review |
| REQ-F-22 | P+H | MUST | One structured, image-free log entry per event | Log inspection |
| REQ-F-23 | P+H | MUST | Out-of-range config rejected; safe default or refuse-start | Unit test |
| REQ-F-24 | H | MUST | Unlock via OS auth, gated on liveness; password independent | Integration test |
| REQ-F-25 | P+H | MUST | Disable/panic works; N fails -> cool-down + password | Functional test |
| REQ-F-26 | P+H | MUST | Killed process restarts; stays locked meanwhile | Fault-injection |
| REQ-F-27 | P+H | SHOULD | Indicator/LED reflects camera use | Manual/observability |
| REQ-NF-01 | P+H | MUST | Active loop >=5 fps on CPU | Benchmark |
| REQ-NF-02 | P+H | MUST | P95 face-to-decision <=2 s; per-frame <=200 ms | Benchmark |
| REQ-NF-03 | P+H | MUST | Away-lock within dwell+2 s | Timed test |
| REQ-NF-04 | P+H | MUST | Stranger-lock within 2 s | Timed test |
| REQ-NF-05 | P+H | MAY | Works fully on CPU with GPU disabled | Benchmark |
| REQ-NF-06 | P+H | MUST | Idle CPU <=15% one core | Profiling |
| REQ-NF-07 | P+H | SHOULD | Active CPU <=1 core sustained | Profiling |
| REQ-NF-08 | P+H | MUST | RSS <=500 MB | Profiling |
| REQ-NF-09 | P+H | SHOULD | Install <=300 MB; logs rotated/capped | Inspection |
| REQ-NF-10 | P+H | MUST | Meets phase FMR/FNMR target with CIs | Eval protocol |
| REQ-NF-11 | H | MUST | APCER<=5% @ BPCER<=5% per PAI | ISO 30107-3-style test |
| REQ-NF-12 | P+H | MUST | Zero outbound network connections | Network monitor |
| REQ-NF-13 | P+H | MUST | No raw frames on disk; only template | FS scan |
| REQ-NF-14 | P/H | MUST | 0600 (P) / encrypted+integrity (H); tamper->fail closed | Inspection + fault inject |
| REQ-NF-15 | P+H | MUST | Erasure leaves no biometric artefact | FS scan |
| REQ-NF-16 | P+H | SHOULD | Camera indicator accurate; greeting name-only | Observation |
| REQ-NF-17 | P+H | SHOULD | Enroll <=2 min; greeting <=1 s; password recovery works | Timed/UX test |
| REQ-NF-18 | P+H | SHOULD | No lock while owner present under co-presence | Scenario test |
| REQ-NF-19 | P+H | MUST | Lock backend swappable via config | Config test |
| REQ-NF-20 | P/H | SHOULD | No X11-only coupling; Wayland backend addable | Arch review |
| REQ-NF-21 | P+H | MUST | Pinned deps reproduce working system | Clean-env install |
| REQ-NF-22 | P+H | MUST | All faults -> stays locked, never auto-unlock | Fault-injection matrix |
| REQ-NF-23 | P+H | MUST | Crashes auto-recover; lock state preserved | Fault-injection |
| REQ-NF-24 | P+H | SHOULD | Logs reconstruct any decision, no images | Log inspection |
| REQ-NF-25 | H | MUST | Tamper-evident audit of security events | Inspection |
| REQ-NF-26 | P+H | MUST | One-command build from pinned env | Build test |
| REQ-NF-27 | P+H | MUST | Prototype non-root; H privileged surface minimal | Inspection/review |

---

## 11. Failure-mode inventory (per feature)

Each failure mode gives: trigger, detection, impact, handling, and the phase in which the handling applies. The task-mandated failure modes are covered by FM-01, FM-02, FM-03, FM-04, FM-05, FM-06, FM-07, FM-08; additional ones (FM-09..FM-16) harden the design. Every handling obeys the Safety Invariant: uncertainty -> stay locked.

| FM-ID | Feature / trigger | How detected | Impact | Handling (and phase) | REQ links |
|---|---|---|---|---|---|
| **FM-01** | **Camera unavailable / busy** (missing device, permission denied, in use by a video call) | V4L2 open error / EBUSY; frame timeout | Cannot detect/verify | Retry with exponential backoff; notify owner; if locked, face-unlock unavailable and **password fallback** is used (SI); if unlocked and camera lost, log and (config) lock after a grace period rather than run blind. Never crash. (P+H) | REQ-F-05, REQ-NF-22, REQ-F-16 |
| **FM-02** | **False accept** (impostor scored >=τ) | Post-hoc via logs/audit; bounded by threshold | Unauthorized screen unlock | Prototype: bounded to screensaver only (CST-3, REQ-F-18); single-owner template; τ set for FMR<=1e-2 (ASM-05). Hardening: liveness gate (REQ-F-19) + tighter τ (FMR<=1e-3) + audit alert (REQ-NF-25). (P+H) | REQ-NF-10/11, REQ-F-18/19 |
| **FM-03** | **False reject** (owner scored <τ) | Verification returns reject | Owner annoyed | Retry across N frames / a short time window before giving up; **password always available** (SI); optional re-enroll prompt if chronic (REQ-F-03); never lower τ to compensate. Cool-down after N fails avoids self-DoS (REQ-F-25, ASM-11). (P+H) | REQ-F-03/16/25, REQ-NF-10/17 |
| **FM-04** | **Presentation / spoofing attack** (printed photo, phone/monitor video replay, 2D/3D mask) | Prototype: minimal (optional blink only). Hardening: passive texture/moire/reflectance + active challenge-response [R1][R5] | Impostor unlocks by spoof | **Prototype: KNOWN LIMITATION, explicitly disclosed (REQ-F-17)** and mitigated only by SI (screensaver-only, not a password) + optional blink challenge (ASM-04); this matches Howdy's documented photo-spoof exposure [R2][R3]. **Hardening: PAD to APCER<=5%@BPCER<=5% (REQ-NF-11) [R1]**, challenge-response, IR/depth if hardware added (ASM-06). (P disclose / H mitigate) | REQ-F-17/19, REQ-NF-11 |
| **FM-05** | **Poor lighting** (dark, backlit, glare, IR-free low light) | Low detection confidence / low frame brightness / low match score | False reject or failed detection | Auto exposure/gain; histogram normalization; if confidence below floor, **do not accept (fail closed)** and use password; NEVER lower τ in bad light; optionally raise capture to 720p (ASM-08). (P+H) | REQ-NF-22, REQ-F-02/07, ASM-08 |
| **FM-06** | **Multiple faces** (owner + others, or several strangers) | Detector returns >1 box above floor | Ambiguity / covert unlock with stranger present | **Fail closed:** do not unlock while >1 face present (REQ-F-08); while unlocked, apply stranger policy (ASM-03) and lock. Choose owner match only when it is the sole detected face. (P+H) | REQ-F-08/11, ASM-03 |
| **FM-07** | **Owner absent for long periods** | Continuous absence beyond dwell; long-idle timer | Wasted CPU/power; camera held unnecessarily | Auto-lock (REQ-F-10); throttle loop to <=1 fps and, after extended absence, **release the camera** (privacy + power) and re-acquire on activity/wake; bound resources (REQ-NF-06). (P+H) | REQ-F-10, REQ-NF-01/06, REQ-F-27 |
| **FM-08** | **Process crash / monitor absent** | Watchdog/health miss; systemd notices exit | If mishandled, screen could stay unlocked with no monitor | **Fail closed:** an absent/crashed monitor NEVER auto-unlocks; on crash while locked, session stays locked; watchdog restarts the service; on restart require live owner or password (REQ-F-26). (P+H) | REQ-F-26, REQ-NF-22/23 |
| FM-09 | **Camera privacy shutter closed / covered** | Uniformly dark/blank frames, no face for long | Cannot verify | Treat as owner-absent/no-face; stay locked; notify "camera blocked"; password available. (P+H) | REQ-NF-22, REQ-F-16 |
| FM-10 | **Template file corrupt / tampered** | Load/format/integrity check fails | Cannot verify or (worse) accept wrong data | Fail closed: disable face-unlock, require password, alert; Hardening integrity check makes tamper evident (REQ-NF-14). (P+H) | REQ-NF-14/22, REQ-F-04 |
| FM-11 | **Model file missing / corrupt** | ONNX load error | Pipeline dead | Fail closed; service reports unhealthy; password unlock unaffected; reproducible build re-provisions models (REQ-NF-26). (P+H) | REQ-NF-22/26 |
| FM-12 | **Disk full (logs)** | Write error / capacity check | Logging fails, possible instability | Log rotation with hard cap (REQ-NF-09); on write failure, degrade logging but keep locking correct (SI). (P+H) | REQ-NF-09/22 |
| FM-13 | **Suspend / resume, lid, DPMS off** | Session/power events | Camera state lost; stale decisions | On resume, re-initialize camera, invalidate stale frames, require fresh verification; default to locked on resume. (P+H) | REQ-F-26, REQ-NF-22 |
| FM-14 | **Appearance drift** (glasses on/off, beard, haircut) | Rising false-reject rate | Owner friction | Multi-sample template (REQ-F-01), re-enroll (REQ-F-03), never auto-relax τ; password fallback. (P+H) | REQ-F-01/03, REQ-NF-10 |
| FM-15 | **Repeated failed attempts / brute force at the camera** | Counter of consecutive fails | Nuisance / spoof probing | Cool-down + fall-through to password (REQ-F-25, ASM-11); Hardening logs to audit (REQ-NF-25). Never lock out the password (SI). (P+H) | REQ-F-25, REQ-NF-25 |
| FM-16 | **Lock-backend API change / conflicting screensaver** | Backend call error / no lock observed | Lock may not engage | Abstracted backend with fallbacks (REQ-NF-19); verify lock actually engaged and retry alternate backend; alert if none succeed and default to a safe state. (P+H) | REQ-NF-19/22 |

---

## 12. Risk assessment (severity x likelihood)

Severity: Low / Med / High / Critical. Likelihood: Low / Med / High.

| Risk | Sev | Like | Exposure | Mitigation |
|---|---|---|---|---|
| Photo/video spoof unlocks Prototype | High | Med | FM-04 | SI bounds to screensaver; disclosure (REQ-F-17); Hardening PAD (REQ-NF-11) |
| False accept in Hardening PAM path | Critical | Low | FM-02 | Liveness gate + tighter τ + audit + password fallback (REQ-F-19/24, REQ-NF-11/25) |
| Owner locked out by chronic false reject | Med | Med | FM-03 | Password always works (SI); cool-down; re-enroll |
| Camera busy during video calls | Med | High | FM-01 | Backoff; password fallback; do not fight for device |
| Nuisance locks annoy owner | Low | Med | FM-06/07 | Lenient stranger policy default (ASM-03); presence hysteresis |
| Privacy leak (frames/network) | High | Low | REQ-NF-12/13 | No disk frames, no network, local-only (design) |
| Crash leaves screen unlocked | High | Low | FM-08 | Fail closed + watchdog (REQ-NF-22/23) |
| GPU driver dependence breaks perf | Med | Med | ASM-07 | CPU-only targets; GPU optional (REQ-NF-05) |
| Privileged PAM helper vulnerability | High | Low | REQ-F-21 | Minimal least-privilege helper, input validation, review |
| Template theft / tamper | High | Low | FM-10 | 0600 (P), encrypted+integrity (H) (REQ-NF-14) |

---

## 13. Prior art / literature review (cited)

- **Presentation Attack Detection standard — ISO/IEC 30107-3.** Defines the PAD evaluation methodology and the APCER / BPCER / ACER metrics used here for the Hardening targets (REQ-NF-11). BPCER is reported at fixed APCER thresholds (e.g., 1/5/10%). [R1]
- **Howdy (Windows-Hello-style face auth for Linux).** The most relevant existing tool. Documented as convenience, not security: a well-printed photo or a look-alike can fool it, and some versions stored webcam snapshots on disk (an exploitable weakness). This directly motivates the Safety Invariant (SI), the no-raw-frame rule (REQ-NF-13), and the Prototype disclosure (REQ-F-17). Howdy also demonstrates the feasible PAM-integration path for Hardening (REQ-F-20). [R2][R3]
- **NIST FRVT / FRTE-FATE (face recognition benchmarking).** Establishes FMR/FNMR as the standard 1:1 verification metrics and shows server-grade algorithms reaching FNMR ~0.1-1% at FMR 1e-5. Our lightweight local targets (ASM-05, REQ-NF-10) are deliberately looser and framed against this reference. [R4]
- **Face liveness / anti-spoofing surveys.** Enumerate passive (texture, moire, reflectance, micro-motion) and active (blink, head-turn, challenge-response) methods, plus hardware IR/depth. Since this machine has no IR sensor (ASM-06), the Hardening PAD is software/behavioural-first, optionally augmented by IR hardware if added. [R5]

### References
- [R1] ISO/IEC 30107-3 PAD testing and APCER/BPCER metrics — Busch, "Presentation Attack Detection - ISO/IEC 30107": https://www.christoph-busch.de/files/Busch-PAD-240701.pdf ; NIST overview: https://www.nist.gov/system/files/documents/2020/09/15/12_buschthieme-ibpc-pad-160504.pdf
- [R2] Howdy project (facial authentication for Linux) and ArchWiki security notes: https://github.com/boltgolt/howdy ; https://wiki.archlinux.org/title/Howdy
- [R3] Howdy security discussion (photo spoof, snapshot storage; "not more secure than a password") — Linux Magazine "Howdy, Friend": https://www.linux-magazine.com/Issues/2022/256/Howdy
- [R4] NIST Face Recognition Vendor Test / FRTE-FATE, FMR/FNMR definitions and benchmark accuracy: https://pages.nist.gov/frvt/reports/demographics/nistir_8429.pdf ; https://nvlpubs.nist.gov/nistpubs/ir/2022/NIST.IR.8429.ipd.pdf
- [R5] Face anti-spoofing / liveness detection survey (passive + active + IR/depth methods): https://www.mdpi.com/2076-3417/15/12/6891

Note (per user rule R6): these are standards and technical reports cited by URL for a requirements document; no BibTeX entries are hand-authored here. When any of these enter the eventual research paper, the authoritative BibTeX must be downloaded and verified per R6 before citation.

---

## 14. Traceability matrix (requirement -> acceptance -> failure mode -> risk)

| REQ-ID | Acceptance | Failure modes | Risk row |
|---|---|---|---|
| REQ-F-01/02/03 | AC-F-01/02/03 | FM-14 | Locked out (drift) |
| REQ-F-04 | AC-F-04 | FM-10 | Template theft |
| REQ-F-05 | AC-F-05 | FM-01, FM-09, FM-13 | Camera busy |
| REQ-F-06/07 | AC-F-06/07 | FM-02, FM-03, FM-05 | False accept/reject |
| REQ-F-08 | AC-F-08 | FM-06 | Nuisance / covert unlock |
| REQ-F-09/10/11/12 | AC-F-09..12 | FM-06, FM-07 | Nuisance locks |
| REQ-F-13 | AC-F-13 | FM-16 | Lock not engaged |
| REQ-F-14/24 | AC-F-14/24 | FM-02, FM-08 | False accept |
| REQ-F-15 | AC-F-15 | n/a | (UX) |
| REQ-F-16 | AC-F-16 | FM-02, FM-04 | Spoof / false accept |
| REQ-F-17/18 | AC-F-17/18 | FM-04 | Spoof (Prototype) |
| REQ-F-19 | AC-F-19 | FM-04 | Spoof |
| REQ-F-20/21 | AC-F-20/21 | FM-02, FM-08 | PAM false accept / privileged bug |
| REQ-F-22 | AC-F-22 | FM-12 | Privacy / disk |
| REQ-F-23 | AC-F-23 | FM-11 | Config error |
| REQ-F-25 | AC-F-25 | FM-03, FM-15 | Self-DoS |
| REQ-F-26 | AC-F-26 | FM-08, FM-13 | Crash unlocked |
| REQ-F-27 | AC-F-27 | FM-07 | Privacy transparency |
| REQ-NF-01..09 | AC-NF-01..09 | FM-07 | Performance/resource |
| REQ-NF-10 | AC-NF-10 | FM-02, FM-03 | Accuracy trade-off |
| REQ-NF-11 | AC-NF-11 | FM-04 | Spoof (Hardening) |
| REQ-NF-12..16 | AC-NF-12..16 | FM-09, FM-12 | Privacy leak |
| REQ-NF-17/18 | AC-NF-17/18 | FM-03, FM-06 | Nuisance / usability |
| REQ-NF-19/20/21 | AC-NF-19..21 | FM-16 | Portability |
| REQ-NF-22/23/24/25 | AC-NF-22..25 | FM-01/08/10/11/13/16 | Fail-closed / audit |
| REQ-NF-26/27 | AC-NF-26/27 | FM-11 | Reproducibility / least privilege |

Coverage check: every FM-01..FM-16 maps to at least one requirement; every REQ maps to at least one acceptance criterion; the task-mandated failure list (camera unavailable/busy, false accept, false reject, presentation/spoofing attack, poor lighting, multiple faces, owner absent long, process crash) is covered by FM-01, FM-02, FM-03, FM-04, FM-05, FM-06, FM-07, FM-08 respectively, each with an explicit handling and phase.

---

## 15. Open Questions / Assumptions (for human review to resolve)

Because this is an autonomous dispatch, the persona's "ask the user" step is recorded here. Each item has a working assumption already applied so no downstream work is blocked; the human review gate should confirm or override.

- **OQ-1 (ASM-02) Away-lock dwell default.** Assumed 30 s. Confirm: is a shorter (10-15 s) or longer value preferred? Trade-off: security vs nuisance.
- **OQ-2 (ASM-03) Stranger-present policy default.** Assumed "lenient" (lock only when a stranger persists and the owner is absent). Confirm whether "strict" (lock on ANY non-owner face, even with the owner present) should be the default for a higher-security posture.
- **OQ-3 (ASM-05, REQ-NF-10) Accuracy operating point.** Assumed Prototype FMR<=1e-2 / FNMR<=5%. Confirm acceptable false-accept tolerance for a screensaver-only scope; Hardening tightens to FMR<=1e-3.
- **OQ-4 (ASM-09, REQ-NF-16) Greeting form.** Assumed a transient notification/overlay with the owner name for <=3 s. Confirm the exact UX (toast vs full-screen) and whether the name should be shown at all on a shared/observed screen.
- **OQ-5 (REQ-F-20/24, CST-6) Hardening OS-auth scope.** Assumed PAM integration is OPT-IN and liveness-gated, never removing the password. Confirm whether Hardening should integrate with `sudo`/login at all, or remain screensaver-only for the strongest blast-radius bound.
- **OQ-6 (ASM-10) Autostart mechanism.** Assumed a systemd user service. Confirm systemd-user vs XDG autostart preference.
- **OQ-7 (ASM-08) Capture resolution.** Assumed 640x480 default (confirmed available). Confirm whether to prefer 1280x720 (MJPG) for better low-light accuracy at higher CPU cost.
- **OQ-8 (ASM-06) IR/depth hardware.** Assumed none (Brio 500 RGB-only). Confirm whether adding a Windows-Hello-class IR camera is on the table for Hardening; if yes, hardware liveness requirements would be added.
- **OQ-9 (CST-1) Single owner.** Assumed one enrolled identity. Confirm whether multiple household/team members will ever need enrollment (would expand to 1:N identification and change FM-06/accuracy targets).
- **OQ-10 Wayland.** Assumed X11-only Prototype with Wayland deferred. Confirm the timeline for Wayland support (affects REQ-NF-20 priority).

---

## 16. Out of scope (explicit)

- Multi-user 1:N identification (pilot is single-owner 1:1, CST-1; see OQ-9).
- Replacing GDM login or `sudo` in the Prototype (CST-3); Hardening PAM is opt-in only (OQ-5).
- Cloud/remote face processing or any network feature (CST-2).
- Emotion/age/gender or any face analysis beyond identity verification and liveness.
- Hardware IR/depth liveness in the Prototype (no sensor, ASM-06).
- Mobile / cross-device sync of templates.
- Continuous video recording or surveillance features (privacy, REQ-NF-13).

---

## 17. Researcher Quality Gate (self-check before hand-off)

- [x] Every requirement has a unique ID (REQ-F-01..27, REQ-NF-01..27).
- [x] Every requirement has at least one testable acceptance criterion (Section 10).
- [x] Gap analysis has zero unresolved Partial/No rows (each maps to an ASM + Open Question).
- [x] Failure modes documented for every mandated feature, >= 3 per subsystem overall (16 FMs, Section 11), each with detection + handling + phase.
- [x] No implicit assumptions: all recorded as ASM-01..11 / CST-1..8 and surfaced as OQ-1..10.
- [x] Prior art reviewed and cited (Howdy, ISO/IEC 30107-3, NIST FRVT, liveness survey — Section 13).
- [x] Prototype vs Hardening clearly tagged on every requirement (Phase column).
- [x] Safety Invariant (fail-closed, never weaken password) threaded through requirements, failure modes, and risks.
- [ ] Human review gate to confirm Open Questions OQ-1..10 (owned by orchestrator + human, NOT the researcher).

*Note:* Per the Output Contract, the researcher does NOT advance the workflow, create checkpoints, or mark deliverables complete. The final unchecked item is intentionally left for the human review gate.
