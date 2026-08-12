# Changelog

All notable changes to facelock are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). The
version is single-sourced from `pyproject.toml`.

## [0.2.0] - 2026-08-12

First packaged, installable release: `pipx install facelock` → `facelock setup`
→ `facelock enroll` works on a fresh machine.

### Added
- **`facelock setup` command** — one-shot, offline-verifiable provisioning.
  Downloads the SHA-256-**pinned** YuNet + SFace models (from the authoritative
  OpenCV Zoo) into the XDG models dir, verifying every byte against the pinned
  hash **before** it is trusted; refuses on a mismatch (fail-closed, R6/FM-11).
  Idempotent (a present, correct-hash model is skipped). Installs the default
  `config.toml` (0600) if absent, never clobbering an existing one. `--systemd`
  installs the `--user` units **without** enabling auto-start (enroll-first
  safety). The HTTP fetch is injected so the whole command is unit-tested with
  no network.
- **Packaging for pipx/PyPI and GitHub Releases.** The wheel now ships the
  default `config/facelock.toml`, the `systemd/*.service` units, and the bundled
  Inter typeface, so `facelock setup` finds everything on a bare install.
- **Release CI** (`.github/workflows/release.yml`) — on a `v*` tag: build wheel
  + sdist, run the test suite as a gate, publish a GitHub Release with the
  artifacts, then publish to PyPI (skipped gracefully when no PyPI token secret
  is configured; the GitHub Release still happens).
- **One-line installer** `scripts/install-pipx.sh` (installs pipx if missing,
  `pipx install`, then `facelock setup`).
- **Evaluation harness** (`facelock/eval/*`, `facelock-eval`) — FMR/FNMR/EER
  metrics through the exact deployed matcher, with Wilson 95% CIs and a
  provenance-stamped JSON report; LFW dataset embedding with a resize path so
  faces clear `min_face_px`.
- **Recognition-model metrics** documented honestly in
  [`docs/MODEL_METRICS.md`](docs/MODEL_METRICS.md). These are the recognition
  **model's** score-separability numbers (SFace + the deployed matcher on
  captured embeddings) — **not** a certified biometric, not an
  application-level deployment FMR/FNMR, and not a spoof-resistance claim. See
  the scope box in that document. No "biometric-grade" product claim is made.
- **Premium Face-ID-style enrollment UI** — a native-resolution fullscreen
  preview with a multi-monitor `--screen` picker, 720p capture, and a polished
  HUD renderer using the bundled Inter typeface (Pillow), with a graceful
  `cv2.putText` fallback when Pillow/fonts are unavailable.
- **Model-free RGB PAD core** (liveness T1–T4) behind a golden-vector security
  gate (Hardening groundwork; anti-spoofing remains **off** by default).

### Changed
- Version bumped to **0.2.0**; `__version__` is now read from the installed
  package metadata (single source of truth = `pyproject.toml`).
- Install docs rewritten around `pipx install facelock` → `facelock setup` →
  `facelock enroll`, with an honestly-scoped recognition-metrics line.
- Security contract: the runtime unlock path remains network-free (REQ-NF-12);
  the one-shot `setup` provisioning verb is the single, tested exception, and a
  compensating-control test proves that network capability can never leak into
  the daemon/guardian unlock loop.

### Fixed
- **Stabilization pass** — eliminated the lockout / screen-flapping /
  recognition bugs surfaced in live testing: bounded camera cycling under
  continuous owner-absence (FM-07), debounced DPMS and slowed the screen-off
  re-assert cadence (FM-DPMS), and fixed a grant-escalation leak and an
  expired-grant acceptance (pinned by regression tests).
- **Shield grab is verified or fails closed** — the guardian now fails closed on
  a shield-raise exception (not only a `False` return), and the shield verifies
  its X11 input grab or fails closed (REQ-F-14 / SI-P5).
- **Dry-run safety** — added a SAFE no-OS-lock dry-run mode for lockout-free
  live testing; a persisted dry-run config is loudly flagged by `config-check`
  because it does not protect the session.

### Security
- **Scope, stated honestly:** facelock is a **screensaver-class**, local
  face-unlock (RGB webcam, CPU-only, **no PAM**). With anti-spoofing **off** (the
  default), it is documentedly bypassable by a printed photo or a phone/monitor
  video of the owner. Your OS password lock is never removed or weakened.
- Models are SHA-256-pinned and verified on download; an unpinned or mismatched
  model is refused (fail-closed).

## [0.1.0] - 2026-07-28

### Added
- Initial screensaver-only face-unlock prototype: two-process fail-closed
  architecture (`facelockd` perception daemon + `facelock-guardian` session
  guardian), YuNet detection + SFace embeddings, per-owner τ calibration with a
  safety floor and k-of-n voting, presence/lock state machine, nonce-bound
  control IPC, X11 shield + greeter, enrollment/delete/calibrate CLI, TOML
  config with fail-closed validation, and `systemd --user` units.

[0.2.0]: https://github.com/Yash-Sukhdeve/facelock/releases/tag/v0.2.0
[0.1.0]: https://github.com/Yash-Sukhdeve/facelock/releases/tag/v0.1.0
