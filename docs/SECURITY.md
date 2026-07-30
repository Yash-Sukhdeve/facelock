# facelock — security posture & audit notes

This document records the threat model, the hardening applied in this review, and
the **residual limitations** that remain in the Prototype (`security.phase = P`).
It is deliberately blunt: a face-unlock tool that oversells its guarantees is
worse than one that states them plainly.

## Threat model

facelock is a **convenience layer over the screen lock** of an already-logged-in
Linux session. It is designed to resist:

1. **A physical attacker at a locked screen with no shell** as the owner — the
   primary case. They see the input-grabbing shield (or the real OS lock) and
   must present the owner's face or type the OS password.
2. **A returning owner** — verified by camera and let straight back in.

It is **not** designed to be a password replacement, and it is **not** wired into
PAM, login, or sudo (REQ-F-18). The OS password path is never removed or
weakened and always works.

### What it explicitly does *not* defend against (documented, by design)

- **Presentation attacks with liveness off** (the prototype default): a printed
  photo or a phone/monitor video of the owner can pass recognition. This is the
  same limitation Howdy documents. Turn on liveness/PAD in the Hardening phase.
- **A same-uid attacker who already has code execution as the owner.** Any process
  running as the user can read the user's files, log keystrokes, and interact
  with the session regardless of facelock. A screen lock cannot defend a session
  against code that is already the user. See "Residual limitations" below for the
  specific same-uid gaps.
- **An offline attacker with raw disk access and no full-disk encryption.** The
  template is 0600 but not encrypted in phase P (see M-KEY below). Use FDE, or
  the Hardening keyring path, if this is in scope.

## Hardening applied in this review

| ID | Fix | File(s) |
|----|-----|---------|
| Escape fail-open | `Esc` now dismisses the shield **only if the OS lock is confirmed engaged**; otherwise it holds the shield and logs `lock_critical` instead of exposing the bare desktop. | `guardian.py` |
| Shield-raise fail-open | If the shield cannot obtain an effective input grab (`raise_shield()` returns `False`), the guardian now **escalates to the OS lock** even for the normally face-dismissable reasons (away/stranger/cooldown), instead of leaving a cosmetic window over a live desktop. | `guardian.py` |
| τ floor bypass | An explicit `recognition.tau` override below `recognition.tau_floor` is now a **hard, fail-closed config error** (security-critical, REQ-NF-22), and the daemon **clamps** `tau` to the floor as defense-in-depth. Previously a value like `tau = 0.05` silently defeated the calibrated operating point. | `config.py`, `daemon.py` |
| Grant sanity | The guardian now independently rejects a grant whose own `score < tau` (a self-inconsistent/forged decision) **before** consuming the nonce. Defense-in-depth for SI-P1. | `guardian.py` |
| PATH hijack | `loginctl` / `gdbus` / `xdg-screensaver` are resolved to **absolute paths in trusted system directories** and never via inherited `$PATH`, so a same-uid `~/.local/bin` trojan cannot subvert both `lock()` and the `is_locked()` confirmation (SI-P5). | `lock_backend.py` |
| Directory perms | `ensure_dir` now tightens **every** directory it creates (not just the leaf), rejects symlinked or foreign-owned directories, and thereby guards the predictable `/tmp/facelock-<uid>` runtime fallback against pre-planting. | `paths.py` |
| Temp-file writes | `secure_write_bytes` creates its temp file with a unique name and `O_EXCL | O_NOFOLLOW`, so a pre-planted symlink can never redirect a 0600 write. | `paths.py` |
| systemd sandbox | Both units add `RestrictAddressFamilies=AF_UNIX AF_NETLINK` (enforces REQ-NF-12 "no network surface"), `UMask=0077`, a pinned `PATH`, `SystemCallFilter=@system-service`, and the kernel/namespace/`RestrictSUIDSGID`/`LockPersonality` protections. | `systemd/*.service` |

All fixes ship with regression tests in `tests/test_security_hardening.py`.

## Residual limitations (accepted in phase P)

These are **known and intentional** for the prototype. Each is closed by the
Hardening phase or by an OS-level control (FDE); they are documented here so no
one mistakes the prototype's guarantees.

- **SAME-UID grant path (`L1`).** The guardian's unlock authority is nonce +
  epoch + owner-uid (`SO_PEERCRED`). The nonce is exposed to any owner-uid client
  via `get_grant_nonce`, and the biometric decision is made by `facelockd`. A
  process **already running as the owner** can therefore request a grant without
  presenting a face. This is inherent to a same-uid two-process split — a
  same-uid attacker can also grab input directly — and is out of scope for a
  screen lock. The `score < tau` sanity gate added above raises the bar against
  *buggy/forged* grants but cannot stop a same-uid attacker who fabricates a
  consistent grant. Closing this fully requires privilege separation (distinct
  uids) — a Hardening item.
- **Template integrity key co-location (`M-KEY`).** The HMAC key
  (`templates/.integrity.key`) and the audit key (`state/audit.key`) live in the
  same owner-only directory as the data they protect. So the `.sig` provides
  **corruption/bit-rot evidence, not anti-tamper**: anyone who can write the
  template can read the key and re-sign. Real tamper-resistance is the Hardening
  `template_encryption = keyring` path (Secret Service). Treat phase-P integrity
  as bit-rot detection only.
- **Multi-pose FMR (`M-POSE`).** τ is calibrated against the centroid, but at
  match time a probe passes if it clears τ against the best of up to `pose_max`
  enrolled sub-templates. Each added anchor widens the acceptance region at the
  same τ, so the **operational false-match rate is somewhat higher** than the
  calibrated FMR reports. k-of-n voting bounds it, but the shipped number is
  optimistic. Hardening should calibrate τ against the same best-of-bank rule
  used at match time.
- **Secure delete (`L-SHRED`).** `facelock delete` overwrites-then-unlinks, which
  is **best-effort only** on SSD/CoW/journaled filesystems (wear-leveling and
  copy-on-write can retain the original blocks). For a hard guarantee, use
  full-disk / `fscrypt` encryption for the data directory.
- **Dependency pinning (`M-DEPS`).** `requirements.txt` pins by version, not by
  hash. For a supply-chain-hardened install, generate a hashed lockfile and
  `pip install --require-hashes`. Models are already SHA-256 pinned
  (`scripts/models.sha256`).
- **Model download (`L-TOFU`).** `download_models.sh` fetches from the OpenCV Zoo
  `raw/main` ref with trust-on-first-use pinning; the shipped pins are already
  populated and enforced on every run. Prefer a pinned commit/tag URL and verify
  recorded hashes against a second source before committing new pins.

## Verified sound (no action needed)

`np.load(..., allow_pickle=False)` everywhere (no pickle RCE); nonces via
`secrets.token_hex`; constant-time `hmac.compare_digest` / `secrets.compare_digest`
for all secret comparisons; no `shell=True` and no attacker-interpolated shell
strings anywhere; `SO_PEERCRED` is kernel-filled and unspoofable; the FSM has no
path to a grant that bypasses the owner (+ liveness when required) and every
error/edge resolves to LOCKED; config treats security keys as always
refuse-on-invalid.
