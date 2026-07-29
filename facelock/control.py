"""Control-socket IPC (design section 10.3) + nonce-bound grant authority.

The two processes communicate over ONE local Unix-domain stream socket
(``$XDG_RUNTIME_DIR/facelock/control.sock``, mode 0600, dir 0700). There is no
network surface (REQ-NF-12). Framing is one JSON object per line.

Security (SI-P1):
  * The socket is owner-only: every accepted connection's peer uid is checked
    via ``SO_PEERCRED`` and any uid != the server's uid is rejected.
  * Unlock is a *grant*, not a decision. The guardian issues a fresh
    ``grant_nonce`` bound to the current ``lock_epoch`` each time it raises the
    shield, and honours an ``unlock_grant`` ONLY if the nonce + epoch match and
    it arrives within the challenge window. A stale/forged/expired grant is
    ignored and the shield stays up. This is realized by :class:`GrantAuthority`
    (pure, unit-testable -- no sockets).
"""

from __future__ import annotations

import json
import os
import secrets
import socket
import struct
import threading
import time
from pathlib import Path
from typing import Any, Callable

from . import paths as _paths

# SO_PEERCRED returns struct ucred { pid_t pid; uid_t uid; gid_t gid; } = 3 ints.
_UCRED_FMT = "3i"
_UCRED_SIZE = struct.calcsize(_UCRED_FMT)


# --------------------------------------------------------------------------- #
# Nonce-bound grant authority (guardian side, pure logic).
# --------------------------------------------------------------------------- #
class GrantAuthority:
    """Owns the lock epoch + the currently valid unlock nonce (SI-P1).

    ``locked`` is True whenever the shield is up. Raising the shield bumps the
    epoch and mints a fresh nonce; a successful unlock also bumps the epoch so a
    replayed grant can never dismiss a future shield.
    """

    def __init__(self, *, window_s: float = 4.0, now_fn: Callable[[], float] = time.monotonic) -> None:
        self.window_s = float(window_s)
        self._now = now_fn
        self.locked = True
        self.epoch = 0
        self.nonce: str | None = None
        self.issued_at = 0.0
        self._lock = threading.Lock()
        self._mint()  # start locked with a valid nonce (fail-closed default)

    def _mint(self) -> tuple[int, str]:
        self.epoch += 1
        self.nonce = secrets.token_hex(16)
        self.issued_at = self._now()
        self.locked = True
        return self.epoch, self.nonce

    def raise_shield(self) -> tuple[int, str]:
        """Called when the guardian raises/keeps the shield; mints a nonce."""
        with self._lock:
            return self._mint()

    def current(self) -> tuple[bool, int, str | None]:
        with self._lock:
            return self.locked, self.epoch, (self.nonce if self.locked else None)

    def refresh_challenge(self) -> tuple[bool, int, str | None]:
        """Issue a fresh challenge window for an unlock attempt.

        The ``window_s`` bound is meant to cap the *response* latency (daemon
        fetches the nonce, then submits a grant within a few ms), NOT the age of
        the shield. Minting stamped ``issued_at`` at lock time, so any return
        more than ``window_s`` after locking made every grant "expired" and
        face-unlock could never succeed. Resetting ``issued_at`` when the daemon
        requests the nonce makes the window measure response latency. The nonce
        and epoch are unchanged, so replay protection (epoch-bound, single-use)
        is preserved; the socket is already owner-only (SO_PEERCRED).
        """
        with self._lock:
            if self.locked and self.nonce is not None:
                self.issued_at = self._now()
            return self.locked, self.epoch, (self.nonce if self.locked else None)

    def validate_grant(self, grant_nonce: str, lock_epoch: int) -> tuple[bool, str]:
        """Validate + consume an unlock grant. Returns ``(ok, reason)``."""
        with self._lock:
            if not self.locked:
                return False, "not_locked"
            if self.nonce is None:
                return False, "no_nonce"
            if lock_epoch != self.epoch:
                return False, "epoch_mismatch"
            if not secrets.compare_digest(str(grant_nonce), str(self.nonce)):
                return False, "stale_nonce"
            if (self._now() - self.issued_at) > self.window_s:
                return False, "expired"
            # Consume: dismiss shield and bump epoch so replay fails.
            self.locked = False
            self.epoch += 1
            self.nonce = None
            return True, "ok"

    def force_locked(self) -> tuple[int, str]:
        """Re-lock (e.g. heartbeat miss); equivalent to raising the shield."""
        return self.raise_shield()


# --------------------------------------------------------------------------- #
# Client helper (daemon + CLI).
# --------------------------------------------------------------------------- #
def send_command(
    socket_path: Path | str,
    message: dict[str, Any],
    *,
    timeout: float = 3.0,
) -> dict[str, Any]:
    """Send one command and return the parsed response.

    Fail-closed: any transport/parse error returns
    ``{"ok": False, "reason": "transport", ...}`` -- the caller (daemon) then
    cannot obtain a grant and the session stays locked.
    """
    socket_path = str(socket_path)
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect(socket_path)
            sock.sendall((json.dumps(message) + "\n").encode("utf-8"))
            buf = b""
            while b"\n" not in buf:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                buf += chunk
                if len(buf) > 1 << 20:  # 1 MiB guard
                    return {"ok": False, "reason": "oversize_response"}
            line = buf.split(b"\n", 1)[0]
            if not line:
                return {"ok": False, "reason": "empty_response"}
            return json.loads(line.decode("utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return {"ok": False, "reason": "transport", "error": str(exc)}


# --------------------------------------------------------------------------- #
# Server (guardian side).
# --------------------------------------------------------------------------- #
Handler = Callable[[dict[str, Any], int], dict[str, Any]]


class ControlServer:
    """Owner-only Unix-domain control server (guardian's ControlServer, C8)."""

    def __init__(
        self,
        socket_path: Path | str | None = None,
        *,
        handler: Handler,
        logger: Any = None,
        owner_uid: int | None = None,
    ) -> None:
        self.socket_path = Path(socket_path or _paths.control_socket_path())
        self.handler = handler
        self.logger = logger
        self.owner_uid = os.getuid() if owner_uid is None else owner_uid
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        _paths.ensure_dir(self.socket_path.parent, 0o700)
        # Remove a stale socket from a previous run.
        try:
            if self.socket_path.exists():
                self.socket_path.unlink()
        except OSError:
            pass
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(str(self.socket_path))
        os.chmod(self.socket_path, 0o600)
        sock.listen(8)
        sock.settimeout(0.5)
        self._sock = sock
        self._stop.clear()
        self._thread = threading.Thread(target=self._serve, name="facelock-control", daemon=True)
        self._thread.start()

    def _peer_uid(self, conn: socket.socket) -> int:
        creds = conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, _UCRED_SIZE)
        _pid, uid, _gid = struct.unpack(_UCRED_FMT, creds)
        return uid

    def _serve(self) -> None:
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with conn:
                try:
                    self._handle_conn(conn)
                except Exception as exc:  # one bad client never kills the server
                    if self.logger is not None:
                        try:
                            self.logger.warning({"event": "control_error", "error": str(exc)})
                        except Exception:
                            pass

    def _handle_conn(self, conn: socket.socket) -> None:
        conn.settimeout(2.0)
        try:
            uid = self._peer_uid(conn)
        except OSError:
            self._reply(conn, {"ok": False, "reason": "no_peer_cred"})
            return
        if uid != self.owner_uid:
            self._reply(conn, {"ok": False, "reason": "forbidden"})
            return
        buf = b""
        while b"\n" not in buf:
            chunk = conn.recv(4096)
            if not chunk:
                break
            buf += chunk
            if len(buf) > 1 << 20:
                self._reply(conn, {"ok": False, "reason": "oversize"})
                return
        line = buf.split(b"\n", 1)[0]
        if not line:
            self._reply(conn, {"ok": False, "reason": "empty"})
            return
        try:
            message = json.loads(line.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._reply(conn, {"ok": False, "reason": "malformed"})
            return
        if not isinstance(message, dict) or not isinstance(message.get("cmd"), str):
            self._reply(conn, {"ok": False, "reason": "malformed"})
            return
        try:
            response = self.handler(message, uid)
        except Exception as exc:
            response = {"ok": False, "reason": "handler_error", "error": str(exc)}
        self._reply(conn, response)

    @staticmethod
    def _reply(conn: socket.socket, obj: dict[str, Any]) -> None:
        try:
            conn.sendall((json.dumps(obj) + "\n").encode("utf-8"))
        except OSError:
            pass

    def stop(self) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            if self.socket_path.exists():
                self.socket_path.unlink()
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# DecisionEmitter + HeartbeatSender (perception daemon side, C7).
# --------------------------------------------------------------------------- #
class DecisionEmitter:
    """Thin client the perception daemon uses to talk to the guardian.

    The daemon can only *request* actions; it never holds lock authority
    (SI-P1). ``request_unlock`` fetches the guardian's current nonce and
    submits a bound grant -- a transport failure just means "stay locked".
    """

    def __init__(self, socket_path: Path | str | None = None, *, timeout: float = 3.0) -> None:
        self.socket_path = Path(socket_path or _paths.control_socket_path())
        self.timeout = timeout

    def _send(self, message: dict[str, Any]) -> dict[str, Any]:
        return send_command(self.socket_path, message, timeout=self.timeout)

    def request_lock(self, reason: str) -> dict[str, Any]:
        return self._send({"cmd": "lock", "reason": reason})

    def get_grant_nonce(self) -> dict[str, Any]:
        return self._send({"cmd": "get_grant_nonce"})

    def request_unlock(self, score: float, tau: float, live: bool) -> dict[str, Any]:
        """Fetch the current nonce/epoch and submit a bound unlock grant."""
        info = self.get_grant_nonce()
        if not info.get("ok") or not info.get("locked"):
            return {"ok": False, "reason": info.get("reason", "not_locked")}
        return self._send({
            "cmd": "unlock_grant",
            "grant_nonce": info.get("grant_nonce"),
            "lock_epoch": info.get("lock_epoch"),
            "score": score,
            "tau": tau,
            "live": bool(live),
        })

    def greet(self, name: str) -> dict[str, Any]:
        return self._send({"cmd": "greet", "name": name})

    def shield_status(
        self,
        phase: str,
        reason: str | None = None,
        *,
        progress: float | None = None,
        votes_k: int | None = None,
        votes_need: int | None = None,
        frames: int | None = None,
        frames_need: int | None = None,
    ) -> dict[str, Any]:
        """Push a cosmetic shield-status phase to the guardian (best-effort).

        ``phase`` is one of ``recognizing`` (checking) | ``denied`` | ``locked``.
        For the checking phase the real k-of-n verification progress is included
        (``progress`` in [0,1] plus the raw vote/frame counts) so the shield can
        draw a truthful progress bar. This carries NO lock authority and a
        transport failure is irrelevant. A short timeout never stalls perception.
        """
        msg: dict[str, Any] = {"cmd": "shield_status", "phase": phase, "reason": reason}
        if progress is not None:
            msg["progress"] = float(progress)
        if votes_k is not None:
            msg["votes_k"] = int(votes_k)
        if votes_need is not None:
            msg["votes_need"] = int(votes_need)
        if frames is not None:
            msg["frames"] = int(frames)
        if frames_need is not None:
            msg["frames_need"] = int(frames_need)
        return send_command(self.socket_path, msg, timeout=min(self.timeout, 0.5))

    def heartbeat(self, seq: int, state: str, health: dict[str, Any]) -> dict[str, Any]:
        return self._send({"cmd": "heartbeat", "seq": seq, "state": state, "health": health})
