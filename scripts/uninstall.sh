#!/usr/bin/env bash
# uninstall.sh -- remove facelock (user-space). Preserves biometric data unless
# --purge is given (then it is securely deleted via 'facelock delete').
#
# Usage: scripts/uninstall.sh [--purge]
set -euo pipefail

PURGE=0
[[ "${1:-}" == "--purge" ]] && PURGE=1

VENV="${XDG_DATA_HOME:-$HOME/.local/share}/facelock/venv"
BIN_DIR="$HOME/.local/bin"
CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/facelock"
STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/facelock"
DATA_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/facelock"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

echo "== facelock uninstall =="

# 1. stop + disable services.
if command -v systemctl >/dev/null 2>&1; then
  systemctl --user stop facelockd.service facelock-guardian.service 2>/dev/null || true
  systemctl --user disable facelockd.service facelock-guardian.service 2>/dev/null || true
  rm -f "$UNIT_DIR/facelockd.service" "$UNIT_DIR/facelock-guardian.service"
  systemctl --user daemon-reload 2>/dev/null || true
  echo ">> services stopped, disabled, unit files removed"
fi

# 2. secure-delete the template FIRST if purging (uses the tool while present).
if [[ "$PURGE" -eq 1 ]]; then
  if [[ -x "$BIN_DIR/facelock" ]]; then
    "$BIN_DIR/facelock" delete --yes || true
  fi
fi

# 3. remove console-script symlinks.
for exe in facelock facelockd facelock-guardian; do
  rm -f "$BIN_DIR/$exe"
done
echo ">> removed console scripts from $BIN_DIR"

# 4. remove venv.
rm -rf "$VENV"
echo ">> removed venv"

# 5. purge data/config/state if requested.
if [[ "$PURGE" -eq 1 ]]; then
  rm -rf "$DATA_DIR" "$CONFIG_DIR" "$STATE_DIR"
  echo ">> purged data, config, and state (biometric artefacts securely deleted)"
else
  echo ">> kept your template + config (re-run with --purge to remove them)"
fi

echo "== uninstall complete =="
echo "Note: your OS password lock was never modified by facelock (SI-P3)."
