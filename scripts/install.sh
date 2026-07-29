#!/usr/bin/env bash
# install.sh -- one-command, reproducible, user-space install (REQ-NF-26/27).
#
# Creates an isolated venv, installs pinned deps + the facelock package, wires
# the console scripts into ~/.local/bin, installs the config (0600), downloads
# the SHA-pinned models, and installs + enables the systemd --user units.
#
# Runs entirely as the unprivileged user (no root, CST-3/REQ-NF-27).
set -euo pipefail

if [[ "$(id -u)" -eq 0 ]]; then
  echo "Refusing to run as root (facelock is user-space only, REQ-NF-27)." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

VENV="${XDG_DATA_HOME:-$HOME/.local/share}/facelock/venv"
BIN_DIR="$HOME/.local/bin"
CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/facelock"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

echo "== facelock install =="
echo "project: $PROJECT_DIR"

# 1. venv + pinned deps + package.
echo ">> creating venv at $VENV"
python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip >/dev/null
echo ">> installing pinned dependencies"
"$VENV/bin/pip" install -r "$PROJECT_DIR/requirements.txt"
echo ">> installing facelock package"
"$VENV/bin/pip" install "$PROJECT_DIR"

# 2. console scripts into ~/.local/bin.
mkdir -p "$BIN_DIR"
for exe in facelock facelockd facelock-guardian; do
  ln -sf "$VENV/bin/$exe" "$BIN_DIR/$exe"
  echo ">> linked $BIN_DIR/$exe"
done

# 3. config (never overwrite an existing one).
mkdir -p "$CONFIG_DIR"; chmod 0700 "$CONFIG_DIR"
if [[ ! -f "$CONFIG_DIR/config.toml" ]]; then
  install -m 0600 "$PROJECT_DIR/config/facelock.toml" "$CONFIG_DIR/config.toml"
  echo ">> installed config -> $CONFIG_DIR/config.toml (0600)"
else
  echo ">> config exists; leaving $CONFIG_DIR/config.toml untouched"
fi

# 4. models (SHA-pinned).
echo ">> downloading models"
bash "$SCRIPT_DIR/download_models.sh"

# 5. tkinter check (system package for the shield).
if ! python3 -c "import tkinter" >/dev/null 2>&1; then
  echo "!! python3-tk is not installed; the shield window needs it."
  echo "   Install with: sudo apt-get install python3-tk"
fi

# 6. systemd --user units.
mkdir -p "$UNIT_DIR"
install -m 0644 "$PROJECT_DIR/systemd/facelock-guardian.service" "$UNIT_DIR/"
install -m 0644 "$PROJECT_DIR/systemd/facelockd.service" "$UNIT_DIR/"
if command -v systemctl >/dev/null 2>&1; then
  systemctl --user daemon-reload
  systemctl --user enable facelock-guardian.service facelockd.service
  echo ">> systemd --user units installed + enabled"
else
  echo "!! systemctl not found; units copied but not enabled."
fi

echo
echo "== install complete =="
"$BIN_DIR/facelock" disclosure || true
echo
echo "Next steps:"
echo "  1) Enroll your face:      facelock enroll --name Yash"
echo "  2) Start the services:    systemctl --user start facelock-guardian facelockd"
echo "  3) Check status:          facelock status"
echo "  4) Self-test the camera:  facelock test --seconds 5"
echo
echo "Ensure ~/.local/bin is on your PATH."
