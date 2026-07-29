#!/usr/bin/env bash
# install.sh -- one-command, reproducible, user-space install (REQ-NF-26/27).
#
# Creates an isolated venv, installs pinned deps + the facelock package, wires
# the console scripts into ~/.local/bin, installs the config (0600), downloads
# the SHA-pinned models, and installs the systemd --user units. It does NOT
# enable auto-start until you have enrolled a face -- otherwise the next login
# would lock the screen with no face able to unlock it. When run interactively
# it offers to enroll and activate for you.
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

# 6. systemd --user units -- INSTALLED but NOT enabled yet. Enabling before a
#    face is enrolled would auto-start the guardian on the next login and lock
#    the screen with no face able to clear it (you'd fall back to the OS
#    password). Enrollment happens first (step 7), then we enable + start.
mkdir -p "$UNIT_DIR"
install -m 0644 "$PROJECT_DIR/systemd/facelock-guardian.service" "$UNIT_DIR/"
install -m 0644 "$PROJECT_DIR/systemd/facelockd.service" "$UNIT_DIR/"
HAVE_SYSTEMCTL=false
if command -v systemctl >/dev/null 2>&1; then
  systemctl --user daemon-reload || true
  HAVE_SYSTEMCTL=true
  echo ">> systemd --user units installed (not enabled yet)"
else
  echo "!! systemctl not found; units copied but cannot be enabled here."
fi

echo
echo "== install complete =="
FACELOCK="$BIN_DIR/facelock"
"$FACELOCK" disclosure || true
echo

# 7. Guided enroll-then-activate. We NEVER enable auto-start before a template
#    exists (safety: a locked screen with no enrolled face needs the OS password).
TEMPLATE_FILE="${XDG_DATA_HOME:-$HOME/.local/share}/facelock/templates/owner.tmpl"
enrolled() { [[ -f "$TEMPLATE_FILE" ]]; }

activate() {
  if ! enrolled; then
    local name="${SUDO_USER:-$USER}" reply=""
    read -r -p "Name to greet you by [${name}]: " reply || true
    [[ -n "$reply" ]] && name="$reply"
    echo ">> enrolling '${name}' -- look at the camera and follow the prompts..."
    if ! "$FACELOCK" enroll --name "$name"; then
      echo "!! enrollment did not complete; NOT enabling auto-start."
      echo "   Re-run:  facelock enroll --name ${name}   then enable (see below)."
      return 1
    fi
  fi
  if [[ "$HAVE_SYSTEMCTL" == true ]]; then
    systemctl --user enable --now facelock-guardian.service facelockd.service
    echo ">> face-unlock is enrolled, enabled, and running. Check: facelock status"
  fi
}

if [[ -t 0 && -t 1 ]]; then
  ans=""
  read -r -p "Enroll your face now and activate face-unlock? [Y/n] " ans || true
  case "${ans:-Y}" in
    [Nn]*) echo ">> Skipped. Activate later with the steps below." ;;
    *)     activate || true ;;
  esac
else
  echo "(non-interactive install: skipping enroll/enable -- do the steps below when ready.)"
fi

echo
echo "To activate later -- ENROLL FIRST, then enable (order matters):"
echo "  1) Enroll your face:   facelock enroll --name <YourName>"
echo "  2) Enable + start:     systemctl --user enable --now facelock-guardian facelockd"
echo "  3) Check status:       facelock status"
echo "  4) Camera self-test:   facelock test --seconds 5"
echo
echo "⚠  Do NOT enable the services before enrolling: a locked screen with no"
echo "   enrolled face can only be cleared with your OS password."
echo
echo "Ensure ~/.local/bin is on your PATH."
