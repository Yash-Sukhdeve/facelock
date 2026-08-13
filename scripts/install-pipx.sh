#!/usr/bin/env bash
# install-pipx.sh -- one-line install via pipx, then provision.
#
#   pipx install facelock-linux  ->  facelock setup  ->  facelock enroll
#
# Ensures pipx is available (user-space), installs facelock into its own
# isolated venv, then runs `facelock setup` to fetch the SHA-pinned models and
# install the default config. It does NOT enable auto-start and does NOT enroll
# -- enrollment must come first (a locked screen with no enrolled face can only
# be cleared with your OS password). Runs entirely as the unprivileged user.
#
# Usage:
#   scripts/install-pipx.sh            # install from PyPI: pipx install facelock-linux
#   scripts/install-pipx.sh --local    # install this checkout: pipx install .
#   scripts/install-pipx.sh --systemd  # also install the --user systemd units
set -euo pipefail

if [[ "$(id -u)" -eq 0 ]]; then
  echo "Refusing to run as root (facelock is user-space only, REQ-NF-27)." >&2
  exit 1
fi

SOURCE_SPEC="facelock-linux"   # default: install the published package from PyPI (command is still `facelock`)
WITH_SYSTEMD=0
for arg in "$@"; do
  case "$arg" in
    --local)   SOURCE_SPEC="." ;;
    --systemd) WITH_SYSTEMD=1 ;;
    -h|--help)
      grep -E '^#( |$)' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
# When installing this checkout, hand pipx the project directory explicitly.
[[ "$SOURCE_SPEC" == "." ]] && SOURCE_SPEC="$PROJECT_DIR"

echo "== facelock install (pipx) =="

# 1. Ensure pipx (user-space). Install it with the --user pip if missing.
if ! command -v pipx >/dev/null 2>&1; then
  echo ">> pipx not found; installing it with 'python -m pip install --user pipx'"
  python3 -m pip install --user pipx
  python3 -m pipx ensurepath || true
  # Make pipx usable in THIS shell even before the PATH change takes effect.
  export PATH="$HOME/.local/bin:$PATH"
  PIPX="python3 -m pipx"
else
  PIPX="pipx"
fi

# 2. Install (or upgrade) facelock into its own isolated venv.
echo ">> pipx install $SOURCE_SPEC"
$PIPX install --force "$SOURCE_SPEC"

# 3. Provision: download the SHA-pinned models + install the default config.
FACELOCK="$(command -v facelock || echo "$HOME/.local/bin/facelock")"
echo ">> $FACELOCK setup$([[ "$WITH_SYSTEMD" -eq 1 ]] && echo ' --systemd')"
if [[ "$WITH_SYSTEMD" -eq 1 ]]; then
  "$FACELOCK" setup --systemd
else
  "$FACELOCK" setup
fi

echo
echo "== install complete =="
echo "Next -- ENROLL FIRST, then enable (order matters):"
echo "  1) facelock enroll --name <YourName>"
echo "  2) systemctl --user enable --now facelock-guardian facelockd"
echo "  3) facelock status"
echo
echo "Ensure ~/.local/bin is on your PATH (run 'pipx ensurepath' and re-open the shell)."
