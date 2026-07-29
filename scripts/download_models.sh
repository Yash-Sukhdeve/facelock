#!/usr/bin/env bash
# download_models.sh -- fetch YuNet + SFace from the OpenCV Zoo, SHA-256 pinned.
#
# Design section 3.1 / 11.4 (REQ-NF-26). Downloads are from the AUTHORITATIVE
# OpenCV Zoo GitHub locations. SHA-256 handling is trust-on-first-use (TOFU)
# with optional enforcement:
#   * If scripts/models.sha256 contains a pinned hash for a file, the download
#     MUST match it or the script aborts (fail-closed provisioning).
#   * If no pin exists yet, the actual SHA-256 is computed, printed, and
#     appended to scripts/models.sha256 for review. Per user rule R6, we NEVER
#     hard-code a guessed hash: you pin the value AFTER verifying it against the
#     OpenCV Zoo. Re-runs then enforce it.
#
# Usage: scripts/download_models.sh [--dest DIR]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/facelock/models"
REPO_DIR="$PROJECT_DIR/models"
PIN_FILE="$SCRIPT_DIR/models.sha256"

DEST="$DATA_DIR"
if [[ "${1:-}" == "--dest" && -n "${2:-}" ]]; then DEST="$2"; fi

ZOO_BASE="https://github.com/opencv/opencv_zoo/raw/main/models"
declare -A MODELS=(
  ["face_detection_yunet_2023mar.onnx"]="$ZOO_BASE/face_detection_yunet/face_detection_yunet_2023mar.onnx"
  ["face_recognition_sface_2021dec.onnx"]="$ZOO_BASE/face_recognition_sface/face_recognition_sface_2021dec.onnx"
)

mkdir -p "$DEST" "$REPO_DIR"
chmod 0700 "$(dirname "$DEST")" 2>/dev/null || true
touch "$PIN_FILE"

fetch() {
  local url="$1" out="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -fSL --retry 3 -o "$out" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -q -O "$out" "$url"
  else
    echo "ERROR: need curl or wget to download models." >&2
    return 1
  fi
}

# NOTE: trailing `|| true` — on first run the pin file is empty, so grep finds
# nothing and (under set -euo pipefail) would abort the script before the
# trust-on-first-use branch. Guard it so "no pin yet" yields an empty string.
pinned_hash() { grep -E "  $1\$" "$PIN_FILE" 2>/dev/null | awk '{print $1}' | head -n1 || true; }

for name in "${!MODELS[@]}"; do
  url="${MODELS[$name]}"
  target="$DEST/$name"
  tmp="$(mktemp)"
  echo ">> downloading $name"
  if ! fetch "$url" "$tmp"; then echo "ERROR: download failed for $name" >&2; rm -f "$tmp"; exit 1; fi
  if [[ ! -s "$tmp" ]]; then echo "ERROR: $name downloaded empty" >&2; rm -f "$tmp"; exit 1; fi

  actual="$(sha256sum "$tmp" | awk '{print $1}')"
  expected="$(pinned_hash "$name")"
  if [[ -n "$expected" ]]; then
    if [[ "$actual" != "$expected" ]]; then
      echo "ERROR: SHA-256 mismatch for $name (fail-closed, FM-11)" >&2
      echo "  expected $expected" >&2
      echo "  actual   $actual" >&2
      rm -f "$tmp"; exit 1
    fi
    echo "   verified against pinned SHA-256."
  else
    echo "   no pinned hash yet; recording actual SHA-256 (REVIEW then keep):"
    echo "   $actual  $name"
    echo "$actual  $name" >> "$PIN_FILE"
  fi

  mv "$tmp" "$target"
  chmod 0644 "$target"
  echo "$actual  $name" > "$target.sha256"
  # Also mirror into the repo-local models/ dir for a repo-local run.
  if [[ "$DEST" != "$REPO_DIR" ]]; then cp -f "$target" "$REPO_DIR/$name"; cp -f "$target.sha256" "$REPO_DIR/$name.sha256"; fi
  echo "   installed -> $target"
done

echo
echo "Models ready in: $DEST"
echo "Pins recorded in: $PIN_FILE"
echo "IMPORTANT (R6): verify the recorded SHA-256 values against the OpenCV Zoo,"
echo "then keep scripts/models.sha256 so future runs enforce them."
