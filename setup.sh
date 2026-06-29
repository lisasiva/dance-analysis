#!/usr/bin/env bash
# One-command setup: checks Python + ffmpeg, builds the venv, installs everything.
# Re-running is safe — it verifies what's already there.
set -e
cd "$(dirname "$0")"

echo "🩰 dance-analysis setup"

# --- Python ---
if ! command -v python3 >/dev/null 2>&1; then
  echo "❌ Python 3 not found. Install Python 3.11 or 3.12 from https://python.org and re-run."
  exit 1
fi
PYV=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "✓ Python $PYV"

# --- ffmpeg (needed to read audio for the beat grid) ---
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "… ffmpeg missing — installing"
  if command -v brew >/dev/null 2>&1; then
    brew install ffmpeg
  elif command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update && sudo apt-get install -y ffmpeg
  else
    echo "❌ Couldn't auto-install ffmpeg. Install it manually, then re-run."
    exit 1
  fi
fi
echo "✓ ffmpeg"

# --- virtualenv + deps ---
if [ ! -d .venv ]; then
  echo "… creating virtual environment"
  python3 -m venv .venv
fi
echo "… installing dependencies (first run downloads PyTorch — a few minutes)"
./.venv/bin/python -m pip install -q --upgrade pip
./.venv/bin/pip install -q -e .

echo ""
echo "✅ All set. Start the app with:   ./run.sh"
