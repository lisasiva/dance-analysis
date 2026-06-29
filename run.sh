#!/usr/bin/env bash
# Launch the local web UI. Opens in your browser.
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "Run ./setup.sh first."
  exit 1
fi
exec ./.venv/bin/streamlit run app.py
