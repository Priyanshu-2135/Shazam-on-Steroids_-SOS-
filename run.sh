#!/usr/bin/env bash
# Shazam-on-Steroids (SOS) Runner
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR"

if [ -f ".venv/bin/streamlit" ]; then
    echo "Starting SOS using local .venv environment..."
    .venv/bin/streamlit run app.py
elif command -v streamlit &> /dev/null; then
    echo "Starting SOS using system streamlit..."
    streamlit run app.py
else
    echo "Error: Streamlit not found. Please run ./setup.sh first to set up the environment."
    exit 1
fi
