#!/usr/bin/env bash
# Shazam-on-Steroids (SOS) Automated Setup Script
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR"

echo "=========================================================="
echo "    Setting up Shazam-on-Steroids (SOS) Environment       "
echo "=========================================================="

# Create virtual environment if it doesn't already exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment in .venv..."
    python3 -m venv .venv
else
    echo "Existing .venv found."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip and install dependencies
echo "Installing dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=========================================================="
echo "    Setup completed successfully!                         "
echo "    To start the application, run:                        "
echo "      ./run.sh                                            "
echo "    or:                                                   "
echo "      source .venv/bin/activate && streamlit run app.py   "
echo "=========================================================="
