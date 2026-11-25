#!/usr/bin/env bash
# ReviewLens: Smart Product Review Intelligence Platform Launcher
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  ReviewLens: Smart Product Review Intelligence Platform"
echo "============================================================"

# Check or create virtual environment
if [ ! -d "venv" ]; then
    echo "[1/3] Creating virtual environment (venv)..."
    python3 -m venv venv
fi

# Activate venv if present
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

echo "[2/3] Setting up NLTK corpora..."
python3 setup_nltk.py

echo "[3/3] Starting ReviewLens on http://localhost:8501..."
python3 -m streamlit run app.py --server.port=8501 --server.headless=true
