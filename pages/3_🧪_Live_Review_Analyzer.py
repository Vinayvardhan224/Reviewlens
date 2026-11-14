"""
Streamlit Multipage Entrypoint: ReviewLens Live Review Diagnostic Lab & Anomaly Detector.
"""

import sys
from pathlib import Path
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Pre-select Live Review Analyzer in session state
st.session_state["nav_view"] = "🧪 Live Review Analyzer"
st.session_state["sidebar_nav_radio"] = "🧪 Live Review Analyzer"

import runpy
app_script = PROJECT_ROOT / "app.py"
runpy.run_path(str(app_script), run_name="__main__")
