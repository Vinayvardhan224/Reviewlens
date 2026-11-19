"""
Streamlit Multipage Entrypoint: Original ReviewLens Analytics Dashboard.
Preserved per technical requirement: Keep the existing analytics dashboard accessible; do not delete it.
Delegates to the unified ReviewLens application hub with the Original Dashboard view preselected.
"""

import sys
from pathlib import Path
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.session_state["nav_view"] = "📊 Original Analytics Dashboard"

try:
    st.switch_page("app.py")
except Exception:
    # Fallback in non-multipage bare mode
    from reviewlens.dashboard import render_original_dashboard
    render_original_dashboard()
