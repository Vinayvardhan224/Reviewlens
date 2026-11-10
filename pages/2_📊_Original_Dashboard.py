"""
Streamlit Multipage Entrypoint: Original ReviewLens Analytics Dashboard.
Preserved per technical requirement: Keep the existing analytics dashboard accessible; do not delete it.
"""

import sys
from pathlib import Path
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reviewlens.dashboard import render_original_dashboard

st.set_page_config(
    page_title="ReviewLens | Original Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

render_original_dashboard()
