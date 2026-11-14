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
from modules.auth import get_current_user, render_auth_interface

st.set_page_config(
    page_title="ReviewLens | Original Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

current_user = get_current_user()
if not current_user.get("authenticated"):
    st.markdown("""
    <style>
        [data-testid="stSidebar"],
        section[data-testid="stSidebar"],
        [data-testid="collapsedControl"],
        div[data-testid="stSidebarCollapsedControl"],
        [data-testid="stSidebarCollapseButton"] {
            display: none !important;
        }
        .main .block-container {
            max-width: 1320px !important;
            padding-top: 1.8rem !important;
            padding-bottom: 3rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }
    </style>
    """, unsafe_allow_html=True)
    render_auth_interface(on_navigate_home=lambda: st.rerun())
    st.stop()

render_original_dashboard()
