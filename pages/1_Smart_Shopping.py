"""
Streamlit Multipage Entrypoint: ReviewLens Smart Shopping & Product Intelligence.
Redirects to the primary application hub.
"""

import sys
from pathlib import Path
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Switch directly to main app.py without runpy recursion
try:
    st.switch_page("app.py")
except Exception:
    st.info("Redirecting to ReviewLens main application hub...")
