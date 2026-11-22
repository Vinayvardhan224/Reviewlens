"""Embed the animated scrolling landing page (landing.html) inside Streamlit."""

from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

LANDING_PATH = Path(__file__).resolve().parent.parent / "landing.html"


def render_landing(height: int = 900) -> None:
    """Render landing.html as a scrollable section.

    Usage (e.g. at the top of the Home view in app.py):
        from components.landing import render_landing
        render_landing()
    """
    if not LANDING_PATH.exists():
        st.info("Landing page file (landing.html) not found.")
        return
    components.html(LANDING_PATH.read_text(encoding="utf-8"), height=height, scrolling=True)
