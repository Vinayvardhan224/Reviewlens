"""
ReviewLens Navigation & Header Component.
Amazon-inspired utility header with department navigation and comparison tray counter.
"""

from typing import List, Optional
import streamlit as st


def render_header(
    compare_count: int = 0,
    active_nav: str = "Catalogue"
) -> None:
    """
    Renders the modern utility header with styling, logo, and comparison tray indicator.
    """
    badge_bg = "rgba(255, 153, 0, 0.2)" if compare_count > 0 else "rgba(255, 255, 255, 0.08)"
    badge_border = "#FF9900" if compare_count > 0 else "rgba(255, 255, 255, 0.15)"
    badge_color = "#FF9900" if compare_count > 0 else "#CBD5E1"
    badge_pill_bg = "#FF9900" if compare_count > 0 else "#64748B"
    badge_pill_color = "#131921" if compare_count > 0 else "#FFFFFF"

    # We format HTML without leading indentation and without blank lines so Markdown never creates a code block
    header_html = (
        '<div style="background: linear-gradient(135deg, #131921 0%, #1e293b 100%); '
        'padding: 1.1rem 1.8rem; border-radius: 12px; color: #FFFFFF; margin-bottom: 1.5rem; '
        'box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15); display: flex; flex-wrap: wrap; '
        'align-items: center; justify-content: space-between; gap: 1rem;">'
        '<div style="display: flex; align-items: center; gap: 1rem;">'
        '<div style="background: linear-gradient(135deg, #FF9900 0%, #E47A24 100%); width: 44px; '
        'height: 44px; border-radius: 10px; display: flex; align-items: center; justify-content: center; '
        'font-size: 24px; font-weight: bold; box-shadow: 0 2px 8px rgba(255, 153, 0, 0.4);">'
        '🔍'
        '</div>'
        '<div>'
        '<div style="font-size: 1.45rem; font-weight: 800; letter-spacing: -0.5px; color: #FFFFFF;">'
        'Review<span style="color: #FF9900;">Lens</span>'
        '</div>'
        '<div style="font-size: 0.8rem; color: #94A3B8; font-weight: 500;">'
        'Smart Shopping Platform • Explainable NLP Intelligence'
        '</div>'
        '</div>'
        '</div>'
        '<div style="display: flex; align-items: center; gap: 0.8rem;">'
        f'<div style="background: {badge_bg}; border: 1px solid {badge_border}; '
        f'padding: 0.45rem 1rem; border-radius: 20px; font-size: 0.88rem; font-weight: 600; '
        f'color: {badge_color}; display: flex; align-items: center; gap: 0.5rem;">'
        '<span>⚔️ Compare Tray:</span>'
        f'<span style="background: {badge_pill_bg}; color: {badge_pill_color}; '
        f'padding: 0.1rem 0.55rem; border-radius: 10px; font-size: 0.8rem; font-weight: 700;">'
        f'{compare_count}</span>'
        '</div>'
        '<div style="background: rgba(37, 99, 235, 0.18); border: 1px solid rgba(37, 99, 235, 0.4); '
        'padding: 0.45rem 0.9rem; border-radius: 20px; font-size: 0.82rem; font-weight: 500; color: #93C5FD;">'
        '🛡️ Prepared Demo Data'
        '</div>'
        '</div>'
        '</div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)
