"""
ReviewLens Glassmorphism & Bento Grid Shared Design System.
Defines unified CSS variables, reusable frosted glass tiles, KPI card builders,
and Plotly chart glass styling for high-contrast dark purple SaaS aesthetics.
"""

import html
from typing import Any, Dict, Optional, Union
import streamlit as st

# Core theme tokens conforming to ReviewLens specifications
THEME_TOKENS: Dict[str, str] = {
    "--bg": "#07051A",
    "--bg2": "#0D0A24",
    "--glass": "rgba(22, 17, 46, 0.62)",
    "--glass-strong": "rgba(30, 23, 62, 0.78)",
    "--border": "rgba(255, 255, 255, 0.10)",
    "--border-hover": "rgba(155, 92, 255, 0.55)",
    "--primary": "#7138F5",
    "--primary-2": "#9B5CFF",
    "--accent": "#22D3EE",
    "--text": "#F1F5F9",
    "--muted": "#A5A3C2",
    "--pos": "#34D399",
    "--neg": "#F87171",
    "--mixed": "#FBBF24",
    "--radius": "18px",
    "--blur": "14px",
}

CHART_COLOURWAY = ["#9B5CFF", "#22D3EE", "#34D399", "#FBBF24", "#F87171", "#F472B6"]


def get_theme_css(reduce_effects: bool = False) -> str:
    """
    Generates the global stylesheet for ReviewLens.
    Supports reduce_effects toggle for low-power or low-end mobile devices.
    """
    blur_val = "0px" if reduce_effects else "var(--blur)"
    backdrop_filter = "none" if reduce_effects else f"blur({blur_val})"

    return f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

        :root {{
            --bg: #07051A;
            --bg2: #0D0A24;
            --glass: rgba(22, 17, 46, 0.62);
            --glass-strong: rgba(30, 23, 62, 0.78);
            --border: rgba(255, 255, 255, 0.10);
            --border-hover: rgba(155, 92, 255, 0.55);
            --primary: #7138F5;
            --primary-2: #9B5CFF;
            --accent: #22D3EE;
            --text: #F1F5F9;
            --muted: #A5A3C2;
            --pos: #34D399;
            --neg: #F87171;
            --mixed: #FBBF24;
            --radius: 18px;
            --blur: {blur_val};
        }}

        /* Base Typography */
        html, body, [class*="css"], .stApp {{
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
            color: var(--text) !important;
        }}

        /* Page Background with 3 soft ambient colour blobs (fixed, pointer-events: none) */
        body, .stApp {{
            background-color: var(--bg) !important;
            background-image: 
                radial-gradient(circle at 12% 16%, rgba(113, 56, 245, 0.28) 0%, transparent 45%),
                radial-gradient(circle at 88% 18%, rgba(155, 92, 255, 0.22) 0%, transparent 42%),
                radial-gradient(circle at 50% 85%, rgba(34, 211, 238, 0.12) 0%, transparent 50%) !important;
            background-attachment: fixed !important;
        }}

        /* Headings */
        h1, h2, h3, h4, h5, h6 {{
            color: #FFFFFF !important;
            letter-spacing: -0.025em !important;
            font-weight: 800 !important;
        }}
        h1 {{ font-size: 2.1rem !important; }}
        h2 {{ font-size: 1.6rem !important; }}
        h3 {{ font-size: 1.25rem !important; }}

        /* Text Contrast & Scale (body text >= 4.5:1) */
        p, span, label, div {{
            color: var(--text);
        }}
        .stCaption, caption {{
            color: var(--muted) !important;
            font-size: 0.82rem !important;
        }}

        /* ====================================================
           REUSABLE GLASSMORPHIC & BENTO CLASSES
           ==================================================== */
        .rl-glass {{
            background: var(--glass) !important;
            backdrop-filter: {backdrop_filter} !important;
            -webkit-backdrop-filter: {backdrop_filter} !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius) !important;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.45) !important;
            padding: 1.25rem !important;
            color: var(--text) !important;
            transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.25s ease, box-shadow 0.25s ease !important;
            position: relative;
            overflow: hidden;
            margin-bottom: 0.9rem;
        }}

        .rl-glass:hover {{
            border-color: var(--border-hover) !important;
            box-shadow: 0 14px 35px rgba(113, 56, 245, 0.35) !important;
            transform: translateY(-2px);
        }}

        .rl-glass-strong {{
            background: var(--glass-strong) !important;
            backdrop-filter: {backdrop_filter} !important;
            -webkit-backdrop-filter: {backdrop_filter} !important;
            border: 1px solid rgba(155, 92, 255, 0.35) !important;
            border-radius: var(--radius) !important;
            box-shadow: 0 14px 40px rgba(0, 0, 0, 0.55) !important;
            padding: 1.35rem !important;
            color: var(--text) !important;
            margin-bottom: 1rem;
        }}

        .rl-tile-title {{
            font-size: 0.78rem !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.08em !important;
            color: var(--muted) !important;
            margin: 0 0 0.45rem 0 !important;
            display: flex;
            align-items: center;
            gap: 0.45rem;
        }}

        .rl-section-title {{
            font-size: 1.25rem !important;
            font-weight: 800 !important;
            color: #FFFFFF !important;
            letter-spacing: -0.02em !important;
            margin: 0 0 0.85rem 0 !important;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .rl-kpi {{
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            min-height: 96px;
        }}
        .rl-kpi-value {{
            font-size: 1.95rem !important;
            font-weight: 800 !important;
            color: #FFFFFF !important;
            line-height: 1.15 !important;
            font-variant-numeric: tabular-nums !important;
            letter-spacing: -0.03em !important;
        }}
        .rl-kpi-label {{
            font-size: 0.78rem !important;
            font-weight: 700 !important;
            color: var(--muted) !important;
            text-transform: uppercase !important;
            letter-spacing: 0.06em !important;
            margin-bottom: 0.25rem !important;
        }}
        .rl-kpi-sub {{
            font-size: 0.76rem !important;
            color: var(--muted) !important;
            margin-top: 0.35rem !important;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }}

        /* Chips with semantic colours only */
        .rl-chip {{
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            font-size: 0.74rem !important;
            font-weight: 700 !important;
            border-radius: 9999px !important;
            padding: 0.18rem 0.6rem !important;
            line-height: 1.2 !important;
            white-space: nowrap;
        }}
        .rl-chip-pos {{
            color: var(--pos) !important;
            background: rgba(52, 211, 153, 0.14) !important;
            border: 1px solid rgba(52, 211, 153, 0.35) !important;
        }}
        .rl-chip-neg {{
            color: var(--neg) !important;
            background: rgba(248, 113, 113, 0.14) !important;
            border: 1px solid rgba(248, 113, 113, 0.35) !important;
        }}
        .rl-chip-mixed {{
            color: var(--mixed) !important;
            background: rgba(251, 191, 36, 0.14) !important;
            border: 1px solid rgba(251, 191, 36, 0.35) !important;
        }}
        .rl-chip-neutral {{
            color: var(--muted) !important;
            background: rgba(255, 255, 255, 0.06) !important;
            border: 1px solid var(--border) !important;
        }}

        /* Backward compatibility with existing card styles, upgraded to glassmorphism */
        .metric-card, .topic-card {{
            background: var(--glass) !important;
            backdrop-filter: {backdrop_filter} !important;
            -webkit-backdrop-filter: {backdrop_filter} !important;
            border: 1px solid var(--border) !important;
            border-left: 4px solid var(--primary-2) !important;
            border-radius: var(--radius) !important;
            padding: 1.25rem !important;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.45) !important;
            margin-bottom: 0.9rem !important;
            transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.25s ease, box-shadow 0.25s ease !important;
        }}
        .metric-card:hover, .topic-card:hover {{
            transform: translateY(-2px) !important;
            border-color: var(--border-hover) !important;
            box-shadow: 0 14px 35px rgba(113, 56, 245, 0.35) !important;
        }}
        .metric-card .metric-label {{
            font-size: 0.78rem !important;
            text-transform: uppercase !important;
            letter-spacing: 0.06em !important;
            color: var(--muted) !important;
            font-weight: 700 !important;
            margin-bottom: 0.3rem !important;
        }}
        .metric-card .metric-value {{
            font-size: 1.85rem !important;
            font-weight: 800 !important;
            color: #FFFFFF !important;
            line-height: 1.2 !important;
            font-variant-numeric: tabular-nums !important;
        }}
        .metric-card .metric-sub {{
            font-size: 0.76rem !important;
            color: var(--muted) !important;
            margin-top: 0.35rem !important;
        }}

        .metric-card-orange {{ border-left-color: var(--mixed) !important; }}
        .metric-card-green {{ border-left-color: var(--pos) !important; }}
        .metric-card-purple {{ border-left-color: var(--primary-2) !important; }}

        .hero-banner {{
            background: linear-gradient(135deg, rgba(30, 23, 62, 0.85) 0%, rgba(13, 10, 36, 0.9) 100%) !important;
            backdrop-filter: {backdrop_filter} !important;
            -webkit-backdrop-filter: {backdrop_filter} !important;
            border-radius: var(--radius) !important;
            padding: 2rem 2.2rem !important;
            color: #FFFFFF !important;
            margin-bottom: 1.5rem !important;
            box-shadow: 0 16px 40px rgba(0, 0, 0, 0.5) !important;
            border: 1px solid rgba(155, 92, 255, 0.3) !important;
            position: relative;
            overflow: hidden;
        }}

        .provenance-banner {{
            background: rgba(113, 56, 245, 0.16) !important;
            backdrop-filter: {backdrop_filter} !important;
            border: 1px solid rgba(155, 92, 255, 0.35) !important;
            border-radius: 12px !important;
            padding: 0.85rem 1.2rem !important;
            color: #C4B5FD !important;
            font-size: 0.85rem !important;
            margin-bottom: 1.2rem !important;
            display: flex;
            align-items: center;
            gap: 0.8rem;
        }}

        .topic-pill {{
            display: inline-block;
            background: rgba(113, 56, 245, 0.22);
            border: 1px solid rgba(155, 92, 255, 0.38);
            color: #C4B5FD;
            font-size: 0.78rem;
            font-weight: 700;
            padding: 0.22rem 0.65rem;
            border-radius: 12px;
            margin-right: 0.4rem;
            margin-bottom: 0.4rem;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }}
        .topic-pill:hover {{
            transform: scale(1.05);
            border-color: var(--primary-2);
            color: #FFFFFF;
        }}

        .fig-tag {{
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.72rem !important;
            font-weight: 800 !important;
            letter-spacing: 0.08em !important;
            color: #C4B5FD !important;
            background: rgba(155, 92, 255, 0.16) !important;
            border: 1px solid rgba(155, 92, 255, 0.35) !important;
            padding: 0.2rem 0.6rem !important;
            border-radius: 6px !important;
            display: inline-block !important;
            margin-bottom: 0.5rem !important;
            text-transform: uppercase !important;
        }}

        .mono-badge {{
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.75rem !important;
            padding: 0.2rem 0.55rem !important;
            border-radius: 6px !important;
            background: rgba(255, 255, 255, 0.06) !important;
            border: 1px solid rgba(255, 255, 255, 0.12) !important;
            color: #CBD5E1 !important;
        }}

        .layer-card, .spec-box {{
            background: var(--glass) !important;
            backdrop-filter: {backdrop_filter} !important;
            -webkit-backdrop-filter: {backdrop_filter} !important;
            border: 1px solid var(--border) !important;
            border-left: 4px solid var(--primary-2) !important;
            border-radius: 12px !important;
            padding: 0.95rem 1.2rem !important;
            color: var(--text) !important;
            font-size: 0.9rem !important;
            line-height: 1.55 !important;
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.35) !important;
            margin-bottom: 0.85rem !important;
        }}

        .bdd-block {{
            font-family: 'JetBrains Mono', monospace !important;
            background: var(--glass-strong) !important;
            backdrop-filter: {backdrop_filter} !important;
            border: 1px solid rgba(155, 92, 255, 0.35) !important;
            border-left: 4px solid var(--pos) !important;
            border-radius: 12px !important;
            padding: 1.1rem 1.4rem !important;
            color: #F1F5F9 !important;
            font-size: 0.88rem !important;
            line-height: 1.65 !important;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.45) !important;
            margin-bottom: 1rem !important;
        }}

        .aspect-card {{
            background: var(--glass) !important;
            backdrop-filter: {backdrop_filter} !important;
            -webkit-backdrop-filter: {backdrop_filter} !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius) !important;
            padding: 1.15rem 0.95rem !important;
            text-align: center !important;
            box-shadow: 0 8px 24px rgba(0,0,0,0.45) !important;
            transition: transform 0.2s ease, border-color 0.2s ease !important;
        }}
        .aspect-card:hover {{
            transform: translateY(-2px) !important;
            border-color: var(--border-hover) !important;
        }}
        .aspect-card .aspect-label {{
            font-size: 0.78rem !important;
            font-weight: 800 !important;
            color: #C4B5FD !important;
            text-transform: uppercase !important;
            letter-spacing: 0.6px !important;
        }}
        .aspect-card .aspect-val {{
            font-size: 1.55rem !important;
            font-weight: 800 !important;
            margin: 0.35rem 0 !important;
            font-variant-numeric: tabular-nums !important;
        }}
        .aspect-card .aspect-sub {{
            font-size: 0.74rem !important;
            color: var(--muted) !important;
        }}

        .strength-card {{
            background: rgba(16, 185, 129, 0.12) !important;
            backdrop-filter: {backdrop_filter} !important;
            border: 1px solid rgba(16, 185, 129, 0.4) !important;
            border-left: 4px solid var(--pos) !important;
            border-radius: 12px !important;
            padding: 0.95rem 1.1rem !important;
            margin-bottom: 0.75rem !important;
            box-shadow: 0 6px 18px rgba(0,0,0,0.35) !important;
        }}
        .strength-card .strength-title {{
            font-size: 0.9rem !important;
            font-weight: 800 !important;
            color: var(--pos) !important;
            margin-bottom: 0.3rem !important;
        }}
        .strength-card .strength-quote {{
            font-size: 0.85rem !important;
            color: #F1F5F9 !important;
            font-style: italic !important;
            line-height: 1.55 !important;
        }}

        .painpoint-card {{
            background: rgba(239, 68, 68, 0.12) !important;
            backdrop-filter: {backdrop_filter} !important;
            border: 1px solid rgba(239, 68, 68, 0.4) !important;
            border-left: 4px solid var(--neg) !important;
            border-radius: 12px !important;
            padding: 0.95rem 1.1rem !important;
            margin-bottom: 0.75rem !important;
            box-shadow: 0 6px 18px rgba(0,0,0,0.35) !important;
        }}
        .painpoint-card .painpoint-title {{
            font-size: 0.9rem !important;
            font-weight: 800 !important;
            color: var(--neg) !important;
            margin-bottom: 0.3rem !important;
        }}
        .painpoint-card .painpoint-quote {{
            font-size: 0.85rem !important;
            color: #F1F5F9 !important;
            font-style: italic !important;
            line-height: 1.55 !important;
        }}

        /* ====================================================
           SIDEBAR GLASS PANEL & NAVIGATION PILLS
           ==================================================== */
        div[data-testid="stSidebarNav"],
        ul[data-testid="stSidebarNavItems"],
        div[data-testid="stSidebarNavSeparator"],
        [data-testid="stSidebarNav"] *,
        nav[data-testid="stSidebarNav"] {{
            display: none !important;
        }}

        section[data-testid="stSidebar"] {{
            background: rgba(13, 10, 36, 0.78) !important;
            backdrop-filter: {backdrop_filter} !important;
            -webkit-backdrop-filter: {backdrop_filter} !important;
            border-right: 1px solid var(--border) !important;
            box-shadow: 10px 0 35px rgba(0, 0, 0, 0.6) !important;
            transition: width 0.3s ease !important;
        }}

        section[data-testid="stSidebar"] .block-container {{
            padding-top: 1.2rem !important;
            padding-bottom: 2rem !important;
        }}

        div[data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child {{
            display: none !important;
        }}
        div[data-testid="stRadio"] div[role="radiogroup"] > label {{
            background: rgba(255, 255, 255, 0.03) !important;
            border: 1px solid var(--border) !important;
            border-radius: 12px !important;
            padding: 0.58rem 0.85rem !important;
            margin-bottom: 0.35rem !important;
            color: var(--text) !important;
            font-weight: 600 !important;
            font-size: 0.86rem !important;
            transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
            cursor: pointer !important;
            display: flex !important;
            align-items: center !important;
            min-height: 44px !important;
            position: relative;
        }}
        div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {{
            background: rgba(113, 56, 245, 0.2) !important;
            border-color: var(--border-hover) !important;
            color: #FFFFFF !important;
            transform: translateX(3px) !important;
            box-shadow: 0 4px 16px rgba(113, 56, 245, 0.25) !important;
        }}
        div[data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"],
        div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {{
            background: linear-gradient(135deg, rgba(113, 56, 245, 0.45) 0%, rgba(155, 92, 255, 0.28) 100%) !important;
            border: 1px solid var(--primary-2) !important;
            border-left: 4px solid var(--accent) !important;
            color: #FFFFFF !important;
            font-weight: 700 !important;
            box-shadow: 0 4px 18px rgba(155, 92, 255, 0.35) !important;
        }}

        /* ====================================================
           STREAMLIT FORM CONTROLS & BUTTONS
           ==================================================== */
        div[data-testid="stTextInput"] input,
        div[data-testid="stTextArea"] textarea,
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
            background: rgba(13, 10, 36, 0.72) !important;
            border: 1px solid var(--border) !important;
            border-radius: 12px !important;
            color: #FFFFFF !important;
            font-family: inherit !important;
            min-height: 44px !important;
            transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
        }}
        div[data-testid="stTextInput"] input:focus,
        div[data-testid="stTextArea"] textarea:focus,
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within {{
            border-color: var(--primary-2) !important;
            box-shadow: 0 0 15px rgba(155, 92, 255, 0.35) !important;
        }}

        /* Buttons: tap targets >= 44px */
        .stButton > button,
        div[data-testid="stButton"] button,
        div[data-testid="stDownloadButton"] button {{
            min-height: 44px !important;
            border-radius: 12px !important;
            font-weight: 700 !important;
            font-size: 0.92rem !important;
            padding: 0.65rem 1.25rem !important;
            transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
        }}
        div[data-testid="stButton"] button[kind="primary"] {{
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-2) 100%) !important;
            border: none !important;
            color: #FFFFFF !important;
            box-shadow: 0 6px 20px rgba(113, 56, 245, 0.4) !important;
        }}
        div[data-testid="stButton"] button[kind="primary"]:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 10px 28px rgba(155, 92, 255, 0.55) !important;
        }}
        div[data-testid="stButton"] button[kind="secondary"],
        div[data-testid="stButton"] button:not([kind="primary"]),
        div[data-testid="stDownloadButton"] button {{
            background: var(--glass) !important;
            border: 1px solid var(--border) !important;
            color: #FFFFFF !important;
        }}
        div[data-testid="stButton"] button[kind="secondary"]:hover,
        div[data-testid="stButton"] button:not([kind="primary"]):hover,
        div[data-testid="stDownloadButton"] button:hover {{
            background: rgba(113, 56, 245, 0.28) !important;
            border-color: var(--border-hover) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(113, 56, 245, 0.3) !important;
        }}

        /* Streamlit Alerts styled as frosted tiles */
        div[data-testid="stAlert"] {{
            background: var(--glass-strong) !important;
            backdrop-filter: {backdrop_filter} !important;
            -webkit-backdrop-filter: {backdrop_filter} !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius) !important;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4) !important;
            color: var(--text) !important;
        }}

        /* Expander styling */
        details {{
            background: var(--glass) !important;
            border: 1px solid var(--border) !important;
            border-radius: 12px !important;
            padding: 0.4rem 0.8rem !important;
            margin-bottom: 0.6rem !important;
        }}
        details summary {{
            color: var(--text) !important;
            font-weight: 700 !important;
        }}

        /* ====================================================
           MOBILE RESPONSIVENESS (<= 768px & 390px)
           ==================================================== */
        @media (max-width: 768px) {{
            .block-container {{
                padding-left: 0.9rem !important;
                padding-right: 0.9rem !important;
                padding-top: 1rem !important;
                max-width: 100% !important;
            }}
            div[data-testid="column"],
            div[data-testid="stColumn"] {{
                width: 100% !important;
                flex: 1 1 100% !important;
                min-width: 100% !important;
                margin-bottom: 0.6rem !important;
            }}
            .rl-kpi-value {{
                font-size: 1.55rem !important;
            }}
            .stButton > button,
            div[data-testid="stButton"] button {{
                width: 100% !important;
            }}
        }}

        @media (max-width: 390px) {{
            .rl-glass, .rl-glass-strong {{
                padding: 1rem !important;
            }}
            html, body {{
                overflow-x: hidden !important;
            }}
        }}

        /* Accessibility: Prefers Reduced Motion */
        @media (prefers-reduced-motion: reduce) {{
            *, *::before, *::after {{
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
                scroll-behavior: auto !important;
            }}
        }}
    </style>
    """


def inject_theme(reduce_effects: Optional[bool] = None) -> None:
    """
    Renders the unified theme CSS into Streamlit.
    Automatically respects session state preference 'rl_reduce_effects'.
    """
    if reduce_effects is None:
        reduce_effects = bool(st.session_state.get("rl_reduce_effects", False))
    css = get_theme_css(reduce_effects=reduce_effects)
    st.markdown(css, unsafe_allow_html=True)


def glass_card(
    html_inner: str,
    span: int = 1,
    extra_class: str = "",
    style: str = ""
) -> str:
    """
    Wraps content inside a frosted glass tile container (.rl-glass).
    Text inside html_inner should be appropriately escaped by callers.
    """
    classes = f"rl-glass {extra_class}".strip()
    custom_style = f"grid-column: span {span}; {style}".strip()
    return f'<div class="{classes}" style="{custom_style}">{html_inner}</div>'


def kpi_tile(
    label: str,
    value: Any,
    delta: Optional[str] = None,
    tone: str = "neutral",
    sublabel: Optional[str] = None
) -> str:
    """
    Builds a high-contrast Bento KPI tile with tabular number formatting and semantic chip.
    Tone: 'pos' (green), 'neg' (red), 'mixed' (amber), 'neutral' (muted).
    """
    esc_label = html.escape(str(label))
    esc_val = html.escape(str(value))

    tone_clean = str(tone).lower().strip()
    if tone_clean in ("pos", "positive", "good", "pass"):
        chip_cls = "rl-chip-pos"
        chip_icon = "▲ "
    elif tone_clean in ("neg", "negative", "bad", "spam"):
        chip_cls = "rl-chip-neg"
        chip_icon = "▼ "
    elif tone_clean in ("mixed", "warning", "amber", "neutral_tone"):
        chip_cls = "rl-chip-mixed"
        chip_icon = "● "
    else:
        chip_cls = "rl-chip-neutral"
        chip_icon = ""

    delta_html = ""
    if delta is not None:
        esc_delta = html.escape(str(delta))
        delta_html = f'<span class="rl-chip {chip_cls}">{chip_icon}{esc_delta}</span>'

    sub_html = ""
    if sublabel is not None:
        esc_sub = html.escape(str(sublabel))
        sub_html = f'<div class="rl-kpi-sub">{esc_sub}</div>'

    html_content = f"""
    <div class="rl-glass rl-kpi">
        <div>
            <div class="rl-kpi-label">{esc_label}</div>
            <div class="rl-kpi-value">{esc_val}</div>
        </div>
        <div style="display: flex; align-items: center; justify-content: space-between; gap: 0.5rem; margin-top: 0.5rem;">
            {sub_html}
            {delta_html}
        </div>
    </div>
    """
    return html_content.strip()


def apply_glass_layout(fig: Any) -> Any:
    """
    Styles a Plotly figure to harmonize with the glassmorphism dark purple identity:
    - Transparent paper and plot background
    - Font family 'Plus Jakarta Sans' with #F1F5F9 font color
    - Subtle gridlines rgba(255, 255, 255, 0.08)
    - ReviewLens colorway [#9B5CFF, #22D3EE, #34D399, #FBBF24, #F87171, #F472B6]
    - Dark frosted hover labels with purple border
    """
    if fig is None:
        return fig

    try:
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            colorway=CHART_COLOURWAY,
            font=dict(
                family="'Plus Jakarta Sans', sans-serif",
                color="#F1F5F9",
                size=12
            ),
            margin=dict(l=24, r=24, t=36, b=24),
            legend=dict(
                bgcolor="rgba(22, 17, 46, 0.65)",
                bordercolor="rgba(255, 255, 255, 0.12)",
                borderwidth=1,
                font=dict(family="'Plus Jakarta Sans', sans-serif", color="#F1F5F9", size=11)
            ),
            hoverlabel=dict(
                bgcolor="rgba(18, 14, 38, 0.95)",
                bordercolor="rgba(155, 92, 255, 0.5)",
                font=dict(family="'Plus Jakarta Sans', sans-serif", color="#FFFFFF", size=12)
            )
        )

        fig.update_xaxes(
            gridcolor="rgba(255, 255, 255, 0.08)",
            zerolinecolor="rgba(255, 255, 255, 0.15)",
            tickfont=dict(color="#A5A3C2", size=11)
        )
        fig.update_yaxes(
            gridcolor="rgba(255, 255, 255, 0.08)",
            zerolinecolor="rgba(255, 255, 255, 0.15)",
            tickfont=dict(color="#A5A3C2", size=11)
        )
    except Exception:
        pass

    return fig
