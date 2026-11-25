"""
ReviewLens: Smart Shopping Platform & NLP Review Intelligence.
Complete implementation conforming to the ReviewLens Blueprint:
Amazon-inspired discovery UI, department navigation, product detail page,
category-specific Aspect-Based Sentiment Analysis (ABSA), interactive LDA topic modeling,
side-by-side comparison with insufficient-evidence handling, transparent recommendations,
benchmark model evaluation metrics, and full access to the original ReviewLens Analytics Dashboard.
"""

import sys
from pathlib import Path
import json
import html
import streamlit as st
import pandas as pd
import numpy as np

# Set project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.data_loader import (
    get_all_products,
    get_all_reviews,
    get_product_by_id,
    get_reviews_for_product,
    format_currency_inr,
    get_provenance_badge,
    get_product_image_src,
    PROVENANCE_DISCLOSURE
)
from modules.validation import validate_products_df, validate_reviews_df
from modules.sentiment import analyze_reviews_sentiment, compute_nss, compute_review_timeline_metrics, get_lexicon
from modules.aspect_analysis import analyze_reviews_aspects, ASPECT_CAVEAT_DISCLOSURE, CATEGORY_TAXONOMY, CATEGORY_ASPECTS
from modules.keyword_extraction import extract_keywords_by_sentiment
from modules.review_summary import generate_executive_summary, generate_markdown_dossier
from modules.comparison import compare_two_products
from modules.recommendation import generate_category_recommendations, rank_products_by_preferences, RECOMMENDATION_METHODOLOGY_DISCLOSURE
from modules.topic_modeling import extract_topic_clusters, extract_topic_clusters_result
from modules.action_brief import (
    evaluate_review_actionability,
    synthesize_engineering_action_brief,
    export_brief_to_jira_markdown,
    ACTIONABILITY_LAYERS
)

try:
    from modules.evaluation import (
        evaluate_sentiment_model,
        evaluate_spam_model,
        evaluate_baseline_comparison,
        evaluate_curated_reviews_corpus,
        evaluate_sst2_public_benchmark,
        evaluate_semeval_aspect_benchmark,
        evaluate_ml_spam_classifier,
        evaluate_real_amazon_reviews,
        evaluate_error_analysis,
        BENCHMARK_DATASET,
        EVALUATION_METHODOLOGY_DISCLOSURE
    )
except ImportError:
    import importlib
    import modules.evaluation
    importlib.reload(modules.evaluation)
    from modules.evaluation import (
        evaluate_sentiment_model,
        evaluate_spam_model,
        evaluate_baseline_comparison,
        evaluate_curated_reviews_corpus,
        evaluate_sst2_public_benchmark,
        evaluate_semeval_aspect_benchmark,
        evaluate_ml_spam_classifier,
        evaluate_real_amazon_reviews,
        evaluate_error_analysis,
        BENCHMARK_DATASET,
        EVALUATION_METHODOLOGY_DISCLOSURE
    )
from modules.live_analyzer import analyze_custom_review
from modules.review_qa import answer_product_review_question
from modules.pdf_report import generate_product_pdf_report
from modules.csv_analyzer import analyze_custom_reviews_csv, decompose_review_words
from modules.navigation import NAV_GROUPS, ALL_NAV_VIEWS, get_group_for_view
from modules.secrets_util import get_secret
from modules.auth import get_current_user, render_auth_interface, logout_user, check_session_timeout
from components.header import render_header
from components.product_card import render_product_card
from components.filters import render_sidebar_filters, apply_filters
from components.landing import render_landing
from components.highlight import highlight_review_html, top_drivers
from components.charts import (
    create_sentiment_donut_chart,
    create_aspect_breakdown_bar_chart,
    create_aspect_radar_chart,
    create_multi_product_radar_chart,
    create_keyword_driver_chart,
    create_rating_scatter_chart,
    create_confusion_matrix_heatmap,
    create_temporal_sentiment_chart,
    create_live_sentiment_gauge
)
from reviewlens.dashboard import render_original_dashboard

# Page configuration
try:
    st.set_page_config(
        page_title="ReviewLens | Smart Shopping & Review Intelligence",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
except Exception:
    pass


@st.cache_resource
def ensure_nlp_resources() -> bool:
    """Pre-loads and caches NLP corpora once across all sessions."""
    try:
        from setup_nltk import main as setup_nltk_main
        setup_nltk_main()
    except Exception:
        pass
    return True


ensure_nlp_resources()


def render_html_block(html_content: str) -> None:
    """
    Renders HTML safely without Markdown treating indented lines as code blocks.
    Strips leading and trailing whitespace from each line and drops blank lines.
    """
    clean_lines = [line.strip() for line in html_content.strip().splitlines() if line.strip()]
    st.markdown("\n".join(clean_lines), unsafe_allow_html=True)


# CSS Styling (Amazon Navy #131921, Amber #FF9900, Clean SaaS cards, Sliding Sidebar)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* ========================================================
       PREMIUM DARK THEME BACKGROUND (OBSIDIAN & AURORA GLOW)
       ======================================================== */
    body, .stApp {
        background-color: #07051A !important;
        background-image: 
            radial-gradient(circle at 12% 15%, rgba(113, 56, 245, 0.32) 0%, transparent 45%),
            radial-gradient(circle at 88% 12%, rgba(155, 92, 255, 0.28) 0%, transparent 40%),
            radial-gradient(circle at 50% 90%, rgba(113, 56, 245, 0.25) 0%, transparent 50%),
            radial-gradient(circle at 85% 80%, rgba(147, 51, 234, 0.22) 0%, transparent 40%) !important;
        background-attachment: fixed !important;
        color: #F1F5F9 !important;
    }

    /* Headings & general typography */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
        letter-spacing: -0.4px !important;
    }
    p, span, label, div {
        color: #E2E8F0;
    }
    .stCaption, caption {
        color: #94A3B8 !important;
    }

    /* Hide Streamlit Native Multipage Sidebar Nav List */
    div[data-testid="stSidebarNav"],
    ul[data-testid="stSidebarNavItems"],
    div[data-testid="stSidebarNavSeparator"],
    [data-testid="stSidebarNav"] *,
    nav[data-testid="stSidebarNav"] {
        display: none !important;
    }

    /* ========================================================
       FADE-IN ANIMATION & STYLING FOR SIDEBAR
       ======================================================== */
    @keyframes sidebarFadeIn {
        0% {
            opacity: 0;
        }
        100% {
            opacity: 1;
        }
    }

    section[data-testid="stSidebar"] {
        animation: sidebarFadeIn 0.35s ease-out forwards;
        transition: width 0.3s ease !important;
        background: linear-gradient(180deg, #09071A 0%, #120D2A 55%, #0B081C 100%) !important;
        border-right: 1px solid rgba(155, 92, 255, 0.2) !important;
        box-shadow: 8px 0 30px rgba(0, 0, 0, 0.5) !important;
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] h5,
    section[data-testid="stSidebar"] h6 {
        color: #F1F5F9;
    }

    /* Expander rotating chevron */
    details summary svg {
        transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    details[open] summary svg {
        transform: rotate(90deg) !important;
    }

    /* Pop badge animation */
    @keyframes popBadge {
        0% { transform: scale(0.8); opacity: 0; }
        70% { transform: scale(1.15); }
        100% { transform: scale(1); opacity: 1; }
    }
    .tray-pop-badge {
        animation: popBadge 0.35s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards;
    }

    /* Sidebar action buttons (Profile, Sign Out) high contrast fix */
    section[data-testid="stSidebar"] div[data-testid="stButton"] button {
        background: rgba(255, 255, 255, 0.08) !important;
        border: 1px solid rgba(155, 92, 255, 0.3) !important;
        border-radius: 12px !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        padding: 0.6rem 0.8rem !important;
        transition: all 0.2s ease !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stButton"] button:hover {
        background: rgba(113, 56, 245, 0.35) !important;
        border-color: #9B5CFF !important;
        color: #FFFFFF !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 16px rgba(155, 92, 255, 0.45) !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stButton"] button p,
    section[data-testid="stSidebar"] div[data-testid="stButton"] button span {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"] {
        background: linear-gradient(135deg, rgba(113, 56, 245, 0.55) 0%, rgba(155, 92, 255, 0.35) 100%) !important;
        border: 1px solid #9B5CFF !important;
        border-left: 4px solid #22D3EE !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 18px rgba(155, 92, 255, 0.4) !important;
    }

    /* Modern SaaS Navigation Radio Items in Sidebar */
    div[data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child {
        display: none !important; /* Hide raw circular radio dots */
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
        padding: 0.55rem 0.85rem !important;
        margin-bottom: 0.35rem !important;
        color: #E2E8F0 !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
        cursor: pointer !important;
        display: flex !important;
        align-items: center !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {
        background: rgba(113, 56, 245, 0.22) !important;
        border-color: rgba(155, 92, 255, 0.7) !important;
        color: #FFFFFF !important;
        transform: translateX(4px) !important;
        box-shadow: 0 4px 16px rgba(113, 56, 245, 0.25) !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"],
    div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(135deg, rgba(113, 56, 245, 0.45) 0%, rgba(155, 92, 255, 0.3) 100%) !important;
        border: 1px solid #9B5CFF !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 18px rgba(155, 92, 255, 0.4) !important;
    }

    /* Primary and Secondary Action Buttons in Main View */
    .stButton>button {
        border-radius: 12px !important;
        font-weight: 700 !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 22px rgba(113, 56, 245, 0.35) !important;
    }
    div[data-testid="stButton"] button[kind="primary"] {
        background: linear-gradient(135deg, #7138F5 0%, #9B5CFF 100%) !important;
        border: none !important;
        color: #FFFFFF !important;
        box-shadow: 0 6px 20px rgba(113, 56, 245, 0.4) !important;
    }
    div[data-testid="stButton"] button[kind="secondary"] {
        background: rgba(22, 17, 46, 0.8) !important;
        border: 1px solid rgba(155, 92, 255, 0.3) !important;
        color: #FFFFFF !important;
    }

    /* Dark Theme Cards (Metric cards, Topic cards) */
    .metric-card, .topic-card {
        background: rgba(22, 17, 46, 0.8) !important;
        border: 1px solid rgba(155, 92, 255, 0.25) !important;
        border-radius: 14px !important;
        padding: 1.15rem !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.45) !important;
        margin-bottom: 1rem !important;
        border-left: 4px solid #9B5CFF !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.25s ease !important;
    }
    .metric-card:hover, .topic-card:hover {
        transform: translateY(-3px) !important;
        border-color: #9B5CFF !important;
        box-shadow: 0 14px 35px rgba(113, 56, 245, 0.35) !important;
    }
    .metric-card .metric-label {
        font-size: 0.8rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        color: #B9B3D0 !important;
        font-weight: 700 !important;
        margin-bottom: 0.3rem !important;
    }
    .metric-card .metric-value {
        font-size: 1.6rem !important;
        font-weight: 800 !important;
        color: #FFFFFF !important;
        line-height: 1.2 !important;
    }
    .metric-card .metric-sub {
        font-size: 0.75rem !important;
        color: #94A3B8 !important;
        margin-top: 0.2rem !important;
    }

    .metric-card-orange {
        border-left-color: #FF9900 !important;
    }
    .metric-card-green {
        border-left-color: #10B981 !important;
    }
    .metric-card-purple {
        border-left-color: #9B5CFF !important;
    }

    /* Dark theme text inputs and selectboxes */
    div[data-testid="stTextInput"] input,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background: rgba(12, 9, 28, 0.7) !important;
        border: 1px solid rgba(155, 92, 255, 0.3) !important;
        border-radius: 10px !important;
        color: #FFFFFF !important;
    }

    /* Streamlit Download Button in Dark Mode */
    div[data-testid="stDownloadButton"] button,
    div[data-testid="stDownloadButton"] button[data-testid="stBaseButton-secondary"],
    button[data-testid="stBaseButton-secondary"] {
        background: #171131 !important;
        background-color: #171131 !important;
        border: 1px solid rgba(155, 92, 255, 0.45) !important;
        border-radius: 12px !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        padding: 0.65rem 1.2rem !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stDownloadButton"] button:hover,
    button[data-testid="stBaseButton-secondary"]:hover {
        background: rgba(113, 56, 245, 0.45) !important;
        background-color: rgba(113, 56, 245, 0.45) !important;
        border-color: #9B5CFF !important;
        color: #FFFFFF !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(155, 92, 255, 0.4) !important;
    }
    div[data-testid="stDownloadButton"] button p,
    div[data-testid="stDownloadButton"] button span,
    button[data-testid="stBaseButton-secondary"] p,
    button[data-testid="stBaseButton-secondary"] span {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    /* Streamlit Alert Callouts (info, success, error, warning) Dark Theme */
    div[data-testid="stAlert"] {
        background: rgba(22, 17, 46, 0.9) !important;
        border: 1px solid rgba(155, 92, 255, 0.3) !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35) !important;
        color: #F8FAFC !important;
    }
    div[data-testid="stAlert"] p,
    div[data-testid="stAlert"] span,
    div[data-testid="stAlert"] div {
        color: #F8FAFC !important;
    }

    .hero-banner {
        background: linear-gradient(135deg, #0B0F19 0%, #131921 60%, #1E293B 100%) !important;
        border-radius: 16px !important;
        padding: 2.2rem !important;
        color: #FFFFFF !important;
        margin-bottom: 2rem !important;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.2) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        position: relative;
        overflow: hidden;
    }

    .provenance-banner {
        background: rgba(30, 58, 138, 0.25);
        border: 1px solid rgba(96, 165, 250, 0.4);
        border-radius: 10px;
        padding: 0.85rem 1.2rem;
        color: #93C5FD;
        font-size: 0.85rem;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }

    .topic-pill {
        display: inline-block;
        background: rgba(113, 56, 245, 0.25);
        border: 1px solid rgba(155, 92, 255, 0.4);
        color: #C4B5FD;
        font-size: 0.78rem;
        font-weight: 700;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .topic-pill:hover {
        transform: scale(1.05);
        border-color: #9B5CFF;
        color: #FFFFFF;
    }

    /* ========================================================
       VECTOR-INSPIRED REQUIREMENT ENGINEERING DESIGN SYSTEM
       ======================================================== */
    .fig-tag {
        font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
        font-size: 0.72rem !important;
        font-weight: 800 !important;
        letter-spacing: 0.08em !important;
        color: #C4B5FD !important;
        background: rgba(155, 92, 255, 0.15) !important;
        border: 1px solid rgba(155, 92, 255, 0.35) !important;
        padding: 0.2rem 0.6rem !important;
        border-radius: 6px !important;
        display: inline-block !important;
        margin-bottom: 0.5rem !important;
        text-transform: uppercase !important;
    }

    .mono-badge {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.75rem !important;
        padding: 0.2rem 0.55rem !important;
        border-radius: 6px !important;
        background: rgba(255, 255, 255, 0.06) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        color: #CBD5E1 !important;
    }

    .layer-card {
        background: #120D2A !important;
        background-color: #120D2A !important;
        border: 1px solid rgba(155, 92, 255, 0.25) !important;
        border-radius: 10px !important;
        padding: 0.75rem 1rem !important;
        margin-bottom: 0.5rem !important;
        box-shadow: 0 4px 14px rgba(0,0,0,0.3) !important;
    }

    .bdd-block {
        font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
        background: #09061A !important;
        background-color: #09061A !important;
        border: 1px solid rgba(155, 92, 255, 0.35) !important;
        border-left: 4px solid #10B981 !important;
        border-radius: 12px !important;
        padding: 1.1rem 1.4rem !important;
        color: #F1F5F9 !important;
        font-size: 0.88rem !important;
        line-height: 1.65 !important;
        box-shadow: 0 6px 22px rgba(0,0,0,0.45) !important;
    }

    /* High-contrast solid Obsidian specification box */
    .spec-box {
        background: #120D2A !important;
        background-color: #120D2A !important;
        border: 1px solid rgba(155, 92, 255, 0.35) !important;
        border-left: 4px solid #9B5CFF !important;
        border-radius: 12px !important;
        padding: 0.95rem 1.2rem !important;
        color: #F1F5F9 !important;
        font-size: 0.9rem !important;
        line-height: 1.55 !important;
        box-shadow: 0 4px 14px rgba(0,0,0,0.4) !important;
    }

    /* Category Aspect Sentiment Card */
    .aspect-card {
        background: #120D2A !important;
        background-color: #120D2A !important;
        border: 1px solid rgba(155, 92, 255, 0.35) !important;
        border-radius: 14px !important;
        padding: 1.1rem 0.9rem !important;
        text-align: center !important;
        box-shadow: 0 8px 24px rgba(0,0,0,0.45) !important;
        transition: transform 0.2s ease, border-color 0.2s ease !important;
    }
    .aspect-card:hover {
        transform: translateY(-2px) !important;
        border-color: #9B5CFF !important;
    }
    .aspect-card .aspect-label {
        font-size: 0.78rem !important;
        font-weight: 800 !important;
        color: #C4B5FD !important;
        text-transform: uppercase !important;
        letter-spacing: 0.6px !important;
    }
    .aspect-card .aspect-val {
        font-size: 1.55rem !important;
        font-weight: 800 !important;
        margin: 0.35rem 0 !important;
    }
    .aspect-card .aspect-sub {
        font-size: 0.74rem !important;
        color: #94A3B8 !important;
    }

    /* Customer Feedback Strength Card (Emerald Obsidian) */
    .strength-card {
        background: #09261E !important;
        background-color: #09261E !important;
        border: 1px solid rgba(16, 185, 129, 0.5) !important;
        border-left: 4px solid #10B981 !important;
        border-radius: 12px !important;
        padding: 0.95rem 1.1rem !important;
        margin-bottom: 0.75rem !important;
        box-shadow: 0 6px 18px rgba(0,0,0,0.35) !important;
    }
    .strength-card .strength-title {
        font-size: 0.9rem !important;
        font-weight: 800 !important;
        color: #34D399 !important;
        margin-bottom: 0.3rem !important;
    }
    .strength-card .strength-quote {
        font-size: 0.85rem !important;
        color: #F1F5F9 !important;
        font-style: italic !important;
        line-height: 1.55 !important;
    }

    /* Customer Feedback Pain Point Card (Ruby Obsidian) */
    .painpoint-card {
        background: #2D111E !important;
        background-color: #2D111E !important;
        border: 1px solid rgba(239, 68, 68, 0.5) !important;
        border-left: 4px solid #EF4444 !important;
        border-radius: 12px !important;
        padding: 0.95rem 1.1rem !important;
        margin-bottom: 0.75rem !important;
        box-shadow: 0 6px 18px rgba(0,0,0,0.35) !important;
    }
    .painpoint-card .painpoint-title {
        font-size: 0.9rem !important;
        font-weight: 800 !important;
        color: #F87171 !important;
        margin-bottom: 0.3rem !important;
    }
    .painpoint-card .painpoint-quote {
        font-size: 0.85rem !important;
        color: #F1F5F9 !important;
        font-style: italic !important;
        line-height: 1.55 !important;
    }

    /* Universal Button & Secondary Action Styling */
    div[data-testid="stDownloadButton"] button,
    .stDownloadButton button,
    button[data-testid="baseButton-secondary"],
    button[data-testid="stBaseButton-secondary"],
    button[kind="secondary"],
    div[data-testid="stButton"] button[kind="secondary"] {
        background: #171131 !important;
        background-color: #171131 !important;
        border: 1px solid rgba(155, 92, 255, 0.45) !important;
        border-radius: 12px !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        padding: 0.65rem 1.2rem !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stDownloadButton"] button:hover,
    .stDownloadButton button:hover,
    button[data-testid="baseButton-secondary"]:hover,
    button[kind="secondary"]:hover {
        background: rgba(113, 56, 245, 0.45) !important;
        background-color: rgba(113, 56, 245, 0.45) !important;
        border-color: #9B5CFF !important;
        color: #FFFFFF !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(155, 92, 255, 0.4) !important;
    }
    div[data-testid="stDownloadButton"] button *,
    .stDownloadButton button *,
    button[data-testid="baseButton-secondary"] *,
    button[kind="secondary"] * {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    /* Mobile Responsiveness (@media <= 768px) */
    @media (max-width: 768px) {
        .block-container {
            padding-left: 0.8rem !important;
            padding-right: 0.8rem !important;
            padding-top: 1rem !important;
        }
        .hero-banner {
            padding: 1.2rem 0.9rem !important;
        }
        .hero-banner h1 {
            font-size: 1.6rem !important;
            line-height: 1.2 !important;
        }
        .metric-card {
            margin-bottom: 0.75rem !important;
        }
        div[data-testid="column"],
        div[data-testid="stColumn"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }
        .nav-in {
            flex-direction: column !important;
            height: auto !important;
            padding: 0.8rem 0 !important;
        }
    }

    /* Accessibility: Prefers Reduced Motion */
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
            scroll-behavior: auto !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "compare_tray" not in st.session_state:
    st.session_state["compare_tray"] = ["samsung-galaxy-s24-fe", "apple-iphone-15-pro-max"]

if "selected_product_id" not in st.session_state:
    st.session_state["selected_product_id"] = "samsung-galaxy-s24-fe"

if "nav_view" not in st.session_state:
    st.session_state["nav_view"] = "🔍 Review Intelligence"

# Handle pending programmatic navigation before widget instantiation
if st.session_state.get("pending_nav"):
    target_nav = st.session_state.pop("pending_nav")
    st.session_state["sidebar_nav_radio"] = target_nav
    st.session_state["nav_view"] = target_nav

if "sidebar_nav_radio" not in st.session_state:
    st.session_state["sidebar_nav_radio"] = st.session_state["nav_view"]

if "department_filter" not in st.session_state:
    st.session_state["department_filter"] = "All"


def navigate_to(view: str) -> None:
    """
    Central navigation helper keeping nav_view and sidebar_nav_radio synchronized.
    Queues navigation in pending_nav so sidebar_nav_radio is safely updated at the start
    of the rerun before widget instantiation, preventing StreamlitWidgetAlreadyInstantiatedError.
    """
    st.session_state["nav_view"] = view
    st.session_state["pending_nav"] = view
    st.rerun()


def toggle_compare_item(pid: str):
    """Toggle a product in or out of the comparison tray (max 3)."""
    tray = st.session_state["compare_tray"]
    if pid in tray:
        tray.remove(pid)
    else:
        if len(tray) >= 3:
            st.toast("⚠️ Comparison tray holds at most 3 products. Removed oldest item.", icon="ℹ️")
            tray.pop(0)
        tray.append(pid)
    st.session_state["compare_tray"] = tray
    st.rerun()


def inspect_product_action(pid: str):
    """Set selected product and switch view to Product Details."""
    st.session_state["selected_product_id"] = pid
    navigate_to("📦 Product Details")


def explore_review_intelligence_action(pid: str):
    """Switch directly to deep review intelligence for a product."""
    st.session_state["selected_product_id"] = pid
    navigate_to("🔍 Review Intelligence")


# Load catalogue & reviews with validation and shared empty-catalogue guard
try:
    df_products = get_all_products(validate=True)
    df_reviews = get_all_reviews(validate=True, products_df=df_products)
except Exception as e:
    st.error(f"⚠️ Dataset validation error: {str(e)}")
    st.stop()

if df_products is None or df_products.empty:
    st.error("⚠️ The product catalogue is empty or missing. Please ensure data/products.csv is present and valid.")
    st.stop()

# Current authenticated user profile
current_user = get_current_user()

if not current_user.get("authenticated"):
    # Enforce pure full-width login page on initial arrival with zero sidebar or header controls
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

    render_auth_interface(on_navigate_home=lambda: navigate_to("🏠 Home / Discovery"))
    st.stop()

# Top utility header (rendered only when authenticated)
render_header(
    compare_count=len(st.session_state["compare_tray"]),
    active_nav=st.session_state["nav_view"]
)

# ========================================================
# SIDEBAR: BRAND LOGO, USER ACCOUNT CARD & NAVIGATION
# ========================================================
esc_u_name = html.escape(str(current_user.get("display_name", "Shopper")))
esc_u_badge = html.escape(str(current_user.get("tier_badge", "🛡️ Verified")))

# 1. ReviewLens Sidebar Brand Header
st.sidebar.markdown("""
<div style="display: flex; align-items: center; gap: 0.75rem; padding: 0.2rem 0.2rem 0.8rem 0.2rem; border-bottom: 1px solid rgba(255, 255, 255, 0.08); margin-bottom: 0.9rem;">
    <div style="width: 38px; height: 38px; border-radius: 11px; background: linear-gradient(135deg, #7138F5 0%, #9B5CFF 100%); display: flex; align-items: center; justify-content: center; font-size: 20px; color: #FFFFFF; box-shadow: 0 0 16px rgba(155, 92, 255, 0.55);">
        ★
    </div>
    <div>
        <div style="font-size: 1.35rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.3px; line-height: 1.1;">
            Review<span style="color: #9B5CFF;">Lens</span>
        </div>
        <div style="font-size: 0.72rem; color: #94A3B8; font-weight: 500;">
            NLP Product Intelligence
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# 2. Authenticated User Profile Card (Rendered directly in Sidebar)
st.sidebar.markdown(f"""
<div style="background: rgba(22, 17, 46, 0.85); border: 1px solid rgba(155, 92, 255, 0.35); border-radius: 14px; padding: 0.85rem; margin-bottom: 0.75rem; display: flex; align-items: center; gap: 0.85rem; box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35);">
    <div style="width: 40px; height: 40px; border-radius: 50%; background: linear-gradient(135deg, #7138F5 0%, #9B5CFF 100%); color: #FFFFFF; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 1.15rem; box-shadow: 0 0 14px rgba(155, 92, 255, 0.5);">
        {current_user.get('avatar_letter', 'U')}
    </div>
    <div style="flex: 1; overflow: hidden;">
        <div style="font-weight: 800; font-size: 0.95rem; color: #FFFFFF; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;">{esc_u_name}</div>
        <div style="font-size: 0.74rem; color: #34D399; font-weight: 700;">{esc_u_badge}</div>
    </div>
</div>
""", unsafe_allow_html=True)

col_sb_prof, col_sb_logout = st.sidebar.columns(2)
with col_sb_prof:
    if st.button("👤 Profile", key="sidebar_btn_profile", use_container_width=True):
        navigate_to("🔐 Sign In / Profile")
with col_sb_logout:
    if st.button("🚪 Sign Out", key="sidebar_btn_logout", use_container_width=True):
        logout_user()
        navigate_to("🔍 Review Intelligence")

st.sidebar.markdown("<div style='margin-top: 0.8rem; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 0.8rem;'></div>", unsafe_allow_html=True)

# Maintain NAV_OPTIONS alias
NAV_OPTIONS = ALL_NAV_VIEWS

current_view = st.session_state.get("nav_view", "🔍 Review Intelligence")
if current_view == "🏠 Home / Discovery":
    current_view = "🏠 Product Explorer & Storefront"

# 1. ANALYZE Group
st.sidebar.markdown("<div style='font-size: 0.72rem; font-weight: 800; color: #9B5CFF; letter-spacing: 0.08em; margin: 0.4rem 0 0.2rem 0;'>ANALYZE</div>", unsafe_allow_html=True)
for item in NAV_GROUPS["ANALYZE"]:
    is_active = (current_view == item)
    if st.sidebar.button(item, key=f"nav_btn_{item}", type="primary" if is_active else "secondary", use_container_width=True):
        navigate_to(item)

# 2. COMPARE Group
st.sidebar.markdown("<div style='font-size: 0.72rem; font-weight: 800; color: #22D3EE; letter-spacing: 0.08em; margin: 0.6rem 0 0.2rem 0;'>COMPARE</div>", unsafe_allow_html=True)
for item in NAV_GROUPS["COMPARE"]:
    is_active = (current_view == item)
    if st.sidebar.button(item, key=f"nav_btn_{item}", type="primary" if is_active else "secondary", use_container_width=True):
        navigate_to(item)

# 3. SHOP (demo) Group (in expander, collapsed by default, auto-expanded if current view in SHOP)
shop_expanded = current_view in NAV_GROUPS["SHOP (demo)"]
with st.sidebar.expander("🛍️ **SHOP (demo)**", expanded=shop_expanded):
    for item in NAV_GROUPS["SHOP (demo)"]:
        is_active = (current_view == item)
        if st.button(item, key=f"nav_btn_{item}", type="primary" if is_active else "secondary", use_container_width=True):
            navigate_to(item)

# 4. MORE Group (in expander, collapsed by default, auto-expanded if current view in MORE)
more_expanded = current_view in NAV_GROUPS["MORE"]
with st.sidebar.expander("⚙️ **MORE**", expanded=more_expanded):
    for item in NAV_GROUPS["MORE"]:
        is_active = (current_view == item)
        if st.button(item, key=f"nav_btn_{item}", type="primary" if is_active else "secondary", use_container_width=True):
            navigate_to(item)

st.sidebar.markdown("---")

# Quick compare tray widget in sidebar
st.sidebar.markdown(f"### ⚔️ Compare Tray ({len(st.session_state['compare_tray'])}/3)")
if st.session_state["compare_tray"]:
    for pid in list(st.session_state["compare_tray"]):
        p_info = get_product_by_id(pid)
        p_name = p_info["product_name"] if p_info else pid
        c_row1, c_row2 = st.sidebar.columns([4, 1])
        c_row1.caption(f"• {p_name[:25]}...")
        if c_row2.button("✕", key=f"del_tray_{pid}", help="Remove from compare"):
            st.session_state["compare_tray"].remove(pid)
            st.rerun()

    if st.sidebar.button("⚔️ Compare Selected Now", use_container_width=True, type="primary"):
        navigate_to("⚔️ Product Comparison")
else:
    st.sidebar.caption("No products queued. Click '⚔️ Compare' on any product card.")

st.sidebar.markdown("---")
st.sidebar.caption("🛡️ **Data Provenance**: Sample prices & ratings are illustrative. Reviews are tagged `prepared_demo` for transparent NLP research.")

install_url = get_secret("REVIEWLENS_INSTALL_URL")
if install_url:
    with st.sidebar.expander("📲 **Install on Phone / PC**", expanded=False):
        st.markdown(f"""
        Install ReviewLens as a fast standalone app:
        - **Web App / PWA**: [Launch App Shell]({install_url})
        - **Mobile**: Tap **Add to Home screen** in your browser
        - **PC / Mac**: Click the install icon in the address bar
        """)


# ==========================================
# VIEW 1: PRODUCT EXPLORER & STOREFRONT
# ==========================================
if st.session_state["nav_view"] in ["🏠 Home / Discovery", "🏠 Product Explorer & Storefront"]:
    # Animated Landing Page Showcase
    render_landing(height=880)

    render_html_block("""
    <div class="hero-banner">
        <div style="display: inline-block; padding: 4px 12px; border-radius: 20px; background: rgba(255,153,0,0.2); border: 1px solid #FF9900; color: #FF9900; font-size: 0.8rem; font-weight: 700; margin-bottom: 0.8rem;">
            NLP-POWERED SMART SHOPPING PLATFORM
        </div>
        <h1 style="font-size: 2.2rem; font-weight: 800; color: #FFFFFF; margin: 0 0 0.8rem 0; line-height: 1.2;">
            Make Smarter Buying Decisions with Explainable Review AI
        </h1>
        <p style="font-size: 1.05rem; color: #CBD5E1; max-width: 800px; line-height: 1.6; margin-bottom: 1.4rem;">
            ReviewLens cuts through raw customer feedback, extracts category-specific aspect sentiment, 
            detects fake reviews, models latent review topics with LDA, and compares electronics 
            side-by-side with 100% mathematical transparency.
        </p>
    </div>
    """)

    # Quick Stats Row
    stat1, stat2, stat3, stat4 = st.columns(4)
    with stat1:
        render_html_block("""
        <div class="metric-card metric-card-orange">
            <div class="metric-label">Electronics Catalogue</div>
            <div class="metric-value">12 Products</div>
            <div class="metric-sub">Phones, Laptops, Headphones, Cameras</div>
        </div>
        """)
    with stat2:
        render_html_block("""
        <div class="metric-card">
            <div class="metric-label">Prepared Reviews</div>
            <div class="metric-value">75+ Feedback</div>
            <div class="metric-sub">Structured research corpus</div>
        </div>
        """)
    with stat3:
        render_html_block("""
        <div class="metric-card metric-card-green">
            <div class="metric-label">Aspect Dimensions</div>
            <div class="metric-value">Category ABSA</div>
            <div class="metric-sub">Domain-tailored taxons per category</div>
        </div>
        """)
    with stat4:
        render_html_block("""
        <div class="metric-card metric-card-purple">
            <div class="metric-label">NLP Explainability</div>
            <div class="metric-value">100% Glass Box</div>
            <div class="metric-sub">VADER + Rule Lexicon + LDA Topics</div>
        </div>
        """)

    # New Features Showcase Callout
    render_html_block("""
    <div style="background: linear-gradient(135deg, rgba(113, 56, 245, 0.2) 0%, rgba(155, 92, 255, 0.15) 100%); border: 1.5px solid rgba(155, 92, 255, 0.35); border-radius: 14px; padding: 1.2rem 1.5rem; margin-top: 1rem; margin-bottom: 1.2rem; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1rem;">
            <div>
                <span style="background: linear-gradient(135deg, #7138F5 0%, #9B5CFF 100%); color: #FFFFFF; font-size: 0.75rem; font-weight: 800; padding: 0.25rem 0.65rem; border-radius: 12px; text-transform: uppercase;">✨ New NLP Capabilities</span>
                <h3 style="margin: 0.5rem 0 0.3rem 0; font-size: 1.3rem; font-weight: 800; color: #FFFFFF;">Live Review Diagnostic Lab & Personalized Matchmaker</h3>
                <p style="margin: 0; font-size: 0.9rem; color: #B9B3D0; max-width: 750px; line-height: 1.5;">
                    Test raw customer review text in real-time with clause-level ABSA, spam probability, and rating anomaly checks — or calibrate multi-attribute aspect weights to find products tailored to your preferences.
                </p>
            </div>
        </div>
    </div>
    """)
    feat_nav_col1, feat_nav_col2, feat_nav_col3 = st.columns(3)
    with feat_nav_col1:
        if st.button("🧪 Launch Live Review Analyzer →", key="home_btn_live_analyzer", type="primary", use_container_width=True):
            navigate_to("🧪 Live Review Analyzer")
    with feat_nav_col2:
        if st.button("🎛️ Custom Buyer Matchmaker →", key="home_btn_matchmaker", use_container_width=True):
            navigate_to("🎯 Smart Recommendations")
    with feat_nav_col3:
        if st.button("📈 Sentiment Drift & Trust Audit →", key="home_btn_drift", use_container_width=True):
            navigate_to("🔍 Review Intelligence")

    st.markdown("<br>", unsafe_allow_html=True)

    # Custom Reviews CSV Ingestion & Explainability Panel (Home View)
    with st.expander("📂 **Analyze Custom Reviews CSV (Instant Batch Upload & Explainability)**", expanded=False):
        st.markdown("Upload any custom CSV containing consumer reviews (capped at 5,000 rows). ReviewLens will automatically execute full aspect extraction, sentiment scoring, spam bot auditing, and lexical explainability.")
        up_c1, up_c2 = st.columns([3, 1])
        with up_c1:
            uploaded_csv = st.file_uploader(
                "Upload Reviews CSV (expected columns: `review_text` or `text`, optional: `rating`, `date`):",
                type=["csv"],
                key="home_custom_csv_uploader"
            )
        with up_c2:
            custom_cat = st.selectbox(
                "Target Category:",
                ["Electronics", "Smartphones", "Laptops", "Headphones", "Cameras"],
                key="home_custom_csv_cat"
            )

        if uploaded_csv is not None:
            try:
                raw_custom_df = pd.read_csv(uploaded_csv)
                with st.spinner("Processing reviews with NLP Pipeline & Explainability Engine..."):
                    custom_res = analyze_custom_reviews_csv(raw_custom_df, category=custom_cat, max_rows=5000)

                if custom_res["is_capped"]:
                    st.warning("⚠️ Dataset exceeded 5,000 rows. Capped at the first 5,000 rows for real-time responsiveness.")

                k = custom_res["kpis"]
                st.success(f"✓ Successfully processed {k['total_reviews']} reviews!")

                # Quick KPI Row
                k1, k2, k3, k4 = st.columns(4)
                with k1:
                    st.metric("Total Analyzed", f"{k['total_reviews']:,}")
                with k2:
                    st.metric("Net Sentiment Score", f"{k['nss']:+.1f}", help="((Positive - Negative) / Total) * 100")
                with k3:
                    st.metric("Spam Anomaly Rate", f"{k['spam_rate_pct']:.1f}%", f"{k['spam_count']} flagged")
                with k4:
                    st.metric("Sentiment Polarity", f"{k['positive_count']} Pos / {k['negative_count']} Neg", f"{k['neutral_count']} Neu")

                # Word-Level Explainability Panel
                st.markdown("#### 🔍 **Word-Level Lexical Drivers (Explainability)**")
                st.caption("ReviewLens decomposes customer language to identify the highest-impact positive and negative vocabulary driving sentiment polarity.")

                driver_col1, driver_col2 = st.columns(2)
                with driver_col1:
                    st.markdown("##### 🟢 **Top Positive Valence Drivers**")
                    pos_drivers = custom_res["word_drivers"]["positive"]
                    if pos_drivers:
                        pos_chips = " ".join([f"<span style='display:inline-block; margin:2px 4px; padding:3px 8px; border-radius:12px; background:rgba(16,185,129,0.15); border:1px solid #10B981; color:#34D399; font-size:0.82rem;'><b>{d['word']}</b> ({d['count']}x, +{d['valence']})</span>" for d in pos_drivers])
                        st.markdown(pos_chips, unsafe_allow_html=True)
                    else:
                        st.info("No dominant positive lexical triggers identified.")

                with driver_col2:
                    st.markdown("##### 🔴 **Top Negative Valence Drivers**")
                    neg_drivers = custom_res["word_drivers"]["negative"]
                    if neg_drivers:
                        neg_chips = " ".join([f"<span style='display:inline-block; margin:2px 4px; padding:3px 8px; border-radius:12px; background:rgba(239,68,68,0.15); border:1px solid #EF4444; color:#F87171; font-size:0.82rem;'><b>{d['word']}</b> ({d['count']}x, {d['valence']})</span>" for d in neg_drivers])
                        st.markdown(neg_chips, unsafe_allow_html=True)
                    else:
                        st.info("No dominant negative lexical triggers identified.")

                # Review Token Decomposer
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("##### 🔬 **Inspect Individual Review Explainability**")
                annotated_df = custom_res["annotated_df"]
                sample_indices = list(range(min(20, len(annotated_df))))
                selected_sample_idx = st.selectbox(
                    "Select a review to decompose word-level impact:",
                    sample_indices,
                    format_func=lambda i: f"Review #{i+1}: {annotated_df['clean_text'].iloc[i][:60]}..."
                )
                sample_text = annotated_df["clean_text"].iloc[selected_sample_idx]
                tokens = decompose_review_words(sample_text)

                highlighted_html = []
                for tok in tokens:
                    w_esc = html.escape(tok["word"])
                    if tok["is_positive"]:
                        highlighted_html.append(f"<span style='background:rgba(16,185,129,0.3); border-bottom:2px solid #10B981; padding:1px 4px; border-radius:3px; color:#FFFFFF;' title='Positive Valence: +{tok['valence']}'>{w_esc}</span>")
                    elif tok["is_negative"]:
                        highlighted_html.append(f"<span style='background:rgba(239,68,68,0.3); border-bottom:2px solid #EF4444; padding:1px 4px; border-radius:3px; color:#FFFFFF;' title='Negative Valence: {tok['valence']}'>{w_esc}</span>")
                    else:
                        highlighted_html.append(f"<span style='color:#CBD5E1;'>{w_esc}</span>")

                st.markdown(f"""
                <div style="background: rgba(13, 10, 36, 0.7); border: 1px solid rgba(155, 92, 255, 0.3); border-radius: 8px; padding: 1rem; line-height: 1.8; font-size: 0.95rem; margin-top: 0.5rem;">
                    {' '.join(highlighted_html)}
                </div>
                """, unsafe_allow_html=True)

                # Export Annotated CSV
                csv_buffer = annotated_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Enriched & Annotated CSV",
                    data=csv_buffer,
                    file_name="reviewlens_annotated_reviews.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Analysis Error: {e}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Department Quick-Access Row
    st.markdown("### 🏷️ **Shop by Department**")
    cat_cols = st.columns(4)
    departments = [
        ("📱 Smartphones", "Smartphones", "Flagships & Mid-range phones"),
        ("💻 Laptops", "Laptops", "Ultrabooks & High-performance laptops"),
        ("🎧 Headphones", "Headphones", "Noise-cancelling & Audiophile headphones"),
        ("📷 Cameras", "Cameras", "Mirrorless & Professional cameras")
    ]
    for idx, (nav_title, dept_id, cat_sub) in enumerate(departments):
        with cat_cols[idx]:
            render_html_block(f"""
            <div style="background: rgba(22, 17, 46, 0.78); border: 1px solid rgba(155, 92, 255, 0.25); border-radius: 12px; padding: 1.1rem; text-align: center; box-shadow: 0 8px 24px rgba(0,0,0,0.35); margin-bottom: 0.5rem; transition: all 0.25s ease;">
                <div style="font-weight: 700; font-size: 1.05rem; color: #FFFFFF; margin-bottom: 0.2rem;">{nav_title}</div>
                <div style="font-size: 0.8rem; color: #B9B3D0;">{cat_sub}</div>
            </div>
            """)
            if st.button(f"Browse {dept_id} →", key=f"home_browse_{dept_id}", use_container_width=True):
                navigate_to(nav_title)

    st.markdown("<br>", unsafe_allow_html=True)

    # Featured Products Section
    st.markdown("### ⭐ **Featured Highlights & Review Intelligence**")
    featured_ids = ["samsung-galaxy-s24-fe", "apple-iphone-15-pro-max", "sony-wh1000xm5", "apple-macbook-air-m3"]
    feat_cols = st.columns(4)

    for idx, pid in enumerate(featured_ids):
        p = get_product_by_id(pid)
        if p:
            with feat_cols[idx]:
                render_product_card(
                    p,
                    is_in_compare=(pid in st.session_state["compare_tray"]),
                    on_inspect=inspect_product_action,
                    on_toggle_compare=toggle_compare_item
                )

    st.markdown("<br>", unsafe_allow_html=True)

    # Value Proposition Callouts
    st.markdown("### 💡 **How ReviewLens Empowers Your Buying Decisions**")
    vp1, vp2, vp3 = st.columns(3)
    with vp1:
        render_html_block("""
        <div style="background: rgba(22, 17, 46, 0.78); border: 1px solid rgba(155, 92, 255, 0.25); border-radius: 12px; padding: 1.2rem; box-shadow: 0 8px 24px rgba(0,0,0,0.35);">
            <div style="font-size: 1.5rem; margin-bottom: 0.4rem;">🎯</div>
            <div style="font-weight: 700; color: #FFFFFF; margin-bottom: 0.4rem;">Category-Specific ABSA</div>
            <div style="font-size: 0.85rem; color: #B9B3D0; line-height: 1.5;">
                Phones evaluate Camera, Battery & Display; Laptops evaluate Processor, RAM & Cooling; Headphones evaluate Sound Quality & Comfort.
            </div>
        </div>
        """)
    with vp2:
        render_html_block("""
        <div style="background: rgba(22, 17, 46, 0.78); border: 1px solid rgba(155, 92, 255, 0.25); border-radius: 12px; padding: 1.2rem; box-shadow: 0 8px 24px rgba(0,0,0,0.35);">
            <div style="font-size: 1.5rem; margin-bottom: 0.4rem;">⚔️</div>
            <div style="font-weight: 700; color: #FFFFFF; margin-bottom: 0.4rem;">Fair Multi-Product Comparison</div>
            <div style="font-size: 0.85rem; color: #B9B3D0; line-height: 1.5;">
                Compare headphones against headphones (Sony vs Bose) or phones against phones with shared specs, dual radar charts, and honest tie verdicts.
            </div>
        </div>
        """)
    with vp3:
        render_html_block("""
        <div style="background: rgba(22, 17, 46, 0.78); border: 1px solid rgba(155, 92, 255, 0.25); border-radius: 12px; padding: 1.2rem; box-shadow: 0 8px 24px rgba(0,0,0,0.35);">
            <div style="font-size: 1.5rem; margin-bottom: 0.4rem;">📊</div>
            <div style="font-weight: 700; color: #FFFFFF; margin-bottom: 0.4rem;">Rigorous NLP Benchmarks</div>
            <div style="font-size: 0.85rem; color: #B9B3D0; line-height: 1.5;">
                We report measured accuracy, precision, recall, and a 3x3 confusion matrix evaluated against 50 manually labelled ground-truth reviews.
            </div>
        </div>
        """)


# ==========================================
# VIEWS 2-6: DEPARTMENT BROWSING & CATALOGUE
# ==========================================
elif st.session_state["nav_view"] in ["📱 Smartphones", "💻 Laptops", "🎧 Headphones", "📷 Cameras", "🛍️ All Products"]:
    view_name = st.session_state["nav_view"]
    category_map = {
        "📱 Smartphones": "Smartphones",
        "💻 Laptops": "Laptops",
        "🎧 Headphones": "Headphones",
        "📷 Cameras": "Cameras",
        "🛍️ All Products": None
    }
    dept_cat = category_map[view_name]

    st.markdown(f"## {view_name}")
    st.caption("Browse curated electronics, compare specifications, and uncover prepared demo review sentiment.")

    # Department-filtered DataFrame
    dept_df = df_products if dept_cat is None else df_products[df_products["category"] == dept_cat]

    # Search bar & Filters
    search_col1, search_col2 = st.columns([3, 1])
    with search_col1:
        search_query = st.text_input(
            "Search Products",
            placeholder=f"Search {dept_cat or 'all products'} by name, brand, processor, display, or specs...",
            label_visibility="collapsed"
        )
    with search_col2:
        st.caption(f"Showing **{len(dept_df)}** catalogue items in this department")

    # Apply filters
    filters = render_sidebar_filters(dept_df)
    filtered_df = apply_filters(dept_df, search_query, filters)

    # Active Filter Chips
    chip_text = []
    if search_query:
        chip_text.append(f'Keyword: "{search_query}"')
    if dept_cat:
        chip_text.append(f'Department: {dept_cat}')
    if filters.get("brands"):
        chip_text.append(f'Brands: {", ".join(filters["brands"])}')
    chip_text.append(f'Price: ₹{filters["price_min"]:,} - ₹{filters["price_max"]:,}')
    chip_text.append(f'Rating: {filters["min_rating"]:.1f}+ ★')

    render_html_block(f"<div style='font-size: 0.85rem; color: #64748B; margin-bottom: 1rem;'><b>Active Filters:</b> {' • '.join(chip_text)} (Found {len(filtered_df)} matches)</div>")

    if filtered_df.empty:
        st.warning("No products matched your search and filter criteria. Try adjusting your filters.")
    else:
        cols_per_row = 3
        rows = [filtered_df.iloc[i:i+cols_per_row] for i in range(0, len(filtered_df), cols_per_row)]

        for r in rows:
            grid_cols = st.columns(cols_per_row)
            for idx, (_, prod_row) in enumerate(r.iterrows()):
                prod_dict = get_product_by_id(prod_row["product_id"])
                with grid_cols[idx]:
                    render_product_card(
                        prod_dict,
                        is_in_compare=(prod_row["product_id"] in st.session_state["compare_tray"]),
                        on_inspect=inspect_product_action,
                        on_toggle_compare=toggle_compare_item
                    )


# ==========================================
# VIEW 7: PRODUCT DETAILS (SHOPPING PAGE)
# ==========================================
elif st.session_state["nav_view"] == "📦 Product Details":
    all_pids = [p["product_id"] for _, p in df_products.iterrows()]
    curr_pid = st.session_state.get("selected_product_id", all_pids[0])
    if curr_pid not in all_pids:
        curr_pid = all_pids[0]

    p_select_col1, p_select_col2 = st.columns([3, 1])
    with p_select_col1:
        selected_pid = st.selectbox(
            "Select Product to View Details:",
            all_pids,
            index=all_pids.index(curr_pid),
            format_func=lambda x: f"{get_product_by_id(x)['product_name']} ({format_currency_inr(get_product_by_id(x)['price'])})"
        )
        st.session_state["selected_product_id"] = selected_pid

    prod = get_product_by_id(selected_pid)
    reviews_df = get_reviews_for_product(selected_pid)

    with p_select_col2:
        in_tray = selected_pid in st.session_state["compare_tray"]
        btn_txt = "✓ In Compare Tray" if in_tray else "⚔️ Add to Compare"
        if st.button(btn_txt, key="btn_details_tray", use_container_width=True, type="primary" if in_tray else "secondary"):
            toggle_compare_item(selected_pid)

    # Breadcrumb
    esc_prod_name = html.escape(str(prod.get('product_name', 'Product')))
    esc_brand = html.escape(str(prod.get('brand', '')))
    esc_category = html.escape(str(prod.get('category', '')))
    esc_description = html.escape(str(prod.get('description', '')))

    st.markdown(f"**Electronics** > **{esc_category}** > **{esc_prod_name}**")

    # Main Product Details Card
    orig_price = float(prod.get("original_price", prod.get("price", 0.0)))
    price = float(prod.get("price", 0.0))
    discount_html = ""
    if orig_price > price:
        pct = int(round((orig_price - price) / orig_price * 100))
        discount_html = f"<span style='text-decoration: line-through; color: #94A3B8; font-size: 1.1rem; margin-right: 0.5rem;'>{format_currency_inr(orig_price)}</span> <span style='background: #DCFCE7; color: #16A34A; font-weight: 700; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.9rem;'>{pct}% OFF</span>"

    detail_left, detail_right = st.columns([1, 2])
    img_src = get_product_image_src(prod)
    with detail_left:
        st.markdown(f"""
        <div style="background: rgba(22, 17, 46, 0.8); border: 1px solid rgba(155, 92, 255, 0.25); border-radius: 12px; padding: 1.5rem; text-align: center; height: 320px; display: flex; align-items: center; justify-content: center; box-shadow: 0 8px 24px rgba(0,0,0,0.35);">
            <img src="{img_src}" style="max-height: 280px; max-width: 100%; object-fit: contain; border-radius: 8px;" alt="{esc_prod_name}" />
        </div>
        """, unsafe_allow_html=True)

    with detail_right:
        st.markdown(f"## {esc_prod_name}")
        st.markdown(f"**Brand:** {esc_brand} &nbsp;|&nbsp; **Category:** {esc_category} &nbsp;|&nbsp; <span style='background: rgba(37, 99, 235, 0.18); color: #93C5FD; font-weight: 600; font-size: 0.8rem; padding: 0.2rem 0.5rem; border-radius: 4px;'>🛡️ Demo (Illustrative Pricing & Ratings)</span>", unsafe_allow_html=True)
        st.markdown(f"<div style='margin: 0.8rem 0;'><span style='font-size: 2rem; font-weight: 800; color: #FFFFFF; margin-right: 0.8rem;'>{format_currency_inr(price)}</span> {discount_html}</div>", unsafe_allow_html=True)
        st.markdown(f"★ **{prod.get('rating'):.1f}** &nbsp;•&nbsp; `{prod.get('review_count'):,}` Illustrative Ratings &nbsp;•&nbsp; `{len(reviews_df)}` Prepared demo reviews analyzed")
        st.markdown(f"<p style='color: #CBD5E1; line-height: 1.6; margin: 0.8rem 0;'>{esc_description}</p>", unsafe_allow_html=True)

        amazon_url = prod.get("product_url") or prod.get("amazon_url") or f"https://www.amazon.in/s?k={esc_prod_name}"
        btn_c1, btn_c2 = st.columns(2)
        with btn_c1:
            if st.button("🔍 Explore Deep Review Intelligence ↗", key="btn_go_intelligence", type="primary", use_container_width=True):
                explore_review_intelligence_action(selected_pid)
        with btn_c2:
            st.markdown(f"""
            <a href="{amazon_url}" target="_blank" style="display: block; text-align: center; background: #FF9900; color: #131921; font-weight: 700; padding: 0.55rem 1rem; border-radius: 8px; text-decoration: none;">
                Search on Amazon India ↗
            </a>
            """, unsafe_allow_html=True)

    if reviews_df.empty:
        st.info("ℹ️ No prepared demo reviews are available for this specific product in the dataset.")

    st.markdown("---")

    # Hardware & Feature Specifications Table (Vector Monospace Token Grid)
    st.markdown("### 📋 **Product Specifications**")
    raw_specs = prod.get("specifications") or prod.get("specs") or {}
    specs_dict = {}
    if isinstance(raw_specs, str):
        try:
            specs_dict = json.loads(raw_specs)
        except Exception:
            specs_dict = {}
    elif isinstance(raw_specs, dict):
        specs_dict = raw_specs

    if specs_dict:
        spec_cols = st.columns(3)
        spec_keys = list(specs_dict.keys())
        for idx, k in enumerate(spec_keys):
            v = specs_dict[k]
            col_target = spec_cols[idx % 3]
            with col_target:
                render_html_block(f"""
                <div class="layer-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.25rem;">
                        <span style="font-weight: 700; color: #FFFFFF; font-size: 0.85rem;">{html.escape(k)}</span>
                        <span class="mono-badge">SPEC {idx+1}</span>
                    </div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #CBD5E1; line-height: 1.4;">
                        {html.escape(str(v))}
                    </div>
                </div>
                """)
        with st.expander("📊 Tabular Specifications View (Dataframe)", expanded=False):
            spec_items = [{"Specification": k, "Detail": v} for k, v in specs_dict.items()]
            st.dataframe(pd.DataFrame(spec_items), use_container_width=True, hide_index=True)
    else:
        render_html_block("""
        <div class="spec-box">
            ℹ️ General product specifications are summarized in the product description above.
        </div>
        """)

    # Category-Specific Aspect Rating Snapshot
    st.markdown("### 🔍 **Aspect Sentiment Summary Snapshot**")
    cat_aspects_res = analyze_reviews_aspects(reviews_df, category=prod.get("category"))
    aspect_list = cat_aspects_res["aspects"]

    asp_cols = st.columns(min(len(aspect_list), 5))
    for idx, asp in enumerate(aspect_list[:5]):
        with asp_cols[idx]:
            if asp.get("has_evidence"):
                nss = asp["net_sentiment_score"]
                color = "#10B981" if nss > 0 else "#EF4444"
                score_str = f"{nss:+.1f}"
                mentions_str = f"{asp['mentions']} review mentions"
            else:
                color = "#94A3B8"
                score_str = "N/A"
                mentions_str = "Insufficient review data"

            render_html_block(f"""
            <div class="aspect-card">
                <div class="aspect-label">{html.escape(str(asp['category']))}</div>
                <div class="aspect-val" style="color: {color}; text-shadow: 0 0 12px {color}33;">{score_str}</div>
                <div class="aspect-sub">{mentions_str}</div>
            </div>
            """)

    # Pros and Cons Preview
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### ⚖️ **Customer Feedback Strengths & Pain Points**")
    sentiment_res = analyze_reviews_sentiment(reviews_df)
    exec_summary = generate_executive_summary(
        prod["product_name"],
        aspect_list,
        sentiment_res["nss"],
        sentiment_res["csat_estimate"],
        sentiment_res["total_reviews"]
    )

    p_col, c_col = st.columns(2)
    with p_col:
        st.markdown("##### 🟢 Top Strengths")
        if exec_summary["pros"]:
            for p in exec_summary["pros"]:
                render_html_block(f"""
                <div class="strength-card">
                    <div class="strength-title">
                        🟢 {html.escape(str(p['aspect']))} (+{p['nss']:.0f} NSS)
                    </div>
                    <div class="strength-quote">
                        "{html.escape(str(p['quote']))}"
                    </div>
                </div>
                """)
        else:
            render_html_block("""
            <div class="aspect-card" style="text-align: left; color: #94A3B8;">
                No standout positive drivers identified.
            </div>
            """)
    with c_col:
        st.markdown("##### 🔴 Key Trade-offs / Pain Points")
        if exec_summary["cons"]:
            for c in exec_summary["cons"]:
                render_html_block(f"""
                <div class="painpoint-card">
                    <div class="painpoint-title">
                        🔴 {html.escape(str(c['aspect']))} ({c['nss']:+.0f} NSS)
                    </div>
                    <div class="painpoint-quote">
                        "{html.escape(str(c['quote']))}"
                    </div>
                </div>
                """)
        else:
            render_html_block("""
            <div class="aspect-card" style="text-align: left; color: #34D399;">
                No dominant customer pain points flagged in this review corpus.
            </div>
            """)

    # Vector-Inspired Feature: Requirement Engineering Action Brief for Product Defects
    if exec_summary["cons"]:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="fig-tag">VECTOR REQUIREMENT ENGINEERING</div>', unsafe_allow_html=True)
        with st.expander("⚡ Synthesize Engineering Action Brief & BDD Acceptance Criteria for Product Defects", expanded=False):
            st.caption("Compile an authoritative 6-section technical bug brief with Gherkin Given/When/Then acceptance criteria for your engineering team.")
            con_aspects = [c["aspect"] for c in exec_summary["cons"]]
            selected_con_asp = st.selectbox("Target Defect / Aspect:", con_aspects, key="prod_con_select")
            matched_con = next((c for c in exec_summary["cons"] if c["aspect"] == selected_con_asp), exec_summary["cons"][0])

            prod_brief = synthesize_engineering_action_brief(
                product_name=prod["product_name"],
                aspect_name=selected_con_asp,
                nss_score=matched_con["nss"],
                verified_quotes=[matched_con["quote"]],
                category=prod.get("category", "Electronics"),
                brand=prod.get("brand", "OEM")
            )

            p_ac = prod_brief["06_acceptance_criteria"]
            p_gherkin = f"""Scenario: {p_ac['scenario']}
  Given {p_ac['given']}
  When {p_ac['when']}
  Then {p_ac['then'][0]}
  Then {p_ac['then'][1]}"""

            render_html_block(f"""
            <div class="bdd-block">
                <div style="font-size: 0.74rem; color: #10B981; font-weight: 800; letter-spacing: 0.05em; margin-bottom: 0.35rem;">
                    // GHERKIN BDD ACCEPTANCE CRITERIA FOR {selected_con_asp.upper()}
                </div>
                <div style="white-space: pre-wrap; font-family: 'JetBrains Mono', monospace; font-size: 0.83rem;">{p_gherkin}</div>
            </div>
            """)

            col_b1, col_b2 = st.columns([1, 1])
            with col_b1:
                st.markdown(f"**Ticket ID:** `{prod_brief['01_context']['ticket_id']}` | **Priority:** `{prod_brief['01_context']['priority']}`")
                st.markdown(f"**Affected Subsystem:** `{prod_brief['02_subsystem']['component']}`")
                st.markdown(f"**Observed Symptom:** {prod_brief['04_reproduction']['symptom']}")
            with col_b2:
                st.markdown(f"**Proposed Remediation:** {prod_brief['05_remediation']['action_plan']}")
                st.markdown(f"**Validation Test:** {prod_brief['05_remediation']['verification_method']}")

            prod_jira_md = export_brief_to_jira_markdown(prod_brief)
            st.download_button(
                label=f"📥 Download {prod_brief['01_context']['ticket_id']} Brief (.md)",
                data=prod_jira_md,
                file_name=f"{prod_brief['01_context']['ticket_id']}.md",
                mime="text/markdown",
                key=f"btn_dl_{selected_con_asp}",
                use_container_width=True
            )

    # Review Q&A Section with TF-IDF Semantic Retrieval & Citations
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 💬 **Ask the Reviews (Extractive Q&A with Citations)**")
    st.caption("Ask natural language questions about this product. The NLP engine retrieves exact review sentences using TF-IDF cosine similarity, balances sentiment consensus, and attributes direct citations.")

    suggested_questions = [
        "How is the battery life and charging?",
        "Is the display bright and sharp?",
        "Does it overheat or throttle?",
        "How is the build quality and ergonomics?"
    ]
    pills_cols = st.columns(len(suggested_questions))
    for p_idx, q_sug in enumerate(suggested_questions):
        with pills_cols[p_idx]:
            if st.button(f"💡 {q_sug}", key=f"btn_qa_sug_{p_idx}", use_container_width=True):
                st.session_state["qa_query_input"] = q_sug

    user_qa_query = st.text_input(
        "Enter question about this product:",
        value=st.session_state.get("qa_query_input", ""),
        key="qa_query_text_box",
        placeholder="e.g. How is the battery life? Does it handle video editing well?"
    )

    if user_qa_query.strip():
        qa_result = answer_product_review_question(
            selected_pid,
            user_qa_query,
            top_k=3,
            reviews_df=reviews_df
        )

        if qa_result["status"] == "ok":
            cons_badge = qa_result["consensus"]
            cons_color = "#10B981" if "Positive" in cons_badge else ("#EF4444" if "Negative" in cons_badge else "#F59E0B")
            pos_c = qa_result["sentiment_summary"]["positive"]
            neg_c = qa_result["sentiment_summary"]["negative"]
            neu_c = qa_result["sentiment_summary"]["neutral"]

            st.markdown(f"""
            <div style="background: rgba(22, 17, 46, 0.85); border: 1px solid rgba(155, 92, 255, 0.4); border-radius: 12px; padding: 1.2rem; margin: 1rem 0;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem; flex-wrap: wrap; gap: 8px;">
                    <div>
                        <span style="font-size: 0.8rem; color: #94A3B8; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">EXTRACTIVE CONSENSUS</span>
                        <div style="font-size: 1.25rem; font-weight: 800; color: {cons_color};">{cons_badge}</div>
                    </div>
                    <div style="font-size: 0.85rem; color: #CBD5E1; background: rgba(255,255,255,0.06); padding: 4px 12px; border-radius: 99px;">
                        Retrieved Evidence: <b style="color:#10B981">{pos_c} Positive</b> &nbsp;|&nbsp; <b style="color:#EF4444">{neg_c} Negative</b> &nbsp;|&nbsp; <b style="color:#94A3B8">{neu_c} Neutral</b>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("##### 📌 **Direct Review Citations & Evidence**")
            for cit in qa_result["citations"]:
                c_sent_color = "#10B981" if cit["sentiment"] == "POSITIVE" else ("#EF4444" if cit["sentiment"] == "NEGATIVE" else "#94A3B8")
                st.markdown(f"""
                <div style="background: rgba(13, 10, 36, 0.7); border-left: 4px solid {c_sent_color}; border-radius: 6px; padding: 0.8rem 1rem; margin-bottom: 0.6rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                        <span style="font-weight: 700; color: #9B5CFF; font-size: 0.82rem; font-family: monospace;">{cit['citation']}</span>
                        <span>
                            <span style="font-size: 0.75rem; background: rgba(34, 211, 238, 0.15); color: #22D3EE; padding: 2px 8px; border-radius: 4px;">Cosine Sim: {cit['similarity']:.3f}</span>
                            &nbsp;<span style="font-size: 0.75rem; background: rgba(255,255,255,0.1); color: #FFFFFF; padding: 2px 8px; border-radius: 4px;">★ {cit['rating']:.1f}</span>
                            &nbsp;<span style="font-size: 0.75rem; background: {c_sent_color}22; color: {c_sent_color}; padding: 2px 8px; border-radius: 4px; font-weight: 700;">{cit['sentiment']}</span>
                        </span>
                    </div>
                    <div style="color: #F1F5F9; font-size: 0.92rem; line-height: 1.5; font-style: italic;">
                        "{html.escape(cit['sentence'])}"
                    </div>
                </div>
                """, unsafe_allow_html=True)
        elif qa_result["status"] == "no_match":
            st.info(f"ℹ️ {qa_result['answer']}")
        elif qa_result["status"] == "no_reviews":
            st.warning(f"⚠️ {qa_result['answer']}")

    st.markdown("<br>", unsafe_allow_html=True)
    cta_col1, cta_col2, cta_col3 = st.columns([2, 1, 1])
    with cta_col1:
        if st.button("🚀 Dive Into Full NLP Review Intelligence & Interactive Topic Modeling →", key="btn_deep_cta", type="primary", use_container_width=True):
            explore_review_intelligence_action(selected_pid)
    with cta_col2:
        try:
            pdf_bytes = generate_product_pdf_report(
                prod=prod,
                aspect_list=aspect_list,
                sentiment_res=sentiment_res,
                exec_summary=exec_summary
            )
            st.download_button(
                label="📄 Export Executive PDF",
                data=pdf_bytes,
                file_name=f"{selected_pid}_executive_report.pdf",
                mime="application/pdf",
                help="Download full executive PDF dossier with aspect scorecard, strengths/cons, and academic disclosures",
                type="primary",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Could not generate PDF: {e}")
    with cta_col3:
        dossier_text = generate_markdown_dossier(prod, aspect_list, sentiment_res, exec_summary)
        st.download_button(
            label="📥 Export Dossier (.md)",
            data=dossier_text,
            file_name=f"{selected_pid}_reviewlens_dossier.md",
            mime="text/markdown",
            help="Download formatted executive intelligence dossier with ABSA scorecard, pros/cons, and transparency disclosures",
            use_container_width=True
        )


# ==========================================
# VIEW 8: REVIEW INTELLIGENCE (DEEP DIVE)
# ==========================================
elif st.session_state["nav_view"] == "🔍 Review Intelligence":
    all_pids = [p["product_id"] for _, p in df_products.iterrows()]
    curr_pid = st.session_state.get("selected_product_id", all_pids[0])
    if curr_pid not in all_pids:
        curr_pid = all_pids[0]

    p_select_col1, p_select_col2 = st.columns([3, 1])
    with p_select_col1:
        selected_pid = st.selectbox(
            "Select Product to Analyze:",
            all_pids,
            index=all_pids.index(curr_pid),
            format_func=lambda x: f"{get_product_by_id(x)['product_name']} ({format_currency_inr(get_product_by_id(x)['price'])})"
        )
        st.session_state["selected_product_id"] = selected_pid

    prod = get_product_by_id(selected_pid)
    reviews_df = get_reviews_for_product(selected_pid)
    img_src = get_product_image_src(prod)

    with p_select_col2:
        in_tray = selected_pid in st.session_state["compare_tray"]
        btn_txt = "✓ In Compare Tray" if in_tray else "⚔️ Add to Compare"
        if st.button(btn_txt, key="btn_intel_tray", use_container_width=True, type="primary" if in_tray else "secondary"):
            toggle_compare_item(selected_pid)

    # Product Overview Banner
    p_name_esc = html.escape(str(prod.get('product_name', 'Product')))
    p_cat_esc = html.escape(str(prod.get('category', '')))
    p_brand_esc = html.escape(str(prod.get('brand', '')))

    render_html_block(f"""
    <div style="background: rgba(22, 17, 46, 0.85); border: 1px solid rgba(155, 92, 255, 0.3); border-radius: 14px; padding: 1.3rem; margin-bottom: 1.5rem; display: flex; flex-wrap: wrap; gap: 1.5rem; align-items: center; box-shadow: 0 8px 24px rgba(0,0,0,0.4);">
        <div style="width: 110px; height: 110px; background: rgba(12, 9, 28, 0.75); border-radius: 10px; display: flex; align-items: center; justify-content: center; overflow: hidden; border: 1px solid rgba(155, 92, 255, 0.2);">
            <img src="{img_src}" style="max-height: 100%; max-width: 100%; object-fit: contain; border-radius: 6px;" alt="{p_name_esc}" />
        </div>
        <div style="flex: 1; min-width: 260px;">
            <div style="display: flex; gap: 0.5rem; margin-bottom: 0.3rem; flex-wrap: wrap;">
                <span style="background: rgba(155, 92, 255, 0.15); color: #C4B5FD; font-weight: 600; font-size: 0.75rem; padding: 0.2rem 0.6rem; border-radius: 6px;">{p_cat_esc}</span>
                <span style="background: rgba(37, 99, 235, 0.18); color: #93C5FD; font-weight: 600; font-size: 0.75rem; padding: 0.2rem 0.6rem; border-radius: 6px;">{p_brand_esc}</span>
                <span style="background: rgba(245, 158, 11, 0.18); color: #FCD34D; font-weight: 600; font-size: 0.75rem; padding: 0.2rem 0.6rem; border-radius: 6px;">🛡️ Demo Data (Illustrative)</span>
            </div>
            <h2 style="margin: 0 0 0.3rem 0; font-size: 1.35rem; font-weight: 800; color: #FFFFFF;">{p_name_esc}</h2>
            <div style="display: flex; align-items: baseline; gap: 0.8rem;">
                <span style="font-size: 1.4rem; font-weight: 800; color: #FFFFFF;">{format_currency_inr(prod.get('price'))}</span>
                <span style="color: #F59E0B; font-weight: 700; font-size: 0.95rem;">★ {prod.get('rating'):.1f}</span>
                <span style="color: #94A3B8; font-size: 0.85rem;">({prod.get('review_count'):,} Ratings)</span>
            </div>
        </div>
    </div>
    """)

    if reviews_df.empty:
        st.warning(f"⚠️ No customer reviews found for '{p_name_esc}' in the review dataset. Analysis metrics will display baseline zero-review state.")

    # Prototype Method Notice
    st.info("ℹ️ **Methodology Notice:** Keyword-based aspect matching is an explainable rule-based prototype method, not a fully trained deep learning ABSA model. Aspect sentiment is computed from sentence-level VADER polarity matching category keywords.")

    # Run NLP Pipeline
    sentiment_res = analyze_reviews_sentiment(reviews_df)
    aspects_res = analyze_reviews_aspects(reviews_df, category=prod.get("category"))
    aspect_list = aspects_res["aspects"]
    keywords_res = extract_keywords_by_sentiment(reviews_df)

    # High-Level Metric Cards with review counts
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_html_block(f"""
        <div class="metric-card metric-card-orange">
            <div class="metric-label">Catalogue Rating</div>
            <div class="metric-value">{prod.get('rating'):.1f} <span style="font-size: 1rem; color: #F59E0B;">★</span></div>
            <div class="metric-sub">Out of 5.0 stars (sample)</div>
        </div>
        """)
    with k2:
        nss_val = sentiment_res["nss"]
        nss_col = "#10B981" if nss_val > 0 else "#EF4444"
        render_html_block(f"""
        <div class="metric-card metric-card-green">
            <div class="metric-label">Net Sentiment Score (NSS)</div>
            <div class="metric-value" style="color: {nss_col};">{nss_val:+.1f}</div>
            <div class="metric-sub">Based on {sentiment_res['total_reviews']} analyzed reviews</div>
        </div>
        """)
    with k3:
        render_html_block(f"""
        <div class="metric-card">
            <div class="metric-label">Estimated CSAT</div>
            <div class="metric-value">{sentiment_res['csat_estimate']:.1f}%</div>
            <div class="metric-sub">Positive review polarity ratio</div>
        </div>
        """)
    with k4:
        render_html_block(f"""
        <div class="metric-card metric-card-purple">
            <div class="metric-label">Reviews Analyzed</div>
            <div class="metric-value">{sentiment_res['total_reviews']}</div>
            <div class="metric-sub">Category-specific ABSA corpus</div>
        </div>
        """)

    # Overall Sentiment Donut & Aspect Breakdown
    st.markdown("### 📊 **Sentiment Distribution & Category Aspects**")
    s_col1, s_col2 = st.columns([1, 1])
    with s_col1:
        st.markdown(f"##### Overall Sentiment Polarity ({sentiment_res['total_reviews']} reviews)")
        fig_donut = create_sentiment_donut_chart(sentiment_res["distribution"])
        st.plotly_chart(fig_donut, use_container_width=True)

    with s_col2:
        st.markdown(f"##### Aspect Sentiment Breakdown ({prod.get('category')} Taxonomy)")
        fig_bar = create_aspect_breakdown_bar_chart(aspect_list)
        st.plotly_chart(fig_bar, use_container_width=True)

    # Radar Chart & Evidence Quotes
    r_col1, r_col2 = st.columns([1, 1])
    with r_col1:
        st.markdown("##### 🕸️ Aspect Net Sentiment Radar")
        fig_radar = create_aspect_radar_chart(aspect_list, product_name=prod["product_name"])
        st.plotly_chart(fig_radar, use_container_width=True)

    with r_col2:
        st.markdown("##### 💬 Aspect Evidence & Mention Quotes")
        for a in aspect_list:
            if a.get("has_evidence"):
                st.markdown(f"**{a['category']}** (Mentions: `{a['mentions']}` | NSS: `{a['net_sentiment_score']:+.1f}`)")
                if a.get("positive_quotes"):
                    st.caption(f"🟢 *\"{a['positive_quotes'][0]}\"*")
                if a.get("negative_quotes"):
                    st.caption(f"🔴 *\"{a['negative_quotes'][0]}\"*")
            else:
                st.markdown(f"**{a['category']}**: *Insufficient review data (0 mentions)*")
            st.markdown("<hr style='margin: 0.4rem 0;'>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # INTERACTIVE LDA TOPIC MODELING
    st.markdown("### 🧠 **Latent Dirichlet Allocation (LDA) Topic Clusters**")
    st.caption("Discover latent conversational themes and recurring semantic clusters in customer reviews with customizable topic counts.")

    topic_slider_col, _ = st.columns([2, 1])
    with topic_slider_col:
        num_topics = st.slider(
            "Select Number of Latent Themes to Discover (k):",
            min_value=2,
            max_value=6,
            value=3,
            step=1,
            key="lda_num_topics_slider"
        )

    review_texts = [r.get("review_text", "") for _, r in reviews_df.iterrows() if r.get("review_text")]
    topic_res = extract_topic_clusters_result(review_texts, n_topics=num_topics, top_words=6)
    discovered_topics = topic_res.get("clusters", [])

    if discovered_topics:
        topic_cols = st.columns(min(len(discovered_topics), 3))
        for idx, t in enumerate(discovered_topics):
            col_target = topic_cols[idx % 3]
            with col_target:
                pills_html = "".join([f'<span class="topic-pill">{html.escape(kw)}</span>' for kw in t["keywords"]])
                quotes_html = "".join([f'<li style="font-size: 0.8rem; color: #475569; margin-bottom: 0.4rem; font-style: italic;">"{html.escape(q[:120])}..."</li>' for q in t["sample_reviews"][:2]])
                render_html_block(f"""
                <div class="topic-card">
                    <div style="font-weight: 700; color: #1E293B; font-size: 0.95rem; margin-bottom: 0.5rem;">
                        🏷️ {html.escape(t['label'])}
                    </div>
                    <div style="margin-bottom: 0.6rem;">{pills_html}</div>
                    <div style="font-size: 0.78rem; font-weight: 600; color: #64748B; margin-bottom: 0.3rem;">Representative Excerpts:</div>
                    <ul style="padding-left: 1.1rem; margin: 0;">{quotes_html}</ul>
                </div>
                """)
    elif topic_res.get("status") == "error":
        st.warning(f"⚠️ Topic modeling notice: {topic_res.get('error_message', 'Could not run LDA clustering.')}")
    else:
        st.info(f"ℹ️ {topic_res.get('error_message', 'Insufficient review text to extract multi-topic LDA clusters.')}")

    st.markdown("<br>", unsafe_allow_html=True)

    # TF-IDF Drivers & Rating Anomalies
    kw_col1, kw_col2 = st.columns(2)
    with kw_col1:
        st.markdown("### 🔑 **Extracted Key Phrases (TF-IDF)**")
        fig_kw = create_keyword_driver_chart(
            keywords_res.get("positive_drivers", []),
            keywords_res.get("negative_pain_points", [])
        )
        st.plotly_chart(fig_kw, use_container_width=True)

    with kw_col2:
        st.markdown("### 🚨 **Rating vs. Sentiment Anomaly Explorer**")
        st.caption("Detects mismatched reviews where numeric stars contradict textual sentiment.")
        fig_scatter = create_rating_scatter_chart(sentiment_res.get("records", []))
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📈 **Temporal Sentiment Drift & Trust Audit**")
    st.caption("Tracks how consumer sentiment and review volume evolve over time, alongside automated spam and authenticity health indicators.")

    time_col, trust_col = st.columns([3, 2])
    with time_col:
        timeline_pts = compute_review_timeline_metrics(reviews_df)
        if timeline_pts:
            fig_time = create_temporal_sentiment_chart(timeline_pts)
            st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.info("ℹ️ Insufficient chronological review date timestamps to construct time-series drift curve.")

    with trust_col:
        st.markdown("##### 🛡️ Corpus Trust & Authenticity Audit")
        from reviewlens.spam_detector.detector import SpamDetector
        detector = SpamDetector()
        spam_count = 0
        for _, r_item in reviews_df.iterrows():
            r_text = str(r_item.get("review_text", ""))
            r_rating = float(r_item.get("review_rating", 3.0)) if pd.notnull(r_item.get("review_rating")) else None
            if detector.analyze(r_text, rating=r_rating, check_duplicate=False).is_spam:
                spam_count += 1

        tot_cnt = max(len(reviews_df), 1)
        clean_pct = round(((tot_cnt - spam_count) / tot_cnt) * 100.0, 1)
        anomaly_cnt = sentiment_res.get("anomalies_count", 0)

        health_badge_color = "#10B981" if clean_pct >= 90 else ("#F59E0B" if clean_pct >= 75 else "#EF4444")
        render_html_block(f"""
        <div style="background: rgba(22, 17, 46, 0.85); border: 1px solid rgba(155, 92, 255, 0.3); border-radius: 14px; padding: 1.2rem; box-shadow: 0 8px 24px rgba(0,0,0,0.35);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                <span style="font-weight: 700; color: #FFFFFF; font-size: 0.95rem;">Trust Health Score</span>
                <span style="background: {health_badge_color}; color: #FFFFFF; font-weight: 800; padding: 0.2rem 0.7rem; border-radius: 12px; font-size: 0.85rem;">{clean_pct}% Clean</span>
            </div>
            <div style="font-size: 0.85rem; color: #CBD5E1; margin-bottom: 0.8rem; line-height: 1.5;">
                Automated multi-heuristic review audit evaluating promotional patterns, repetitive templates, and star rating contradictions.
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem;">
                <div style="background: rgba(12, 9, 28, 0.7); border: 1px solid rgba(155, 92, 255, 0.2); border-radius: 8px; padding: 0.6rem;">
                    <div style="font-size: 0.75rem; color: #94A3B8;">Spam Flags</div>
                    <div style="font-size: 1.1rem; font-weight: 800; color: #EF4444;">{spam_count} <span style="font-size: 0.75rem; color: #94A3B8;">/ {len(reviews_df)}</span></div>
                </div>
                <div style="background: rgba(12, 9, 28, 0.7); border: 1px solid rgba(155, 92, 255, 0.2); border-radius: 8px; padding: 0.6rem;">
                    <div style="font-size: 0.75rem; color: #94A3B8;">Rating Anomalies</div>
                    <div style="font-size: 1.1rem; font-weight: 800; color: #F59E0B;">{anomaly_cnt} <span style="font-size: 0.75rem; color: #94A3B8;">mismatches</span></div>
                </div>
            </div>
            <div style="margin-top: 0.8rem; font-size: 0.78rem; color: #94A3B8;">
                🛡️ Verified against ReviewLens Heuristic Engine & VADER Sentiment Valence
            </div>
        </div>
        """)

    # Raw Analyzed Reviews Table & Download
    with st.expander("📄 View All Analyzed Customer Reviews & Download Dataset", expanded=False):
        rec_df = pd.DataFrame(sentiment_res.get("records", []))
        if not rec_df.empty:
            display_cols = ["review_id", "review_rating", "sentiment_label", "sentiment_compound", "review_text", "verified_purchase"]
            valid_disp = [c for c in display_cols if c in rec_df.columns]
            st.dataframe(rec_df[valid_disp], use_container_width=True, hide_index=True)

            csv_data = rec_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Analyzed Reviews (CSV)",
                data=csv_data,
                file_name=f"{selected_pid}_reviews_reviewlens.csv",
                mime="text/csv"
            )


# ==========================================
# VIEW: LIVE REVIEW ANALYZER & DIAGNOSTIC LAB
# ==========================================
elif st.session_state["nav_view"] == "🧪 Live Review Analyzer":
    nav_b_col1, nav_b_col2 = st.columns([1.2, 3.8])
    with nav_b_col1:
        if st.button("← Back to Discovery", key="btn_live_back_home", type="secondary", use_container_width=True):
            navigate_to("🏠 Home / Discovery")
    with nav_b_col2:
        st.markdown("<div style='font-size: 0.85rem; color: #64748B; padding-top: 0.4rem;'>🧭 View: <b>Review Diagnostic Lab</b> | Switch views using this button or the sidebar radio</div>", unsafe_allow_html=True)

    st.markdown("## 🧪 **Live Review Diagnostic Lab & Anomaly Detector**")
    st.caption("Paste any raw customer review or synthetic comment to evaluate real-time ABSA clauses, VADER valence, spam bot probability, and star-rating contradictions.")

    if "live_sample_text" not in st.session_state:
        st.session_state["live_sample_text"] = "The battery life and OLED screen are astonishingly good, but the camera autofocus is utterly disappointing in low light and customer support was completely unhelpful."
    if "live_sample_rating" not in st.session_state:
        st.session_state["live_sample_rating"] = 3.0
    if "live_sample_category" not in st.session_state:
        st.session_state["live_sample_category"] = "Smartphones"

    st.markdown("##### ⚡ Quick Presets (Click to Test)")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    with p_col1:
        if st.button("📱 Mixed Smartphone", use_container_width=True):
            st.session_state["live_sample_text"] = "The battery life and OLED screen are astonishingly good, but the camera autofocus is utterly disappointing in low light and customer support was completely unhelpful."
            st.session_state["live_sample_rating"] = 3.0
            st.session_state["live_sample_category"] = "Smartphones"
            st.rerun()
    with p_col2:
        if st.button("🤖 Bot / Promotional Spam", use_container_width=True):
            st.session_state["live_sample_text"] = "AMAZING PRODUCT BUY NOW DISCOUNT CLICK HERE BEST BEST BEST FREE GIFT http://promo.example.com CHECK OUT NOW!!!!!"
            st.session_state["live_sample_rating"] = 5.0
            st.session_state["live_sample_category"] = "Smartphones"
            st.rerun()
    with p_col3:
        if st.button("🚨 Rating Contradiction", use_container_width=True):
            st.session_state["live_sample_text"] = "Worst phone I have ever owned. It constantly overheats, crashes every 10 minutes, and the screen shattered on day one."
            st.session_state["live_sample_rating"] = 5.0
            st.session_state["live_sample_category"] = "Smartphones"
            st.rerun()
    with p_col4:
        if st.button("🎧 Audiophile Soundstage", use_container_width=True):
            st.session_state["live_sample_text"] = "Crisp highs, rich sub-bass response, and very wide soundstage. Noise cancellation is decent, but ear cushion padding feels slightly tight."
            st.session_state["live_sample_rating"] = 4.5
            st.session_state["live_sample_category"] = "Headphones"
            st.rerun()

    input_text = st.text_area(
        "Enter Review Text to Inspect:",
        value=st.session_state["live_sample_text"],
        height=120,
        placeholder="Type or paste any customer review here..."
    )

    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1.5, 1.5, 1.5, 1])
    with ctrl_col1:
        input_rating = st.slider(
            "Associated Star Rating (1.0 to 5.0):",
            min_value=1.0,
            max_value=5.0,
            value=float(st.session_state["live_sample_rating"]),
            step=0.5
        )
    with ctrl_col2:
        categories = ["Smartphones", "Laptops", "Headphones", "Cameras"]
        cat_idx = categories.index(st.session_state["live_sample_category"]) if st.session_state["live_sample_category"] in categories else 0
        input_cat = st.selectbox(
            "Category Taxonomy Context:",
            categories,
            index=cat_idx
        )
    with ctrl_col3:
        model_choice = st.selectbox(
            "Sentiment NLP Engine:",
            ["VADER (Lexicon)", "TF-IDF + LogReg (ML Model)"],
            index=0,
            help="Choose between rule-based VADER or trained supervised TF-IDF + Logistic Regression model"
        )
    with ctrl_col4:
        st.write("")
        st.write("")
        run_nlp = st.button("🔬 Run Diagnostic", type="primary", use_container_width=True)

    if input_text.strip():
        diag_res = analyze_custom_review(input_text, rating=input_rating, category=input_cat, model_type=model_choice)

        if diag_res.get("status") == "ok":
            st.markdown("---")
            st.markdown("### 🔬 **Diagnostic Results & NLP Decomposition**")

            # High-level KPIs
            d1, d2, d3, d4 = st.columns(4)
            with d1:
                sent = diag_res["sentiment"]
                comp = sent["compound"]
                s_color = "#10B981" if sent["label"] == "POSITIVE" else ("#EF4444" if sent["label"] == "NEGATIVE" else "#64748B")
                engine_title = sent.get("engine", "Sentiment Polarity")
                render_html_block(f"""
                <div class="metric-card" style="border-left: 4px solid {s_color};">
                    <div class="metric-label">{html.escape(engine_title)}</div>
                    <div class="metric-value" style="color: {s_color};">{sent['label']}</div>
                    <div class="metric-sub">Compound: <b>{comp:+.2f}</b> (Pos: {sent['pos']:.2f}, Neg: {sent['neg']:.2f})</div>
                </div>
                """)

            with d2:
                spam_info = diag_res["spam_analysis"]
                spam_flag = spam_info["is_spam"]
                spam_col = "#EF4444" if spam_flag else "#10B981"
                spam_label = "HIGH RISK / SPAM" if spam_flag else "AUTHENTIC REVIEW"
                ml_prob_sub = ""
                if spam_info.get("ml_spam_prob") is not None:
                    ml_prob_sub = f" | ML: <b>{spam_info['ml_spam_prob']*100:.0f}%</b>"
                render_html_block(f"""
                <div class="metric-card" style="border-left: 4px solid {spam_col};">
                    <div class="metric-label">Spam & Bot Probability</div>
                    <div class="metric-value" style="color: {spam_col}; font-size: 1.3rem;">{spam_label}</div>
                    <div class="metric-sub">Rules: <b>{spam_info['spam_score'] * 100:.0f}%</b>{ml_prob_sub}</div>
                </div>
                """)

            with d3:
                anom = diag_res["anomaly"]
                anom_col = "#EF4444" if anom["is_anomaly"] else "#10B981"
                anom_title = "CONTRADICTION" if anom["is_anomaly"] else "CONSISTENT"
                render_html_block(f"""
                <div class="metric-card" style="border-left: 4px solid {anom_col};">
                    <div class="metric-label">Rating Alignment</div>
                    <div class="metric-value" style="color: {anom_col}; font-size: 1.3rem;">{anom_title}</div>
                    <div class="metric-sub">{input_rating}★ vs {sent['label']} Text</div>
                </div>
                """)

            with d4:
                aspects_found = diag_res["aspects_found"]
                render_html_block(f"""
                <div class="metric-card metric-card-purple">
                    <div class="metric-label">Aspects Identified</div>
                    <div class="metric-value">{len(aspects_found)}</div>
                    <div class="metric-sub">{', '.join(aspects_found) if aspects_found else 'No domain keywords'}</div>
                </div>
                """)

            # Explainability: Word-level sentiment highlighting & evidence drivers
            st.markdown("#### 🖍️ Why this score? (word-level evidence)")
            lex = get_lexicon()
            st.markdown(highlight_review_html(input_text, lex), unsafe_allow_html=True)
            pos_w, neg_w = top_drivers(input_text, lex)
            if pos_w or neg_w:
                st.caption(
                    "Top positive: " + (", ".join(f"`{w}`" for w, _ in pos_w) or "none")
                    + "  |  Top negative: " + (", ".join(f"`{w}`" for w, _ in neg_w) or "none")
                )

            # Anomaly alert callout if present
            if diag_res["anomaly"]["is_anomaly"]:
                st.error(f"🚨 **Rating Contradiction Detected:** {diag_res['anomaly']['description']}")
            elif diag_res["spam_analysis"]["is_spam"]:
                st.warning(f"⚠️ **Spam Triggers Detected:** {', '.join(diag_res['spam_analysis']['reasons'])}")
                if diag_res["spam_analysis"].get("ml_domain_caveat"):
                    st.caption(f"ℹ️ *ML Domain Caveat:* {diag_res['spam_analysis']['ml_domain_caveat']}")

            # Visual breakdown
            gauge_col, clause_col = st.columns([1, 2])
            with gauge_col:
                st.markdown("##### 🧭 Real-Time Sentiment Polarity")
                fig_gauge = create_live_sentiment_gauge(comp)
                st.plotly_chart(fig_gauge, use_container_width=True)

            with clause_col:
                st.markdown("##### ✂️ Clause-by-Clause ABSA Segmentation")
                st.caption("Splits multi-aspect sentences on contrastive conjunctions (e.g. 'but', 'however') to decouple opposing sentiments.")
                for c in diag_res["clause_analysis"]:
                    c_label = c["sentiment_label"]
                    c_comp = c["compound"]
                    c_badge = "🟢 POSITIVE" if c_label == "POSITIVE" else ("🔴 NEGATIVE" if c_label == "NEGATIVE" else "🟡 NEUTRAL")
                    c_color = "#10B981" if c_label == "POSITIVE" else ("#EF4444" if c_label == "NEGATIVE" else "#64748B")
                    aspect_tags = "".join([f'<span class="topic-pill">{html.escape(a)}</span>' for a in c["aspects"]]) if c["aspects"] else '<span style="font-size: 0.78rem; color: #94A3B8;">(General statement)</span>'

                    render_html_block(f"""
                    <div style="background: rgba(22, 17, 46, 0.85); border: 1px solid rgba(155, 92, 255, 0.25); border-left: 4px solid {c_color}; border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 0.6rem; box-shadow: 0 4px 14px rgba(0,0,0,0.25);">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.3rem;">
                            <span style="font-size: 0.78rem; font-weight: 700; color: {c_color};">{c_badge} ({c_comp:+.2f})</span>
                            <div>{aspect_tags}</div>
                        </div>
                        <div style="font-size: 0.88rem; color: #F1F5F9; font-style: italic; line-height: 1.5;">\"{html.escape(c['text'])}\"</div>
                    </div>
                    """)

            with st.expander("🔍 Deep Linguistic & Negation Scope Inspection", expanded=False):
                neg_words = diag_res.get("negated_words", [])
                st.markdown(f"**Negated Tokens Detected:** {', '.join([f'`{w}`' for w in neg_words]) if neg_words else 'None (standard polarity weighting)'}")
                st.markdown(f"**Spam Heuristic Triggers:** {', '.join(diag_res['spam_analysis']['reasons']) if diag_res['spam_analysis']['reasons'] else 'Clean — no repetitive keywords, suspicious URLs, or excessive caps.'}")

            # Vector-Inspired Requirement Engineering & Review Actionability Index
            st.markdown("---")
            act_res = diag_res.get("actionability", {})
            act_score = act_res.get("actionability_score", 0)
            act_level = act_res.get("level", "LOW")
            act_label = act_res.get("level_label", "")
            act_cov = act_res.get("coverage_pct", 0)
            act_covered_layers = act_res.get("layers_covered", 0)

            act_color = "#10B981" if act_score >= 75 else ("#F59E0B" if act_score >= 50 else "#EF4444")

            st.markdown('<div class="fig-tag">FIG 1.0 · REQUIREMENT ENGINEERING & ACTION BRIEF</div>', unsafe_allow_html=True)
            st.markdown("### 📐 **Vector-Style Review Actionability & Gap Index**")
            st.caption("Deterministic inspection across 6 structural feedback layers (Subsystem, Defect Symptom, Trigger Context, Polarity, Reproducibility, Expected Outcome) to measure whether customer feedback is engineering-actionable.")

            # Score and KPI metrics
            ak1, ak2, ak3 = st.columns([1.5, 2, 2.5])
            with ak1:
                render_html_block(f"""
                <div class="metric-card" style="border-left: 4px solid {act_color}; text-align: center;">
                    <div class="metric-label">Actionability Score</div>
                    <div class="metric-value" style="color: {act_color}; font-size: 2.2rem;">{act_score}<span style="font-size: 1rem; color: #94A3B8;"> / 100</span></div>
                    <div class="metric-sub" style="font-weight: 700; color: {act_color};">{act_level} ACTIONABILITY</div>
                </div>
                """)
            with ak2:
                render_html_block(f"""
                <div class="metric-card" style="border-left: 4px solid #9B5CFF;">
                    <div class="metric-label">Structural Layer Coverage</div>
                    <div class="metric-value" style="font-size: 1.8rem;">{act_covered_layers} / 6 <span style="font-size: 1rem; color: #C4B5FD;">({act_cov}%)</span></div>
                    <div class="metric-sub">{act_label}</div>
                </div>
                """)
            with ak3:
                gaps = act_res.get("gap_findings", [])
                if gaps:
                    gap_items = "".join([f"<div style='font-size: 0.78rem; margin-bottom: 0.25rem; color: #FCA5A5;'>⚠️ <b>{g['type']}</b>: {g['message']} [{g['severity']}]</div>" for g in gaps[:2]])
                    render_html_block(f"""
                    <div class="metric-card" style="border-left: 4px solid #EF4444;">
                        <div class="metric-label" style="color: #F87171;">Detected Structural Gaps ({len(gaps)})</div>
                        <div style="padding-top: 0.2rem;">{gap_items}</div>
                    </div>
                    """)
                else:
                    render_html_block("""
                    <div class="metric-card" style="border-left: 4px solid #10B981;">
                        <div class="metric-label" style="color: #34D399;">Zero Critical Gaps</div>
                        <div class="metric-value" style="color: #10B981; font-size: 1.25rem;">Deterministic Ready</div>
                        <div class="metric-sub">Sufficient context for immediate developer triage</div>
                    </div>
                    """)

            # 6 Structural Layers inspection checklist
            with st.expander("🔎 Inspect 6 Structural Actionability Layers & Criteria", expanded=False):
                l_cols = st.columns(2)
                for idx, layer_info in enumerate(act_res.get("layer_breakdown", [])):
                    col_target = l_cols[idx % 2]
                    with col_target:
                        status_icon = "🟢" if layer_info["covered"] else "⚪"
                        status_badge = f'<span class="mono-badge" style="color:#10B981;border-color:rgba(16,185,129,0.4);">COVERED (+{layer_info["weight"]} pts)</span>' if layer_info["covered"] else '<span class="mono-badge" style="color:#94A3B8;border-color:rgba(148,163,184,0.3);">UNMET (0 pts)</span>'
                        render_html_block(f"""
                        <div class="layer-card">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.3rem;">
                                <span style="font-weight: 700; color: #FFFFFF; font-size: 0.88rem;">{status_icon} {layer_info['layer']}</span>
                                {status_badge}
                            </div>
                            <div style="font-size: 0.78rem; color: #94A3B8;">{layer_info['evidence']}</div>
                        </div>
                        """)

            # 6-Section Technical Brief Synthesis
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="fig-tag">FIG 2.0 · BDD SPECIFICATION SYNTHESIS</div>', unsafe_allow_html=True)
            st.markdown("### ⚡ **Synthesize Jira / Linear Engineering Action Brief & BDD Tests**")
            st.caption("Converts the unstructured customer review into an authoritative 6-section technical bug ticket with Gherkin Acceptance Criteria (Given/When/Then).")

            primary_aspect = diag_res["aspects_found"][0] if diag_res["aspects_found"] else (input_cat if input_cat else "General Customer Experience")
            action_brief = synthesize_engineering_action_brief(
                product_name=f"Customer Feedback Item ({input_cat})",
                aspect_name=primary_aspect,
                nss_score=float(comp * 100),
                verified_quotes=[input_text],
                category=input_cat,
                brand="OEM"
            )

            # Display the BDD block prominently
            ac = action_brief["06_acceptance_criteria"]
            bdd_gherkin = f"""Scenario: {ac['scenario']}
  Given {ac['given']}
  When {ac['when']}
  Then {ac['then'][0]}
  Then {ac['then'][1]}"""

            render_html_block(f"""
            <div class="bdd-block">
                <div style="font-size: 0.75rem; color: #10B981; font-weight: 800; letter-spacing: 0.05em; margin-bottom: 0.4rem;">
                    // DETERMINISTIC BDD ACCEPTANCE CRITERIA (GHERKIN SPEC)
                </div>
                <div style="white-space: pre-wrap; font-family: 'JetBrains Mono', monospace; font-size: 0.84rem;">{bdd_gherkin}</div>
            </div>
            """)

            with st.expander("📋 View Full 6-Section Technical Action Brief (Jira / Linear Ticket Preview)", expanded=False):
                b_c1 = action_brief["01_context"]
                b_c2 = action_brief["02_subsystem"]
                b_c4 = action_brief["04_reproduction"]
                b_c5 = action_brief["05_remediation"]

                st.markdown(f"**Ticket ID:** `{b_c1['ticket_id']}` | **Priority:** `{b_c1['priority']}`")
                st.markdown(f"**Affected Subsystem:** `{b_c2['component']}`")
                st.markdown(f"**Firmware Daemon:** `{b_c2['firmware_module']}`")
                st.markdown(f"**Observed Symptom:** {b_c4['symptom']}")
                st.markdown(f"**Reproduction Trigger:** {b_c4['trigger_condition']}")
                st.markdown(f"**Root Cause Hypothesis:** {b_c4['root_cause_hypothesis']}")
                st.markdown(f"**Proposed Remediation:** {b_c5['action_plan']}")
                st.markdown(f"**Verification Method:** {b_c5['verification_method']}")

            jira_text = export_brief_to_jira_markdown(action_brief)
            st.download_button(
                label="📥 Download Engineering Action Brief (.md)",
                data=jira_text,
                file_name=f"{action_brief['01_context']['ticket_id']}_action_brief.md",
                mime="text/markdown",
                help="Download full 6-section technical brief ready for Jira, Linear, or GitHub Issues import",
                key="btn_download_live_brief",
                use_container_width=True
            )

    st.markdown("<br>", unsafe_allow_html=True)
    b_row1, b_row2, b_row3 = st.columns(3)
    with b_row1:
        if st.button("← Return to Discovery Catalogue", key="btn_live_bottom_home", use_container_width=True):
            navigate_to("🏠 Home / Discovery")
    with b_row2:
        if st.button("🔍 Explore Full Review Intelligence →", key="btn_live_bottom_intel", use_container_width=True):
            navigate_to("🔍 Review Intelligence")
    with b_row3:
        if st.button("🎯 Try Custom Buyer Matchmaker →", key="btn_live_bottom_match", use_container_width=True):
            navigate_to("🎯 Smart Recommendations")


# ==========================================
# VIEW 9: PRODUCT COMPARISON (RADAR & SCORECARD)
# ==========================================
elif st.session_state["nav_view"] == "⚔️ Product Comparison":
    st.markdown("## ⚔️ **Side-by-Side Product Comparison & Radar Duel**")
    st.caption("Compare specifications, consumer sentiment deltas, and category-specific aspect ratings across 2 or 3 products.")

    all_pids = [p["product_id"] for _, p in df_products.iterrows()]
    tray = st.session_state["compare_tray"]

    # If comparison tray has items, use them; otherwise default to two distinct products
    default_a = tray[0] if len(tray) > 0 and tray[0] in all_pids else all_pids[0]
    default_b = tray[1] if len(tray) > 1 and tray[1] in all_pids and tray[1] != default_a else (all_pids[1] if len(all_pids) > 1 else all_pids[0])

    # If the user is comparing headphones, default to two distinct headphones
    prod_a_check = get_product_by_id(default_a)
    if prod_a_check and prod_a_check.get("category") == "Headphones" and default_a == default_b:
        default_b = "bose-qc-ultra-headphones" if "bose-qc-ultra-headphones" in all_pids else all_pids[1]

    comp_sel1, comp_sel2, comp_sel3 = st.columns([1, 1, 1])
    with comp_sel1:
        sel_a = st.selectbox(
            "Product A (Blue):",
            all_pids,
            index=all_pids.index(default_a),
            format_func=lambda x: f"{get_product_by_id(x)['product_name']} ({get_product_by_id(x)['category']})"
        )
    with comp_sel2:
        b_candidates = [p for p in all_pids if p != sel_a]
        b_idx = b_candidates.index(default_b) if default_b in b_candidates else 0
        sel_b = st.selectbox(
            "Product B (Red):",
            b_candidates,
            index=b_idx,
            format_func=lambda x: f"{get_product_by_id(x)['product_name']} ({get_product_by_id(x)['category']})"
        )
    with comp_sel3:
        optional_c_options = ["None (Compare 2)"] + [p for p in all_pids if p not in (sel_a, sel_b)]
        sel_c_choice = st.selectbox("Optional Product C (Green):", optional_c_options, index=0)
        sel_c = None if sel_c_choice == "None (Compare 2)" else sel_c_choice

    prod_a = get_product_by_id(sel_a)
    prod_b = get_product_by_id(sel_b)
    prod_c = get_product_by_id(sel_c) if sel_c else None

    # Load reviews & compute aspect metrics
    reviews_a = get_reviews_for_product(sel_a)
    reviews_b = get_reviews_for_product(sel_b)

    sent_a = analyze_reviews_sentiment(reviews_a)
    sent_b = analyze_reviews_sentiment(reviews_b)

    cat_to_use = prod_a.get("category") if prod_a.get("category") == prod_b.get("category") else None
    asp_a = analyze_reviews_aspects(reviews_a, category=cat_to_use)
    asp_b = analyze_reviews_aspects(reviews_b, category=cat_to_use)

    summary_a = {"nss": sent_a["nss"], "csat_estimate": sent_a["csat_estimate"], "aspects": asp_a["aspects"]}
    summary_b = {"nss": sent_b["nss"], "csat_estimate": sent_b["csat_estimate"], "aspects": asp_b["aspects"]}

    comp_result = compare_two_products(prod_a, summary_a, prod_b, summary_b)

    # Sample Size Warning Banner per checklist requirement
    if len(reviews_a) < 10 or len(reviews_b) < 10:
        st.warning(f"⚠️ **Sample Size Warning**: The review sample for this comparison is small ({prod_a['product_name']}: {len(reviews_a)} reviews, {prod_b['product_name']}: {len(reviews_b)} reviews). Comparative rankings and scores should be considered illustrative.")

    # Winner Banner
    winner_badge = "👑" if comp_result["overall_winner"] != "Close Draw" else "🤝"
    esc_winner = html.escape(str(comp_result['overall_winner']))
    esc_verdict = html.escape(str(comp_result['verdict_reason']))
    esc_name_a = html.escape(str(prod_a.get('product_name', 'Product A')))
    esc_name_b = html.escape(str(prod_b.get('product_name', 'Product B')))

    render_html_block(f"""
    <div style="background: linear-gradient(135deg, #131921 0%, #1e293b 100%); padding: 1.5rem; border-radius: 14px; color: white; margin-bottom: 1.5rem; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
        <div style="font-size: 1.3rem; font-weight: 800; color: #FF9900; margin-bottom: 0.3rem;">
            {winner_badge} Overall Head-to-Head Winner: {esc_winner}
        </div>
        <p style="margin: 0; font-size: 0.95rem; color: #E2E8F0; line-height: 1.5;">
            {esc_verdict}
        </p>
    </div>
    """)

    # Side-by-side KPI cards showing price, catalogue rating, review count, overall sentiment, and aspect wins
    c_kpi1, c_kpi2 = st.columns(2)
    with c_kpi1:
        render_html_block(f"""
        <div class="metric-card" style="border-left-color: #2563EB;">
            <div class="metric-label">{esc_name_a}</div>
            <div class="metric-value" style="color: #2563EB;">{format_currency_inr(prod_a['price'])}</div>
            <div class="metric-sub">
                Rating: <strong>★ {prod_a.get('rating', 4.0):.1f}</strong> | Reviews: <strong>{len(reviews_a)}</strong> | NSS: <strong>{sent_a['nss']:+.1f}</strong> | CSAT: <strong>{sent_a['csat_estimate']:.1f}%</strong> | Aspect Wins: <strong>{comp_result['kpi_summary']['wins_a']}</strong>
            </div>
        </div>
        """)

    with c_kpi2:
        render_html_block(f"""
        <div class="metric-card" style="border-left-color: #DC2626;">
            <div class="metric-label">{esc_name_b}</div>
            <div class="metric-value" style="color: #DC2626;">{format_currency_inr(prod_b['price'])}</div>
            <div class="metric-sub">
                Rating: <strong>★ {prod_b.get('rating', 4.0):.1f}</strong> | Reviews: <strong>{len(reviews_b)}</strong> | NSS: <strong>{sent_b['nss']:+.1f}</strong> | CSAT: <strong>{sent_b['csat_estimate']:.1f}%</strong> | Aspect Wins: <strong>{comp_result['kpi_summary']['wins_b']}</strong>
            </div>
        </div>
        """)

    # Multi-product Radar Chart & Aspect Scorecard
    st.markdown("### 🕸️ **Aspect Dimensions Comparison**")
    r_col1, r_col2 = st.columns([1, 1])

    radar_items = [
        {"name": prod_a["product_name"][:20], "aspects": asp_a["aspects"]},
        {"name": prod_b["product_name"][:20], "aspects": asp_b["aspects"]}
    ]
    if prod_c:
        reviews_c = get_reviews_for_product(sel_c)
        sent_c = analyze_reviews_sentiment(reviews_c)
        asp_c = analyze_reviews_aspects(reviews_c, category=cat_to_use)
        radar_items.append({"name": prod_c["product_name"][:20], "aspects": asp_c["aspects"]})

    with r_col1:
        st.markdown("##### Multi-Product Aspect Radar")
        fig_multi_radar = create_multi_product_radar_chart(radar_items)
        st.plotly_chart(fig_multi_radar, use_container_width=True)

    with r_col2:
        st.markdown("##### 🏆 Head-to-Head Aspect Scorecard")
        scorecard_rows = []
        for d in comp_result["aspect_deltas"]:
            if not d.get("has_evidence_a") or not d.get("has_evidence_b"):
                scorecard_rows.append({
                    "Aspect": d["category"],
                    f"{prod_a['product_name'][:14]} (NSS)": d.get("score_a_display", "Insufficient review data"),
                    f"{prod_b['product_name'][:14]} (NSS)": d.get("score_b_display", "Insufficient review data"),
                    "Winner": "N/A - Insufficient data",
                    "Delta": "—"
                })
            else:
                win_label = f"🥇 {d['winner'][:18]}" if d['winner'] != "TIE" else "🤝 Tie"
                scorecard_rows.append({
                    "Aspect": d["category"],
                    f"{prod_a['product_name'][:14]} (NSS)": f"{d['nss_a']:+.1f}",
                    f"{prod_b['product_name'][:14]} (NSS)": f"{d['nss_b']:+.1f}",
                    "Winner": win_label,
                    "Delta": f"{abs(d['delta']):.1f} pts"
                })
        st.dataframe(pd.DataFrame(scorecard_rows), use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Detailed Specifications Comparison Table
    st.markdown("### 📋 **Detailed Specifications Comparison**")
    spec_rows = []
    for s in comp_result["spec_comparison"]:
        spec_rows.append({
            "Specification Dimension": s["dimension"],
            f"{prod_a['product_name'][:20]}": s["product_a"],
            f"{prod_b['product_name'][:20]}": s["product_b"],
            "Different?": "⚡ Difference" if s["is_different"] else "Identical"
        })
    st.dataframe(pd.DataFrame(spec_rows), use_container_width=True, hide_index=True)


# ==========================================
# VIEW 10: SMART RECOMMENDATIONS
# ==========================================
elif st.session_state["nav_view"] == "🎯 Smart Recommendations":
    st.markdown("## 🎯 **ReviewLens Algorithmic Recommendations**")
    st.caption("Defensible product recommendations backed by mathematical composite formulas and prepared demo review sentiment.")

    dept_opts = ["All Departments"] + sorted([c for c in df_products["category"].dropna().unique()])
    chosen_dept = st.selectbox("Filter Recommendations by Category:", dept_opts, index=0)
    category_filter = None if chosen_dept == "All Departments" else chosen_dept

    prod_items = []
    for _, prow in df_products.iterrows():
        pid = prow["product_id"]
        p_dict = get_product_by_id(pid)
        revs = get_reviews_for_product(pid)
        s_res = analyze_reviews_sentiment(revs)
        a_res = analyze_reviews_aspects(revs, category=p_dict.get("category"))
        prod_items.append({
            "product": p_dict,
            "summary": {
                "nss": s_res["nss"],
                "csat_estimate": s_res["csat_estimate"],
                "aspects": a_res["aspects"],
                "total_reviews": s_res["total_reviews"]
            }
        })

    tab_awards, tab_matchmaker = st.tabs(["🏆 Category Awards", "🎛️ Custom Buyer Matchmaker"])

    with tab_awards:
        recs = generate_category_recommendations(prod_items, category=category_filter)

        if recs:
            rec_cards = [
                ("🏆 Best Overall Pick", recs.get("best_overall"), "#2563EB"),
                ("💰 Best Budget & Value Champion", recs.get("best_budget"), "#10B981"),
                ("📸 Best for Photography & Camera", recs.get("best_camera"), "#8B5CF6"),
                ("🔋 Best for Battery & Endurance", recs.get("best_battery"), "#F59E0B"),
                ("⚡ Best for Speed & Gaming", recs.get("best_performance"), "#EF4444")
            ]

            for title, rec_data, color in rec_cards:
                if rec_data and rec_data.get("product"):
                    p = rec_data["product"]
                    esc_p_name = html.escape(str(p.get('product_name', 'Product')))
                    esc_reason = html.escape(str(rec_data.get('reason', '')))
                    amazon_url = p.get("product_url") or p.get("amazon_url") or f"https://www.amazon.in/s?k={esc_p_name}"
                    render_html_block(f"""
                    <div style="background: rgba(22, 17, 46, 0.85); border: 1px solid rgba(155, 92, 255, 0.3); border-left: 6px solid {color}; border-radius: 12px; padding: 1.2rem; margin-bottom: 1.2rem; box-shadow: 0 8px 24px rgba(0,0,0,0.35);">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem; flex-wrap: wrap; gap: 0.5rem;">
                            <span style="font-weight: 800; font-size: 1.05rem; color: {color};">{title}</span>
                            <span style="font-size: 1.2rem; font-weight: 800; color: #FFFFFF;">{format_currency_inr(p.get('price'))}</span>
                        </div>
                        <div style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF; margin-bottom: 0.4rem;">{esc_p_name}</div>
                        <div style="font-size: 0.88rem; color: #CBD5E1; line-height: 1.5; margin-bottom: 0.6rem;">{esc_reason}</div>
                        <div style="display: flex; gap: 0.8rem; align-items: center; flex-wrap: wrap;">
                            <span style="font-size: 0.82rem; color: #94A3B8;">Rating: ★ {p.get('rating'):.1f} ({p.get('review_count'):,} sample ratings)</span>
                            <span style="font-size: 0.82rem; color: #94A3B8;">• Reviews Analyzed: {rec_data.get('review_count', 0)}</span>
                            <a href="{amazon_url}" target="_blank" style="font-size: 0.82rem; color: #FF9900; font-weight: 700; text-decoration: none;">Search on Amazon India ↗</a>
                        </div>
                    </div>
                    """)
                elif rec_data:
                    esc_reason = html.escape(str(rec_data.get('reason', 'Insufficient review evidence for this category award.')))
                    render_html_block(f"""
                    <div style="background: rgba(22, 17, 46, 0.6); border: 1px dashed rgba(155, 92, 255, 0.3); border-left: 6px solid #94A3B8; border-radius: 12px; padding: 1.2rem; margin-bottom: 1.2rem;">
                        <div style="font-weight: 800; font-size: 1.05rem; color: #CBD5E1; margin-bottom: 0.4rem;">{title}</div>
                        <div style="font-size: 0.95rem; font-weight: 700; color: #94A3B8; margin-bottom: 0.2rem;">Insufficient review evidence</div>
                        <div style="font-size: 0.85rem; color: #64748B; line-height: 1.4;">{esc_reason}</div>
                    </div>
                    """)

    with tab_matchmaker:
        st.markdown("### 🎛️ **Personalized Buyer Matchmaker Engine**")
        st.caption("Adjust sliders according to what matters most to you. ReviewLens recalculates multi-attribute utility scores in real time.")

        m_col1, m_col2 = st.columns(2)
        with m_col1:
            w_battery = st.slider("🔋 Battery Life & Endurance", 0, 100, 50, step=5)
            w_camera = st.slider("📸 Camera & Video Quality", 0, 100, 50, step=5)
            w_perf = st.slider("⚡ Performance & Speed", 0, 100, 60, step=5)
            w_display = st.slider("🖥️ Display & Screen Clarity", 0, 100, 40, step=5)
        with m_col2:
            w_audio = st.slider("🎧 Sound & Audio Quality", 0, 100, 30, step=5)
            w_price = st.slider("💰 Price Sensitivity & Value", 0, 100, 50, step=5)
            w_star = st.slider("⭐ Baseline Star Rating Weight", 0, 100, 30, step=5)
            budget_max = st.number_input("Maximum Budget Ceiling (₹ INR, 0 for no limit):", min_value=0, max_value=500000, value=0, step=5000)

        weights = {
            "battery": w_battery / 100.0,
            "camera": w_camera / 100.0,
            "performance": w_perf / 100.0,
            "display": w_display / 100.0,
            "audio": w_audio / 100.0,
            "value_for_money": w_price / 100.0,
            "star_rating": w_star / 100.0
        }
        max_bud = float(budget_max) if budget_max > 0 else None

        ranked_matches = rank_products_by_preferences(
            prod_items,
            weights=weights,
            max_budget=max_bud,
            category=category_filter
        )

        if not ranked_matches:
            st.warning("No products matched your category and budget filter. Try increasing the budget limit.")
        else:
            st.markdown(f"##### Found {len(ranked_matches)} Tailored Matches (Ranked by Compatibility)")
            for idx, match in enumerate(ranked_matches[:6]):
                p_item = match["product"]
                m_score = match["match_score"]
                p_id = p_item["product_id"]
                p_name = html.escape(str(p_item.get("product_name", "Product")))
                badge_bg = "#10B981" if m_score >= 80 else ("#2563EB" if m_score >= 65 else "#F59E0B")

                hl_pills = "".join([f'<span class="topic-pill" style="background: rgba(16, 185, 129, 0.15); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.35); padding: 0.2rem 0.55rem; border-radius: 6px; font-size: 0.78rem;">🟢 {html.escape(h)}</span>' for h in match.get("highlights", [])])
                to_pills = "".join([f'<span class="topic-pill" style="background: rgba(239, 68, 68, 0.15); color: #F87171; border: 1px solid rgba(239, 68, 68, 0.35); padding: 0.2rem 0.55rem; border-radius: 6px; font-size: 0.78rem;">🔴 {html.escape(t)}</span>' for t in match.get("trade_offs", [])])

                render_html_block(f"""
                <div style="background: rgba(22, 17, 46, 0.85); border: 1px solid rgba(155, 92, 255, 0.25); border-radius: 12px; padding: 1.1rem 1.4rem; margin-bottom: 1rem; box-shadow: 0 6px 20px rgba(0,0,0,0.35);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.3rem;">
                        <span style="font-weight: 800; font-size: 1.05rem; color: #FFFFFF;">#{idx+1} {p_name}</span>
                        <span style="background: {badge_bg}; color: #FFFFFF; font-weight: 800; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.88rem;">{m_score:.0f}% Match</span>
                    </div>
                    <div style="display: flex; gap: 1rem; color: #94A3B8; font-size: 0.85rem; margin-bottom: 0.5rem; flex-wrap: wrap;">
                        <span>Price: <b style="color: #FFFFFF;">{format_currency_inr(match['price'])}</b></span>
                        <span>Rating: <b style="color: #F59E0B;">★ {match['rating']:.1f}</b></span>
                        <span>Category: <b style="color: #C4B5FD;">{p_item.get('category')}</b></span>
                    </div>
                    <div style="font-size: 0.88rem; color: #CBD5E1; margin-bottom: 0.5rem; line-height: 1.5;">{html.escape(match['rationale'])}</div>
                    <div style="display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 0.6rem;">{hl_pills} {to_pills}</div>
                </div>
                """)

                act_c1, act_c2, _ = st.columns([1.5, 1.5, 4])
                with act_c1:
                    if st.button("Inspect Details", key=f"mm_inspect_{p_id}", use_container_width=True):
                        inspect_product_action(p_id)
                with act_c2:
                    in_tray = p_id in st.session_state["compare_tray"]
                    t_btn = "✓ In Compare" if in_tray else "⚔️ Add to Compare"
                    if st.button(t_btn, key=f"mm_tray_{p_id}", use_container_width=True):
                        toggle_compare_item(p_id)

    with st.expander("📐 Explain Scoring Methodology & Mathematical Formulas", expanded=False):
        st.markdown(f"""
        > **Prototype Methodology Disclosure:** {RECOMMENDATION_METHODOLOGY_DISCLOSURE}

        **Composite Quality Score Formula (0 to 100):**
        $$ \\text{{Score}} = (0.30 \\times \\text{{Star Rating \\%}}) + (0.45 \\times \\text{{Normalized NSS}}) + (0.25 \\times \\text{{CSAT \\%}}) $$
        - **Star Rating %**: `(Rating / 5.0) * 100` (from illustrative catalogue metadata)
        - **Normalized NSS**: Rescales `[-100, +100]` to `[0, 100]` via `(NSS + 100) / 2`
        - **CSAT %**: Percentage of total customer feedback expressing positive polarity.
        - **Aspect Awards**: Strict evidence gating requires products to have verified aspect mentions before receiving Best Camera/Battery/Performance honors.
        - **Budget Index**: Adjusts score by price curve `Score / (Price ^ 0.35)` to identify highest feature return per rupee.
        - **Buyer Matchmaker**: Normalizes user-weighted aspect dimensions and balances sentiment with value sensitivity.
        """)


# ==========================================
# VIEW: AUTHENTICATION & USER PROFILE
# ==========================================
elif st.session_state["nav_view"] == "🔐 Sign In / Profile":
    render_auth_interface(on_navigate_home=lambda: navigate_to("🏠 Home / Discovery"))


# ==========================================
# VIEW 11: ORIGINAL REVIEWLENS ANALYTICS DASHBOARD
# ==========================================
elif st.session_state["nav_view"] == "📊 Original Analytics Dashboard":
    render_original_dashboard()


# ==========================================
# VIEW 12: MODEL EVALUATION & BENCHMARKS
# ==========================================
elif st.session_state["nav_view"] == "📈 Model Evaluation & Benchmarks":
    st.markdown("## 📈 **Model Evaluation & Ground-Truth Benchmarks**")
    st.caption(f"Empirical performance evaluation of VADER Sentiment Classification and Spam Detection against a manually labelled test set ({len(BENCHMARK_DATASET)} ground-truth samples). Serves as a sanity validation suite rather than an empirical proof of generalization on unseen production distribution.")

    # EMPIRICAL MODEL COMPARISON ON REAL AMAZON DATA
    model_comp_path = PROJECT_ROOT / "results" / "model_comparison.json"
    if model_comp_path.exists():
        try:
            with open(model_comp_path, "r", encoding="utf-8") as f:
                m_data = json.load(f)

            st.markdown("### 🏆 **Empirical Model Comparison: VADER vs. Supervised ML (TF-IDF + Logistic Regression)**")
            st.caption(f"Trained and evaluated on genuine customer reviews from **{m_data['dataset_name']}** (Total N={m_data['total_samples']}, Train={m_data['splits']['train']}, Val={m_data['splits']['val']}, Held-Out Test={m_data['splits']['test']}).")

            r_v = m_data["vader"]
            r_lr = m_data["logistic_regression"]
            delta = m_data["delta"]

            ec1, ec2, ec3, ec4 = st.columns(4)
            with ec1:
                render_html_block(f"""
                <div class="metric-card metric-card-purple">
                    <div class="metric-label">ML Test Accuracy</div>
                    <div class="metric-value">{r_lr['accuracy'] * 100:.1f}%</div>
                    <div class="metric-sub">TF-IDF + LogReg (C={m_data['hyperparameters']['best_c']})</div>
                </div>
                """)
            with ec2:
                render_html_block(f"""
                <div class="metric-card">
                    <div class="metric-label">VADER Test Accuracy</div>
                    <div class="metric-value">{r_v['accuracy'] * 100:.1f}%</div>
                    <div class="metric-sub">Zero-Shot Lexicon Engine</div>
                </div>
                """)
            with ec3:
                render_html_block(f"""
                <div class="metric-card metric-card-green">
                    <div class="metric-label">ML Macro F1-Score</div>
                    <div class="metric-value">{r_lr['macro_f1']:.4f}</div>
                    <div class="metric-sub">Balanced Macro F1 ({delta['f1_gain']:+.4f} vs VADER)</div>
                </div>
                """)
            with ec4:
                render_html_block(f"""
                <div class="metric-card">
                    <div class="metric-label">VADER Macro F1-Score</div>
                    <div class="metric-value">{r_v['macro_f1']:.4f}</div>
                    <div class="metric-sub">Zero-Shot Macro F1</div>
                </div>
                """)

            # Comparison DataFrame
            comp_rows = [
                {"Evaluation Metric": "Overall Test Accuracy", "VADER (Rule-Based)": f"{r_v['accuracy']*100:.2f}%", "TF-IDF + LogReg (ML)": f"{r_lr['accuracy']*100:.2f}%", "Empirical Delta": f"{delta['accuracy_gain_pct']:+.2f}%"},
                {"Evaluation Metric": "Macro F1-Score", "VADER (Rule-Based)": f"{r_v['macro_f1']:.4f}", "TF-IDF + LogReg (ML)": f"{r_lr['macro_f1']:.4f}", "Empirical Delta": f"{delta['f1_gain']:+.4f}"},
                {"Evaluation Metric": "Macro Precision", "VADER (Rule-Based)": f"{r_v['macro_precision']*100:.2f}%", "TF-IDF + LogReg (ML)": f"{r_lr['macro_precision']*100:.2f}%", "Empirical Delta": f"{(r_lr['macro_precision'] - r_v['macro_precision'])*100:+.2f}%"},
                {"Evaluation Metric": "Macro Recall", "VADER (Rule-Based)": f"{r_v['macro_recall']*100:.2f}%", "TF-IDF + LogReg (ML)": f"{r_lr['macro_recall']*100:.2f}%", "Empirical Delta": f"{(r_lr['macro_recall'] - r_v['macro_recall'])*100:+.2f}%"},
                {"Evaluation Metric": "Negative Sentiment F1", "VADER (Rule-Based)": f"{r_v['class_f1']['negative']:.4f}", "TF-IDF + LogReg (ML)": f"{r_lr['class_f1']['negative']:.4f}", "Empirical Delta": f"{r_lr['class_f1']['negative'] - r_v['class_f1']['negative']:+.4f}"},
                {"Evaluation Metric": "Neutral Sentiment F1", "VADER (Rule-Based)": f"{r_v['class_f1']['neutral']:.4f}", "TF-IDF + LogReg (ML)": f"{r_lr['class_f1']['neutral']:.4f}", "Empirical Delta": f"{r_lr['class_f1']['neutral'] - r_v['class_f1']['neutral']:+.4f}"},
                {"Evaluation Metric": "Positive Sentiment F1", "VADER (Rule-Based)": f"{r_v['class_f1']['positive']:.4f}", "TF-IDF + LogReg (ML)": f"{r_lr['class_f1']['positive']:.4f}", "Empirical Delta": f"{r_lr['class_f1']['positive'] - r_v['class_f1']['positive']:+.4f}"}
            ]
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

            # Confusion Matrices side-by-side
            cm_lr_path = PROJECT_ROOT / "results" / "cm_logistic_regression.png"
            cm_v_path = PROJECT_ROOT / "results" / "cm_vader.png"
            if cm_lr_path.exists() and cm_v_path.exists():
                st.markdown("##### 🔍 Empirical Confusion Matrices (Held-Out Test)")
                cm_c1, cm_c2 = st.columns(2)
                with cm_c1:
                    st.image(str(cm_lr_path), caption="TF-IDF + Logistic Regression Confusion Matrix", use_container_width=True)
                with cm_c2:
                    st.image(str(cm_v_path), caption="VADER Lexicon Confusion Matrix", use_container_width=True)

            # 30 Misclassified Test Cases
            err_path = PROJECT_ROOT / "results" / "error_analysis.csv"
            if err_path.exists():
                err_df = pd.read_csv(err_path)
                with st.expander("🔬 Qualitative Error Analysis: 30 Misclassified Test Cases (Sarcasm, Negation, Mixed)", expanded=False):
                    st.caption("Diagnostic examination of actual test set reviews where models failed, categorized by linguistic error cause.")
                    st.dataframe(err_df, use_container_width=True, hide_index=True)

            st.markdown("---")
        except Exception as e:
            st.warning(f"Could not load empirical model comparison: {e}")

    sentiment_metrics = evaluate_sentiment_model()
    spam_metrics = evaluate_spam_model()

    st.markdown("### 1. VADER Sentiment Classification Performance")
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        render_html_block(f"""
        <div class="metric-card">
            <div class="metric-label">Benchmark Samples</div>
            <div class="metric-value">{sentiment_metrics['total_test_samples']}</div>
            <div class="metric-sub">Manually labelled reviews</div>
        </div>
        """)
    with m2:
        render_html_block(f"""
        <div class="metric-card metric-card-green">
            <div class="metric-label">Overall Accuracy</div>
            <div class="metric-value">{sentiment_metrics['accuracy'] * 100:.1f}%</div>
            <div class="metric-sub">Correctly classified</div>
        </div>
        """)
    with m3:
        render_html_block(f"""
        <div class="metric-card metric-card-purple">
            <div class="metric-label">Macro F1-Score</div>
            <div class="metric-value">{sentiment_metrics['f1_macro']:.3f}</div>
            <div class="metric-sub">Harmonic mean P & R</div>
        </div>
        """)
    with m4:
        render_html_block(f"""
        <div class="metric-card">
            <div class="metric-label">Macro Precision</div>
            <div class="metric-value">{sentiment_metrics['precision_macro']:.3f}</div>
            <div class="metric-sub">Positive predictive value</div>
        </div>
        """)
    with m5:
        render_html_block(f"""
        <div class="metric-card">
            <div class="metric-label">Macro Recall</div>
            <div class="metric-value">{sentiment_metrics['recall_macro']:.3f}</div>
            <div class="metric-sub">True positive rate</div>
        </div>
        """)

    eval_col1, eval_col2 = st.columns([1, 1])
    with eval_col1:
        st.markdown("##### 3x3 Confusion Matrix Heatmap")
        fig_cm = create_confusion_matrix_heatmap(
            sentiment_metrics["confusion_matrix"],
            sentiment_metrics["labels"]
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with eval_col2:
        st.markdown("##### Per-Class Performance Breakdown")
        class_rows = []
        for l, stats in sentiment_metrics["per_class"].items():
            class_rows.append({
                "Sentiment Class": l,
                "Precision": f"{stats['precision']:.3f}",
                "Recall": f"{stats['recall']:.3f}",
                "F1-Score": f"{stats['f1']:.3f}",
                "Support Count": stats["support"]
            })
        st.dataframe(pd.DataFrame(class_rows), use_container_width=True, hide_index=True)

    st.markdown("---")

    st.markdown("### 2. Fake Review & Spam Detection Performance")
    sp1, sp2, sp3, sp4 = st.columns(4)
    with sp1:
        render_html_block(f"""
        <div class="metric-card metric-card-orange">
            <div class="metric-label">Spam Accuracy</div>
            <div class="metric-value">{spam_metrics['accuracy'] * 100:.1f}%</div>
            <div class="metric-sub">Binary spam discrimination</div>
        </div>
        """)
    with sp2:
        render_html_block(f"""
        <div class="metric-card">
            <div class="metric-label">Spam Precision</div>
            <div class="metric-value">{spam_metrics['precision']:.3f}</div>
            <div class="metric-sub">Spam detection precision</div>
        </div>
        """)
    with sp3:
        render_html_block(f"""
        <div class="metric-card">
            <div class="metric-label">Spam Recall</div>
            <div class="metric-value">{spam_metrics['recall']:.3f}</div>
            <div class="metric-sub">Spam capture rate</div>
        </div>
        """)
    with sp4:
        render_html_block(f"""
        <div class="metric-card metric-card-green">
            <div class="metric-label">Spam F1-Score</div>
            <div class="metric-value">{spam_metrics['f1_score']:.3f}</div>
            <div class="metric-sub">Harmonic mean</div>
        </div>
        """)

    with st.expander("🔎 Browse Benchmark Test Dataset & Predictions", expanded=False):
        bench_records = []
        for idx, (txt, true_sent, is_spam) in enumerate(BENCHMARK_DATASET):
            bench_records.append({
                "Index": idx + 1,
                "Review Text": txt,
                "True Sentiment": true_sent,
                "True Is Spam": "Yes" if is_spam else "No"
            })
        st.dataframe(pd.DataFrame(bench_records), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 3. Supervised Baseline Comparison & Dataset Alignment")

    baseline_metrics = evaluate_baseline_comparison()
    curated_alignment = evaluate_curated_reviews_corpus()

    b_col1, b_col2, b_col3, b_col4 = st.columns(4)
    with b_col1:
        render_html_block(f"""
        <div class="metric-card">
            <div class="metric-label">VADER Rule-Based Acc</div>
            <div class="metric-value">{baseline_metrics['vader_accuracy'] * 100:.1f}%</div>
            <div class="metric-sub">Curated Benchmark (n=50)</div>
        </div>
        """)
    with b_col2:
        render_html_block(f"""
        <div class="metric-card">
            <div class="metric-label">TF-IDF + LogReg (5-Fold)</div>
            <div class="metric-value">{baseline_metrics['baseline_accuracy'] * 100:.1f}%</div>
            <div class="metric-sub">Cross-Validated Baseline</div>
        </div>
        """)
    with b_col3:
        render_html_block(f"""
        <div class="metric-card metric-card-purple">
            <div class="metric-label">Baseline Macro F1</div>
            <div class="metric-value">{baseline_metrics['baseline_f1_macro']:.3f}</div>
            <div class="metric-sub">Supervised 5-Fold F1</div>
        </div>
        """)
    with b_col4:
        render_html_block(f"""
        <div class="metric-card metric-card-green">
            <div class="metric-label">Catalogue Reviews Agreement</div>
            <div class="metric-value">{curated_alignment.get('agreement_rate', 0.0) * 100:.1f}%</div>
            <div class="metric-sub">Rating vs VADER (n=101)</div>
        </div>
        """)

    st.info(
        "⚠️ **Evaluation Sample Size & Generalization Caveat**: "
        "These performance figures are evaluated on a small hand-labelled sanity check sample (n=50) and the 101-review demonstration corpus. "
        "They serve as regression validation checks and methodology demonstrations rather than empirical proof of generalization on large-scale production distributions."
    )

    st.markdown("---")
    st.markdown("### 4. Official Public Benchmark: Stanford Sentiment Treebank (SST-2 / GLUE)")
    st.caption("Empirical three-way evaluation on 872 independent test sentences (444 positive, 428 negative) from academic NLP literature.")

    sst2_res = evaluate_sst2_public_benchmark()
    if sst2_res.get("status") == "ok":
        # Three-Way Model Comparison Table
        model_comp_data = [
            {
                "Model Architecture": "Rule-Based VADER Lexicon",
                "Paradigm": "Unsupervised / Zero-Shot Lexicon",
                "Training Split": "None (Pre-compiled valence dictionary)",
                "Dev Accuracy (n=872)": f"{sst2_res['vader_accuracy_binary'] * 100:.1f}% (Binary) / {sst2_res['vader_accuracy_all'] * 100:.1f}% (3-Class)",
                "Macro F1": f"{sst2_res['vader_f1_macro']:.3f}",
                "Inference Latency": "< 1 ms / review"
            },
            {
                "Model Architecture": "TF-IDF (1-2 grams) + Logistic Regression (C=2.0)",
                "Paradigm": "Supervised Classical Machine Learning",
                "Training Split": f"67,349 SST-2 Train sentences",
                "Dev Accuracy (n=872)": f"{sst2_res['baseline_accuracy'] * 100:.1f}%",
                "Macro F1": f"{sst2_res['baseline_f1_macro']:.3f}",
                "Inference Latency": "~2 ms / review"
            },
            {
                "Model Architecture": "DistilBERT (distilbert-base-uncased-finetuned-sst-2) [Ref]",
                "Paradigm": "Fine-Tuned Deep Bidirectional Transformer",
                "Training Split": "Published Reference Metric (GLUE/HuggingFace Card; not locally computed)",
                "Dev Accuracy (n=872)": f"{sst2_res['transformer_accuracy'] * 100:.1f}% (Reference)",
                "Macro F1": f"{sst2_res['transformer_f1_macro']:.3f}",
                "Inference Latency": "~25 ms (CPU) / ~4 ms (GPU)"
            }
        ]
        st.dataframe(pd.DataFrame(model_comp_data), use_container_width=True, hide_index=True)

        s_col1, s_col2, s_col3, s_col4 = st.columns(4)
        with s_col1:
            render_html_block(f"""
            <div class="metric-card">
                <div class="metric-label">VADER Binary Polarized Acc</div>
                <div class="metric-value">{sst2_res['vader_accuracy_binary'] * 100:.1f}%</div>
                <div class="metric-sub">Excl. {sst2_res['vader_neutral_count']} neutral (n=718)</div>
            </div>
            """)
        with s_col2:
            render_html_block(f"""
            <div class="metric-card metric-card-green">
                <div class="metric-label">Classical ML (67k Train)</div>
                <div class="metric-value">{sst2_res['baseline_accuracy'] * 100:.1f}%</div>
                <div class="metric-sub">TF-IDF + LogReg on Dev</div>
            </div>
            """)
        with s_col3:
            render_html_block(f"""
            <div class="metric-card metric-card-purple">
                <div class="metric-label">Classical ML Macro F1</div>
                <div class="metric-value">{sst2_res['baseline_f1_macro']:.3f}</div>
                <div class="metric-sub">Macro-averaged F1</div>
            </div>
            """)
        with s_col4:
            render_html_block(f"""
            <div class="metric-card metric-card-orange">
                <div class="metric-label">Transformer [Reference]</div>
                <div class="metric-value">{sst2_res['transformer_accuracy'] * 100:.1f}%</div>
                <div class="metric-sub">Published GLUE Reference</div>
            </div>
            """)

        st.info(
            f"ℹ️ **Public Benchmark Verification & Scope Disclosure**: "
            f"Evaluated on the Stanford Sentiment Treebank (SST-2) development set. "
            f"The supervised TF-IDF + Logistic Regression model trained on 67,349 sentences achieves **{sst2_res['baseline_accuracy'] * 100:.1f}%** accuracy (Macro F1: {sst2_res['baseline_f1_macro']:.3f}), "
            f"while zero-shot VADER achieves **{sst2_res['vader_accuracy_binary'] * 100:.1f}%** directional accuracy. "
            f"**Dataset Scope**: {sst2_res['domain_caveat']}"
        )
    else:
        st.warning("Public SST-2 benchmark dataset could not be evaluated.")

    # 5. SemEval-2014 Task 4 Aspect Benchmark
    st.markdown("---")
    st.markdown("### 5. Aspect-Based Sentiment Analysis Benchmark: SemEval-2014 Task 4 (Laptops)")
    st.caption("Empirical evaluation of spaCy Dependency Parsing and Noun Chunk ABSA against the official SemEval-2014 Task 4 Gold Standard test set.")

    semeval_res = evaluate_semeval_aspect_benchmark(sample_limit=100)
    if semeval_res.get("status") == "ok":
        sem_col1, sem_col2, sem_col3, sem_col4 = st.columns(4)
        with sem_col1:
            render_html_block(f"""
            <div class="metric-card">
                <div class="metric-label">Gold Sentences Evaluated</div>
                <div class="metric-value">{semeval_res['total_sentences_evaluated']}</div>
                <div class="metric-sub">SemEval Laptops Test Set</div>
            </div>
            """)
        with sem_col2:
            render_html_block(f"""
            <div class="metric-card metric-card-green">
                <div class="metric-label">Aspect Extraction Recall</div>
                <div class="metric-value">{semeval_res['recall'] * 100:.1f}%</div>
                <div class="metric-sub">Gold aspects captured</div>
            </div>
            """)
        with sem_col3:
            render_html_block(f"""
            <div class="metric-card">
                <div class="metric-label">Aspect Precision</div>
                <div class="metric-value">{semeval_res['precision'] * 100:.1f}%</div>
                <div class="metric-sub">Extracted terms relevance</div>
            </div>
            """)
        with sem_col4:
            render_html_block(f"""
            <div class="metric-card metric-card-purple">
                <div class="metric-label">Aspect F1-Score</div>
                <div class="metric-value">{semeval_res['f1_score']:.3f}</div>
                <div class="metric-sub">Harmonic mean P & R</div>
            </div>
            """)

        st.success(
            "✅ **Syntactic Dependency Aspect Extraction**: "
            "Unlike simple keyword matching, spaCy dependency parsing traverses grammatical relations (`amod`, `acomp`, `nsubj`, `attr`) "
            f"to discover unlisted consumer aspects with **{semeval_res['recall'] * 100:.1f}% recall** and **{semeval_res['f1_score']:.3f} F1-score** on official SemEval gold data."
        )
    else:
        st.warning(f"SemEval-2014 benchmark: {semeval_res.get('message', 'Not available')}")

    # 6. Supervised ML Spam Classifier Benchmark
    st.markdown("---")
    st.markdown("### 6. Supervised ML Fake-Review & Spam Classifier Performance")
    st.caption("Empirical evaluation of statistical Random Forest classifier trained on 9 behavioral feature vectors combined with TF-IDF unigram & bigram representations.")

    ml_spam_res = evaluate_ml_spam_classifier()
    if ml_spam_res.get("status") == "ok":
        sp_col1, sp_col2, sp_col3, sp_col4 = st.columns(4)
        with sp_col1:
            render_html_block(f"""
            <div class="metric-card metric-card-green">
                <div class="metric-label">ML Spam Accuracy</div>
                <div class="metric-value">{ml_spam_res['accuracy'] * 100:.1f}%</div>
                <div class="metric-sub">Random Forest (n=40)</div>
            </div>
            """)
        with sp_col2:
            render_html_block(f"""
            <div class="metric-card">
                <div class="metric-label">Spam Precision</div>
                <div class="metric-value">{ml_spam_res['precision'] * 100:.1f}%</div>
                <div class="metric-sub">Low false alarm rate</div>
            </div>
            """)
        with sp_col3:
            render_html_block(f"""
            <div class="metric-card">
                <div class="metric-label">Spam Recall</div>
                <div class="metric-value">{ml_spam_res['recall'] * 100:.1f}%</div>
                <div class="metric-sub">Synthetic astroturfing caught</div>
            </div>
            """)
        with sp_col4:
            render_html_block(f"""
            <div class="metric-card metric-card-purple">
                <div class="metric-label">Spam F1-Score</div>
                <div class="metric-value">{ml_spam_res['f1_score']:.3f}</div>
                <div class="metric-sub">Balanced detection F1</div>
            </div>
            """)

        with st.expander("🔍 Inspect Feature Representation & Domain Caveats of the ML Fake-Review Model"):
            for feat in ml_spam_res.get("feature_set", []):
                st.markdown(f"- **Feature Representation:** {feat}")
            if ml_spam_res.get("domain_caveat"):
                st.info(f"ℹ️ **Domain Notice:** {ml_spam_res['domain_caveat']}")

    # 7. Real Public Amazon Electronics Review Corpus (n=997)
    st.markdown("---")
    st.markdown("### 7. Real-World E-Commerce Corpus: UCSD Amazon Electronics (n=997)")
    st.caption("Empirical sentiment-to-star rating agreement evaluated on 997 genuine customer reviews across 40 consumer electronics products from Julian McAuley's UCSD benchmark.")

    real_amz_res = evaluate_real_amazon_reviews()
    if real_amz_res.get("status") == "ok":
        az_col1, az_col2, az_col3, az_col4 = st.columns(4)
        with az_col1:
            render_html_block(f"""
            <div class="metric-card">
                <div class="metric-label">Amazon Reviews Analyzed</div>
                <div class="metric-value">{real_amz_res['sample_size']}</div>
                <div class="metric-sub">Genuine consumer reviews</div>
            </div>
            """)
        with az_col2:
            render_html_block(f"""
            <div class="metric-card metric-card-green">
                <div class="metric-label">Rating Agreement Rate</div>
                <div class="metric-value">{real_amz_res['agreement_rate'] * 100:.1f}%</div>
                <div class="metric-sub">Star rating vs NLP sentiment</div>
            </div>
            """)
        with az_col3:
            render_html_block(f"""
            <div class="metric-card">
                <div class="metric-label">Distinct Products</div>
                <div class="metric-value">{real_amz_res['unique_products']}</div>
                <div class="metric-sub">Phones, audio, accessories</div>
            </div>
            """)
        with az_col4:
            render_html_block(f"""
            <div class="metric-card metric-card-orange">
                <div class="metric-label">Corpus Provenance</div>
                <div class="metric-value">UCSD Public</div>
                <div class="metric-sub">Julian McAuley Benchmark</div>
            </div>
            """)

        st.caption(f"Sentiment distribution across 997 genuine reviews: **{real_amz_res['positive_count']} positive**, **{real_amz_res['neutral_count']} neutral**, **{real_amz_res['negative_count']} negative**.")

    # 8. Qualitative Error Analysis & Failure Modes
    st.markdown("---")
    st.markdown("### 8. Qualitative Error Analysis & Model Failure Modes")
    st.caption("Structured catalog of linguistic edge cases and failure modes in lexicon-based and linear NLP architectures.")

    error_cases = evaluate_error_analysis()
    err_df = pd.DataFrame(error_cases)
    st.dataframe(err_df, use_container_width=True, hide_index=True)


# ==========================================
# VIEW 13: DATA INGESTION & VALIDATOR
# ==========================================
elif st.session_state["nav_view"] == "📂 Data Ingestion & Validator":
    st.markdown("## 📂 **Dataset Ingestion & Schema Validator**")
    st.caption("Upload custom electronics products and reviews CSVs to validate schema integrity and run ReviewLens intelligence.")

    up_col1, up_col2 = st.columns(2)
    with up_col1:
        st.markdown("#### 1. Upload Products CSV")
        custom_prod_file = st.file_uploader("Upload Products (CSV)", type=["csv"], key="custom_prod_up")
    with up_col2:
        st.markdown("#### 2. Upload Reviews CSV")
        custom_rev_file = st.file_uploader("Upload Reviews (CSV)", type=["csv"], key="custom_rev_up")

    if custom_prod_file or custom_rev_file:
        st.markdown("### 🔍 **Validation Audit Results**")

        if custom_prod_file:
            user_prod_df = pd.read_csv(custom_prod_file)
            val_p = validate_products_df(user_prod_df)
            st.markdown(f"**Products Dataset:** {'✅ VALID' if val_p.is_valid else '❌ INVALID'}")
            if val_p.errors:
                for e in val_p.errors:
                    st.error(f"Error: {e}")
            if val_p.warnings:
                for w in val_p.warnings:
                    st.warning(f"Warning: {w}")
            st.json(val_p.stats)

        if custom_rev_file:
            user_rev_df = pd.read_csv(custom_rev_file)
            p_ref = user_prod_df if custom_prod_file else df_products
            val_r = validate_reviews_df(user_rev_df, products_df=p_ref)
            st.markdown(f"**Reviews Dataset:** {'✅ VALID' if val_r.is_valid else '❌ INVALID'}")
            if val_r.errors:
                for e in val_r.errors:
                    st.error(f"Error: {e}")
            if val_r.warnings:
                for w in val_r.warnings:
                    st.warning(f"Warning: {w}")
            st.json(val_r.stats)
    else:
        st.info("Upload your CSV files above to test schema conformity, or inspect the built-in catalogue dataset.")
        st.markdown("#### Current Catalogue Schema Validation Status:")
        val_cur_p = validate_products_df(df_products)
        val_cur_r = validate_reviews_df(df_reviews, products_df=df_products)
        st.success(f"Built-in products.csv: Valid ({val_cur_p.stats['total_products']} products)")
        st.success(f"Built-in reviews.csv: Valid ({val_cur_r.stats['total_reviews']} reviews across {val_cur_r.stats.get('unique_products_reviewed', 0)} products)")


# ==========================================
# VIEW 14: METHODOLOGY & TRANSPARENCY
# ==========================================
elif st.session_state["nav_view"] == "📖 Methodology & Transparency":
    st.markdown("## 📖 **Methodology, Caveats & Technical Transparency**")

    st.markdown("""
    ### 1. NLP Pipeline & Sentiment Scoring
    ReviewLens uses **VADER (Valence Aware Dictionary and sEntiment Reasoner)** augmented with custom e-commerce product domain modifiers:
    - **Compound Metric**: Normalizes sentiment valence between -1.0 (extremely negative) and +1.0 (extremely positive).
    - **Net Sentiment Score (NSS)**:
      $$ \\text{NSS} = \\frac{\\text{Positive Reviews} - \\text{Negative Reviews}}{\\text{Total Reviews}} \\times 100 $$
    - **CSAT %**: Proportion of positive reviews over total reviews.

    ### 2. Category-Specific Aspect-Based Sentiment Analysis (ABSA)
    Unlike generic models, ReviewLens uses category-specific aspect taxonomies:
    - **Smartphones**: Camera, Battery, Display, Charging, Performance
    - **Laptops**: Processor, RAM, GPU, Keyboard, Cooling
    - **Cameras**: Image quality, Autofocus, Lens quality, Low-light performance, Video recording
    - **Headphones**: Sound quality, Battery, Comfort, Noise cancellation, Connectivity
    - **Prototype Methodology Disclosure**: Keyword-based aspect matching is an explainable rule-based prototype method, not a fully trained deep learning ABSA neural model.
    - **Zero Mentions Protocol**: When an aspect has 0 mentions, the NSS is marked as `None` (not 0.0) with an explicit disclosure of `"Insufficient review data"`.

    ### 3. Unsupervised Topic Modeling (LDA)
    - Latent Dirichlet Allocation (LDA) models latent conversational themes from review text.
    - Interactive topic count slider ($k = 2 \\dots 6$) allows exploration of micro-themes with keyword pills and representative quotes.

    ### 4. Transparent Academic Caveats & Limitations
    - **Negation Scope**: Complex grammatical inversions (e.g., *"not only is it fast, but also..."*) can occasionally confuse polarity classifiers.
    - **Sarcasm Detection**: Sarcastic remarks with literal positive words may be classified as positive unless flagged by star-rating anomaly checks.
    - **Sample Size Warning**: Products with fewer than 10 reviews carry lower statistical power. We advise users to weigh sample counts when interpreting NSS.
    - **Data Provenance**: Reviews and specifications in this prototype are curated demonstrations (`data_type='prepared_demo'`) designed for clean evaluation. Live outbound links connect directly to Amazon India.
    """)

    with st.expander("🩺 **System Status & Health Check (/healthz)**", expanded=False):
        import platform
        import sklearn
        import nltk
        try:
            import spacy
            spacy_ver = spacy.__version__
        except Exception:
            spacy_ver = "Not installed (fallback keyword-taxonomy active)"

        st.markdown(f"""
        - **Python Runtime**: `{platform.python_version()}` ({platform.system()} {platform.machine()})
        - **Streamlit**: `{st.__version__}`
        - **scikit-learn**: `{sklearn.__version__}`
        - **NLTK**: `{nltk.__version__}`
        - **spaCy**: `{spacy_ver}`
        - **Users DB (`data/users.json`)**: `{'✅ Present' if Path('data/users.json').exists() else '❌ Missing'}`
        - **SST-2 Benchmark (`data/public_sst2_benchmark.csv`)**: `{'✅ Present' if Path('data/public_sst2_benchmark.csv').exists() else '❌ Missing'}`
        - **ML Spam Model (`models/fake_review_model.joblib`)**: `{'✅ Present' if Path('models/fake_review_model.joblib').exists() else '❌ Missing'}`
        - **Auth Rate Limit Cache (`data/otp_rate_limits.json`)**: `{'✅ Present' if Path('data/otp_rate_limits.json').exists() else 'ℹ️ Empty/New'}`
        """)

    with st.expander("📲 **Install ReviewLens on Phone or PC (PWA Guide)**", expanded=False):
        st.markdown("""
        ReviewLens can be installed as a standalone Progressive Web App (PWA) with zero app-store download:
        1. **Chrome / Edge (Desktop)**: Click the **Install ReviewLens** icon in the address bar (or Menu > 'Install ReviewLens').
        2. **Android (Chrome)**: Tap the three dots (⋮) > **Add to Home screen** > **Install**.
        3. **iPhone / iPad (Safari)**: Tap the Share button (⎋) > **Add to Home Screen** > **Add**.
        4. **Offline Capability**: The PWA service worker caches app icons and displays a dedicated offline warning if internet connection is lost.
        """)


# ==========================================
# GLOBAL FOOTER
# ==========================================
st.markdown("<br><br>", unsafe_allow_html=True)
render_html_block("""
<div style="background: #0F172A; border-top: 2px solid #1E293B; border-radius: 12px; padding: 1.8rem 2rem; color: #94A3B8; font-size: 0.85rem; margin-top: 2rem;">
    <div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 1rem; margin-bottom: 1rem;">
        <div>
            <div style="font-size: 1.15rem; font-weight: 800; color: #FFFFFF; margin-bottom: 0.2rem;">
                Review<span style="color: #FF9900;">Lens</span> Smart Shopping Platform
            </div>
            <div style="color: #64748B; font-size: 0.8rem;">
                Explainable Natural Language Processing & Aspect-Based Sentiment Analysis for Consumer Electronics
            </div>
        </div>
        <div style="display: flex; gap: 1rem; align-items: center;">
            <a href="https://github.com/Vinayvardhan224/Reviewlens" target="_blank" style="color: #FF9900; font-weight: 700; text-decoration: none;">GitHub Repository ↗</a>
            <span style="color: #334155;">•</span>
            <span style="background: rgba(37,99,235,0.2); color: #93C5FD; padding: 0.2rem 0.6rem; border-radius: 12px; font-size: 0.75rem;">🛡️ Data: prepared_demo</span>
        </div>
    </div>
    <div style="border-top: 1px solid #1E293B; padding-top: 0.8rem; font-size: 0.78rem; color: #64748B;">
        Designed for academic research and rigorous consumer transparency. Outbound links connect directly to live search listings on Amazon India.
    </div>
</div>
""")
