"""
ReviewLens: Smart Shopping Platform & NLP Review Intelligence.
Complete implementation conforming to the ReviewLens Blueprint:
Amazon-inspired discovery UI, explainable Aspect-Based Sentiment Analysis (ABSA),
multi-product comparison, transparent recommendations, and dataset validation.
"""

import sys
from pathlib import Path
import json
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
    PROVENANCE_DISCLOSURE
)
from modules.validation import validate_products_df, validate_reviews_df
from modules.sentiment import analyze_reviews_sentiment, compute_nss
from modules.aspect_analysis import analyze_reviews_aspects, ASPECT_CAVEAT_DISCLOSURE
from modules.keyword_extraction import extract_keywords_by_sentiment
from modules.review_summary import generate_executive_summary
from modules.comparison import compare_two_products
from modules.recommendation import generate_category_recommendations
from components.header import render_header
from components.product_card import render_product_card
from components.filters import render_sidebar_filters, apply_filters
from components.charts import (
    create_sentiment_donut_chart,
    create_aspect_breakdown_bar_chart,
    create_aspect_radar_chart,
    create_multi_product_radar_chart,
    create_keyword_driver_chart,
    create_rating_scatter_chart
)

# Page configuration
st.set_page_config(
    page_title="ReviewLens | Smart Shopping & Review Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


def render_html_block(html_content: str) -> None:
    """
    Renders HTML safely without Markdown treating indented lines as code blocks.
    Strips leading and trailing whitespace from each line and drops blank lines.
    """
    clean_lines = [line.strip() for line in html_content.strip().splitlines() if line.strip()]
    st.markdown("\n".join(clean_lines), unsafe_allow_html=True)


# CSS Styling (Amazon Navy #131921, Amber #FF9900, Clean SaaS cards)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Primary buttons */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }

    /* KPI Cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.1rem;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
        margin-bottom: 1rem;
        border-left: 4px solid #2563EB;
    }
    .metric-card-orange {
        border-left-color: #FF9900;
    }
    .metric-card-green {
        border-left-color: #10B981;
    }
    .metric-card-purple {
        border-left-color: #8B5CF6;
    }
    .metric-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 800;
        color: #0F172A;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #94A3B8;
        margin-top: 0.2rem;
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #131921 0%, #1e293b 100%);
        border-radius: 14px;
        padding: 2.2rem;
        color: #FFFFFF;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* Provenance alert badge */
    .provenance-banner {
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 10px;
        padding: 0.85rem 1.2rem;
        color: #1E40AF;
        font-size: 0.85rem;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "compare_tray" not in st.session_state:
    st.session_state["compare_tray"] = ["samsung-galaxy-s24-fe", "apple-iphone-15-pro-max"]

if "selected_product_id" not in st.session_state:
    st.session_state["selected_product_id"] = "samsung-galaxy-s24-fe"

if "nav_view" not in st.session_state:
    st.session_state["nav_view"] = "🏠 Home / Discovery"


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


def inspect_product_action(pid: str):
    """Set selected product and switch view to Product Intelligence."""
    st.session_state["selected_product_id"] = pid
    st.session_state["nav_view"] = "🔍 Product Intelligence & ABSA"


# Load catalogue & reviews
df_products = get_all_products()
df_reviews = get_all_reviews()

# Top utility header
render_header(
    compare_count=len(st.session_state["compare_tray"]),
    active_nav=st.session_state["nav_view"]
)

# Sidebar Navigation & Controls
st.sidebar.markdown("## 🧭 **Navigation**")
NAV_OPTIONS = [
    "🏠 Home / Discovery",
    "🛍️ Catalogue & Search",
    "🔍 Product Intelligence & ABSA",
    "⚔️ Product Comparison",
    "🎯 Smart Recommendations",
    "📂 Data Upload & Validator",
    "📖 Methodology & Transparency"
]

selected_view = st.sidebar.radio(
    "Go to",
    NAV_OPTIONS,
    index=NAV_OPTIONS.index(st.session_state["nav_view"]) if st.session_state["nav_view"] in NAV_OPTIONS else 0,
    key="sidebar_nav_radio"
)
st.session_state["nav_view"] = selected_view

st.sidebar.markdown("---")

# Quick compare tray widget in sidebar
st.sidebar.markdown(f"### ⚔️ Compare Tray ({len(st.session_state['compare_tray'])}/3)")
if st.session_state["compare_tray"]:
    for pid in st.session_state["compare_tray"]:
        p_info = get_product_by_id(pid)
        p_name = p_info["product_name"] if p_info else pid
        c_row1, c_row2 = st.sidebar.columns([4, 1])
        c_row1.caption(f"• {p_name[:25]}...")
        if c_row2.button("✕", key=f"del_tray_{pid}", help="Remove from compare"):
            st.session_state["compare_tray"].remove(pid)
            st.rerun()

    if st.sidebar.button("⚔️ Compare Selected Now", use_container_width=True, type="primary"):
        st.session_state["nav_view"] = "⚔️ Product Comparison"
        st.rerun()
else:
    st.sidebar.caption("No products queued. Click '⚔️ Compare' on any product card.")

st.sidebar.markdown("---")
st.sidebar.caption("🛡️ **Data Provenance**: All products & reviews are labeled `prepared_demo` for transparent NLP analysis.")


# ==========================================
# VIEW 1: HOME / DISCOVERY
# ==========================================
if st.session_state["nav_view"] == "🏠 Home / Discovery":
    render_html_block("""
    <div class="hero-banner">
        <div style="display: inline-block; padding: 4px 12px; border-radius: 20px; background: rgba(255,153,0,0.2); border: 1px solid #FF9900; color: #FF9900; font-size: 0.8rem; font-weight: 700; margin-bottom: 0.8rem;">
            NLP-POWERED SMART SHOPPING INTELLIGENCE
        </div>
        <h1 style="font-size: 2.2rem; font-weight: 800; color: #FFFFFF; margin: 0 0 0.8rem 0; line-height: 1.2;">
            Make Smarter Buying Decisions with Explainable Review AI
        </h1>
        <p style="font-size: 1.05rem; color: #CBD5E1; max-width: 800px; line-height: 1.6; margin-bottom: 1.4rem;">
            ReviewLens cuts through raw customer feedback, extracts aspect-level sentiment 
            (Camera, Battery, Performance, Display), detects fake rating anomalies, and 
            compares electronics side-by-side with 100% mathematical transparency.
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
            <div class="metric-sub">Phones, Audio, Laptops, Cameras</div>
        </div>
        """)
    with stat2:
        render_html_block("""
        <div class="metric-card">
            <div class="metric-label">Verified Reviews</div>
            <div class="metric-value">70+ Feedback</div>
            <div class="metric-sub">Prepared & validated records</div>
        </div>
        """)
    with stat3:
        render_html_block("""
        <div class="metric-card metric-card-green">
            <div class="metric-label">Aspect Dimensions</div>
            <div class="metric-value">8 Key Pillars</div>
            <div class="metric-sub">Speed, Display, Battery, Value...</div>
        </div>
        """)
    with stat4:
        render_html_block("""
        <div class="metric-card metric-card-purple">
            <div class="metric-label">NLP Explainability</div>
            <div class="metric-value">100% Glass Box</div>
            <div class="metric-sub">VADER + Rule-based Lexicon</div>
        </div>
        """)

    # Department Quick-Access Row
    st.markdown("### 🏷️ **Shop by Department**")
    cat_cols = st.columns(5)
    departments = [
        ("Smartphones", "📱", "Flagships & Mid-range"),
        ("Audio", "🎧", "Headphones & Earbuds"),
        ("Laptops", "💻", "Ultrabooks & Gaming"),
        ("Wearables", "⌚", "Smartwatches & Fitness"),
        ("Cameras", "📷", "Mirrorless & Full-Frame")
    ]
    for idx, (cat_name, cat_icon, cat_sub) in enumerate(departments):
        with cat_cols[idx]:
            render_html_block(f"""
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 1rem; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
                <div style="font-size: 2rem; margin-bottom: 0.3rem;">{cat_icon}</div>
                <div style="font-weight: 700; font-size: 0.95rem; color: #0F172A;">{cat_name}</div>
                <div style="font-size: 0.75rem; color: #64748B;">{cat_sub}</div>
            </div>
            """)

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
        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 1.2rem;">
            <div style="font-size: 1.5rem; margin-bottom: 0.4rem;">🔍</div>
            <div style="font-weight: 700; color: #0F172A; margin-bottom: 0.4rem;">Aspect-Based Sentiment</div>
            <div style="font-size: 0.85rem; color: #475569; line-height: 1.5;">
                A phone may have a 4.5-star overall rating, but terrible 25W charging speeds. ReviewLens deconstructs feedback into 8 separate dimensions.
            </div>
        </div>
        """)
    with vp2:
        render_html_block("""
        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 1.2rem;">
            <div style="font-size: 1.5rem; margin-bottom: 0.4rem;">⚔️</div>
            <div style="font-weight: 700; color: #0F172A; margin-bottom: 0.4rem;">Side-by-Side Duel & Radar</div>
            <div style="font-size: 0.85rem; color: #475569; line-height: 1.5;">
                Compare 2 or 3 products on shared specs and customer satisfaction deltas with dual radar charts and honest winner rationales.
            </div>
        </div>
        """)
    with vp3:
        render_html_block("""
        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 1.2rem;">
            <div style="font-size: 1.5rem; margin-bottom: 0.4rem;">🛡️</div>
            <div style="font-weight: 700; color: #0F172A; margin-bottom: 0.4rem;">Transparent Data Provenance</div>
            <div style="font-size: 0.85rem; color: #475569; line-height: 1.5;">
                No deceptive claims. Prepared demonstration datasets are clearly flagged, with outbound links to live Amazon India product listings.
            </div>
        </div>
        """)


# ==========================================
# VIEW 2: CATALOGUE & SEARCH
# ==========================================
elif st.session_state["nav_view"] == "🛍️ Catalogue & Search":
    st.markdown("## 🛍️ **Electronics Catalogue & Smart Search**")

    # Search bar & Filters
    search_col1, search_col2 = st.columns([3, 1])
    with search_col1:
        search_query = st.text_input(
            "Search Products",
            placeholder="Search by product name, brand, processor, camera, or keyword (e.g. 'AMOLED', 'Titanium', 'Bose')...",
            label_visibility="collapsed"
        )
    with search_col2:
        st.caption(f"Showing **{len(df_products)}** verified electronics items")

    # Sidebar filter application
    filters = render_sidebar_filters(df_products)
    filtered_df = apply_filters(df_products, search_query, filters)

    # Active Filter Chips
    chip_text = []
    if search_query:
        chip_text.append(f'Keyword: "{search_query}"')
    if filters.get("category"):
        chip_text.append(f'Dept: {filters["category"]}')
    if filters.get("brands"):
        chip_text.append(f'Brands: {", ".join(filters["brands"])}')
    chip_text.append(f'Price: ₹{filters["price_min"]:,} - ₹{filters["price_max"]:,}')
    chip_text.append(f'Rating: {filters["min_rating"]:.1f}+ ★')

    render_html_block(f"<div style='font-size: 0.85rem; color: #64748B; margin-bottom: 1rem;'><b>Active Filters:</b> {' • '.join(chip_text)} (Found {len(filtered_df)} matches)</div>")

    if filtered_df.empty:
        st.warning("No products matched your search and filter criteria. Try broadening your price range or clearing keyword filters.")
    else:
        # Display products in a 3-column responsive grid
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
# VIEW 3: PRODUCT INTELLIGENCE & ABSA (DEEP DIVE)
# ==========================================
elif st.session_state["nav_view"] == "🔍 Product Intelligence & ABSA":
    all_pids = [p["product_id"] for _, p in df_products.iterrows()]
    curr_pid = st.session_state.get("selected_product_id", all_pids[0])
    if curr_pid not in all_pids:
        curr_pid = all_pids[0]

    # Product Selector Dropdown
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

    with p_select_col2:
        in_tray = selected_pid in st.session_state["compare_tray"]
        btn_txt = "✓ In Compare Tray" if in_tray else "⚔️ Add to Compare"
        if st.button(btn_txt, key="btn_inspect_tray", use_container_width=True, type="primary" if in_tray else "secondary"):
            toggle_compare_item(selected_pid)
            st.rerun()

    # Product Header & Info Banner
    render_html_block(f"""
    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 14px; padding: 1.5rem; margin-bottom: 1.5rem; display: flex; flex-wrap: wrap; gap: 1.5rem; align-items: center; box-shadow: 0 4px 12px rgba(0,0,0,0.03);">
        <div style="width: 140px; height: 140px; background: #F8FAFC; border-radius: 10px; display: flex; align-items: center; justify-content: center; overflow: hidden; border: 1px solid #E2E8F0;">
            <img src="{prod.get('image_url')}" style="max-height: 100%; max-width: 100%; object-fit: contain;" alt="{prod.get('product_name')}" />
        </div>
        <div style="flex: 1; min-width: 280px;">
            <div style="display: flex; gap: 0.5rem; margin-bottom: 0.4rem;">
                <span style="background: #F1F5F9; color: #475569; font-weight: 600; font-size: 0.75rem; padding: 0.2rem 0.6rem; border-radius: 4px;">{prod.get('category')}</span>
                <span style="background: #EFF6FF; color: #2563EB; font-weight: 600; font-size: 0.75rem; padding: 0.2rem 0.6rem; border-radius: 4px;">{prod.get('brand')}</span>
                <span style="background: #FEF3C7; color: #D97706; font-weight: 600; font-size: 0.75rem; padding: 0.2rem 0.6rem; border-radius: 4px;">🛡️ Prepared Demo Data</span>
            </div>
            <h2 style="margin: 0 0 0.4rem 0; font-size: 1.5rem; font-weight: 800; color: #0F172A;">{prod.get('product_name')}</h2>
            <div style="display: flex; align-items: baseline; gap: 0.8rem; margin-bottom: 0.6rem;">
                <span style="font-size: 1.7rem; font-weight: 800; color: #0F172A;">{format_currency_inr(prod.get('price'))}</span>
                <span style="color: #F59E0B; font-weight: 700; font-size: 1rem;">★ {prod.get('rating'):.1f}</span>
                <span style="color: #64748B; font-size: 0.88rem;">({prod.get('review_count'):,} Amazon India Ratings)</span>
            </div>
            <p style="margin: 0; font-size: 0.88rem; color: #475569; line-height: 1.5;">{prod.get('description')}</p>
            <div style="margin-top: 0.6rem;">
                <a href="{prod.get('amazon_url')}" target="_blank" style="font-size: 0.82rem; color: #FF9900; font-weight: 700; text-decoration: none;">
                    View Live Listing on Amazon India ↗
                </a>
            </div>
        </div>
    </div>
    """)

    # Provenance disclosure notice
    render_html_block(f"""
    <div class="provenance-banner">
        <span style="font-size: 1.2rem;">🛡️</span>
        <div>
            <strong>Transparent Data Provenance:</strong> {PROVENANCE_DISCLOSURE}
        </div>
    </div>
    """)

    # Run NLP Pipeline on this product's reviews
    sentiment_res = analyze_reviews_sentiment(reviews_df)
    aspects_res = analyze_reviews_aspects(reviews_df)
    aspect_list = aspects_res["aspects"]
    keywords_res = extract_keywords_by_sentiment(reviews_df)
    exec_summary = generate_executive_summary(
        prod["product_name"],
        aspect_list,
        sentiment_res["nss"],
        sentiment_res["csat_estimate"],
        sentiment_res["total_reviews"]
    )

    # High-Level Metric Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_html_block(f"""
        <div class="metric-card metric-card-orange">
            <div class="metric-label">Amazon Star Rating</div>
            <div class="metric-value">{prod.get('rating'):.1f} <span style="font-size: 1rem; color: #F59E0B;">★</span></div>
            <div class="metric-sub">Out of 5.0 stars</div>
        </div>
        """)
    with k2:
        nss_val = sentiment_res["nss"]
        nss_col = "#10B981" if nss_val > 0 else "#EF4444"
        render_html_block(f"""
        <div class="metric-card metric-card-green">
            <div class="metric-label">Net Sentiment Score (NSS)</div>
            <div class="metric-value" style="color: {nss_col};">{nss_val:+.1f}</div>
            <div class="metric-sub">% Positive minus % Negative (-100 to +100)</div>
        </div>
        """)
    with k3:
        render_html_block(f"""
        <div class="metric-card">
            <div class="metric-label">Estimated CSAT</div>
            <div class="metric-value">{sentiment_res['csat_estimate']:.1f}%</div>
            <div class="metric-sub">Calculated customer satisfaction</div>
        </div>
        """)
    with k4:
        render_html_block(f"""
        <div class="metric-card metric-card-purple">
            <div class="metric-label">Reviews Analyzed</div>
            <div class="metric-value">{sentiment_res['total_reviews']}</div>
            <div class="metric-sub">Synthesized NLP corpus</div>
        </div>
        """)

    # Executive Summary & Persona Box
    st.markdown("### 📋 **Executive Intelligence Summary**")
    render_html_block(f"""
    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 1.2rem; margin-bottom: 1.5rem;">
        <div style="font-size: 1.05rem; font-weight: 700; color: #0F172A; margin-bottom: 0.4rem;">
            🎯 Bottom Line: {exec_summary['verdict']}
        </div>
        <div style="font-size: 0.88rem; color: #475569; margin-bottom: 0.6rem;">
            <strong>Ideal Buyer Profile:</strong> {exec_summary['persona']}
        </div>
        <div style="font-size: 0.8rem; color: #94A3B8;">
            {exec_summary['sample_disclosure']}
        </div>
    </div>
    """)

    # Pros and Cons Side-by-Side
    pro_col, con_col = st.columns(2)
    with pro_col:
        st.markdown("#### ✅ **Key Product Strengths (Verified Praise)**")
        if exec_summary["pros"]:
            for p in exec_summary["pros"]:
                render_html_block(f"""
                <div style="background: #ECFDF5; border-left: 4px solid #10B981; padding: 0.75rem 1rem; border-radius: 6px; margin-bottom: 0.6rem;">
                    <div style="font-weight: 700; color: #065F46; font-size: 0.88rem;">{p['aspect']} (+{p['nss']:.0f} NSS)</div>
                    <div style="font-size: 0.82rem; color: #047857; font-style: italic;">"{p['quote']}"</div>
                </div>
                """)
        else:
            st.info("No standout positive drivers exceeded threshold.")

    with con_col:
        st.markdown("#### ⚠️ **Key Consumer Pain Points & Trade-offs**")
        if exec_summary["cons"]:
            for c in exec_summary["cons"]:
                render_html_block(f"""
                <div style="background: #FEF2F2; border-left: 4px solid #EF4444; padding: 0.75rem 1rem; border-radius: 6px; margin-bottom: 0.6rem;">
                    <div style="font-weight: 700; color: #991B1B; font-size: 0.88rem;">{c['aspect']} ({c['nss']:+.0f} NSS)</div>
                    <div style="font-size: 0.82rem; color: #B91C1C; font-style: italic;">"{c['quote']}"</div>
                </div>
                """)
        else:
            st.success("No dominant negative pain points detected in this review corpus.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Aspect-Based Sentiment Analysis (ABSA) Section
    st.markdown("### 🔍 **Aspect-Based Sentiment Analysis (ABSA)**")
    st.caption("ABSA deconstructs customer feedback into 8 primary hardware & software dimensions, measuring positive vs negative balance for each.")

    absa_col1, absa_col2 = st.columns(2)
    with absa_col1:
        st.markdown("##### 📊 Aspect-Level Sentiment Breakdown (%)")
        fig_bar = create_aspect_breakdown_bar_chart(aspect_list)
        st.plotly_chart(fig_bar, use_container_width=True)

    with absa_col2:
        st.markdown("##### 🕸️ Aspect Net Sentiment Score Radar (NSS: -100 to +100)")
        fig_radar = create_aspect_radar_chart(aspect_list, product_name=prod["product_name"])
        st.plotly_chart(fig_radar, use_container_width=True)

    # Expandable Aspect Evidence Explorer
    with st.expander("🔎 Browse Evidence Quotes & Review Snippets per Aspect Dimension", expanded=False):
        for a in aspect_list:
            if a.get("mentions", 0) > 0:
                st.markdown(f"**{a['category']}** (Mentions: {a['mentions']} | NSS: `{a['net_sentiment_score']:+.1f}`)")
                e_col1, e_col2 = st.columns(2)
                with e_col1:
                    if a.get("positive_quotes"):
                        st.caption("🟢 **Positive Feedback:**")
                        for q in a["positive_quotes"][:2]:
                            st.markdown(f"- *\"{q}\"*")
                with e_col2:
                    if a.get("negative_quotes"):
                        st.caption("🔴 **Critical Feedback:**")
                        for q in a["negative_quotes"][:2]:
                            st.markdown(f"- *\"{q}\"*")
                st.markdown("---")

    st.markdown("<br>", unsafe_allow_html=True)

    # TF-IDF Keywords & Rating Anomalies
    kw_col1, kw_col2 = st.columns(2)
    with kw_col1:
        st.markdown("### 🧠 **Key Drivers & Extracted Phrases (TF-IDF)**")
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

    # Raw Reviews Explorer
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
# VIEW 4: PRODUCT COMPARISON
# ==========================================
elif st.session_state["nav_view"] == "⚔️ Product Comparison":
    st.markdown("## ⚔️ **Side-by-Side Product Comparison & Radar Duel**")
    st.caption("Compare specifications, consumer sentiment deltas, and aspect ratings across 2 or 3 products.")

    all_pids = [p["product_id"] for _, p in df_products.iterrows()]
    tray = st.session_state["compare_tray"]

    # Select Product A and Product B
    default_a = tray[0] if len(tray) > 0 and tray[0] in all_pids else all_pids[0]
    default_b = tray[1] if len(tray) > 1 and tray[1] in all_pids else (all_pids[1] if len(all_pids) > 1 else all_pids[0])

    comp_sel1, comp_sel2, comp_sel3 = st.columns([1, 1, 1])
    with comp_sel1:
        sel_a = st.selectbox(
            "Product A (Blue):",
            all_pids,
            index=all_pids.index(default_a),
            format_func=lambda x: get_product_by_id(x)["product_name"]
        )
    with comp_sel2:
        # Pick default_b distinct from sel_a
        b_idx = all_pids.index(default_b) if default_b != sel_a else (1 if len(all_pids) > 1 else 0)
        sel_b = st.selectbox(
            "Product B (Red):",
            all_pids,
            index=b_idx,
            format_func=lambda x: get_product_by_id(x)["product_name"]
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

    asp_a = analyze_reviews_aspects(reviews_a)
    asp_b = analyze_reviews_aspects(reviews_b)

    summary_a = {"nss": sent_a["nss"], "csat_estimate": sent_a["csat_estimate"], "aspects": asp_a["aspects"]}
    summary_b = {"nss": sent_b["nss"], "csat_estimate": sent_b["csat_estimate"], "aspects": asp_b["aspects"]}

    comp_result = compare_two_products(prod_a, summary_a, prod_b, summary_b)

    # Winner Banner
    winner_badge = "👑" if comp_result["overall_winner"] != "Close Draw" else "🤝"
    render_html_block(f"""
    <div style="background: linear-gradient(135deg, #131921 0%, #1e293b 100%); padding: 1.5rem; border-radius: 14px; color: white; margin-bottom: 1.5rem; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
        <div style="font-size: 1.3rem; font-weight: 800; color: #FF9900; margin-bottom: 0.3rem;">
            {winner_badge} Overall Head-to-Head Winner: {comp_result['overall_winner']}
        </div>
        <p style="margin: 0; font-size: 0.95rem; color: #E2E8F0; line-height: 1.5;">
            {comp_result['verdict_reason']}
        </p>
    </div>
    """)

    # Side-by-side KPI cards
    c_kpi1, c_kpi2 = st.columns(2)
    with c_kpi1:
        render_html_block(f"""
        <div class="metric-card" style="border-left-color: #2563EB;">
            <div class="metric-label">{prod_a['product_name']}</div>
            <div class="metric-value" style="color: #2563EB;">{format_currency_inr(prod_a['price'])}</div>
            <div class="metric-sub">
                NSS: <strong>{sent_a['nss']:+.1f}</strong> | CSAT: <strong>{sent_a['csat_estimate']:.1f}%</strong> | Aspect Wins: <strong>{comp_result['kpi_summary']['wins_a']}</strong>
            </div>
        </div>
        """)

    with c_kpi2:
        render_html_block(f"""
        <div class="metric-card" style="border-left-color: #DC2626;">
            <div class="metric-label">{prod_b['product_name']}</div>
            <div class="metric-value" style="color: #DC2626;">{format_currency_inr(prod_b['price'])}</div>
            <div class="metric-sub">
                NSS: <strong>{sent_b['nss']:+.1f}</strong> | CSAT: <strong>{sent_b['csat_estimate']:.1f}%</strong> | Aspect Wins: <strong>{comp_result['kpi_summary']['wins_b']}</strong>
            </div>
        </div>
        """)

    # Multi-product Radar Chart & Aspect Scorecard
    st.markdown("### 🕸️ **Shared Aspect Dimensions Comparison**")
    r_col1, r_col2 = st.columns([1, 1])

    radar_items = [
        {"name": prod_a["product_name"][:20], "aspects": asp_a["aspects"]},
        {"name": prod_b["product_name"][:20], "aspects": asp_b["aspects"]}
    ]
    if prod_c:
        reviews_c = get_reviews_for_product(sel_c)
        sent_c = analyze_reviews_sentiment(reviews_c)
        asp_c = analyze_reviews_aspects(reviews_c)
        radar_items.append({"name": prod_c["product_name"][:20], "aspects": asp_c["aspects"]})

    with r_col1:
        st.markdown("##### Multi-Product Aspect Net Sentiment Radar")
        fig_multi_radar = create_multi_product_radar_chart(radar_items)
        st.plotly_chart(fig_multi_radar, use_container_width=True)

    with r_col2:
        st.markdown("##### 🏆 Aspect-by-Aspect Head-to-Head Scorecard")
        scorecard_rows = []
        for d in comp_result["aspect_deltas"]:
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
    st.markdown("### 📋 **Detailed Hardware & Hardware Specifications**")
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
# VIEW 5: SMART RECOMMENDATIONS
# ==========================================
elif st.session_state["nav_view"] == "🎯 Smart Recommendations":
    st.markdown("## 🎯 **ReviewLens Algorithmic Recommendations**")
    st.caption("Defensible product recommendations backed by mathematical composite formulas and verified review sentiment.")

    # Department filter
    dept_opts = ["All Departments"] + sorted([c for c in df_products["category"].dropna().unique()])
    chosen_dept = st.selectbox("Filter Recommendations by Category:", dept_opts, index=0)
    category_filter = None if chosen_dept == "All Departments" else chosen_dept

    # Build products with summaries
    prod_items = []
    for _, prow in df_products.iterrows():
        pid = prow["product_id"]
        p_dict = get_product_by_id(pid)
        revs = get_reviews_for_product(pid)
        s_res = analyze_reviews_sentiment(revs)
        a_res = analyze_reviews_aspects(revs)
        prod_items.append({
            "product": p_dict,
            "summary": {
                "nss": s_res["nss"],
                "csat_estimate": s_res["csat_estimate"],
                "aspects": a_res["aspects"]
            }
        })

    recs = generate_category_recommendations(prod_items, category=category_filter)

    if recs:
        rec_cards = [
            ("🏆 Best Overall Pick", recs.get("best_overall"), "#2563EB", "badge-blue"),
            ("💰 Best Budget & Value Champion", recs.get("best_budget"), "#10B981", "badge-green"),
            ("📸 Best for Photography & Camera", recs.get("best_camera"), "#8B5CF6", "badge-purple"),
            ("🔋 Best for Battery & Endurance", recs.get("best_battery"), "#F59E0B", "badge-amber"),
            ("⚡ Best for Speed & Gaming", recs.get("best_performance"), "#EF4444", "badge-red")
        ]

        for title, rec_data, color, _ in rec_cards:
            if rec_data and rec_data.get("product"):
                p = rec_data["product"]
                render_html_block(f"""
                <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-left: 6px solid {color}; border-radius: 12px; padding: 1.2rem; margin-bottom: 1.2rem; box-shadow: 0 2px 6px rgba(0,0,0,0.03);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                        <span style="font-weight: 800; font-size: 1.05rem; color: {color};">{title}</span>
                        <span style="font-size: 1.2rem; font-weight: 800; color: #0F172A;">{format_currency_inr(p.get('price'))}</span>
                    </div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #0F172A; margin-bottom: 0.4rem;">{p.get('product_name')}</div>
                    <div style="font-size: 0.88rem; color: #475569; line-height: 1.5; margin-bottom: 0.6rem;">{rec_data.get('reason')}</div>
                    <div style="display: flex; gap: 0.8rem; align-items: center;">
                        <span style="font-size: 0.82rem; color: #64748B;">Rating: ★ {p.get('rating'):.1f} ({p.get('review_count'):,} reviews)</span>
                        <a href="{p.get('amazon_url')}" target="_blank" style="font-size: 0.82rem; color: #FF9900; font-weight: 700; text-decoration: none;">View on Amazon India ↗</a>
                    </div>
                </div>
                """)

        # Transparent Scoring Explanation
        with st.expander("📐 Explain Scoring Methodology & Mathematical Formulas", expanded=False):
            st.markdown("""
            **Composite Quality Score Formula (0 to 100):**
            $$ \\text{Score} = (0.30 \\times \\text{Star Rating \\%}) + (0.45 \\times \\text{Normalized NSS}) + (0.25 \\times \\text{CSAT \\%}) $$
            - **Star Rating %**: `(Rating / 5.0) * 100`
            - **Normalized NSS**: Rescales `[-100, +100]` to `[0, 100]` via `(NSS + 100) / 2`
            - **CSAT %**: Percentage of total customer feedback expressing positive polarity.
            - **Budget Index**: Adjusts score by price curve `Score / (Price ^ 0.35)` to identify highest feature return per rupee.
            """)


# ==========================================
# VIEW 6: DATA UPLOAD & SCHEMA VALIDATOR
# ==========================================
elif st.session_state["nav_view"] == "📂 Data Upload & Validator":
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
# VIEW 7: METHODOLOGY & TRANSPARENCY
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

    ### 2. Aspect-Based Sentiment Analysis (ABSA)
    Unlike opaque black-box neural networks, ReviewLens operates on an **explainable rule-based keyword lexicon**:
    - Evaluates reviews sentence-by-sentence.
    - Tags sentences mentioning 8 core consumer electronics aspects: *Performance & Speed*, *Display & Screen*, *Camera & Photography*, *Battery & Charging*, *Build Quality & Design*, *Value for Money & Price*, *Audio & Sound*, and *Software & Features*.
    - Extracts authentic quote evidence snippets for full auditability.

    ### 3. Transparent Academic Caveats & Limitations
    - **Negation Scope**: Complex grammatical inversions (e.g., *"not only is it fast, but also..."*) can occasionally confuse polarity classifiers.
    - **Sarcasm Detection**: Sarcastic remarks with literal positive words may be classified as positive unless flagged by star-rating anomaly checks.
    - **Sample Size Warning**: Products with fewer than 10 reviews carry lower statistical power. We advise users to weigh sample counts when interpreting NSS.
    - **Data Provenance**: Reviews and specifications in this prototype are curated demonstrations (`data_type='prepared_demo'`) designed for clean evaluation. Live outbound links connect directly to Amazon India.
    """)
