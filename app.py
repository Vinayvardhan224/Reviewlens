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
from modules.sentiment import analyze_reviews_sentiment, compute_nss
from modules.aspect_analysis import analyze_reviews_aspects, ASPECT_CAVEAT_DISCLOSURE, CATEGORY_TAXONOMY, CATEGORY_ASPECTS
from modules.keyword_extraction import extract_keywords_by_sentiment
from modules.review_summary import generate_executive_summary
from modules.comparison import compare_two_products
from modules.recommendation import generate_category_recommendations
from modules.topic_modeling import extract_topic_clusters
from modules.evaluation import evaluate_sentiment_model, evaluate_spam_model, BENCHMARK_DATASET
from components.header import render_header
from components.product_card import render_product_card
from components.filters import render_sidebar_filters, apply_filters
from components.charts import (
    create_sentiment_donut_chart,
    create_aspect_breakdown_bar_chart,
    create_aspect_radar_chart,
    create_multi_product_radar_chart,
    create_keyword_driver_chart,
    create_rating_scatter_chart,
    create_confusion_matrix_heatmap
)
from reviewlens.dashboard import render_original_dashboard

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

    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }

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

    .hero-banner {
        background: linear-gradient(135deg, #131921 0%, #1e293b 100%);
        border-radius: 14px;
        padding: 2.2rem;
        color: #FFFFFF;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

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

    .topic-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }

    .topic-pill {
        display: inline-block;
        background: #EEF2FF;
        color: #4F46E5;
        font-size: 0.78rem;
        font-weight: 700;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
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

if "department_filter" not in st.session_state:
    st.session_state["department_filter"] = "All"


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
    st.session_state["nav_view"] = "📦 Product Details"
    st.rerun()


def explore_review_intelligence_action(pid: str):
    """Switch directly to deep review intelligence for a product."""
    st.session_state["selected_product_id"] = pid
    st.session_state["nav_view"] = "🔍 Review Intelligence"
    st.rerun()


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
    "📱 Smartphones",
    "💻 Laptops",
    "🎧 Headphones",
    "📷 Cameras",
    "🛍️ All Products",
    "📦 Product Details",
    "🔍 Review Intelligence",
    "⚔️ Product Comparison",
    "🎯 Smart Recommendations",
    "📊 Original Analytics Dashboard",
    "📈 Model Evaluation & Benchmarks",
    "📂 Data Ingestion & Validator",
    "📖 Methodology & Transparency"
]

current_view_idx = NAV_OPTIONS.index(st.session_state["nav_view"]) if st.session_state["nav_view"] in NAV_OPTIONS else 0
selected_view = st.sidebar.radio(
    "Go to",
    NAV_OPTIONS,
    index=current_view_idx,
    key="sidebar_nav_radio"
)
st.session_state["nav_view"] = selected_view

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
        st.session_state["nav_view"] = "⚔️ Product Comparison"
        st.rerun()
else:
    st.sidebar.caption("No products queued. Click '⚔️ Compare' on any product card.")

st.sidebar.markdown("---")
st.sidebar.caption("🛡️ **Data Provenance**: Sample prices & ratings are illustrative. Reviews are tagged `prepared_demo` for transparent NLP research.")


# ==========================================
# VIEW 1: HOME / DISCOVERY
# ==========================================
if st.session_state["nav_view"] == "🏠 Home / Discovery":
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
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 1.1rem; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.02); margin-bottom: 0.5rem;">
                <div style="font-weight: 700; font-size: 1.05rem; color: #0F172A; margin-bottom: 0.2rem;">{nav_title}</div>
                <div style="font-size: 0.78rem; color: #64748B;">{cat_sub}</div>
            </div>
            """)
            if st.button(f"Browse {dept_id} →", key=f"home_browse_{dept_id}", use_container_width=True):
                st.session_state["nav_view"] = nav_title
                st.rerun()

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
            <div style="font-size: 1.5rem; margin-bottom: 0.4rem;">🎯</div>
            <div style="font-weight: 700; color: #0F172A; margin-bottom: 0.4rem;">Category-Specific ABSA</div>
            <div style="font-size: 0.85rem; color: #475569; line-height: 1.5;">
                Phones evaluate Camera, Battery & Display; Laptops evaluate Processor, RAM & Cooling; Headphones evaluate Sound Quality & Comfort.
            </div>
        </div>
        """)
    with vp2:
        render_html_block("""
        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 1.2rem;">
            <div style="font-size: 1.5rem; margin-bottom: 0.4rem;">⚔️</div>
            <div style="font-weight: 700; color: #0F172A; margin-bottom: 0.4rem;">Fair Multi-Product Comparison</div>
            <div style="font-size: 0.85rem; color: #475569; line-height: 1.5;">
                Compare headphones against headphones (Sony vs Bose) or phones against phones with shared specs, dual radar charts, and honest tie verdicts.
            </div>
        </div>
        """)
    with vp3:
        render_html_block("""
        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 1.2rem;">
            <div style="font-size: 1.5rem; margin-bottom: 0.4rem;">📊</div>
            <div style="font-weight: 700; color: #0F172A; margin-bottom: 0.4rem;">Rigorous NLP Benchmarks</div>
            <div style="font-size: 0.85rem; color: #475569; line-height: 1.5;">
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
    st.caption("Browse curated electronics, compare specifications, and uncover authentic customer review sentiment.")

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
        st.caption(f"Showing **{len(dept_df)}** verified items in this department")

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
    st.markdown(f"**Electronics** > **{prod.get('category')}** > **{prod.get('product_name')}**")

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
        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 1.5rem; text-align: center; height: 320px; display: flex; align-items: center; justify-content: center;">
            <img src="{img_src}" style="max-height: 280px; max-width: 100%; object-fit: contain; border-radius: 8px;" alt="{prod.get('product_name')}" />
        </div>
        """, unsafe_allow_html=True)

    with detail_right:
        st.markdown(f"## {prod.get('product_name')}")
        st.markdown(f"**Brand:** {prod.get('brand')} &nbsp;|&nbsp; **Category:** {prod.get('category')} &nbsp;|&nbsp; <span style='background: #EFF6FF; color: #2563EB; font-weight: 600; font-size: 0.8rem; padding: 0.2rem 0.5rem; border-radius: 4px;'>🛡️ Demo (Illustrative Pricing & Ratings)</span>", unsafe_allow_html=True)
        st.markdown(f"<div style='margin: 0.8rem 0;'><span style='font-size: 2rem; font-weight: 800; color: #0F172A; margin-right: 0.8rem;'>{format_currency_inr(price)}</span> {discount_html}</div>", unsafe_allow_html=True)
        st.markdown(f"★ **{prod.get('rating'):.1f}** &nbsp;•&nbsp; `{prod.get('review_count'):,}` Illustrative Ratings &nbsp;•&nbsp; `{len(reviews_df)}` Prepared demo reviews analyzed")
        st.markdown(f"<p style='color: #475569; line-height: 1.6; margin: 0.8rem 0;'>{prod.get('description')}</p>", unsafe_allow_html=True)

        amazon_url = prod.get("product_url") or prod.get("amazon_url") or f"https://www.amazon.in/s?k={prod.get('product_name')}"
        btn_c1, btn_c2 = st.columns(2)
        with btn_c1:
            if st.button("🔍 Explore Deep Review Intelligence ↗", key="btn_go_intelligence", type="primary", use_container_width=True):
                explore_review_intelligence_action(selected_pid)
        with btn_c2:
            st.markdown(f"""
            <a href="{amazon_url}" target="_blank" style="display: block; text-align: center; background: #FF9900; color: #131921; font-weight: 700; padding: 0.55rem 1rem; border-radius: 8px; text-decoration: none;">
                Search on Amazon India ↗ (Verified Search)
            </a>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Hardware & Feature Specifications Table
    st.markdown("### 📋 **Product Specifications**")
    specs_dict = prod.get("specs", {})
    if specs_dict:
        spec_items = [{"Specification": k, "Detail": v} for k, v in specs_dict.items()]
        st.dataframe(pd.DataFrame(spec_items), use_container_width=True, hide_index=True)
    else:
        st.info("General product specifications are summarized in the product description above.")

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
                color = "#64748B"
                score_str = "N/A"
                mentions_str = "Insufficient review data"

            render_html_block(f"""
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 0.8rem; text-align: center;">
                <div style="font-size: 0.78rem; font-weight: 700; color: #475569; text-transform: uppercase;">{asp['category']}</div>
                <div style="font-size: 1.3rem; font-weight: 800; color: {color}; margin: 0.2rem 0;">{score_str}</div>
                <div style="font-size: 0.72rem; color: #94A3B8;">{mentions_str}</div>
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
                st.success(f"**{p['aspect']}** (+{p['nss']:.0f} NSS): *\"{p['quote']}\"*")
        else:
            st.info("No standout positive drivers identified.")
    with c_col:
        st.markdown("##### 🔴 Key Trade-offs / Pain Points")
        if exec_summary["cons"]:
            for c in exec_summary["cons"]:
                st.error(f"**{c['aspect']}** ({c['nss']:+.0f} NSS): *\"{c['quote']}\"*")
        else:
            st.success("No dominant customer pain points flagged in this review corpus.")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Dive Into Full NLP Review Intelligence & Interactive Topic Modeling →", key="btn_deep_cta", type="primary"):
        explore_review_intelligence_action(selected_pid)


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
    render_html_block(f"""
    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 14px; padding: 1.3rem; margin-bottom: 1.5rem; display: flex; flex-wrap: wrap; gap: 1.5rem; align-items: center; box-shadow: 0 4px 12px rgba(0,0,0,0.03);">
        <div style="width: 110px; height: 110px; background: #F8FAFC; border-radius: 10px; display: flex; align-items: center; justify-content: center; overflow: hidden; border: 1px solid #E2E8F0;">
            <img src="{img_src}" style="max-height: 100%; max-width: 100%; object-fit: contain; border-radius: 6px;" alt="{prod.get('product_name')}" />
        </div>
        <div style="flex: 1; min-width: 260px;">
            <div style="display: flex; gap: 0.5rem; margin-bottom: 0.3rem;">
                <span style="background: #F1F5F9; color: #475569; font-weight: 600; font-size: 0.75rem; padding: 0.2rem 0.6rem; border-radius: 4px;">{prod.get('category')}</span>
                <span style="background: #EFF6FF; color: #2563EB; font-weight: 600; font-size: 0.75rem; padding: 0.2rem 0.6rem; border-radius: 4px;">{prod.get('brand')}</span>
                <span style="background: #FEF3C7; color: #D97706; font-weight: 600; font-size: 0.75rem; padding: 0.2rem 0.6rem; border-radius: 4px;">🛡️ Demo Data (Illustrative)</span>
            </div>
            <h2 style="margin: 0 0 0.3rem 0; font-size: 1.35rem; font-weight: 800; color: #0F172A;">{prod.get('product_name')}</h2>
            <div style="display: flex; align-items: baseline; gap: 0.8rem;">
                <span style="font-size: 1.4rem; font-weight: 800; color: #0F172A;">{format_currency_inr(prod.get('price'))}</span>
                <span style="color: #F59E0B; font-weight: 700; font-size: 0.95rem;">★ {prod.get('rating'):.1f}</span>
                <span style="color: #64748B; font-size: 0.85rem;">({prod.get('review_count'):,} Ratings)</span>
            </div>
        </div>
    </div>
    """)

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

    review_texts = [r.get("review_text", "") for _, r in reviews_df.iterrows()]
    discovered_topics = extract_topic_clusters(review_texts, n_topics=num_topics, top_words=6)

    if discovered_topics:
        topic_cols = st.columns(min(len(discovered_topics), 3))
        for idx, t in enumerate(discovered_topics):
            col_target = topic_cols[idx % 3]
            with col_target:
                pills_html = "".join([f'<span class="topic-pill">{kw}</span>' for kw in t["keywords"]])
                quotes_html = "".join([f'<li style="font-size: 0.8rem; color: #475569; margin-bottom: 0.4rem; font-style: italic;">"{q[:120]}..."</li>' for q in t["sample_reviews"][:2]])
                render_html_block(f"""
                <div class="topic-card">
                    <div style="font-weight: 700; color: #1E293B; font-size: 0.95rem; margin-bottom: 0.5rem;">
                        🏷️ {t['label']}
                    </div>
                    <div style="margin-bottom: 0.6rem;">{pills_html}</div>
                    <div style="font-size: 0.78rem; font-weight: 600; color: #64748B; margin-bottom: 0.3rem;">Representative Excerpts:</div>
                    <ul style="padding-left: 1.1rem; margin: 0;">{quotes_html}</ul>
                </div>
                """)
    else:
        st.info("Insufficient text length in review sample to extract multi-topic LDA clusters.")

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

    # Side-by-side KPI cards showing price, catalogue rating, review count, overall sentiment, and aspect wins
    c_kpi1, c_kpi2 = st.columns(2)
    with c_kpi1:
        render_html_block(f"""
        <div class="metric-card" style="border-left-color: #2563EB;">
            <div class="metric-label">{prod_a['product_name']}</div>
            <div class="metric-value" style="color: #2563EB;">{format_currency_inr(prod_a['price'])}</div>
            <div class="metric-sub">
                Rating: <strong>★ {prod_a.get('rating', 4.0):.1f}</strong> | Reviews: <strong>{len(reviews_a)}</strong> | NSS: <strong>{sent_a['nss']:+.1f}</strong> | CSAT: <strong>{sent_a['csat_estimate']:.1f}%</strong> | Aspect Wins: <strong>{comp_result['kpi_summary']['wins_a']}</strong>
            </div>
        </div>
        """)

    with c_kpi2:
        render_html_block(f"""
        <div class="metric-card" style="border-left-color: #DC2626;">
            <div class="metric-label">{prod_b['product_name']}</div>
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
    st.caption("Defensible product recommendations backed by mathematical composite formulas and verified review sentiment.")

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
                "aspects": a_res["aspects"]
            }
        })

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
                amazon_url = p.get("product_url") or p.get("amazon_url") or f"https://www.amazon.in/s?k={p.get('product_name')}"
                render_html_block(f"""
                <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-left: 6px solid {color}; border-radius: 12px; padding: 1.2rem; margin-bottom: 1.2rem; box-shadow: 0 2px 6px rgba(0,0,0,0.03);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                        <span style="font-weight: 800; font-size: 1.05rem; color: {color};">{title}</span>
                        <span style="font-size: 1.2rem; font-weight: 800; color: #0F172A;">{format_currency_inr(p.get('price'))}</span>
                    </div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #0F172A; margin-bottom: 0.4rem;">{p.get('product_name')}</div>
                    <div style="font-size: 0.88rem; color: #475569; line-height: 1.5; margin-bottom: 0.6rem;">{rec_data.get('reason')}</div>
                    <div style="display: flex; gap: 0.8rem; align-items: center;">
                        <span style="font-size: 0.82rem; color: #64748B;">Rating: ★ {p.get('rating'):.1f} ({p.get('review_count'):,} sample ratings)</span>
                        <a href="{amazon_url}" target="_blank" style="font-size: 0.82rem; color: #FF9900; font-weight: 700; text-decoration: none;">Search on Amazon India ↗</a>
                    </div>
                </div>
                """)

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
# VIEW 11: ORIGINAL REVIEWLENS ANALYTICS DASHBOARD
# ==========================================
elif st.session_state["nav_view"] == "📊 Original Analytics Dashboard":
    render_original_dashboard()


# ==========================================
# VIEW 12: MODEL EVALUATION & BENCHMARKS
# ==========================================
elif st.session_state["nav_view"] == "📈 Model Evaluation & Benchmarks":
    st.markdown("## 📈 **Model Evaluation & Ground-Truth Benchmarks**")
    st.caption("Empirical performance evaluation of VADER Sentiment Classification and Spam Detection against a manually labelled test set (50 ground-truth samples).")

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
