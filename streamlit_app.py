"""
ReviewLens: Interactive Streamlit Web Application Dashboard.
An NLP-based smart product review intelligence and analytics platform.
"""

import sys
from pathlib import Path
import io
import json
import streamlit as st
import pandas as pd

# Set parent dir on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from reviewlens.pipeline import ReviewLensPipeline
from reviewlens.comparison import ProductComparator
from reviewlens.visualization.charts import (
    create_sentiment_donut,
    create_aspect_bar_chart,
    create_aspect_radar_chart,
    create_rating_vs_sentiment_scatter,
    create_keyword_bar_chart,
    generate_wordcloud_image,
    create_competitor_radar_chart,
    create_head_to_head_bar_chart,
)

# Page configuration
st.set_page_config(
    page_title="ReviewLens | Smart Review Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Modern SaaS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Clean modern tab bar */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #F1F5F9;
        padding: 6px;
        border-radius: 14px;
        border: 1px solid #E2E8F0;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 8px 18px;
        font-weight: 600;
        font-size: 0.9rem;
        color: #475569;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #2563EB !important;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.12) !important;
    }

    /* Header Banner */
    .header-banner {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border-radius: 18px;
        padding: 24px 28px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        background: rgba(59, 130, 246, 0.25);
        border: 1px solid rgba(96, 165, 250, 0.4);
        color: #93C5FD;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    .header-title {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        color: #F8FAFC;
    }
    .header-subtitle {
        color: #94A3B8;
        font-size: 0.92rem;
        margin-top: 6px;
        margin-bottom: 0;
    }
    .engine-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        color: #E2E8F0;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10B981;
        box-shadow: 0 0 10px #10B981;
    }

    /* KPI Cards */
    .kpi-card {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 18px 20px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        position: relative;
        overflow: hidden;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.06);
    }
    .kpi-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
    }
    .kpi-blue::before { background: #3B82F6; }
    .kpi-green::before { background: #10B981; }
    .kpi-purple::before { background: #8B5CF6; }
    .kpi-red::before { background: #EF4444; }

    .kpi-label {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.2;
    }
    .kpi-subtext {
        font-size: 0.8rem;
        color: #94A3B8;
        margin-top: 4px;
        font-weight: 500;
    }

    /* SWOT Cards */
    .swot-box {
        border-radius: 14px;
        padding: 18px;
        border: 1px solid transparent;
        height: 100%;
    }
    .swot-s { background: #F0FDF4; border-color: #BBF7D0; color: #166534; }
    .swot-w { background: #FEF2F2; border-color: #FECACA; color: #991B1B; }
    .swot-o { background: #EFF6FF; border-color: #BFDBFE; color: #1E40AF; }
    .swot-t { background: #FFFBEB; border-color: #FDE68A; color: #92400E; }

    .swot-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 24px;
        height: 24px;
        border-radius: 6px;
        color: white;
        font-weight: 800;
        font-size: 0.75rem;
        margin-right: 6px;
    }
    .swot-badge-s { background: #16A34A; }
    .swot-badge-w { background: #DC2626; }
    .swot-badge-o { background: #2563EB; }
    .swot-badge-t { background: #D97706; }

    /* Action Plan Cards */
    .plan-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02);
        margin-bottom: 12px;
    }
    .badge-critical {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.75rem;
    }
    .badge-high {
        background-color: #FFEDD5;
        color: #9A3412;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.75rem;
    }
    .badge-medium {
        background-color: #FEF9C3;
        color: #854D0E;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)


ASPECT_ICONS = {
    "Battery & Power": "🔋",
    "Sound & Audio": "🔊",
    "Build Quality & Durability": "🔨",
    "Comfort & Ergonomics": "🎧",
    "Display & Visuals": "📱",
    "Performance & Speed": "⚡",
    "Price & Value for Money": "💰",
    "Ease of Use & Connectivity": "📶",
    "Customer Support & Warranty": "🤝",
    "Packaging & Delivery": "📦"
}


@st.cache_resource
def get_pipeline():
    """Cache the ReviewLens pipeline instance."""
    return ReviewLensPipeline()


pipeline = get_pipeline()

# Sidebar: Controls & Dataset Selection
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/search-property.png", width=60)
    st.markdown("### **ReviewLens**")
    st.caption("NLP-Based Smart Product Review Analysis")

    st.markdown("---")
    st.markdown("#### 📁 **Review Data Source**")
    data_source = st.radio(
        "Choose Review Dataset:",
        [
            "📡 Real-World Amazon Echo Dot (Scraped)",
            "🎧 Wireless Headphones (Sample)",
            "📱 Flagship Smartphone (Sample)",
            "☕ Smart Coffee Maker (Sample)",
            "🌐 Live Amazon Scraper (URL/ASIN)",
            "📤 Upload Custom File (CSV/Excel/JSON)"
        ],
        label_visibility="collapsed"
    )

    df_reviews = None
    data_dir = Path(__file__).resolve().parent / "data"

    if "Amazon Echo Dot" in data_source:
        df_reviews = pd.read_csv(data_dir / "real_world_amazon_echo_reviews.csv")
    elif "Headphones" in data_source:
        df_reviews = pd.read_csv(data_dir / "sample_headphones_reviews.csv")
    elif "Smartphone" in data_source:
        df_reviews = pd.read_csv(data_dir / "sample_smartphone_reviews.csv")
    elif "Coffee Maker" in data_source:
        df_reviews = pd.read_csv(data_dir / "sample_coffee_maker_reviews.csv")
    elif "Live Amazon Scraper" in data_source:
        from scraper.amazon_scraper import AmazonReviewScraper
        asin_input = st.text_input("Amazon URL or 10-digit ASIN:", value="B084DCJKSL", placeholder="e.g. B084DCJKSL or https://amazon.com/dp/...")
        max_p = st.slider("Max Review Pages to Scrape:", min_value=1, max_value=5, value=2)
        if st.button("🔍 Scrape & Analyze Amazon Reviews", type="primary"):
            scraper = AmazonReviewScraper()
            asin = scraper.extract_asin(asin_input)
            if asin:
                with st.spinner(f"Scraping Amazon reviews for ASIN: {asin}..."):
                    scraped_data = scraper.scrape_product_reviews(asin=asin, max_pages=max_p)
                    if scraped_data:
                        df_reviews = pd.DataFrame(scraped_data)
                        st.session_state["scraped_df"] = df_reviews
                        st.success(f"Successfully scraped {len(scraped_data)} reviews!")
                    else:
                        st.error("No reviews could be retrieved for this ASIN.")
            else:
                st.error("Invalid Amazon URL or ASIN. Please provide a valid 10-character ASIN.")
        elif "scraped_df" in st.session_state:
            df_reviews = st.session_state["scraped_df"]
    else:
        uploaded_file = st.file_uploader("Upload CSV, Excel, or JSON", type=["csv", "xlsx", "json"])
        if uploaded_file is not None:
            if uploaded_file.name.endswith(".csv"):
                df_reviews = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".xlsx"):
                df_reviews = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith(".json"):
                df_reviews = pd.read_json(uploaded_file)

    st.markdown("---")
    st.markdown("#### ⚙️ **NLP Engine Tuning**")
    n_topics = st.slider("Max Topics (LDA)", min_value=2, max_value=6, value=4)
    st.caption("Engine: Augmented VADER Lexicon + ABSA + Latent Dirichlet Allocation.")


# Process Data
if df_reviews is not None and not df_reviews.empty:
    def analyze_dataframe(df_in):
        """Analyze DataFrame with automatic column detection and zero serialization overhead."""
        candidate_t = [c for c in df_in.columns if any(k in c.lower() for k in ["text", "review", "content", "body", "comment"])]
        t_col = candidate_t[0] if candidate_t else df_in.columns[0]
        candidate_r = [c for c in df_in.columns if any(k in c.lower() for k in ["rating", "score", "stars"])]
        r_col = candidate_r[0] if candidate_r else None
        return pipeline.analyze_dataset(df_in, text_column=t_col, rating_column=r_col)

    with st.spinner("🧠 Analyzing reviews with NLP pipeline (Sentiment, ABSA, Topics)..."):
        analysis_result = analyze_dataframe(df_reviews)

    # Executive Header Banner
    st.markdown("""
    <div class="header-banner">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
            <div>
                <span class="badge-pill">✨ NLP-BASED PRODUCT INTELLIGENCE</span>
                <h1 class="header-title">ReviewLens Analytics Dashboard</h1>
                <p class="header-subtitle">Real-Time Aspect Sentiment • Fake Review Shield • Emotion Profiling • Automated SWOT</p>
            </div>
            <div class="engine-pill">
                <span class="status-dot"></span> VADER + ABSA + LDA Active
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Tabs Layout
    tabs = st.tabs([
        "📊 Executive KPIs",
        "🔍 Aspect-Based Sentiment (ABSA)",
        "⚔️ Phone & Competitor Comparison",
        "🧠 Deep NLP & Topics",
        "🎯 SWOT & Action Plan",
        "✍️ Single Review Tester",
        "📋 Review Explorer"
    ])

    # TAB 1: EXECUTIVE DASHBOARD
    with tabs[0]:
        nss = analysis_result.executive_summary.net_sentiment_score
        nss_sign = "+" if nss >= 0 else ""
        nss_color = "kpi-green" if nss >= 30 else ("kpi-blue" if nss >= 0 else "kpi-red")

        # Sleek KPI Cards
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        with kpi_col1:
            st.markdown(f"""
            <div class="kpi-card kpi-blue">
                <div class="kpi-label">TOTAL REVIEWS</div>
                <div class="kpi-value">{analysis_result.total_reviews}</div>
                <div class="kpi-subtext">100% processed</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi_col2:
            st.markdown(f"""
            <div class="kpi-card {nss_color}">
                <div class="kpi-label">NET SENTIMENT SCORE</div>
                <div class="kpi-value">{nss_sign}{nss:.1f}</div>
                <div class="kpi-subtext">Scale: -100 to +100</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi_col3:
            st.markdown(f"""
            <div class="kpi-card kpi-purple">
                <div class="kpi-label">ESTIMATED CSAT</div>
                <div class="kpi-value">{analysis_result.executive_summary.csat_score:.1f}%</div>
                <div class="kpi-subtext">Positive opinion ratio</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi_col4:
            spam_col = "kpi-green" if analysis_result.spam_reviews == 0 else "kpi-red"
            st.markdown(f"""
            <div class="kpi-card {spam_col}">
                <div class="kpi-label">SPAM / FAKE REVIEWS</div>
                <div class="kpi-value">{analysis_result.spam_reviews}</div>
                <div class="kpi-subtext">{analysis_result.valid_reviews} authentic reviews</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts Row
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            fig_donut = create_sentiment_donut(analysis_result.overall_sentiment_distribution)
            st.plotly_chart(fig_donut, use_container_width=True)

        with chart_col2:
            fig_scatter = create_rating_vs_sentiment_scatter(analysis_result.reviews)
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Timeline chart if available
        if analysis_result.timeline:
            st.markdown("#### 📅 **Sentiment Trajectory Over Time**")
            df_time = pd.DataFrame(analysis_result.timeline)
            st.line_chart(df_time.set_index("period")["average_sentiment"])

    # TAB 2: ASPECT-BASED SENTIMENT (ABSA)
    with tabs[1]:
        st.markdown("""
        <div style="background: #FFFFFF; padding: 18px 22px; border-radius: 14px; border: 1px solid #E2E8F0; margin-bottom: 20px;">
            <h3 style="margin: 0; font-size: 1.15rem; font-weight: 700; color: #0F172A;">🔍 Aspect-Based Sentiment Analysis (ABSA)</h3>
            <p style="margin: 4px 0 0 0; font-size: 0.88rem; color: #64748B;">
                ABSA decomposes customer reviews into specific hardware & software dimensions, accurately isolating praise from frustration.
            </p>
        </div>
        """, unsafe_allow_html=True)

        absa_col1, absa_col2 = st.columns([1, 1])
        with absa_col1:
            fig_aspect_bar = create_aspect_bar_chart(analysis_result.aspect_summaries)
            st.plotly_chart(fig_aspect_bar, use_container_width=True)

        with absa_col2:
            fig_radar = create_aspect_radar_chart(analysis_result.aspect_summaries)
            st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("#### 💬 **Voice of Customer: Aspect Deep-Dive & Quotes**")
        for aspect in analysis_result.aspect_summaries:
            icon = ASPECT_ICONS.get(aspect.category, "🏷️")
            nss_badge = f"+{aspect.net_sentiment_score:.0f}" if aspect.net_sentiment_score >= 0 else f"{aspect.net_sentiment_score:.0f}"
            with st.expander(f"{icon} **{aspect.category}** — {aspect.total_mentions} Mentions | NSS: {nss_badge}"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Positive", f"{aspect.positive_mentions} ({aspect.sentiment_ratio.get('positive', 0):.0f}%)")
                c2.metric("Neutral", f"{aspect.neutral_mentions} ({aspect.sentiment_ratio.get('neutral', 0):.0f}%)")
                c3.metric("Negative", f"{aspect.negative_mentions} ({aspect.sentiment_ratio.get('negative', 0):.0f}%)")

                q_col1, q_col2 = st.columns(2)
                with q_col1:
                    st.markdown("**🟢 Top Positive Feedback:**")
                    if aspect.sample_positive_quotes:
                        for q in aspect.sample_positive_quotes:
                            st.info(f"“{q}”")
                    else:
                        st.caption("No positive mentions recorded.")

                with q_col2:
                    st.markdown("**🔴 Top Customer Pain Points:**")
                    if aspect.sample_negative_quotes:
                        for q in aspect.sample_negative_quotes:
                            st.error(f"“{q}”")
                    else:
                        st.caption("No negative mentions recorded.")

    # TAB 3: COMPETITOR COMPARISON
    with tabs[2]:
        st.markdown("""
        <div style="background: #FFFFFF; padding: 18px 22px; border-radius: 14px; border: 1px solid #E2E8F0; margin-bottom: 20px;">
            <h3 style="margin: 0; font-size: 1.15rem; font-weight: 700; color: #0F172A;">⚔️ Side-by-Side Competitor & Phone Benchmarking</h3>
            <p style="margin: 4px 0 0 0; font-size: 0.88rem; color: #64748B;">
                Directly compare customer sentiment across two competing phones/products to identify which brand leads in Battery, Display, Thermals, and Value.
            </p>
        </div>
        """, unsafe_allow_html=True)

        comp_col1, comp_col2 = st.columns([2, 1])
        with comp_col1:
            matchup_type = st.selectbox(
                "Choose Comparison Matchup:",
                [
                    "📱 Flagship Phone A (120Hz & Fast Charge) vs. Flagship Phone B (Pro Cam & Cool Thermals)",
                    "🎧 Wireless Headphones A (Punchy Bass) vs. Headphones B (Lightweight)",
                    "📤 Compare 2 Custom Review Files (CSV)"
                ]
            )

        data_dir = Path(__file__).resolve().parent / "data"
        if "Flagship Phone" in matchup_type:
            name_a = "Flagship Phone A"
            name_b = "Flagship Phone B"
            df_a = pd.read_csv(data_dir / "sample_smartphone_reviews.csv")
            df_b = pd.read_csv(data_dir / "sample_phone_b_reviews.csv")
        elif "Wireless Headphones" in matchup_type:
            name_a = "Headphones A (Punchy Bass)"
            name_b = "Headphones B (Lightweight)"
            df_a = pd.read_csv(data_dir / "sample_headphones_reviews.csv")
            df_b = pd.read_csv(data_dir / "sample_headphones_reviews.csv")
        else:
            u_col1, u_col2 = st.columns(2)
            with u_col1:
                name_a = st.text_input("Product A Name:", "Phone A", placeholder="e.g. iPhone 15 Pro")
                file_a = st.file_uploader("Upload Phone A Reviews (CSV):", type=["csv"], key="file_comp_a")
            with u_col2:
                name_b = st.text_input("Product B Name:", "Phone B", placeholder="e.g. Galaxy S24 Ultra")
                file_b = st.file_uploader("Upload Phone B Reviews (CSV):", type=["csv"], key="file_comp_b")
            df_a = pd.read_csv(file_a) if file_a else pd.read_csv(data_dir / "sample_smartphone_reviews.csv")
            df_b = pd.read_csv(file_b) if file_b else pd.read_csv(data_dir / "sample_phone_b_reviews.csv")

        # Ensure product names are never empty
        name_a = name_a.strip() if name_a and name_a.strip() else "Phone A"
        name_b = name_b.strip() if name_b and name_b.strip() else "Phone B"

        # Run pipeline on both
        res_a = analyze_dataframe(df_a)
        res_b = analyze_dataframe(df_b)

        comp_report = ProductComparator.compare(res_a, res_b, name_a=name_a, name_b=name_b)

        # Winner Banner
        winner_emoji = "👑" if "Draw" not in comp_report.overall_winner else "🤝"
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%); padding: 22px 26px; border-radius: 16px; color: white; margin-bottom: 22px; box-shadow: 0 4px 18px rgba(15, 23, 42, 0.15);">
            <div style="display: flex; align-items: center; gap: 8px; font-size: 1.3rem; font-weight: 800; color: #F59E0B;">
                <span>{winner_emoji} Overall Verdict: {comp_report.overall_winner} Takes the Lead</span>
            </div>
            <p style="margin: 6px 0 0 0; font-size: 0.95rem; color: #E2E8F0; line-height: 1.5;">
                {comp_report.executive_verdict}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Side-by-side KPI cards
        kpi_a_col, kpi_b_col = st.columns(2)
        with kpi_a_col:
            st.markdown(f"""
            <div class="kpi-card kpi-blue">
                <div class="kpi-label">{name_a}</div>
                <div class="kpi-value" style="color: #2563EB;">{comp_report.nss_a:+.1f} <span style="font-size: 1rem; font-weight: 500; color: #64748B;">NSS</span></div>
                <div class="kpi-subtext">Estimated CSAT: <strong>{comp_report.csat_a:.1f}%</strong> | {comp_report.total_reviews_a} reviews analyzed</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi_b_col:
            st.markdown(f"""
            <div class="kpi-card kpi-purple">
                <div class="kpi-label">{name_b}</div>
                <div class="kpi-value" style="color: #D97706;">{comp_report.nss_b:+.1f} <span style="font-size: 1rem; font-weight: 500; color: #64748B;">NSS</span></div>
                <div class="kpi-subtext">Estimated CSAT: <strong>{comp_report.csat_b:.1f}%</strong> | {comp_report.total_reviews_b} reviews analyzed</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Dual Charts Row
        c_chart1, c_chart2 = st.columns(2)
        with c_chart1:
            fig_comp_radar = create_competitor_radar_chart(comp_report.aspect_comparisons, name_a=name_a, name_b=name_b)
            st.plotly_chart(fig_comp_radar, use_container_width=True)

        with c_chart2:
            fig_comp_bar = create_head_to_head_bar_chart(comp_report.aspect_comparisons, name_a=name_a, name_b=name_b)
            st.plotly_chart(fig_comp_bar, use_container_width=True)

        # Aspect Winner Matrix
        st.markdown("#### 🏆 **Head-to-Head Aspect Winner Scorecard**")
        table_rows = []
        for ac in comp_report.aspect_comparisons:
            icon = ASPECT_ICONS.get(ac.category, "🏷️")
            winner_badge = f"🥇 {ac.winner}" if ac.winner != "Tie" else "🤝 Tie"
            diff_str = f"{abs(ac.delta):.1f} pts" if ac.winner != "Tie" else "Even"
            table_rows.append({
                "Aspect Dimension": f"{icon} {ac.category}",
                f"{name_a} (NSS)": f"{ac.score_a:+.1f}",
                f"{name_b} (NSS)": f"{ac.score_b:+.1f}",
                "Category Winner": winner_badge,
                "Margin of Victory": diff_str
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

        # Key Differentiator Cards
        diff_col1, diff_col2 = st.columns(2)
        with diff_col1:
            st.markdown(f"""
            <div class="swot-box swot-o" style="margin-top: 10px;">
                <h4 style="margin: 0 0 8px 0; font-size: 0.95rem; font-weight: 700; color: #1E40AF;">🔹 Where {name_a} Dominates:</h4>
                <div style="font-size: 0.85rem; line-height: 1.6;">
                    {'<br>'.join(f'• {s}' for s in comp_report.a_strengths) if comp_report.a_strengths else '• Balanced performance across dimensions.'}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with diff_col2:
            st.markdown(f"""
            <div class="swot-box swot-s" style="margin-top: 10px;">
                <h4 style="margin: 0 0 8px 0; font-size: 0.95rem; font-weight: 700; color: #166534;">🔸 Where {name_b} Dominates:</h4>
                <div style="font-size: 0.85rem; line-height: 1.6;">
                    {'<br>'.join(f'• {s}' for s in comp_report.b_strengths) if comp_report.b_strengths else '• Balanced performance across dimensions.'}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # TAB 4: DEEP NLP & TOPICS
    with tabs[3]:
        st.markdown("""
        <div style="background: #FFFFFF; padding: 18px 22px; border-radius: 14px; border: 1px solid #E2E8F0; margin-bottom: 20px;">
            <h3 style="margin: 0; font-size: 1.15rem; font-weight: 700; color: #0F172A;">🧠 Key Phrases & Unsupervised Topic Modeling</h3>
            <p style="margin: 4px 0 0 0; font-size: 0.88rem; color: #64748B;">
                Isolate distinct positive drivers vs. negative pain points with TF-IDF, and discover hidden conversation clusters with LDA.
            </p>
        </div>
        """, unsafe_allow_html=True)

        nlp_col1, nlp_col2 = st.columns(2)
        with nlp_col1:
            fig_kw = create_keyword_bar_chart(
                analysis_result.top_positive_keywords,
                analysis_result.top_negative_keywords
            )
            st.plotly_chart(fig_kw, use_container_width=True)

        with nlp_col2:
            st.markdown("#### ☁️ **Customer Review Word Cloud**")
            all_texts = [r.cleaned_text for r in analysis_result.reviews]
            wc_image = generate_wordcloud_image(all_texts)
            if wc_image:
                st.image(wc_image, use_container_width=True)
            else:
                st.caption("Wordcloud visualization rendered.")

        st.markdown("---")
        st.markdown("#### 🏷️ **Discovered Latent Discussion Topics (LDA)**")
        if analysis_result.topics:
            t_cols = st.columns(len(analysis_result.topics))
            for i, topic in enumerate(analysis_result.topics):
                with t_cols[i]:
                    st.markdown(f"**{topic.label}**")
                    st.markdown(f"*Salient Terms:* `{', '.join(topic.keywords[:4])}`")
                    if topic.sample_reviews:
                        with st.popover("Sample Review Snippet"):
                            st.write(topic.sample_reviews[0])
        else:
            st.info("Topic modeling requires at least 4 reviews to extract meaningful latent clusters.")

    # TAB 5: SWOT & RECOMMENDATIONS
    with tabs[4]:
        st.markdown("""
        <div style="background: #FFFFFF; padding: 18px 22px; border-radius: 14px; border: 1px solid #E2E8F0; margin-bottom: 20px;">
            <h3 style="margin: 0; font-size: 1.15rem; font-weight: 700; color: #0F172A;">🎯 Automated Executive SWOT Analysis</h3>
            <p style="margin: 4px 0 0 0; font-size: 0.88rem; color: #64748B;">
                Synthesizes statistical sentiment data into strategic Strengths, Weaknesses, Opportunities, and Threats.
            </p>
        </div>
        """, unsafe_allow_html=True)

        swot = analysis_result.executive_summary

        s_col1, s_col2, s_col3, s_col4 = st.columns(4)
        with s_col1:
            st.markdown(f"""
            <div class="swot-box swot-s">
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    <span class="swot-badge swot-badge-s">S</span>
                    <strong>Strengths</strong>
                </div>
                <div style="font-size: 0.82rem; line-height: 1.5;">
                    {'<br>'.join(f'• {s}' for s in swot.strengths)}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with s_col2:
            st.markdown(f"""
            <div class="swot-box swot-w">
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    <span class="swot-badge swot-badge-w">W</span>
                    <strong>Weaknesses</strong>
                </div>
                <div style="font-size: 0.82rem; line-height: 1.5;">
                    {'<br>'.join(f'• {w}' for w in swot.weaknesses)}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with s_col3:
            st.markdown(f"""
            <div class="swot-box swot-o">
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    <span class="swot-badge swot-badge-o">O</span>
                    <strong>Opportunities</strong>
                </div>
                <div style="font-size: 0.82rem; line-height: 1.5;">
                    {'<br>'.join(f'• {o}' for o in swot.opportunities)}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with s_col4:
            st.markdown(f"""
            <div class="swot-box swot-t">
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    <span class="swot-badge swot-badge-t">T</span>
                    <strong>Threats</strong>
                </div>
                <div style="font-size: 0.82rem; line-height: 1.5;">
                    {'<br>'.join(f'• {t}' for t in swot.threats)}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🛠️ **Prioritized Engineering & Product Management Action Plan**")
        if swot.actionable_recommendations:
            for rec in swot.actionable_recommendations:
                badge_class = f"badge-{rec['priority'].lower()}"
                st.markdown(f"""
                <div class="plan-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-size: 1.05rem; font-weight: 700; color: #0F172A;">{rec['title']}</span>
                        <span class="{badge_class}">{rec['priority']} PRIORITY</span>
                    </div>
                    <div style="color: #334155; font-size: 0.92rem; margin-bottom: 6px;">
                        <strong>Action Item:</strong> {rec['recommended_action']}
                    </div>
                    <div style="font-size: 0.82rem; color: #64748b;">
                        <strong>Aspect:</strong> {rec['category']} &nbsp;|&nbsp; 
                        <strong>Responsible Team:</strong> {rec['responsible_team']} &nbsp;|&nbsp; 
                        <strong>Negative Mentions:</strong> {rec['negative_mentions']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("No critical product vulnerabilities identified! All aspects performing above standard.")

    # TAB 6: SINGLE REVIEW TESTER
    with tabs[5]:
        st.markdown("""
        <div style="background: #FFFFFF; padding: 18px 22px; border-radius: 14px; border: 1px solid #E2E8F0; margin-bottom: 20px;">
            <h3 style="margin: 0; font-size: 1.15rem; font-weight: 700; color: #0F172A;">✍️ Real-Time Single Review Inspector</h3>
            <p style="margin: 4px 0 0 0; font-size: 0.88rem; color: #64748B;">
                Type or paste any product review to test the live sentiment analyzer, ABSA engine, and spam detector.
            </p>
        </div>
        """, unsafe_allow_html=True)

        preset = st.selectbox(
            "Or load a preset example:",
            [
                "-- Type Custom Text --",
                "The battery life is amazing and lasts 3 days, but the earcups hurt after 2 hours and customer service was unhelpful.",
                "Worst smartphone ever! The screen cracked after a small drop and the processor overheats constantly.",
                "CLICK HERE FOR CASH BACK DISCOUNT CODE GET FREE GIFT ON WHATSAPP FIVE STARS FOR FREE!!!!",
                "Sound quality is crystal clear with deep punchy bass. Best investment I've made this year."
            ]
        )

        default_text = "" if preset.startswith("--") else preset
        test_text = st.text_area("Review Text:", value=default_text, height=100)
        test_rating = st.slider("User Star Rating (Optional):", min_value=1.0, max_value=5.0, value=4.0, step=0.5)

        if st.button("🚀 Analyze Single Review", type="primary"):
            if test_text.strip():
                single_res = pipeline.analyze_single(test_text, rating=test_rating)

                r_col1, r_col2, r_col3, r_col4 = st.columns(4)
                with r_col1:
                    lbl = single_res.sentiment.label
                    color = "green" if lbl == "POSITIVE" else ("red" if lbl == "NEGATIVE" else "orange")
                    st.markdown(f"**Sentiment:** :{color}[{lbl}]")
                    st.caption(f"Compound Score: {single_res.sentiment.compound:+.3f}")
                with r_col2:
                    st.markdown(f"**Predicted Rating:** {single_res.predicted_rating:.1f} / 5.0")
                    st.caption(f"Actual: {test_rating:.1f} / 5.0")
                with r_col3:
                    st.markdown(f"**Dominant Emotion:** {single_res.sentiment.primary_emotion}")
                    st.caption(f"Confidence: {single_res.sentiment.confidence:.0%}")
                with r_col4:
                    if single_res.spam.is_spam:
                        st.error(f"⚠️ SPAM (Score: {single_res.spam.spam_score:.2f})")
                        st.caption("; ".join(single_res.spam.reasons))
                    else:
                        st.success("✅ Authentic Review")

                st.markdown("---")
                st.markdown("#### 🎯 Aspect Mentions Extracted:")
                if single_res.aspects:
                    for a in single_res.aspects:
                        a_col = "green" if a.sentiment_label == "POSITIVE" else ("red" if a.sentiment_label == "NEGATIVE" else "orange")
                        st.markdown(f"- **{a.category}** (term: `{a.matched_term}`): :{a_col}[{a.sentiment_label} ({a.sentiment_score:+.2f})] — *“{a.sentence}”*")
                else:
                    st.info("No predefined product aspects identified in this text.")
            else:
                st.warning("Please enter review text to analyze.")

    # TAB 7: REVIEW EXPLORER
    with tabs[6]:
        st.markdown("""
        <div style="background: #FFFFFF; padding: 18px 22px; border-radius: 14px; border: 1px solid #E2E8F0; margin-bottom: 20px;">
            <h3 style="margin: 0; font-size: 1.15rem; font-weight: 700; color: #0F172A;">📋 Review Explorer & Export</h3>
            <p style="margin: 4px 0 0 0; font-size: 0.88rem; color: #64748B;">
                Filter by sentiment, spam flags, or keywords, and export the processed intelligence dataset.
            </p>
        </div>
        """, unsafe_allow_html=True)

        df_processed = pipeline.to_dataframe(analysis_result)

        # Filters
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            sentiment_filter = st.multiselect("Filter by Sentiment:", ["POSITIVE", "NEUTRAL", "NEGATIVE"], default=["POSITIVE", "NEUTRAL", "NEGATIVE"])
        with f_col2:
            spam_filter = st.selectbox("Spam Filter:", ["All Reviews", "Authentic Only", "Spam Only"])
        with f_col3:
            search_query = st.text_input("Keyword Search:", "")

        df_filtered = df_processed[df_processed["sentiment_label"].isin(sentiment_filter)]
        if spam_filter == "Authentic Only":
            df_filtered = df_filtered[~df_filtered["is_spam"]]
        elif spam_filter == "Spam Only":
            df_filtered = df_filtered[df_filtered["is_spam"]]

        if search_query:
            df_filtered = df_filtered[df_filtered["text"].str.contains(search_query, case=False, na=False)]

        st.dataframe(
            df_filtered[["review_id", "text", "rating", "predicted_rating", "sentiment_label", "sentiment_compound", "primary_emotion", "is_spam", "aspects_mentioned"]],
            use_container_width=True,
            height=350
        )

        st.markdown("<br>", unsafe_allow_html=True)
        down_col1, down_col2 = st.columns(2)
        with down_col1:
            csv_data = df_processed.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Processed Dataset (CSV)", csv_data, "reviewlens_processed.csv", "text/csv")

        with down_col2:
            json_report = analysis_result.model_dump_json(indent=2).encode("utf-8")
            st.download_button("📥 Download Full Intelligence Report (JSON)", json_report, "reviewlens_report.json", "application/json")

else:
    st.info("👈 Select a sample dataset from the sidebar or upload your own CSV file to begin analysis.")
