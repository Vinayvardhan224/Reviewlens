"""
ReviewLens: Interactive Streamlit Web Application Dashboard (Original Engine).
An NLP-based smart product review intelligence and analytics platform.
Preserved as an accessible module per ReviewLens technical requirements.
"""

import sys
from pathlib import Path
import io
import json
import streamlit as st
import pandas as pd

# Set project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


def render_original_dashboard():
    """Renders the complete original 7-tab ReviewLens Analytics Dashboard."""
    # Data directory
    data_dir = PROJECT_ROOT / "data"

    # Initialize NLP Pipeline
    pipeline = ReviewLensPipeline()

    st.markdown("""
    <div style="background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%); padding: 1.5rem 1.8rem; border-radius: 14px; color: #FFFFFF; margin-bottom: 1.5rem; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1rem;">
            <div>
                <div style="font-size: 1.5rem; font-weight: 800; color: #FFFFFF;">
                    Review<span style="color: #FF9900;">Lens</span> Original Analytics Dashboard
                </div>
                <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 0.2rem;">
                    Full-pipeline batch text ingestion, VADER compound scoring, ABSA, LDA topics, and SWOT engine
                </div>
            </div>
            <div style="background: rgba(255, 153, 0, 0.18); border: 1px solid #FF9900; padding: 0.4rem 0.9rem; border-radius: 20px; font-size: 0.82rem; font-weight: 700; color: #FF9900;">
                🛠️ Legacy Batch Mode
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Data Source Selection
    ds_col1, ds_col2 = st.columns([2, 1])
    with ds_col1:
        dataset_choice = st.selectbox(
            "Select Review Dataset to Analyze:",
            [
                "📱 Flagship Smartphone Reviews (50 Reviews)",
                "🎧 Noise-Canceling Wireless Headphones (70 Reviews)",
                "☕ Smart Coffee Maker IoT Feedback (40 Reviews)",
                "📦 Real-world Amazon Echo Smart Speaker Corpus (30 Reviews)",
                "📤 Upload Custom File (CSV/Excel/JSON)"
            ],
            index=0
        )

    df_reviews = None
    if "Smartphone" in dataset_choice:
        df_reviews = pd.read_csv(data_dir / "sample_smartphone_reviews.csv")
    elif "Headphones" in dataset_choice:
        df_reviews = pd.read_csv(data_dir / "sample_headphones_reviews.csv")
    elif "Amazon Echo" in dataset_choice:
        df_reviews = pd.read_csv(data_dir / "real_world_amazon_echo_reviews.csv")
    elif "Coffee Maker" in dataset_choice:
        df_reviews = pd.read_csv(data_dir / "sample_coffee_maker_reviews.csv")
    else:
        uploaded_file = st.file_uploader("Upload CSV, Excel, or JSON", type=["csv", "xlsx", "json"])
        if uploaded_file:
            if uploaded_file.name.endswith(".csv"):
                df_reviews = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".xlsx"):
                df_reviews = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith(".json"):
                df_reviews = pd.read_json(uploaded_file)

    if df_reviews is not None and not df_reviews.empty:
        def analyze_dataframe(df_in):
            candidate_t = [c for c in df_in.columns if any(k in c.lower() for k in ["text", "review", "content", "body", "comment"])]
            t_col = candidate_t[0] if candidate_t else df_in.columns[0]
            candidate_r = [c for c in df_in.columns if any(k in c.lower() for k in ["rating", "score", "stars"])]
            r_col = candidate_r[0] if candidate_r else None
            return pipeline.analyze_dataset(df_in, text_column=t_col, rating_column=r_col)

        with st.spinner("🧠 Analyzing reviews with NLP pipeline (Sentiment, ABSA, Topics)..."):
            analysis_result = analyze_dataframe(df_reviews)

        # Tab Navigation
        tabs = st.tabs([
            "📊 Executive KPIs",
            "🔍 Aspect-Based Sentiment (ABSA)",
            "⚔️ Phone & Competitor Comparison",
            "🧠 Deep NLP & Topics",
            "🎯 SWOT & Action Plan",
            "✍️ Single Review Tester",
            "📋 Review Explorer"
        ])

        # TAB 1: EXECUTIVE KPIS
        with tabs[0]:
            st.markdown("### 📊 **Executive Overview & Performance Metrics**")
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("Total Reviews", f"{analysis_result.total_reviews:,}")
            with k2:
                nss = analysis_result.executive_summary.net_sentiment_score
                st.metric("Net Sentiment Score", f"{nss:+.1f}", delta=f"{nss:+.1f}")
            with k3:
                csat = analysis_result.executive_summary.csat_score
                st.metric("Estimated CSAT", f"{csat:.1f}%")
            with k4:
                spam_count = analysis_result.spam_reviews
                st.metric("Spam / Fake Reviews", f"{spam_count} ({spam_count/max(1, analysis_result.total_reviews)*100:.1f}%)")

            c_left, c_right = st.columns(2)
            with c_left:
                st.markdown("##### Sentiment Distribution")
                st.plotly_chart(create_sentiment_donut(analysis_result.overall_sentiment_distribution), use_container_width=True)
            with c_right:
                st.markdown("##### Rating vs Sentiment Anomaly Explorer")
                st.plotly_chart(create_rating_vs_sentiment_scatter(analysis_result.reviews), use_container_width=True)

        # TAB 2: ABSA
        with tabs[1]:
            st.markdown("### 🔍 **Aspect-Based Sentiment Analysis (ABSA)**")
            st.caption("ℹ️ Methodology Notice: Keyword-based aspect matching is a rule-based prototype method, not a fully trained ABSA neural model.")
            ab_c1, ab_c2 = st.columns(2)
            with ab_c1:
                st.plotly_chart(create_aspect_bar_chart(analysis_result.aspect_summaries), use_container_width=True)
            with ab_c2:
                st.plotly_chart(create_aspect_radar_chart(analysis_result.aspect_summaries), use_container_width=True)

            with st.expander("🔎 View Evidence Quotes by Aspect Dimension", expanded=False):
                for asp in analysis_result.aspect_summaries:
                    if asp.total_mentions > 0:
                        st.markdown(f"**{asp.category}** (Mentions: `{asp.total_mentions}` | NSS: `{asp.net_sentiment_score:+.1f}`)")
                        for q in asp.sample_positive_quotes[:2]:
                            st.caption(f"🟢 *\"{q}\"*")
                        for q in asp.sample_negative_quotes[:2]:
                            st.caption(f"🔴 *\"{q}\"*")
                        st.markdown("---")

        # TAB 3: COMPETITOR COMPARISON
        with tabs[2]:
            st.markdown("### ⚔️ **Head-to-Head Competitor Comparison**")
            matchup_type = st.radio(
                "Select Comparison Dataset:",
                [
                    "📱 Flagship Phone A (120Hz & Fast Charge) vs. Flagship Phone B (Pro Cam & Cool Thermals)",
                    "🎧 Wireless Headphones A (Sony) vs. Headphones B (Bose)",
                    "📤 Compare 2 Custom Review Files (CSV)"
                ],
                horizontal=True
            )

            if "Flagship Phone" in matchup_type:
                name_a = "Flagship Phone A"
                name_b = "Flagship Phone B"
                df_a = pd.read_csv(data_dir / "sample_smartphone_reviews.csv")
                df_b = pd.read_csv(data_dir / "sample_phone_b_reviews.csv")
            elif "Wireless Headphones" in matchup_type:
                name_a = "Headphones A (Sony WH-1000XM5)"
                name_b = "Headphones B (Bose QC Ultra)"
                df_a = pd.read_csv(data_dir / "sample_headphones_reviews.csv")
                df_b = pd.read_csv(data_dir / "sample_headphones_b_reviews.csv") if (data_dir / "sample_headphones_b_reviews.csv").exists() else pd.read_csv(data_dir / "sample_headphones_reviews.csv")
            else:
                u_col1, u_col2 = st.columns(2)
                with u_col1:
                    name_a = st.text_input("Product A Name:", "Product A")
                    file_a = st.file_uploader("Upload Product A Reviews (CSV):", type=["csv"], key="dash_file_comp_a")
                with u_col2:
                    name_b = st.text_input("Product B Name:", "Product B")
                    file_b = st.file_uploader("Upload Product B Reviews (CSV):", type=["csv"], key="dash_file_comp_b")
                df_a = pd.read_csv(file_a) if file_a else pd.read_csv(data_dir / "sample_smartphone_reviews.csv")
                df_b = pd.read_csv(file_b) if file_b else pd.read_csv(data_dir / "sample_phone_b_reviews.csv")

            name_a = name_a.strip() if name_a and name_a.strip() else "Product A"
            name_b = name_b.strip() if name_b and name_b.strip() else "Product B"

            res_a = analyze_dataframe(df_a)
            res_b = analyze_dataframe(df_b)

            # Sample size warning
            if len(df_a) < 10 or len(df_b) < 10:
                st.warning(f"⚠️ Sample Size Warning: Prepared sample is small ({name_a}: {len(df_a)} reviews, {name_b}: {len(df_b)} reviews). Comparative rankings are illustrative.")

            comp_report = ProductComparator.compare(res_a, res_b, name_a=name_a, name_b=name_b)

            winner_emoji = "👑" if "Draw" not in comp_report.overall_winner else "🤝"
            verdict_text = getattr(comp_report, "summary", getattr(comp_report, "executive_verdict", ""))
            st.success(f"{winner_emoji} Overall Verdict: **{comp_report.overall_winner} Takes the Lead**\n\n{verdict_text}")

            cp1, cp2 = st.columns(2)
            with cp1:
                st.plotly_chart(create_competitor_radar_chart(comp_report.aspect_comparisons, name_a=comp_report.product_a_name, name_b=comp_report.product_b_name), use_container_width=True)
            with cp2:
                st.plotly_chart(create_head_to_head_bar_chart(comp_report.aspect_comparisons, name_a=comp_report.product_a_name, name_b=comp_report.product_b_name), use_container_width=True)

        # TAB 4: DEEP NLP & TOPICS
        with tabs[3]:
            st.markdown("### 🧠 **Unsupervised Topic Modeling (LDA)**")
            if analysis_result.topics:
                for top in analysis_result.topics:
                    st.markdown(f"##### 🏷️ {top.label}")
                    st.write(f"**Keywords:** {', '.join(top.keywords)}")
                    for s in top.sample_reviews[:2]:
                        st.caption(f"• *\"{s}\"*")
                    st.markdown("---")
            else:
                st.info("Insufficient texts to cluster topics.")

            kw_col1, kw_col2 = st.columns(2)
            with kw_col1:
                st.plotly_chart(create_keyword_bar_chart(analysis_result.top_positive_keywords, "Top Positive Feedback Drivers", "#10B981"), use_container_width=True)
            with kw_col2:
                st.plotly_chart(create_keyword_bar_chart(analysis_result.top_negative_keywords, "Top Customer Pain Points", "#EF4444"), use_container_width=True)

        # TAB 5: SWOT & ACTION PLAN
        with tabs[4]:
            st.markdown("### 🎯 **SWOT Analysis & Strategic Action Plan**")
            swot = analysis_result.executive_summary
            sw_c1, sw_c2 = st.columns(2)
            with sw_c1:
                st.markdown("##### 🟢 Strengths")
                for s in swot.strengths:
                    st.write(f"• {s}")
                st.markdown("##### 🟡 Opportunities")
                for o in swot.opportunities:
                    st.write(f"• {o}")
            with sw_c2:
                st.markdown("##### 🔴 Weaknesses")
                for w in swot.weaknesses:
                    st.write(f"• {w}")
                st.markdown("##### ⚠️ Threats")
                for t in swot.threats:
                    st.write(f"• {t}")

        # TAB 6: SINGLE REVIEW TESTER
        with tabs[5]:
            st.markdown("### ✍️ **Live Single Review NLP Tester**")
            test_txt = st.text_area(
                "Enter custom product review text:",
                "The battery life is phenomenal and easily lasts 2 days, but the fast charger in the box stopped working after a week."
            )
            if st.button("Run Live NLP Inspection", type="primary"):
                test_res = pipeline.analyze_single_review(test_txt)
                st.write(f"**Sentiment Polarity:** `{test_res.sentiment.label}` (Compound Score: `{test_res.sentiment.compound:+.3f}`)")
                st.write(f"**Spam Index:** `{'🚨 IS SPAM' if test_res.spam.is_spam else '✅ Authentic Review'}` (Score: `{test_res.spam.spam_score:.2f}`)")
                if test_res.aspects:
                    st.write("**Extracted Aspects:**")
                    for a in test_res.aspects:
                        st.write(f"- Dimension: **{a.category}** | Term: `{a.matched_term}` | Sentiment: `{a.sentiment_label}` ({a.sentiment_score:+.2f})")

        # TAB 7: REVIEW EXPLORER
        with tabs[6]:
            st.markdown("### 📋 **Processed Review Corpus Table**")
            clean_records = []
            for r in analysis_result.reviews:
                clean_records.append({
                    "Review ID": r.id,
                    "Sentiment": r.sentiment.label,
                    "Compound": round(r.sentiment.compound, 3),
                    "Spam": "Yes" if r.spam.is_spam else "No",
                    "Detected Aspects": ", ".join(r.detected_aspects) if r.detected_aspects else "None",
                    "Review Text": r.original_text
                })
            df_proc = pd.DataFrame(clean_records)
            st.dataframe(df_proc, use_container_width=True, hide_index=True)
            csv_data = df_proc.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Processed Dataset (CSV)", csv_data, "reviewlens_processed.csv", "text/csv")
    else:
        st.info("👈 Select a sample dataset above or upload your own CSV file to begin analysis.")
