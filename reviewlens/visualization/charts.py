"""
Interactive Plotly and visual analytics generation module for ReviewLens.
Produces publication-quality charts for Streamlit, reports, and dashboards.
"""

from typing import List, Dict, Any, Optional
import io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from reviewlens.models import AspectSummary, ReviewAnalysis


# Professional color palette
COLOR_MAP = {
    "POSITIVE": "#10B981",  # Emerald Green
    "NEUTRAL": "#94A3B8",   # Slate Gray
    "NEGATIVE": "#EF4444",  # Crimson Red
}


def create_sentiment_donut(distribution: Dict[str, int]) -> go.Figure:
    """Create an interactive donut chart of sentiment distribution."""
    labels = list(distribution.keys())
    values = list(distribution.values())
    colors = [COLOR_MAP.get(l.upper(), "#3B82F6") for l in labels]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker=dict(colors=colors),
                textinfo="label+percent",
                hoverinfo="label+value+percent",
            )
        ]
    )
    fig.update_layout(
        title=dict(text="Overall Sentiment Distribution", x=0.05, y=0.95),
        showlegend=True,
        height=430,
        margin=dict(t=60, b=80, l=30, r=30),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.12,
            xanchor="center",
            x=0.5
        ),
    )
    return fig


def create_aspect_bar_chart(aspects: List[AspectSummary]) -> go.Figure:
    """
    Create a stacked horizontal bar chart showing % positive, % neutral, % negative per aspect.
    Clean separated legend at bottom to prevent title collision.
    """
    if not aspects:
        return go.Figure()

    categories = [a.category for a in aspects][::-1]  # Reverse for top-down display
    pos_pcts = [a.sentiment_ratio.get("positive", 0.0) for a in aspects][::-1]
    neu_pcts = [a.sentiment_ratio.get("neutral", 0.0) for a in aspects][::-1]
    neg_pcts = [a.sentiment_ratio.get("negative", 0.0) for a in aspects][::-1]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=categories,
        x=pos_pcts,
        name="Positive %",
        orientation="h",
        marker=dict(color=COLOR_MAP["POSITIVE"]),
    ))
    fig.add_trace(go.Bar(
        y=categories,
        x=neu_pcts,
        name="Neutral %",
        orientation="h",
        marker=dict(color=COLOR_MAP["NEUTRAL"]),
    ))
    fig.add_trace(go.Bar(
        y=categories,
        x=neg_pcts,
        name="Negative %",
        orientation="h",
        marker=dict(color=COLOR_MAP["NEGATIVE"]),
    ))

    fig.update_layout(
        barmode="stack",
        title=dict(text="Aspect-Level Sentiment Breakdown (%)", x=0.05, y=0.95),
        height=450,
        xaxis=dict(title="Percentage (%)", range=[0, 100]),
        yaxis=dict(title=""),
        margin=dict(t=60, b=90, l=150, r=30),
        # Place legend cleanly at the bottom, centered, away from title
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5
        ),
    )
    return fig


def create_aspect_radar_chart(aspects: List[AspectSummary]) -> go.Figure:
    """
    Create a Radar / Spider chart showing the Net Sentiment Score (NSS) for each aspect.
    """
    if not aspects:
        return go.Figure()

    categories = [a.category for a in aspects]
    nss_scores = [a.net_sentiment_score for a in aspects]

    # Close the radar polygon loop
    categories_closed = categories + [categories[0]]
    scores_closed = nss_scores + [nss_scores[0]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=scores_closed,
            theta=categories_closed,
            fill="toself",
            name="Net Sentiment Score",
            line=dict(color="#3B82F6", width=2),
            fillcolor="rgba(59, 130, 246, 0.25)"
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[-100, 100],
                tickvals=[-100, -50, 0, 50, 100]
            )
        ),
        showlegend=False,
        height=450,
        title=dict(text="Aspect Net Sentiment Score Radar (NSS: -100 to +100)", x=0.05, y=0.95),
        margin=dict(t=70, b=50, l=60, r=60),
    )
    return fig


def create_rating_vs_sentiment_scatter(reviews: List[ReviewAnalysis]) -> go.Figure:
    """
    Create a scatter plot comparing user star rating vs. NLP compound sentiment.
    Clean separated legend at bottom.
    """
    rated_reviews = [r for r in reviews if r.rating is not None]
    if not rated_reviews:
        return go.Figure()

    df = pd.DataFrame([
        {
            "Rating": r.rating,
            "Predicted Rating": r.predicted_rating,
            "Sentiment Compound": r.sentiment.compound,
            "Label": r.sentiment.label,
            "Is Spam": "Suspicious / Spam" if r.spam.is_spam else "Authentic",
            "Snippet": r.original_text[:80] + ("..." if len(r.original_text) > 80 else "")
        }
        for r in rated_reviews
    ])

    fig = px.scatter(
        df,
        x="Rating",
        y="Sentiment Compound",
        color="Is Spam",
        color_discrete_map={"Authentic": "#3B82F6", "Suspicious / Spam": "#EF4444"},
        hover_data=["Snippet", "Predicted Rating", "Label"],
        title="Actual Rating vs. NLP Sentiment Score",
        labels={"Rating": "User Given Star Rating (1-5)", "Sentiment Compound": "Computed NLP Polarity (-1 to +1)"}
    )

    fig.update_layout(
        title=dict(x=0.05, y=0.95),
        height=430,
        margin=dict(t=60, b=80, l=50, r=30),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.20,
            xanchor="center",
            x=0.5
        )
    )
    return fig


def create_keyword_bar_chart(pos_keywords: List[Dict[str, Any]], neg_keywords: List[Dict[str, Any]]) -> go.Figure:
    """
    Create dual horizontal bar chart comparing top positive vs negative driver keywords.
    Clean separated legend at bottom.
    """
    fig = go.Figure()

    if pos_keywords:
        pos_terms = [k["term"] for k in pos_keywords[:10]][::-1]
        pos_scores = [k["score"] for k in pos_keywords[:10]][::-1]
        fig.add_trace(go.Bar(
            y=pos_terms,
            x=pos_scores,
            name="Positive Drivers",
            orientation="h",
            marker=dict(color=COLOR_MAP["POSITIVE"])
        ))

    if neg_keywords:
        neg_terms = [k["term"] for k in neg_keywords[:10]][::-1]
        neg_scores = [k["score"] for k in neg_keywords[:10]][::-1]
        fig.add_trace(go.Bar(
            y=neg_terms,
            x=neg_scores,
            name="Negative Pain Points",
            orientation="h",
            marker=dict(color=COLOR_MAP["NEGATIVE"])
        ))

    fig.update_layout(
        title=dict(text="Top Positive Drivers vs. Negative Pain Points (TF-IDF)", x=0.05, y=0.95),
        barmode="group",
        height=450,
        xaxis=dict(title="Importance / Frequency Score"),
        yaxis=dict(title=""),
        margin=dict(t=60, b=90, l=150, r=30),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5
        )
    )
    return fig


def generate_wordcloud_image(texts: List[str], max_words: int = 100):
    """
    Generate a WordCloud PIL image from texts with robust fallback.
    """
    if not texts:
        return None

    try:
        from wordcloud import WordCloud
        from reviewlens.preprocessing.tokenizer import FILTERED_STOPWORDS

        corpus = " ".join([t for t in texts if t])
        wc = WordCloud(
            width=800,
            height=450,
            background_color="white",
            colormap="viridis",
            stopwords=FILTERED_STOPWORDS,
            max_words=max_words,
            contour_width=1,
            contour_color="steelblue"
        ).generate(corpus)

        return wc.to_image()
    except Exception:
        try:
            import matplotlib.pyplot as plt
            from collections import Counter
            import re
            from reviewlens.preprocessing.tokenizer import FILTERED_STOPWORDS

            all_words = []
            for t in texts:
                words = re.findall(r"\b[a-zA-Z]{3,}\b", t.lower())
                all_words.extend([w for w in words if w not in FILTERED_STOPWORDS])
            common = Counter(all_words).most_common(12)

            fig, ax = plt.subplots(figsize=(8, 4.5))
            if common:
                terms, counts = zip(*common)
                ax.barh(list(terms)[::-1], list(counts)[::-1], color="#3B82F6")
                ax.set_title("Frequent Keywords in Reviews")
                ax.set_xlabel("Count")
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=100)
            plt.close(fig)
            buf.seek(0)

            from PIL import Image
            return Image.open(buf)
        except Exception:
            return None

def create_competitor_radar_chart(
    comparisons: List[Any],
    name_a: str = "Product A",
    name_b: str = "Product B",
) -> go.Figure:
    """Compare two products' net sentiment scores across aspects."""
    if not comparisons:
        return go.Figure()

    categories = [item.category for item in comparisons]
    scores_a = [item.score_a for item in comparisons]
    scores_b = [item.score_b for item in comparisons]

    # Close the radar polygons.
    categories += [categories[0]]
    scores_a += [scores_a[0]]
    scores_b += [scores_b[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scores_a,
        theta=categories,
        fill="toself",
        name=name_a,
    ))
    fig.add_trace(go.Scatterpolar(
        r=scores_b,
        theta=categories,
        fill="toself",
        name=name_b,
    ))

    fig.update_layout(
        title="Competitor Aspect Sentiment Comparison",
        polar=dict(radialaxis=dict(visible=True, range=[-100, 100])),
        height=450,
        margin=dict(t=70, b=50, l=60, r=60),
        legend=dict(orientation="h", yanchor="top", y=-0.15),
    )
    return fig


def create_head_to_head_bar_chart(
    comparisons: List[Any],
    name_a: str = "Product A",
    name_b: str = "Product B",
) -> go.Figure:
    """Compare two products' net sentiment scores with horizontal bars."""
    if not comparisons:
        return go.Figure()

    categories = [item.category for item in comparisons][::-1]
    scores_a = [item.score_a for item in comparisons][::-1]
    scores_b = [item.score_b for item in comparisons][::-1]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=categories,
        x=scores_a,
        name=name_a,
        orientation="h",
    ))
    fig.add_trace(go.Bar(
        y=categories,
        x=scores_b,
        name=name_b,
        orientation="h",
    ))

    fig.update_layout(
        title="Head-to-Head Net Sentiment Score by Aspect",
        barmode="group",
        xaxis=dict(title="Net Sentiment Score", range=[-100, 100]),
        yaxis=dict(title=""),
        height=450,
        margin=dict(t=60, b=80, l=150, r=30),
        legend=dict(orientation="h", yanchor="top", y=-0.2),
    )
    return fig
