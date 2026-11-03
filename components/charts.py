"""
ReviewLens Visualization & Charts Component.
Engineered for publication-quality visual analytics with zero legend/title overlap,
responsive container support, and accessible color schemes.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Color constants
COLOR_POSITIVE = "#10B981"  # Emerald
COLOR_NEUTRAL = "#94A3B8"   # Slate
COLOR_NEGATIVE = "#EF4444"  # Crimson
COLOR_PRIMARY = "#2563EB"   # Royal Blue
COLOR_ACCENT = "#F59E0B"    # Amber / Orange


def create_sentiment_donut_chart(distribution: Dict[str, int]) -> go.Figure:
    """Create clean sentiment distribution donut chart."""
    labels = ["Positive", "Neutral", "Negative"]
    values = [
        distribution.get("POSITIVE", 0),
        distribution.get("NEUTRAL", 0),
        distribution.get("NEGATIVE", 0)
    ]
    colors = [COLOR_POSITIVE, COLOR_NEUTRAL, COLOR_NEGATIVE]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.60,
                marker=dict(colors=colors),
                textinfo="label+percent",
                hoverinfo="label+value+percent",
                sort=False
            )
        ]
    )
    fig.update_layout(
        showlegend=True,
        height=380,
        margin=dict(t=20, b=50, l=20, r=20),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.08,
            xanchor="center",
            x=0.5
        )
    )
    return fig


def create_aspect_breakdown_bar_chart(aspects: List[Dict[str, Any]]) -> go.Figure:
    """
    Horizontal stacked bar chart showing % Positive, % Neutral, % Negative per aspect.
    Title is handled outside in Streamlit markdown to guarantee 0% title/legend collision.
    """
    if not aspects:
        return go.Figure()

    # Filter to aspects with mentions and reverse for top-down display
    active = [a for a in aspects if a.get("mentions", 0) > 0][::-1]
    if not active:
        active = aspects[::-1]

    categories = [a["category"] for a in active]
    pos_pcts = [a.get("pos_pct", 0.0) for a in active]
    neu_pcts = [a.get("neu_pct", 0.0) for a in active]
    neg_pcts = [a.get("neg_pct", 0.0) for a in active]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=categories,
        x=pos_pcts,
        name="Positive %",
        orientation="h",
        marker=dict(color=COLOR_POSITIVE),
        hovertemplate="%{y}: %{x:.1f}% Positive<extra></extra>"
    ))
    fig.add_trace(go.Bar(
        y=categories,
        x=neu_pcts,
        name="Neutral %",
        orientation="h",
        marker=dict(color=COLOR_NEUTRAL),
        hovertemplate="%{y}: %{x:.1f}% Neutral<extra></extra>"
    ))
    fig.add_trace(go.Bar(
        y=categories,
        x=neg_pcts,
        name="Negative %",
        orientation="h",
        marker=dict(color=COLOR_NEGATIVE),
        hovertemplate="%{y}: %{x:.1f}% Negative<extra></extra>"
    ))

    fig.update_layout(
        barmode="stack",
        height=420,
        xaxis=dict(title="Sentiment Percentage (%)", range=[0, 100]),
        yaxis=dict(title=""),
        margin=dict(t=15, b=65, l=150, r=20),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.20,
            xanchor="center",
            x=0.5
        )
    )
    return fig


def create_aspect_radar_chart(aspects: List[Dict[str, Any]], product_name: str = "Product") -> go.Figure:
    """
    Radar (spider) chart showing Net Sentiment Score (-100 to +100) per aspect.
    """
    if not aspects:
        return go.Figure()

    active = [a for a in aspects if a.get("mentions", 0) > 0]
    if not active:
        active = aspects

    categories = [a["category"] for a in active]
    nss_scores = [a.get("net_sentiment_score", 0.0) for a in active]

    # Close polygon loop
    categories_closed = categories + [categories[0]]
    scores_closed = nss_scores + [nss_scores[0]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=scores_closed,
            theta=categories_closed,
            fill="toself",
            name=product_name,
            line=dict(color="#2563EB", width=2.5),
            fillcolor="rgba(37, 99, 235, 0.22)",
            hovertemplate="<b>%{theta}</b><br>NSS: %{r:+.1f}<extra></extra>"
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[-100, 100],
                tickvals=[-100, -50, 0, 50, 100],
                ticktext=["-100", "-50", "0", "+50", "+100"]
            )
        ),
        showlegend=False,
        height=420,
        margin=dict(t=25, b=30, l=40, r=40)
    )
    return fig


def create_multi_product_radar_chart(products_data: List[Dict[str, Any]]) -> go.Figure:
    """
    Multi-trace radar chart comparing 2 or 3 products on shared aspect dimensions.
    """
    if not products_data:
        return go.Figure()

    colors = ["#2563EB", "#DC2626", "#059669"]

    fig = go.Figure()

    # Collect all categories
    all_cats = []
    for item in products_data:
        for a in item.get("aspects", []):
            if a["category"] not in all_cats:
                all_cats.append(a["category"])

    if not all_cats:
        return go.Figure()

    for idx, item in enumerate(products_data):
        name = item.get("name", f"Product {idx+1}")
        aspect_dict = {a["category"]: a.get("net_sentiment_score", 0.0) for a in item.get("aspects", [])}
        scores = [aspect_dict.get(c, 0.0) for c in all_cats]

        # Close loop
        cats_closed = all_cats + [all_cats[0]]
        scores_closed = scores + [scores[0]]

        c = colors[idx % len(colors)]
        # Extract rgb components for fill
        fill_col = c.replace("#", "")
        r = int(fill_col[0:2], 16)
        g = int(fill_col[2:4], 16)
        b = int(fill_col[4:6], 16)

        fig.add_trace(
            go.Scatterpolar(
                r=scores_closed,
                theta=cats_closed,
                fill="toself",
                name=name,
                line=dict(color=c, width=2.5),
                fillcolor=f"rgba({r}, {g}, {b}, 0.18)",
                hovertemplate=f"<b>{name}</b><br>%{{theta}}: %{{r:+.1f}} NSS<extra></extra>"
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
        showlegend=True,
        height=450,
        margin=dict(t=25, b=65, l=40, r=40),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.12,
            xanchor="center",
            x=0.5
        )
    )
    return fig


def create_keyword_driver_chart(pos_keywords: List[Dict[str, Any]], neg_keywords: List[Dict[str, Any]]) -> go.Figure:
    """
    Dual bar chart comparing top positive driver terms vs negative pain points.
    Title rendered outside in markdown. Legend placed at bottom with margin.
    """
    fig = go.Figure()

    if pos_keywords:
        terms = [k["term"] for k in pos_keywords[:8]][::-1]
        scores = [k["score"] for k in pos_keywords[:8]][::-1]
        fig.add_trace(go.Bar(
            y=terms,
            x=scores,
            name="Positive Drivers",
            orientation="h",
            marker=dict(color=COLOR_POSITIVE),
            hovertemplate="Driver: %{y}<br>Weight: %{x:.2f}<extra></extra>"
        ))

    if neg_keywords:
        terms = [k["term"] for k in neg_keywords[:8]][::-1]
        scores = [k["score"] for k in neg_keywords[:8]][::-1]
        fig.add_trace(go.Bar(
            y=terms,
            x=scores,
            name="Negative Pain Points",
            orientation="h",
            marker=dict(color=COLOR_NEGATIVE),
            hovertemplate="Pain Point: %{y}<br>Weight: %{x:.2f}<extra></extra>"
        ))

    fig.update_layout(
        barmode="group",
        height=400,
        xaxis=dict(title="TF-IDF Importance Score"),
        yaxis=dict(title=""),
        margin=dict(t=15, b=65, l=140, r=20),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5
        )
    )
    return fig


def create_rating_scatter_chart(records: List[Dict[str, Any]]) -> go.Figure:
    """
    Scatter plot comparing star rating vs sentiment compound score to spot anomalies.
    """
    valid = [r for r in records if r.get("review_rating") is not None]
    if not valid:
        return go.Figure()

    df = pd.DataFrame([
        {
            "Rating": float(r.get("review_rating", 3.0)),
            "Compound": float(r.get("sentiment_compound", 0.0)),
            "Anomaly": "Mismatched / Anomaly" if r.get("is_anomaly") else "Authentic / Consistent",
            "Snippet": (r.get("review_text", "")[:75] + "...") if len(r.get("review_text", "")) > 75 else r.get("review_text", "")
        }
        for r in valid
    ])

    fig = px.scatter(
        df,
        x="Rating",
        y="Compound",
        color="Anomaly",
        color_discrete_map={
            "Authentic / Consistent": COLOR_PRIMARY,
            "Mismatched / Anomaly": COLOR_NEGATIVE
        },
        hover_data=["Snippet"],
        labels={"Rating": "User Star Rating (1 to 5)", "Compound": "NLP Sentiment Polarity (-1 to +1)"}
    )

    fig.update_layout(
        height=380,
        margin=dict(t=20, b=65, l=40, r=20),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5
        )
    )
    return fig
