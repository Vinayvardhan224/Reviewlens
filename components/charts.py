"""
ReviewLens Visualization & Charts Component.
Engineered for publication-quality visual analytics with zero legend/title overlap,
responsive container support, category-specific ABSA, and confusion matrix heatmaps.
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
    Filters out aspects with zero mentions to avoid false reporting.
    """
    if not aspects:
        return go.Figure()

    active = [a for a in aspects if a.get("mentions", 0) > 0][::-1]
    if not active:
        return go.Figure()

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
    Radar chart showing Net Sentiment Score per aspect.
    Only includes aspects with verified evidence.
    """
    if not aspects:
        return go.Figure()

    active = [a for a in aspects if a.get("mentions", 0) > 0 and a.get("net_sentiment_score") is not None]
    if len(active) < 3:
        # Fallback to bar chart if too few points to form a polygon
        fig = go.Figure()
        categories = [a["category"] for a in active]
        scores = [a["net_sentiment_score"] for a in active]
        colors = [COLOR_POSITIVE if s >= 0 else COLOR_NEGATIVE for s in scores]
        fig.add_trace(go.Bar(y=categories, x=scores, orientation="h", marker=dict(color=colors)))
        fig.update_layout(height=350, xaxis=dict(range=[-100, 100], title="Net Sentiment Score (-100 to +100)"))
        return fig

    categories = [a["category"] for a in active]
    nss_scores = [a["net_sentiment_score"] for a in active]

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

    # Collect shared categories that have evidence in at least one product
    all_cats = []
    for item in products_data:
        for a in item.get("aspects", []):
            if a.get("mentions", 0) > 0 and a["category"] not in all_cats:
                all_cats.append(a["category"])

    if len(all_cats) < 3:
        all_cats = [a["category"] for a in products_data[0].get("aspects", [])]

    if not all_cats:
        return go.Figure()

    for idx, item in enumerate(products_data):
        name = item.get("name", f"Product {idx+1}")
        aspect_dict = {a["category"]: a.get("net_sentiment_score", 0.0) for a in item.get("aspects", [])}
        scores = [aspect_dict.get(c, 0.0) if aspect_dict.get(c) is not None else 0.0 for c in all_cats]

        # Close loop
        cats_closed = all_cats + [all_cats[0]]
        scores_closed = scores + [scores[0]]

        c = colors[idx % len(colors)]
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
        xaxis=dict(title="TF-IDF Salience Score"),
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


def create_confusion_matrix_heatmap(cm: List[List[int]], labels: List[str]) -> go.Figure:
    """
    Render an interactive confusion matrix heatmap for model evaluation.
    """
    z = cm
    x = [f"Pred: {l}" for l in labels]
    y = [f"True: {l}" for l in labels]

    fig = px.imshow(
        z,
        x=x,
        y=y,
        color_continuous_scale="Blues",
        text_auto=True,
        labels=dict(x="Predicted Class", y="Actual Ground Truth", color="Review Count")
    )
    fig.update_layout(
        height=380,
        margin=dict(t=30, b=40, l=40, r=40)
    )
    return fig


def create_temporal_sentiment_chart(timeline_points: List[Dict[str, Any]]) -> go.Figure:
    """
    Renders dual-axis timeline chart:
      - Primary Axis: Monthly Net Sentiment Score (NSS) line curve
      - Secondary Axis: Monthly Review Volume bar columns
    """
    if not timeline_points:
        return go.Figure()

    periods = [p["period"] for p in timeline_points]
    nss_values = [p["net_sentiment_score"] for p in timeline_points]
    volumes = [p["review_count"] for p in timeline_points]

    fig = go.Figure()

    # Bar chart for review volume (secondary axis)
    fig.add_trace(
        go.Bar(
            x=periods,
            y=volumes,
            name="Review Volume",
            marker=dict(color="rgba(148, 163, 184, 0.35)"),
            yaxis="y2",
            hovertemplate="Period: %{x}<br>Reviews: %{y}<extra></extra>"
        )
    )

    # Line chart for Net Sentiment Score
    fig.add_trace(
        go.Scatter(
            x=periods,
            y=nss_values,
            mode="lines+markers",
            name="Net Sentiment Score (NSS)",
            line=dict(color=COLOR_PRIMARY, width=3),
            marker=dict(size=8, color=[COLOR_POSITIVE if v >= 0 else COLOR_NEGATIVE for v in nss_values]),
            hovertemplate="Period: %{x}<br>NSS: %{y:+.1f}<extra></extra>"
        )
    )

    fig.update_layout(
        title=dict(text="Consumer Sentiment Trend & Review Volume Over Time", x=0.05, y=0.95),
        height=380,
        margin=dict(t=50, b=60, l=40, r=40),
        xaxis=dict(title="Month / Time Period"),
        yaxis=dict(
            title="Net Sentiment Score (-100 to +100)",
            range=[-100, 100],
            zeroline=True,
            zerolinecolor="#94A3B8"
        ),
        yaxis2=dict(
            title="Monthly Reviews",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.20,
            xanchor="center",
            x=0.5
        )
    )
    return fig


def create_live_sentiment_gauge(compound_score: float) -> go.Figure:
    """
    Renders an interactive gauge indicator for real-time review sentiment polarity.
    Compound score ranges from -1.0 to +1.0.
    """
    # Scale -1.0..+1.0 into 0..100 for gauge visualization
    normalized = round((compound_score + 1.0) * 50.0, 1)

    if compound_score >= 0.05:
        bar_color = COLOR_POSITIVE
        sentiment_label = "POSITIVE"
    elif compound_score <= -0.05:
        bar_color = COLOR_NEGATIVE
        sentiment_label = "NEGATIVE"
    else:
        bar_color = COLOR_NEUTRAL
        sentiment_label = "NEUTRAL"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=compound_score,
            delta={"reference": 0.0, "increasing": {"color": COLOR_POSITIVE}, "decreasing": {"color": COLOR_NEGATIVE}},
            number={"valueformat": "+.2f", "font": {"size": 36, "color": bar_color}},
            title={"text": f"Valence: <b>{sentiment_label}</b>", "font": {"size": 16, "color": "#0F172A"}},
            gauge={
                "axis": {"range": [-1.0, 1.0], "tickwidth": 1, "tickcolor": "#64748B"},
                "bar": {"color": bar_color, "thickness": 0.3},
                "bgcolor": "#F8FAFC",
                "borderwidth": 1,
                "bordercolor": "#E2E8F0",
                "steps": [
                    {"range": [-1.0, -0.05], "color": "rgba(239, 68, 68, 0.12)"},
                    {"range": [-0.05, 0.05], "color": "rgba(148, 163, 184, 0.12)"},
                    {"range": [0.05, 1.0], "color": "rgba(16, 185, 129, 0.12)"}
                ],
                "threshold": {
                    "line": {"color": "#0F172A", "width": 3},
                    "thickness": 0.75,
                    "value": compound_score
                }
            }
        )
    )

    fig.update_layout(
        height=260,
        margin=dict(t=40, b=20, l=30, r=30)
    )
    return fig
