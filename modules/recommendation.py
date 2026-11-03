"""
ReviewLens Recommendation Engine.
Produces defensible, rule-based product recommendations with transparent
weighting formulas and statistical sample size disclosures.
"""

from typing import Dict, Any, List, Optional
import pandas as pd


def compute_product_recommendation_score(prod: Dict[str, Any], review_summary: Dict[str, Any]) -> float:
    """
    Compute transparent composite quality score (0 to 100).
    Weights:
      - Star Rating: 30%
      - Overall NLP Net Sentiment Score: 45%
      - CSAT %: 25%
    """
    rating = float(prod.get("rating", 4.0))
    rating_score = (rating / 5.0) * 100.0

    nss = float(review_summary.get("nss", 0.0))
    # Normalize NSS (-100 to +100) into 0 to 100 range
    nss_normalized = (nss + 100.0) / 2.0

    csat = float(review_summary.get("csat_estimate", 50.0))

    composite = (0.30 * rating_score) + (0.45 * nss_normalized) + (0.25 * csat)
    return round(composite, 1)


def generate_category_recommendations(
    products_with_summaries: List[Dict[str, Any]],
    category: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluates a list of (product, review_summary) pairs and determines:
      - Best Overall
      - Best Budget / Value Pick
      - Best for Camera / Photography
      - Best for Battery Endurance
      - Best for Speed & Gaming
    """
    # Filter by category if provided
    items = products_with_summaries
    if category and category.lower() != "all":
        items = [item for item in items if item["product"].get("category", "").lower() == category.lower()]

    if not items:
        return {}

    # Score each product
    scored = []
    for item in items:
        prod = item["product"]
        summary = item["summary"]
        composite = compute_product_recommendation_score(prod, summary)

        # Map aspects by name
        aspect_dict = {a["category"]: a.get("net_sentiment_score", 0.0) for a in summary.get("aspects", [])}

        price = float(prod.get("price", 0.0))
        scored.append({
            "product": prod,
            "summary": summary,
            "composite_score": composite,
            "price": price,
            "aspect_scores": aspect_dict
        })

    # 1. Best Overall
    best_overall = max(scored, key=lambda x: x["composite_score"])

    # 2. Best Budget / Value Pick
    # Value index: composite_score / (log or normalized price)
    best_budget = max(scored, key=lambda x: (x["composite_score"] / (max(x["price"], 1000.0) ** 0.35)))

    # 3. Best Camera
    best_camera = max(scored, key=lambda x: x["aspect_scores"].get("Camera & Photography", -100))

    # 4. Best Battery
    best_battery = max(scored, key=lambda x: x["aspect_scores"].get("Battery & Charging", -100))

    # 5. Best Performance
    best_perf = max(scored, key=lambda x: x["aspect_scores"].get("Performance & Speed", -100))

    return {
        "best_overall": {
            "product": best_overall["product"],
            "score": best_overall["composite_score"],
            "reason": f"Top-ranked composite score ({best_overall['composite_score']}/100) combining stellar star ratings, superior NLP Net Sentiment (+{best_overall['summary'].get('nss', 0):.1f}), and comprehensive customer praise."
        },
        "best_budget": {
            "product": best_budget["product"],
            "score": best_budget["composite_score"],
            "reason": f"Offers exceptional feature-to-price ratio at ₹{best_budget['price']:,.0f} with strong customer satisfaction (+{best_budget['summary'].get('nss', 0):.1f} NSS)."
        },
        "best_camera": {
            "product": best_camera["product"],
            "score": best_camera["aspect_scores"].get("Camera & Photography", 0.0),
            "reason": f"Highest customer satisfaction for camera & photo capture (NSS: +{best_camera['aspect_scores'].get('Camera & Photography', 0.0):.1f})."
        },
        "best_battery": {
            "product": best_battery["product"],
            "score": best_battery["aspect_scores"].get("Battery & Charging", 0.0),
            "reason": f"Highest customer satisfaction for battery endurance and charging speeds (NSS: +{best_battery['aspect_scores'].get('Battery & Charging', 0.0):.1f})."
        },
        "best_performance": {
            "product": best_perf["product"],
            "score": best_perf["aspect_scores"].get("Performance & Speed", 0.0),
            "reason": f"Highest customer satisfaction for processor speed, fluid display, and sustained gaming (NSS: +{best_perf['aspect_scores'].get('Performance & Speed', 0.0):.1f})."
        }
    }
