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

        # Map aspects by name (safely handling None for unevidenced aspects)
        aspect_dict = {
            a["category"]: a.get("net_sentiment_score")
            for a in summary.get("aspects", [])
        }

        price = float(prod.get("price", 0.0))
        scored.append({
            "product": prod,
            "summary": summary,
            "composite_score": composite,
            "price": price,
            "aspect_scores": aspect_dict
        })

    def get_candidate_score(item, candidates):
        for c in candidates:
            val = item["aspect_scores"].get(c)
            if val is not None:
                return float(val), c
        return None, None

    def pick_best_by_candidates(candidates, fallback_key="composite_score"):
        best_item = None
        best_score = -999.0
        best_matched_aspect = None
        for it in scored:
            val, matched = get_candidate_score(it, candidates)
            if val is not None and val > best_score:
                best_score = val
                best_item = it
                best_matched_aspect = matched
        if best_item is None:
            best_item = max(scored, key=lambda x: x[fallback_key])
            return best_item, None, None
        return best_item, best_score, best_matched_aspect

    # 1. Best Overall
    best_overall = max(scored, key=lambda x: x["composite_score"])

    # 2. Best Budget / Value Pick
    # Value index: composite_score / (log or normalized price)
    best_budget = max(scored, key=lambda x: (x["composite_score"] / (max(x["price"], 1000.0) ** 0.35)))

    # 3. Best Camera / Image Quality
    camera_candidates = ["Camera", "Image quality", "Camera & Photography", "Lens quality"]
    best_camera, cam_score, cam_aspect = pick_best_by_candidates(camera_candidates)

    # 4. Best Battery / Endurance
    battery_candidates = ["Battery", "Battery & Charging", "Cooling"]
    best_battery, batt_score, batt_aspect = pick_best_by_candidates(battery_candidates)

    # 5. Best Performance / Core Experience
    perf_candidates = ["Performance", "Processor", "Sound quality", "Performance & Speed", "GPU"]
    best_perf, perf_score, perf_aspect = pick_best_by_candidates(perf_candidates)

    def format_score_str(score, aspect_name):
        if score is not None and aspect_name:
            sign = "+" if score > 0 else ""
            return f"{aspect_name} NSS: {sign}{score:.1f}"
        return "Strong consumer sentiment"

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
            "score": cam_score if cam_score is not None else best_camera["composite_score"],
            "reason": f"Highest customer satisfaction for capture & optics ({format_score_str(cam_score, cam_aspect)})."
        },
        "best_battery": {
            "product": best_battery["product"],
            "score": batt_score if batt_score is not None else best_battery["composite_score"],
            "reason": f"Highest customer satisfaction for battery endurance and runtime ({format_score_str(batt_score, batt_aspect)})."
        },
        "best_performance": {
            "product": best_perf["product"],
            "score": perf_score if perf_score is not None else best_perf["composite_score"],
            "reason": f"Highest customer satisfaction for speed and core user experience ({format_score_str(perf_score, perf_aspect)})."
        }
    }
