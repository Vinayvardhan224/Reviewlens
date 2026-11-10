"""
ReviewLens Recommendation Engine.
Produces rule-based product recommendations with evidence gating,
transparent weighting formulas, review count disclosures, and prototype heuristic labeling.
"""

from typing import Dict, Any, List, Optional
import pandas as pd


RECOMMENDATION_METHODOLOGY_DISCLOSURE = (
    "Methodology Note: The composite recommendation score is an explainable decision-support "
    "prototype heuristic combining catalogue star ratings, VADER Net Sentiment, and CSAT polarity "
    "ratios. It is not an empirically validated consumer quality index."
)


def compute_product_recommendation_score(prod: Dict[str, Any], review_summary: Dict[str, Any]) -> float:
    """
    Compute transparent composite quality score (0 to 100).
    Weights (prototype heuristic):
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
    Evaluates a list of (product, review_summary) pairs with strict evidence gating:
      - Products with 0 reviews are excluded from category awards.
      - Category-specific awards (Camera, Battery, Performance) require verified aspect evidence.
      - If no candidate aspect evidence exists, displays 'Insufficient review evidence' instead of an arbitrary fallback winner.
    """
    items = products_with_summaries
    if category and category.lower() != "all":
        items = [item for item in items if item["product"].get("category", "").lower() == category.lower()]

    if not items:
        return {}

    # Score each product with evidence check
    scored = []
    for item in items:
        prod = item["product"]
        summary = item["summary"]
        
        # Check review evidence volume
        total_revs = summary.get("total_reviews", len(summary.get("aspects", [])))
        aspects_list = summary.get("aspects", [])
        total_mentions = sum(a.get("mentions", 0) for a in aspects_list)

        composite = compute_product_recommendation_score(prod, summary)

        # Map aspects by name (safely handling None for unevidenced aspects)
        aspect_dict = {}
        aspect_mentions = {}
        for a in aspects_list:
            cat_name = a["category"]
            m_count = a.get("mentions", 0)
            aspect_mentions[cat_name] = m_count
            if m_count > 0 and a.get("net_sentiment_score") is not None:
                aspect_dict[cat_name] = float(a["net_sentiment_score"])
            else:
                aspect_dict[cat_name] = None

        price = float(prod.get("price", 0.0))
        scored.append({
            "product": prod,
            "summary": summary,
            "composite_score": composite,
            "price": price,
            "aspect_scores": aspect_dict,
            "aspect_mentions": aspect_mentions,
            "total_reviews": total_revs,
            "total_mentions": total_mentions
        })

    if not scored:
        return {}

    # 1. Best Overall Pick
    best_overall = max(scored, key=lambda x: x["composite_score"])

    # 2. Best Budget / Value Pick (Composite score normalized by price curve)
    best_budget = max(scored, key=lambda x: (x["composite_score"] / (max(x["price"], 1000.0) ** 0.35)))

    def pick_best_by_candidates(candidates):
        """Pick best product for an aspect ONLY if evidence exists."""
        best_item = None
        best_score = -999.0
        best_matched_aspect = None
        best_mention_count = 0

        for it in scored:
            for c in candidates:
                val = it["aspect_scores"].get(c)
                mentions = it["aspect_mentions"].get(c, 0)
                if val is not None and mentions > 0 and val > best_score:
                    best_score = val
                    best_item = it
                    best_matched_aspect = c
                    best_mention_count = mentions

        if best_item is None:
            return None, None, None, 0
        return best_item, best_score, best_matched_aspect, best_mention_count

    # 3. Best Camera / Image Quality
    camera_candidates = ["Camera", "Image quality", "Camera & Photography", "Lens quality"]
    best_camera, cam_score, cam_aspect, cam_mentions = pick_best_by_candidates(camera_candidates)

    # 4. Best Battery / Endurance
    battery_candidates = ["Battery", "Battery & Charging", "Cooling"]
    best_battery, batt_score, batt_aspect, batt_mentions = pick_best_by_candidates(battery_candidates)

    # 5. Best Performance / Core Experience
    perf_candidates = ["Performance", "Processor", "Sound quality", "Performance & Speed", "GPU"]
    best_perf, perf_score, perf_aspect, perf_mentions = pick_best_by_candidates(perf_candidates)

    def build_aspect_award(item, score, aspect_name, mentions, category_label):
        if item is None or score is None:
            return {
                "product": None,
                "score": None,
                "review_count": 0,
                "reason": f"Insufficient review evidence: No products in this selection contain verified {category_label} review mentions."
            }
        sign = "+" if score > 0 else ""
        return {
            "product": item["product"],
            "score": score,
            "review_count": item["total_reviews"],
            "mention_count": mentions,
            "reason": f"Highest consumer satisfaction for {aspect_name} ({sign}{score:.1f} NSS across {mentions} review mentions, {item['total_reviews']} total reviews analyzed)."
        }

    return {
        "methodology_disclosure": RECOMMENDATION_METHODOLOGY_DISCLOSURE,
        "best_overall": {
            "product": best_overall["product"],
            "score": best_overall["composite_score"],
            "review_count": best_overall["total_reviews"],
            "reason": f"Top composite score ({best_overall['composite_score']}/100) combining star rating, NLP Net Sentiment (+{best_overall['summary'].get('nss', 0):.1f}), and CSAT ({best_overall['summary'].get('csat_estimate', 50):.1f}%) based on {best_overall['total_reviews']} prepared demo reviews."
        },
        "best_budget": {
            "product": best_budget["product"],
            "score": best_budget["composite_score"],
            "review_count": best_budget["total_reviews"],
            "reason": f"Exceptional feature-to-price ratio at ₹{best_budget['price']:,.0f} with strong customer satisfaction (+{best_budget['summary'].get('nss', 0):.1f} NSS) based on {best_budget['total_reviews']} prepared demo reviews."
        },
        "best_camera": build_aspect_award(best_camera, cam_score, cam_aspect, cam_mentions, "Camera/Optics"),
        "best_battery": build_aspect_award(best_battery, batt_score, batt_aspect, batt_mentions, "Battery/Endurance"),
        "best_performance": build_aspect_award(best_perf, perf_score, perf_aspect, perf_mentions, "Performance/Speed")
    }
