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


def rank_products_by_preferences(
    products_with_summaries: List[Dict[str, Any]],
    weights: Dict[str, float],
    max_budget: Optional[float] = None,
    category: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Ranks products according to customizable user priorities (Matchmaker Engine).
    Weights dictionary may contain normalized values for:
      - 'battery': float (0.0 to 1.0)
      - 'camera': float (0.0 to 1.0)
      - 'performance': float (0.0 to 1.0)
      - 'display': float (0.0 to 1.0)
      - 'audio': float (0.0 to 1.0)
      - 'value_for_money': float (0.0 to 1.0)
      - 'star_rating': float (0.0 to 1.0)
    Returns ranked list of candidate matches with match_score (0-100%) and personalized rationale.
    """
    items = products_with_summaries
    if category and category.lower() != "all":
        items = [i for i in items if i["product"].get("category", "").lower() == category.lower()]

    if not items:
        return []

    # Map weight keys to aspect category name variants
    aspect_map = {
        "battery": ["Battery", "Battery & Charging", "Cooling"],
        "camera": ["Camera", "Image quality", "Camera & Photography", "Lens quality"],
        "performance": ["Performance", "Processor", "Performance & Speed", "GPU", "RAM"],
        "display": ["Display", "Screen", "Display & Screen"],
        "audio": ["Audio & Sound", "Sound quality", "Audio"],
    }

    # Normalize weights so they sum to 1.0 (if total > 0)
    active_weights = {k: max(0.0, float(v)) for k, v in weights.items() if v > 0}
    total_w = sum(active_weights.values())
    if total_w == 0:
        active_weights = {"star_rating": 0.4, "performance": 0.3, "battery": 0.3}
        total_w = 1.0
    norm_weights = {k: v / total_w for k, v in active_weights.items()}

    # Determine maximum price reference for value-for-money normalization
    all_prices = [float(i["product"].get("price", 0.0)) for i in items]
    ref_max_price = max(all_prices) if all_prices else 100000.0
    if max_budget and max_budget > 0:
        ref_max_price = max(ref_max_price, max_budget)

    ranked_results = []

    for it in items:
        prod = it["product"]
        summary = it["summary"]
        price = float(prod.get("price", 0.0))
        rating = float(prod.get("rating", 4.0))

        # Hard budget filter if user specified a ceiling
        if max_budget and max_budget > 0 and price > max_budget:
            continue

        aspects_list = summary.get("aspects", [])
        aspect_lookup = {a["category"]: a for a in aspects_list}

        component_scores = {}
        highlights = []
        trade_offs = []

        # 1. Star Rating Component (0 - 100)
        component_scores["star_rating"] = min(100.0, (rating / 5.0) * 100.0)

        # 2. Value for money score (lower price relative to budget = higher score)
        val_score = max(10.0, 100.0 - (price / ref_max_price) * 85.0)
        component_scores["value_for_money"] = val_score

        # 3. Aspect Scores
        for key, candidate_cats in aspect_map.items():
            matched_nss = None
            for c_name in candidate_cats:
                if c_name in aspect_lookup and aspect_lookup[c_name].get("has_evidence"):
                    matched_nss = aspect_lookup[c_name].get("net_sentiment_score")
                    break

            if matched_nss is not None:
                # Normalize [-100, +100] to [0, 100]
                norm_score = max(0.0, min(100.0, (matched_nss + 100.0) / 2.0))
                component_scores[key] = norm_score

                if norm_weights.get(key, 0) >= 0.15:
                    if matched_nss >= 40:
                        highlights.append(f"Exceptional {key.title()} (NSS: {matched_nss:+.0f})")
                    elif matched_nss <= -10:
                        trade_offs.append(f"Mixed feedback on {key.title()} (NSS: {matched_nss:+.0f})")
            else:
                # Neutral baseline if unmentioned in reviews
                component_scores[key] = 50.0

        # Compute weighted match score
        total_match = sum(component_scores.get(k, 50.0) * w for k, w in norm_weights.items())
        total_match = max(10.0, min(99.0, round(total_match, 1)))

        # Build personalized rationale
        if highlights:
            rationale = f"Strong match for your priorities with {', '.join(highlights[:2])}."
        elif price <= ref_max_price * 0.7:
            rationale = f"Balanced everyday performer offering excellent value at ₹{price:,.0f}."
        else:
            rationale = f"Well-rounded flagship experience with solid ★{rating:.1f} satisfaction."

        ranked_results.append({
            "product": prod,
            "summary": summary,
            "match_score": total_match,
            "price": price,
            "rating": rating,
            "highlights": highlights,
            "trade_offs": trade_offs,
            "rationale": rationale,
            "component_scores": component_scores
        })

    # Sort descending by match score
    ranked_results.sort(key=lambda x: x["match_score"], reverse=True)
    return ranked_results
