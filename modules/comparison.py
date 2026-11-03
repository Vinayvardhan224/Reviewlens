"""
ReviewLens Multi-Product Comparison Engine.
Compares specifications, customer sentiment scores, and aspect deltas
across 2 or 3 products with transparent head-to-head scoring and winner rationales.
"""

from typing import Dict, Any, List, Optional
import json


def compare_two_products(prod_a: Dict[str, Any], reviews_a_summary: Dict[str, Any],
                         prod_b: Dict[str, Any], reviews_b_summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform deep side-by-side comparison between Product A and Product B.
    """
    name_a = prod_a.get("product_name", "Product A")
    name_b = prod_b.get("product_name", "Product B")

    nss_a = reviews_a_summary.get("nss", 0.0)
    nss_b = reviews_b_summary.get("nss", 0.0)
    csat_a = reviews_a_summary.get("csat_estimate", 0.0)
    csat_b = reviews_b_summary.get("csat_estimate", 0.0)

    # 1. Spec comparison
    specs_a = prod_a.get("parsed_specs", {})
    specs_b = prod_b.get("parsed_specs", {})
    all_spec_keys = list(dict.fromkeys(list(specs_a.keys()) + list(specs_b.keys())))

    spec_comparison = []
    for k in all_spec_keys:
        val_a = str(specs_a.get(k, "N/A"))
        val_b = str(specs_b.get(k, "N/A"))
        spec_comparison.append({
            "dimension": k,
            "product_a": val_a,
            "product_b": val_b,
            "is_different": val_a.strip().lower() != val_b.strip().lower()
        })

    # 2. Aspect sentiment comparison
    aspects_a = {a["category"]: a for a in reviews_a_summary.get("aspects", [])}
    aspects_b = {b["category"]: b for b in reviews_b_summary.get("aspects", [])}
    all_aspect_keys = list(dict.fromkeys(list(aspects_a.keys()) + list(aspects_b.keys())))

    aspect_deltas = []
    wins_a = 0
    wins_b = 0
    ties = 0

    for cat in all_aspect_keys:
        item_a = aspects_a.get(cat, {})
        item_b = aspects_b.get(cat, {})
        nss_val_a = item_a.get("net_sentiment_score", 0.0)
        nss_val_b = item_b.get("net_sentiment_score", 0.0)
        mentions_a = item_a.get("mentions", 0)
        mentions_b = item_b.get("mentions", 0)

        delta = round(nss_val_a - nss_val_b, 1)

        if abs(delta) < 2.0:
            winner = "TIE"
            ties += 1
        elif delta > 0:
            winner = name_a
            wins_a += 1
        else:
            winner = name_b
            wins_b += 1

        aspect_deltas.append({
            "category": cat,
            "nss_a": nss_val_a,
            "nss_b": nss_val_b,
            "mentions_a": mentions_a,
            "mentions_b": mentions_b,
            "delta": delta,
            "winner": winner
        })

    # Sort aspects by total mentions
    aspect_deltas.sort(key=lambda x: (x["mentions_a"] + x["mentions_b"]), reverse=True)

    # 3. Overall Verdict determination
    price_a = float(prod_a.get("price", 0.0))
    price_b = float(prod_b.get("price", 0.0))

    if nss_a > nss_b + 5.0:
        overall_winner = name_a
        verdict_reason = f"{name_a} achieves significantly higher overall customer satisfaction (+{nss_a - nss_b:.1f} NSS) and leads in {wins_a} aspect categories."
    elif nss_b > nss_a + 5.0:
        overall_winner = name_b
        verdict_reason = f"{name_b} achieves significantly higher overall customer satisfaction (+{nss_b - nss_a:.1f} NSS) and leads in {wins_b} aspect categories."
    else:
        # Near tie on sentiment, factor in value
        if price_a < price_b * 0.8:
            overall_winner = name_a
            verdict_reason = f"Sentiment is neck-and-neck, but {name_a} offers drastically superior value for money at ₹{price_a:,.0f} vs ₹{price_b:,.0f}."
        elif price_b < price_a * 0.8:
            overall_winner = name_b
            verdict_reason = f"Sentiment is neck-and-neck, but {name_b} offers drastically superior value for money at ₹{price_b:,.0f} vs ₹{price_a:,.0f}."
        else:
            overall_winner = "Close Draw"
            verdict_reason = f"Both devices perform comparably across consumer satisfaction metrics ({nss_a:.1f} vs {nss_b:.1f} NSS)."

    return {
        "product_a": name_a,
        "product_b": name_b,
        "overall_winner": overall_winner,
        "verdict_reason": verdict_reason,
        "kpi_summary": {
            "nss_a": nss_a,
            "nss_b": nss_b,
            "csat_a": csat_a,
            "csat_b": csat_b,
            "wins_a": wins_a,
            "wins_b": wins_b,
            "ties": ties
        },
        "spec_comparison": spec_comparison,
        "aspect_deltas": aspect_deltas
    }
