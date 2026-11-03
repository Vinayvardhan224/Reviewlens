"""
ReviewLens Multi-Product Comparison Engine.
Compares specifications, customer sentiment scores, and category-specific aspect deltas
across 2 or 3 products. Explicitly reports 'Insufficient review data' when an aspect lacks evidence.
"""

from typing import Dict, Any, List, Optional
import json


def compare_two_products(prod_a: Dict[str, Any], reviews_a_summary: Dict[str, Any],
                         prod_b: Dict[str, Any], reviews_b_summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform deep side-by-side comparison between Product A and Product B.
    Handles missing aspect evidence transparently without assigning false neutral scores.
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

        mentions_a = item_a.get("mentions", 0)
        mentions_b = item_b.get("mentions", 0)
        has_ev_a = item_a.get("has_evidence", mentions_a > 0)
        has_ev_b = item_b.get("has_evidence", mentions_b > 0)

        nss_val_a = item_a.get("net_sentiment_score")
        nss_val_b = item_b.get("net_sentiment_score")

        if has_ev_a and has_ev_b and nss_val_a is not None and nss_val_b is not None:
            delta = round(nss_val_a - nss_val_b, 1)
            delta_display = f"{abs(delta):.1f} pts"
            if abs(delta) < 2.0:
                winner = "Tie"
                ties += 1
            elif delta > 0:
                winner = name_a
                wins_a += 1
            else:
                winner = name_b
                wins_b += 1
        elif has_ev_a and not has_ev_b:
            delta = None
            delta_display = "N/A"
            winner = f"Insufficient data for {name_b[:12]}"
        elif has_ev_b and not has_ev_a:
            delta = None
            delta_display = "N/A"
            winner = f"Insufficient data for {name_a[:12]}"
        else:
            delta = None
            delta_display = "N/A"
            winner = "Insufficient review data"

        score_a_display = f"{nss_val_a:+.1f} ({mentions_a} mentions)" if has_ev_a and nss_val_a is not None else "Insufficient review data"
        score_b_display = f"{nss_val_b:+.1f} ({mentions_b} mentions)" if has_ev_b and nss_val_b is not None else "Insufficient review data"

        aspect_deltas.append({
            "category": cat,
            "has_evidence_a": has_ev_a,
            "has_evidence_b": has_ev_b,
            "nss_a": nss_val_a,
            "nss_b": nss_val_b,
            "score_a_display": score_a_display,
            "score_b_display": score_b_display,
            "mentions_a": mentions_a,
            "mentions_b": mentions_b,
            "delta": delta,
            "delta_display": delta_display,
            "winner": winner
        })

    # Sort aspects by total mentions
    aspect_deltas.sort(key=lambda x: (x["mentions_a"] + x["mentions_b"]), reverse=True)

    # 3. Overall Verdict determination
    price_a = float(prod_a.get("price", 0.0))
    price_b = float(prod_b.get("price", 0.0))

    if nss_a > nss_b + 5.0:
        overall_winner = name_a
        verdict_reason = f"{name_a} achieves higher overall customer satisfaction (+{nss_a - nss_b:.1f} NSS) and leads in {wins_a} aspect categories."
    elif nss_b > nss_a + 5.0:
        overall_winner = name_b
        verdict_reason = f"{name_b} achieves higher overall customer satisfaction (+{nss_b - nss_a:.1f} NSS) and leads in {wins_b} aspect categories."
    else:
        # Near tie on sentiment, factor in value / price
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
