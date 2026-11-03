"""
ReviewLens Review Summary & Synthesis Module.
Produces human-readable executive summaries, key pros/cons with authentic evidence,
and ideal buyer personas.
"""

from typing import Dict, Any, List


def generate_executive_summary(
    product_name: str,
    aspects: List[Dict[str, Any]],
    nss: float,
    csat: float,
    total_reviews: int
) -> Dict[str, Any]:
    """
    Synthesize high-level intelligence into bulleted strengths, weaknesses,
    and a recommended buyer profile.
    """
    pros = []
    cons = []

    # Filter active aspects
    active_aspects = [a for a in aspects if a.get("mentions", 0) > 0]

    # Find highest NSS aspects for Pros
    sorted_pos = sorted(active_aspects, key=lambda x: x.get("net_sentiment_score", 0), reverse=True)
    for a in sorted_pos:
        if a.get("net_sentiment_score", 0) > 10 and a.get("positive_quotes"):
            pros.append({
                "aspect": a["category"],
                "nss": a["net_sentiment_score"],
                "quote": a["positive_quotes"][0]
            })
            if len(pros) >= 3:
                break

    # Find lowest NSS aspects for Cons
    sorted_neg = sorted(active_aspects, key=lambda x: x.get("net_sentiment_score", 0))
    for a in sorted_neg:
        if (a.get("net_sentiment_score", 0) < 40 or a.get("negative_count", 0) > 0) and a.get("negative_quotes"):
            cons.append({
                "aspect": a["category"],
                "nss": a["net_sentiment_score"],
                "quote": a["negative_quotes"][0]
            })
            if len(cons) >= 3:
                break

    # Determine buyer persona
    if nss >= 50:
        verdict = f"{product_name} enjoys overwhelming customer acclaim (NSS: +{nss:.1f}) with exceptional marks across primary dimensions."
        persona = "Power users and enthusiasts seeking uncompromising premium quality and reliable day-to-day satisfaction."
    elif nss >= 20:
        verdict = f"{product_name} delivers a balanced customer experience (NSS: +{nss:.1f}) with standout advantages in specific core features."
        persona = "Smart shoppers seeking great value, practical features, and solid performance without paying ultra-flagship premiums."
    else:
        verdict = f"{product_name} exhibits polarized reviews (NSS: {nss:.1f}), with trade-offs that buyers should weigh carefully."
        persona = "Niche buyers whose specific workflows rely heavily on its unique strengths, while comfortable tolerating its documented limitations."

    return {
        "verdict": verdict,
        "persona": persona,
        "pros": pros,
        "cons": cons,
        "sample_disclosure": f"Synthesized from {total_reviews} verified customer reviews (NSS: {nss:+.1f}, CSAT: {csat:.1f}%)."
    }
