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
        verdict = f"{product_name} exhibits strong positive review sentiment (NSS: +{nss:.1f}) across primary evaluated dimensions."
        persona = "Buyers seeking high performance and reliable features across key product dimensions."
    elif nss >= 20:
        verdict = f"{product_name} delivers balanced review sentiment (NSS: +{nss:.1f}) with standout advantages in specific core features."
        persona = "Value-conscious shoppers seeking practical features and solid everyday performance."
    else:
        verdict = f"{product_name} exhibits mixed review sentiment (NSS: {nss:.1f}), with trade-offs that buyers should weigh carefully."
        persona = "Buyers whose specific use-cases align with its strengths, while accepting its noted trade-offs."

    return {
        "verdict": verdict,
        "persona": persona,
        "pros": pros,
        "cons": cons,
        "sample_disclosure": f"Synthesized from {total_reviews} prepared demo reviews (NSS: {nss:+.1f}, CSAT: {csat:.1f}%)."
    }


def generate_markdown_dossier(
    prod: Dict[str, Any],
    aspect_list: List[Dict[str, Any]],
    sentiment_res: Dict[str, Any],
    exec_summary: Dict[str, Any]
) -> str:
    """
    Generates a structured, publication-ready ReviewLens Intelligence Dossier in Markdown format.
    Suitable for export, reporting, or archival.
    """
    # Gracefully handle either argument order
    if ("verdict" in sentiment_res or "persona" in sentiment_res or "pros" in sentiment_res) and ("nss" in exec_summary or "csat_estimate" in exec_summary):
        exec_summary, sentiment_res = sentiment_res, exec_summary

    p_name = prod.get("product_name", "Product")
    p_brand = prod.get("brand", "Unknown")
    p_cat = prod.get("category", "Electronics")
    price = prod.get("price", 0.0)
    rating = prod.get("rating", 4.0)

    nss = sentiment_res.get("nss", 0.0)
    csat = sentiment_res.get("csat_estimate", 0.0)
    total_revs = sentiment_res.get("total_reviews", 0)

    verdict_text = exec_summary.get("verdict") or exec_summary.get("executive_verdict", "N/A")
    persona_text = exec_summary.get("persona") or exec_summary.get("ideal_persona", "N/A")

    md = [
        f"# ReviewLens Intelligence Dossier: {p_name}",
        f"**Category:** {p_cat} | **Brand:** {p_brand} | **Price:** ₹{price:,.2f} | **Catalogue Rating:** ★ {rating:.1f}/5.0",
        "",
        "---",
        "",
        "## 1. Executive Summary & Buyer Persona",
        f"- **Verdict:** {verdict_text}",
        f"- **Ideal Buyer Persona:** {persona_text}",
        f"- **Aggregate Sentiment:** Net Sentiment Score (NSS): `{nss:+.1f}` | CSAT: `{csat:.1f}%` ({total_revs} analyzed reviews)",
        "",
        "## 2. Aspect-Based Sentiment Analysis (ABSA) Scorecard",
        "| Aspect Dimension | Mentions | Net Sentiment Score (NSS) | Status |",
        "| :--- | :---: | :---: | :--- |"
    ]

    for a in aspect_list:
        m = a.get("mentions", 0)
        if a.get("has_evidence") and a.get("net_sentiment_score") is not None:
            score = f"{a['net_sentiment_score']:+.1f}"
            status = "Positive Driver 🟢" if a["net_sentiment_score"] > 15 else ("Critical Trade-off 🔴" if a["net_sentiment_score"] < -10 else "Neutral / Mixed 🟡")
        else:
            score = "N/A"
            status = "Insufficient Review Data ⚪"
        md.append(f"| {a['category']} | {m} | {score} | {status} |")

    md.extend([
        "",
        "## 3. Customer Feedback Strengths & Key Pain Points",
        "### 🟢 Standout Strengths"
    ])
    if exec_summary.get("pros"):
        for p in exec_summary["pros"]:
            md.append(f"- **{p['aspect']}** (+{p['nss']:.0f} NSS): *\"{p['quote']}\"*")
    else:
        md.append("- *No dominant positive drivers flagged in this sample.*")

    md.extend([
        "",
        "### 🔴 Customer Pain Points & Trade-offs"
    ])
    if exec_summary.get("cons"):
        for c in exec_summary["cons"]:
            md.append(f"- **{c['aspect']}** ({c['nss']:+.0f} NSS): *\"{c['quote']}\"*")
    else:
        md.append("- *No dominant customer pain points flagged in this sample.*")

    md.extend([
        "",
        "---",
        "",
        "## 4. Methodology & Transparency Disclosures",
        "- **ABSA Taxonomy:** Category-specific rule-based aspect keyword classification with contrastive clause splitting.",
        "- **Sentiment Engine:** VADER Lexicon adapted with consumer electronics domain valence modifiers.",
        "- **Data Provenance:** Prepared demonstration reviews (`prepared_demo`). Illustrative retail specifications and pricing.",
        f"- **Generated by:** ReviewLens Smart Shopping Platform (Conversation ID: ReviewLens-NLP-Engine)"
    ])

    return "\n".join(md)
