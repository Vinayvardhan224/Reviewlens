"""
ReviewLens Executive PDF Report Generator.
Builds high-impact, professional executive PDF evaluation dossiers for products
using ReportLab, including aspect sentiment tables, strengths, pain points,
spam audit summaries, and academic provenance disclosures.
"""

import io
import datetime
import html
from typing import Dict, Any, List, Optional

from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable
)


def generate_product_pdf_report(
    prod: Dict[str, Any],
    aspect_list: List[Dict[str, Any]],
    sentiment_res: Dict[str, Any],
    exec_summary: Dict[str, Any],
    spam_summary: Optional[Dict[str, Any]] = None
) -> bytes:
    """
    Generate an executive PDF review intelligence report for a single product.

    Args:
        prod: Product dictionary (product_name, brand, category, price, rating, specs, etc.)
        aspect_list: List of aspect sentiment results (category, net_sentiment_score, mentions)
        sentiment_res: Overall sentiment dictionary (csat_estimate, nss, total_reviews, etc.)
        exec_summary: Strengths and pain points dict with 'pros' and 'cons'
        spam_summary: Optional spam detection audit summary

    Returns:
        bytes: Raw PDF file bytes starting with b"%PDF"
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    # Palette
    c_primary = HexColor("#7138F5")
    c_dark = HexColor("#07051A")
    c_card_bg = HexColor("#16112E")
    c_text = HexColor("#1E293B")
    c_light_bg = HexColor("#F8FAFC")
    c_cyan = HexColor("#0EA5E9")
    c_green = HexColor("#10B981")
    c_red = HexColor("#EF4444")
    c_border = HexColor("#CBD5E1")

    base_styles = getSampleStyleSheet()

    # Custom typography styles
    style_title = ParagraphStyle(
        "DocTitle",
        parent=base_styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=HexColor("#4C1D95")
    )
    style_subtitle = ParagraphStyle(
        "DocSubtitle",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=HexColor("#64748B")
    )
    style_h2 = ParagraphStyle(
        "SectionHeading",
        parent=base_styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=c_primary,
        spaceBefore=12,
        spaceAfter=6
    )
    style_body = ParagraphStyle(
        "BodyDark",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=c_text
    )
    style_cell = ParagraphStyle(
        "TableCell",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=c_text
    )
    style_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=style_cell,
        fontName="Helvetica-Bold"
    )
    style_quote_pro = ParagraphStyle(
        "QuotePro",
        parent=base_styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=12,
        textColor=HexColor("#065F46")
    )
    style_quote_con = ParagraphStyle(
        "QuoteCon",
        parent=base_styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=12,
        textColor=HexColor("#991B1B")
    )
    style_footer = ParagraphStyle(
        "FooterDisclosure",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=10,
        textColor=HexColor("#94A3B8")
    )

    story = []

    # 1. Header Banner
    prod_name = prod.get("product_name", "Product Review Report")
    brand = prod.get("brand", "N/A")
    category = prod.get("category", "Electronics")
    price_val = prod.get("price", 0.0)
    rating_val = prod.get("rating", 0.0)
    now_str = datetime.datetime.now().strftime("%B %d, %Y - %H:%M UTC")

    story.append(Paragraph("ReviewLens Executive Review Dossier", style_title))
    story.append(Paragraph(f"Autonomous NLP Sentiment, Aspect Extraction & Spam Integrity Audit | Generated {now_str}", style_subtitle))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_primary, spaceBefore=2, spaceAfter=12))

    # 2. Executive Product Overview
    csat = sentiment_res.get("csat_estimate", 0)
    nss = sentiment_res.get("nss", 0.0)
    total_revs = sentiment_res.get("total_reviews", 0)

    summary_data = [
        [
            Paragraph("<b>Product Name</b>", style_cell_bold),
            Paragraph(prod_name, style_cell),
            Paragraph("<b>Category / Brand</b>", style_cell_bold),
            Paragraph(f"{category} ({brand})", style_cell)
        ],
        [
            Paragraph("<b>Retail Price</b>", style_cell_bold),
            Paragraph(f"INR {price_val:,.2f}" if isinstance(price_val, (int, float)) else str(price_val), style_cell),
            Paragraph("<b>Catalogue Rating</b>", style_cell_bold),
            Paragraph(f"{rating_val:.1f} / 5.0 Stars", style_cell)
        ],
        [
            Paragraph("<b>Analyzed Reviews</b>", style_cell_bold),
            Paragraph(f"{total_revs} Customer Reviews", style_cell),
            Paragraph("<b>Customer Satisfaction (CSAT)</b>", style_cell_bold),
            Paragraph(f"{csat}% Satisfaction Index", style_cell)
        ],
        [
            Paragraph("<b>Net Sentiment Score (NSS)</b>", style_cell_bold),
            Paragraph(f"{nss:+.1f} (-100 to +100)", style_cell),
            Paragraph("<b>Data Provenance</b>", style_cell_bold),
            Paragraph("Curated demo reviews for transparent viva analysis", style_cell)
        ]
    ]

    t_summary = Table(summary_data, colWidths=[120, 150, 130, 130])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), c_light_bg),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
        ('PADDING', (0, 0), (-1, -1), 5),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
    ]))
    story.append(t_summary)
    story.append(Spacer(1, 14))

    # 3. Category Aspect Sentiment Scorecard
    story.append(Paragraph("Category Aspect Sentiment Scorecard", style_h2))
    story.append(Paragraph("Multi-aspect sentiment classification decomposes user opinions into specific hardware and functional components.", style_body))
    story.append(Spacer(1, 6))

    aspect_table_data = [
        [
            Paragraph("<b>Aspect Dimension</b>", style_cell_bold),
            Paragraph("<b>Mentions</b>", style_cell_bold),
            Paragraph("<b>NSS Score</b>", style_cell_bold),
            Paragraph("<b>Consensus Rating</b>", style_cell_bold)
        ]
    ]

    for asp in aspect_list:
        if asp.get("has_evidence", True):
            score = asp.get("net_sentiment_score", 0.0)
            mentions = asp.get("mentions", 0)
            if score >= 30:
                rating_badge = "Excellent / Highly Praised"
            elif score > 0:
                rating_badge = "Favorable / Positive"
            elif score == 0:
                rating_badge = "Neutral / Balanced"
            else:
                rating_badge = "Critical Defect / Pain Point"
            score_str = f"{score:+.1f}"
        else:
            score_str = "N/A"
            mentions = 0
            rating_badge = "Insufficient Evidence"

        aspect_table_data.append([
            Paragraph(str(asp.get("category", "")), style_cell),
            Paragraph(f"{mentions} mentions", style_cell),
            Paragraph(score_str, style_cell_bold),
            Paragraph(rating_badge, style_cell)
        ])

    t_aspect = Table(aspect_table_data, colWidths=[160, 90, 90, 190])
    t_aspect.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor("#EDE9FE")),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
        ('PADDING', (0, 0), (-1, -1), 4),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor("#FFFFFF"), c_light_bg]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
    ]))
    story.append(t_aspect)
    story.append(Spacer(1, 14))

    # 4. Feedback Strengths & Critical Pain Points
    story.append(Paragraph("Key Strengths & Critical Customer Trade-offs", style_h2))

    pros = exec_summary.get("pros", [])
    cons = exec_summary.get("cons", [])

    feedback_data = [
        [
            Paragraph("<b>Top Strengths (High NSS)</b>", style_cell_bold),
            Paragraph("<b>Trade-offs & Pain Points (Low NSS)</b>", style_cell_bold)
        ]
    ]

    max_len = max(len(pros), len(cons), 1)
    for i in range(max_len):
        pro_text = ""
        if i < len(pros):
            p = pros[i]
            pro_text = f"<b>{html.escape(str(p['aspect']))} (+{p['nss']:.0f} NSS):</b><br/><i>\"{html.escape(str(p['quote']))}\"</i>"
        else:
            pro_text = "<i>No additional major strengths flagged.</i>"

        con_text = ""
        if i < len(cons):
            c = cons[i]
            con_text = f"<b>{html.escape(str(c['aspect']))} ({c['nss']:+.0f} NSS):</b><br/><i>\"{html.escape(str(c['quote']))}\"</i>"
        else:
            con_text = "<i>No dominant customer pain points flagged.</i>"

        feedback_data.append([
            Paragraph(pro_text, style_quote_pro),
            Paragraph(con_text, style_quote_con)
        ])

    t_feedback = Table(feedback_data, colWidths=[265, 265])
    t_feedback.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, 0), HexColor("#D1FAE5")),
        ('BACKGROUND', (1, 0), (1, 0), HexColor("#FEE2E2")),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'TOP')
    ]))
    story.append(t_feedback)
    story.append(Spacer(1, 14))

    # 5. Review Integrity & Spam Audit
    if spam_summary:
        story.append(Paragraph("Review Integrity & Anti-Astroturfing Audit", style_h2))
        spam_flagged = spam_summary.get("flagged_count", 0)
        spam_total = spam_summary.get("total_checked", total_revs)
        spam_pct = (spam_flagged / max(1, spam_total)) * 100

        burst_status = spam_summary.get("burst_status", "Normal review velocity (No sudden spikes)")
        duplicate_status = spam_summary.get("duplicate_status", "Zero identical/near-duplicate review clusters")

        spam_data = [
            [
                Paragraph("<b>Audit Dimension</b>", style_cell_bold),
                Paragraph("<b>Observation & Diagnostic Finding</b>", style_cell_bold)
            ],
            [
                Paragraph("Suspicious / Spam Rate", style_cell),
                Paragraph(f"{spam_flagged} of {spam_total} reviews flagged ({spam_pct:.1f}% anomaly density)", style_cell)
            ],
            [
                Paragraph("Temporal Velocity Spikes", style_cell),
                Paragraph(burst_status, style_cell)
            ],
            [
                Paragraph("Content Plagiarism / Duplicates", style_cell),
                Paragraph(duplicate_status, style_cell)
            ]
        ]
        t_spam = Table(spam_data, colWidths=[180, 350])
        t_spam.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor("#FEF3C7")),
            ('GRID', (0, 0), (-1, -1), 0.5, c_border),
            ('PADDING', (0, 0), (-1, -1), 5),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
        ]))
        story.append(t_spam)
        story.append(Spacer(1, 14))

    # 6. Academic Methodology & Provenance Disclosure
    story.append(HRFlowable(width="100%", thickness=0.8, color=c_border, spaceBefore=8, spaceAfter=8))
    disclosure_text = (
        "<b>Academic & Regulatory Disclosure:</b> This report was compiled by ReviewLens NLP Engine. "
        "Sentiment polarity is computed using rule-based VADER Lexicon with valence shifters and validated against "
        "supervised scikit-learn models. Aspect extraction uses category taxonomies with keyword proximity windowing. "
        "Spam evaluation adheres to the Ott et al. (ACL 2011) opinion fraud methodology. "
        "Catalogue pricing and review subsets are illustrative demonstration samples for academic viva evaluation."
    )
    story.append(Paragraph(disclosure_text, style_footer))

    doc.build(story)
    return buffer.getvalue()
