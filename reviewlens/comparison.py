"""
Competitor Benchmarking and Side-by-Side Product Comparison Engine for ReviewLens.
Compares two products on aspect Net Sentiment Scores (NSS), CSAT, and detects winners.
Handles missing aspect evidence transparently without assigning false neutral zero scores.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from reviewlens.models import BatchAnalysisResult, AspectSummary


class AspectComparison(BaseModel):
    category: str
    score_a: Optional[float] = None
    score_b: Optional[float] = None
    score_a_display: str = "Insufficient review data"
    score_b_display: str = "Insufficient review data"
    has_evidence_a: bool = False
    has_evidence_b: bool = False
    winner: str
    delta: Optional[float] = None
    delta_display: str = "N/A"


class ProductComparisonReport(BaseModel):
    product_a_name: str
    product_b_name: str
    total_reviews_a: int
    total_reviews_b: int
    nss_a: float
    nss_b: float
    csat_a: float
    csat_b: float
    overall_winner: str
    aspect_comparisons: List[AspectComparison]
    a_strengths: List[str]
    b_strengths: List[str]
    executive_verdict: str
    summary: Optional[str] = None


class ProductComparator:
    """Computes aspect-level winner scorecards and comparative benchmarks."""

    @staticmethod
    def compare(
        result_a: BatchAnalysisResult,
        result_b: BatchAnalysisResult,
        name_a: str = "Product A",
        name_b: str = "Product B"
    ) -> ProductComparisonReport:
        nss_a = result_a.executive_summary.net_sentiment_score
        nss_b = result_b.executive_summary.net_sentiment_score

        csat_a = result_a.executive_summary.csat_score
        csat_b = result_b.executive_summary.csat_score

        # Map aspects by category
        map_a = {a.category: a for a in result_a.aspect_summaries}
        map_b = {b.category: b for b in result_b.aspect_summaries}

        all_categories = sorted(list(set(map_a.keys()) | set(map_b.keys())))
        aspect_comparisons: List[AspectComparison] = []

        a_wins = 0
        b_wins = 0
        a_strengths = []
        b_strengths = []

        for cat in all_categories:
            asp_a = map_a.get(cat)
            asp_b = map_b.get(cat)

            m_a = getattr(asp_a, "total_mentions", getattr(asp_a, "mentions", 0)) if asp_a is not None else 0
            m_b = getattr(asp_b, "total_mentions", getattr(asp_b, "mentions", 0)) if asp_b is not None else 0

            has_ev_a = asp_a is not None and m_a > 0
            has_ev_b = asp_b is not None and m_b > 0

            score_a = asp_a.net_sentiment_score if has_ev_a else None
            score_b = asp_b.net_sentiment_score if has_ev_b else None

            score_a_display = f"{score_a:+.1f} ({m_a} mentions)" if has_ev_a and score_a is not None else "Insufficient review data"
            score_b_display = f"{score_b:+.1f} ({m_b} mentions)" if has_ev_b and score_b is not None else "Insufficient review data"

            if has_ev_a and has_ev_b and score_a is not None and score_b is not None:
                diff = score_a - score_b
                delta = round(diff, 1)
                delta_display = f"{abs(delta):.1f} pts"

                if diff > 5.0:
                    winner = name_a
                    a_wins += 1
                    if diff >= 15.0:
                        a_strengths.append(f"{cat} (+{diff:.0f} pts over {name_b})")
                elif diff < -5.0:
                    winner = name_b
                    b_wins += 1
                    if diff <= -15.0:
                        b_strengths.append(f"{cat} (+{abs(diff):.0f} pts over {name_a})")
                else:
                    winner = "Tie"
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

            aspect_comparisons.append(
                AspectComparison(
                    category=cat,
                    score_a=score_a,
                    score_b=score_b,
                    score_a_display=score_a_display,
                    score_b_display=score_b_display,
                    has_evidence_a=has_ev_a,
                    has_evidence_b=has_ev_b,
                    winner=winner,
                    delta=delta,
                    delta_display=delta_display
                )
            )

        # Determine overall winner
        if nss_a > nss_b + 5.0 or a_wins > b_wins + 1:
            overall_winner = name_a
        elif nss_b > nss_a + 5.0 or b_wins > a_wins + 1:
            overall_winner = name_b
        else:
            overall_winner = "Draw / Competitive Tie"

        # Narrative verdict
        if overall_winner == name_a:
            verdict = f"{name_a} outperforms {name_b} with a Net Sentiment advantage of {nss_a - nss_b:+.1f} and {a_wins} aspect victories."
        elif overall_winner == name_b:
            verdict = f"{name_b} outperforms {name_a} with a Net Sentiment advantage of {nss_b - nss_a:+.1f} and {b_wins} aspect victories."
        else:
            verdict = f"{name_a} and {name_b} demonstrate comparable overall sentiment ({nss_a:+.1f} vs {nss_b:+.1f} NSS) across verified customer feedback."

        return ProductComparisonReport(
            product_a_name=name_a,
            product_b_name=name_b,
            total_reviews_a=result_a.total_reviews,
            total_reviews_b=result_b.total_reviews,
            nss_a=nss_a,
            nss_b=nss_b,
            csat_a=csat_a,
            csat_b=csat_b,
            overall_winner=overall_winner,
            aspect_comparisons=aspect_comparisons,
            a_strengths=a_strengths,
            b_strengths=b_strengths,
            executive_verdict=verdict,
            summary=verdict
        )
