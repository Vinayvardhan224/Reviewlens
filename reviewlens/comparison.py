"""
Competitor Benchmarking and Side-by-Side Product Comparison Engine for ReviewLens.
Compares two products on aspect Net Sentiment Scores (NSS), CSAT, and detects winners.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from reviewlens.models import BatchAnalysisResult, AspectSummary


class AspectComparison(BaseModel):
    category: str
    score_a: float
    score_b: float
    winner: str
    delta: float


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
        map_a = {a.category: a.net_sentiment_score for a in result_a.aspect_summaries}
        map_b = {b.category: b.net_sentiment_score for b in result_b.aspect_summaries}

        all_categories = sorted(list(set(map_a.keys()) | set(map_b.keys())))
        aspect_comparisons: List[AspectComparison] = []

        a_wins = 0
        b_wins = 0
        a_strengths = []
        b_strengths = []

        for cat in all_categories:
            score_a = map_a.get(cat, 0.0)
            score_b = map_b.get(cat, 0.0)
            diff = score_a - score_b

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

            aspect_comparisons.append(
                AspectComparison(
                    category=cat,
                    score_a=score_a,
                    score_b=score_b,
                    winner=winner,
                    delta=round(diff, 1)
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
            verdict = f"{name_a} outperforms {name_b} overall with higher Net Sentiment ({nss_a:+.1f} vs {nss_b:+.1f}) and leads in {len(a_strengths)} key dimensions."
        elif overall_winner == name_b:
            verdict = f"{name_b} claims the overall victory with superior customer satisfaction ({csat_b:.1f}% vs {csat_a:.1f}%) and leads in {len(b_strengths)} key dimensions."
        else:
            verdict = f"{name_a} and {name_b} offer balanced trade-offs. Choose based on which specific aspect matters most to your use case."

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
            executive_verdict=verdict
        )
