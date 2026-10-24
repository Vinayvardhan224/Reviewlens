"""
Executive Insights and SWOT Generation Module for ReviewLens.
Translates statistical distributions into high-level business intelligence.
"""

from typing import List, Dict, Any
from reviewlens.models import ReviewAnalysis, AspectSummary, ExecutiveSummary
from reviewlens.insights.recommender import ActionRecommender


class ExecutiveInsightsEngine:
    """Computes executive-level KPIs, SWOT analysis, and strategic summaries."""

    def __init__(self):
        self.recommender = ActionRecommender()

    def generate(
        self,
        reviews: List[ReviewAnalysis],
        aspect_summaries: List[AspectSummary]
    ) -> ExecutiveSummary:
        """
        Synthesize comprehensive executive overview from analyzed reviews.
        """
        valid_reviews = [r for r in reviews if not r.spam.is_spam]
        total_valid = len(valid_reviews) or 1

        # Count sentiment labels
        pos_count = sum(1 for r in valid_reviews if r.sentiment.label == "POSITIVE")
        neg_count = sum(1 for r in valid_reviews if r.sentiment.label == "NEGATIVE")
        neu_count = sum(1 for r in valid_reviews if r.sentiment.label == "NEUTRAL")

        sentiment_dist = {
            "POSITIVE": pos_count,
            "NEGATIVE": neg_count,
            "NEUTRAL": neu_count,
        }

        sentiment_pct = {
            "positive_pct": round((pos_count / total_valid) * 100, 1),
            "negative_pct": round((neg_count / total_valid) * 100, 1),
            "neutral_pct": round((neu_count / total_valid) * 100, 1),
        }

        # Net Sentiment Score (NSS) = % Positive - % Negative, range [-100, 100]
        nss = round(sentiment_pct["positive_pct"] - sentiment_pct["negative_pct"], 1)

        # Customer Satisfaction (CSAT) Estimate: Positive / (Positive + Negative)
        opinionated_total = pos_count + neg_count
        csat = round((pos_count / opinionated_total) * 100, 1) if opinionated_total > 0 else 50.0

        # Automated SWOT Analysis
        strengths = []
        weaknesses = []
        opportunities = []
        threats = []

        for aspect in aspect_summaries:
            pos_ratio = aspect.sentiment_ratio.get("positive", 0.0)
            neg_ratio = aspect.sentiment_ratio.get("negative", 0.0)

            # Strengths: High positive ratio & positive NSS
            if pos_ratio >= 60.0 and aspect.net_sentiment_score > 30.0:
                strengths.append(f"{aspect.category}: {pos_ratio:.0f}% positive feedback (NSS: +{aspect.net_sentiment_score:.0f})")

            # Weaknesses: Noticeable negative ratio
            if neg_ratio >= 25.0 or aspect.net_sentiment_score < 0:
                weaknesses.append(f"{aspect.category}: {neg_ratio:.0f}% negative feedback (NSS: {aspect.net_sentiment_score:.0f})")

            # Opportunities: High mention volume with room for improvement
            if aspect.total_mentions >= 3 and 10.0 <= aspect.net_sentiment_score <= 45.0:
                opportunities.append(f"{aspect.category}: High customer engagement with moderate satisfaction; optimization will turn buyers into brand advocates.")

            # Threats: Severe negative backlash or structural complaints
            if neg_ratio >= 45.0 or aspect.net_sentiment_score <= -30.0:
                sample = f" ('{aspect.sample_negative_quotes[0]}')" if aspect.sample_negative_quotes else ""
                threats.append(f"{aspect.category}: High churn risk due to severe dissatisfaction{sample}")

        # Spam threats check
        spam_count = sum(1 for r in reviews if r.spam.is_spam)
        if spam_count > 0:
            threats.append(f"Review Integrity Alert: {spam_count} reviews flagged as suspicious or potentially fake.")

        if not strengths:
            strengths.append("General reliability across baseline product functions.")
        if not weaknesses:
            weaknesses.append("No critical product deficiencies identified above threshold.")
        if not opportunities:
            opportunities.append("Expand companion feature offerings and highlight top-rated features in marketing.")
        if not threats:
            threats.append("No critical churn or hardware failure threats detected.")

        # Actionable recommendations
        recommendations = self.recommender.generate_recommendations(aspect_summaries)

        return ExecutiveSummary(
            net_sentiment_score=nss,
            csat_score=csat,
            sentiment_distribution=sentiment_dist,
            sentiment_percentages=sentiment_pct,
            strengths=strengths,
            weaknesses=weaknesses,
            opportunities=opportunities,
            threats=threats,
            actionable_recommendations=recommendations,
        )
