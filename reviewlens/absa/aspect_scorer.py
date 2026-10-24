"""
Aspect sentiment aggregator and scoring engine for ReviewLens.
Aggregates fine-grained mentions into aspect-level performance metrics, Net Sentiment Score, and quote highlights.
"""

from typing import List, Dict
from collections import defaultdict
from reviewlens.models import AspectMention, AspectSummary


class AspectScorer:
    """Aggregates aspect-level data into statistical summaries and business KPIs."""

    @staticmethod
    def aggregate(aspect_mentions: List[AspectMention]) -> List[AspectSummary]:
        """
        Aggregate a flat list of AspectMentions across multiple reviews.
        
        Args:
            aspect_mentions: All AspectMention objects extracted from the dataset.
            
        Returns:
            List of AspectSummary sorted by mention count descending.
        """
        grouped = defaultdict(list)
        for m in aspect_mentions:
            grouped[m.category].append(m)

        summaries: List[AspectSummary] = []

        for category, mentions in grouped.items():
            total = len(mentions)
            pos_mentions = [m for m in mentions if m.sentiment_label == "POSITIVE"]
            neg_mentions = [m for m in mentions if m.sentiment_label == "NEGATIVE"]
            neu_mentions = [m for m in mentions if m.sentiment_label == "NEUTRAL"]

            pos_count = len(pos_mentions)
            neg_count = len(neg_mentions)
            neu_count = len(neu_mentions)

            # Net Sentiment Score: % Positive - % Negative, range [-100, 100]
            nss = round(((pos_count - neg_count) / total) * 100, 1) if total > 0 else 0.0

            # Average compound polarity
            avg_compound = round(sum(m.sentiment_score for m in mentions) / total, 3) if total > 0 else 0.0

            # Collect representative high-impact quotes
            sorted_pos = sorted(pos_mentions, key=lambda x: x.sentiment_score, reverse=True)
            sorted_neg = sorted(neg_mentions, key=lambda x: x.sentiment_score)

            pos_quotes = [m.sentence for m in sorted_pos[:3]]
            neg_quotes = [m.sentence for m in sorted_neg[:3]]

            summaries.append(
                AspectSummary(
                    category=category,
                    total_mentions=total,
                    positive_mentions=pos_count,
                    negative_mentions=neg_count,
                    neutral_mentions=neu_count,
                    average_sentiment=avg_compound,
                    net_sentiment_score=nss,
                    sentiment_ratio={
                        "positive": round((pos_count / total) * 100, 1) if total else 0.0,
                        "negative": round((neg_count / total) * 100, 1) if total else 0.0,
                        "neutral": round((neu_count / total) * 100, 1) if total else 0.0,
                    },
                    sample_positive_quotes=pos_quotes,
                    sample_negative_quotes=neg_quotes,
                )
            )

        # Sort primarily by total mentions descending
        summaries.sort(key=lambda s: s.total_mentions, reverse=True)
        return summaries
