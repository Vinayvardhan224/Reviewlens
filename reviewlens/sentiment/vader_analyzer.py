"""
Sentiment Analysis Engine using enhanced VADER with domain-adapted product review lexicon.
"""

from typing import Dict, Any, Optional
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from reviewlens.config import SENTIMENT_THRESHOLDS, DOMAIN_SENTIMENT_LEXICON, RATING_MAPPING
from reviewlens.models import SentimentResult


class SentimentAnalyzer:
    """Enhanced VADER sentiment analyzer tailored for e-commerce and product reviews."""

    def __init__(self, custom_lexicon: Optional[Dict[str, float]] = None):
        try:
            self.sia = SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
            self.sia = SentimentIntensityAnalyzer()

        # Augment with product review specific lexicon
        lexicon_to_add = DOMAIN_SENTIMENT_LEXICON.copy()
        if custom_lexicon:
            lexicon_to_add.update(custom_lexicon)

        self.sia.lexicon.update(lexicon_to_add)

    def analyze(self, text: str) -> SentimentResult:
        """
        Compute polarity scores and classification for review text.
        
        Args:
            text: Cleaned review string.
            
        Returns:
            SentimentResult with label, compound, pos, neu, neg, and predicted rating.
        """
        if not text or not text.strip():
            return SentimentResult(
                label="NEUTRAL",
                compound=0.0,
                pos=0.0,
                neu=1.0,
                neg=0.0,
                predicted_rating=3.0,
                confidence=0.5
            )

        scores = self.sia.polarity_scores(text)
        compound = round(scores["compound"], 4)
        pos = round(scores["pos"], 4)
        neu = round(scores["neu"], 4)
        neg = round(scores["neg"], 4)

        # Determine label based on configured thresholds
        if compound >= SENTIMENT_THRESHOLDS["positive"]:
            label = "POSITIVE"
        elif compound <= SENTIMENT_THRESHOLDS["negative"]:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"

        # Compute predicted rating (1 to 5 stars)
        predicted_rating = self.map_compound_to_rating(compound)

        # Confidence metric derived from polarity strength and margin
        polarity_delta = abs(pos - neg)
        confidence = round(min(1.0, max(0.4, abs(compound) * 0.7 + polarity_delta * 0.3)), 2)

        return SentimentResult(
            label=label,
            compound=compound,
            pos=pos,
            neu=neu,
            neg=neg,
            predicted_rating=predicted_rating,
            confidence=confidence
        )

    @staticmethod
    def map_compound_to_rating(compound: float) -> float:
        """
        Map a continuous compound score [-1.0, 1.0] to a 1.0 - 5.0 star rating.
        Interpolates linearly within rating bins for smooth gradation.
        """
        # Linear normalization: -1.0 maps to 1.0 stars, +1.0 maps to 5.0 stars
        # stars = 1.0 + (compound + 1.0) / 2.0 * 4.0
        stars = 1.0 + ((compound + 1.0) / 2.0) * 4.0
        return round(max(1.0, min(5.0, stars)), 1)
