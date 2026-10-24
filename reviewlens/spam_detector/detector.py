"""
Spam, Bot, and Fake Review Detection Engine for ReviewLens.
Analyzes reviews for synthetic patterns, promotional manipulation, rating mismatches, and copy-paste duplicates.
"""

from typing import List, Optional, Set
import re
from reviewlens.config import SPAM_DETECTION_CONFIG
from reviewlens.models import SpamResult


class SpamDetector:
    """Detects spam, incentivized, or untrustworthy product reviews."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or SPAM_DETECTION_CONFIG
        self.seen_signatures: Set[str] = set()

    def _normalize_signature(self, text: str) -> str:
        """Create a simplified hashable signature for duplicate checking."""
        return re.sub(r"[^a-zA-Z0-9]", "", text.lower())

    def analyze(
        self,
        text: str,
        rating: Optional[float] = None,
        sentiment_compound: Optional[float] = None,
        check_duplicate: bool = True
    ) -> SpamResult:
        """
        Evaluate a single review against multi-point spam heuristics.
        
        Args:
            text: Raw review content.
            rating: Optional user star rating (1-5).
            sentiment_compound: Optional computed sentiment compound score (-1.0 to 1.0).
            check_duplicate: Whether to compare against previously seen reviews in this session.
            
        Returns:
            SpamResult with boolean decision, continuous score, and reasons.
        """
        if not text or not text.strip():
            return SpamResult(is_spam=True, spam_score=1.0, reasons=["Empty review"])

        reasons: List[str] = []
        spam_points = 0.0
        text_len = len(text.strip())

        # 1. Extremely short review
        if text_len < self.config["min_review_length_chars"]:
            spam_points += 0.25
            reasons.append(f"Excessively brief review (<{self.config['min_review_length_chars']} chars)")

        # 2. Uppercase ratio (Shouting)
        alphabetic_chars = [c for c in text if c.isalpha()]
        if alphabetic_chars:
            upper_ratio = sum(1 for c in alphabetic_chars if c.isupper()) / len(alphabetic_chars)
            if upper_ratio > self.config["max_uppercase_ratio"] and len(alphabetic_chars) > 20:
                spam_points += 0.35
                reasons.append(f"Excessive capitalization ({upper_ratio:.0%} uppercase)")

        # 3. Punctuation repetition (e.g. !!!!! or ?????)
        punct_match = re.search(r"([!?,.]){" + str(self.config["max_punctuation_repetition"]) + r",}", text)
        if punct_match:
            spam_points += 0.25
            reasons.append(f"Repetitive punctuation pattern: '{punct_match.group(0)}'")

        # 4. Known promotional / spam triggers
        lower_text = text.lower()
        for phrase in self.config["spam_phrases"]:
            if phrase in lower_text:
                spam_points += 0.50
                reasons.append(f"Contains promotional/spam phrase: '{phrase}'")
                break

        # 5. Rating vs Sentiment Inconsistency (Severe divergence)
        if rating is not None and sentiment_compound is not None:
            # High rating (>=4.5) but deeply negative sentiment (<= -0.4)
            if rating >= 4.5 and sentiment_compound <= -0.40:
                spam_points += 0.55
                reasons.append(f"Rating-sentiment mismatch (5-star rating with negative text score {sentiment_compound:.2f})")
            # Low rating (<=1.5) but deeply positive sentiment (>= 0.6)
            elif rating <= 1.5 and sentiment_compound >= 0.60:
                spam_points += 0.55
                reasons.append(f"Rating-sentiment mismatch (1-star rating with glowing positive text score {sentiment_compound:.2f})")

        # 6. Repetitive character loops (e.g. "sooooo goooood" or "aaaaaaa")
        char_repetition = re.search(r"(.)\1{5,}", text)
        if char_repetition:
            spam_points += 0.20
            reasons.append("Contains unnatural repeated character sequences")

        # 7. Duplicate detection across session/batch
        if check_duplicate:
            sig = self._normalize_signature(text)
            if len(sig) > 20:
                if sig in self.seen_signatures:
                    spam_points += 0.70
                    reasons.append("Exact or near-identical duplicate of an earlier review")
                else:
                    self.seen_signatures.add(sig)

        normalized_score = min(1.0, round(spam_points, 2))
        is_spam = normalized_score >= 0.50

        return SpamResult(
            is_spam=is_spam,
            spam_score=normalized_score,
            reasons=reasons
        )

    def reset_session(self):
        """Clear duplicate signature memory."""
        self.seen_signatures.clear()
