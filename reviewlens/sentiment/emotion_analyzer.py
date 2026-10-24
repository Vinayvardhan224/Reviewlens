"""
Emotion Detection Module for ReviewLens.
Discovers nuanced psychological customer reactions beyond binary polarity.
"""

from typing import Dict, Tuple, Optional
import re
from reviewlens.config import EMOTION_KEYWORDS


class EmotionAnalyzer:
    """Classifies consumer emotional state into core sentiment dimensions."""

    def __init__(self, emotion_dict: Optional[Dict[str, list]] = None):
        self.emotion_map = emotion_dict or EMOTION_KEYWORDS

    def analyze(self, text: str, sentiment_compound: float = 0.0) -> Tuple[Optional[str], Dict[str, float]]:
        """
        Extract emotional distribution and dominant emotion.
        
        Args:
            text: Review text.
            sentiment_compound: Overall compound sentiment score.
            
        Returns:
            Tuple of (dominant_emotion_name, scores_dict).
        """
        if not text:
            return "Neutral / Informative", {k: 0.0 for k in self.emotion_map}

        lower_text = text.lower()
        counts: Dict[str, float] = {}

        for emotion, keywords in self.emotion_map.items():
            count = 0.0
            for kw in keywords:
                # Word boundary match
                pattern = r"\b" + re.escape(kw) + r"\b"
                matches = len(re.findall(pattern, lower_text))
                count += matches
            counts[emotion] = count

        # If sentiment strongly positive or negative, provide baseline anchor
        if sentiment_compound >= 0.5:
            counts["Joy & Delight"] = counts.get("Joy & Delight", 0.0) + 1.0
        elif sentiment_compound <= -0.4:
            if "Frustration & Anger" in counts and counts["Frustration & Anger"] == 0:
                counts["Disappointment"] = counts.get("Disappointment", 0.0) + 1.0

        total = sum(counts.values())
        if total == 0:
            return "Neutral / Informative", {k: 0.0 for k in self.emotion_map}

        # Normalize to percentages
        normalized = {k: round(v / total, 3) for k, v in counts.items()}
        dominant_emotion = max(normalized, key=normalized.get)

        return dominant_emotion, normalized
