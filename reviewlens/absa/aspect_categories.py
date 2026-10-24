"""
Aspect taxonomy definitions and dynamic category registry for ReviewLens ABSA.
"""

from typing import Dict, List, Optional
import re
from reviewlens.config import DEFAULT_ASPECT_TAXONOMY


class AspectTaxonomy:
    """Manages product aspect categories and associated matching keywords."""

    def __init__(self, custom_taxonomy: Optional[Dict[str, List[str]]] = None):
        self.taxonomy: Dict[str, List[str]] = {}
        # Load defaults
        for cat, keywords in DEFAULT_ASPECT_TAXONOMY.items():
            self.taxonomy[cat] = [kw.lower() for kw in keywords]

        # Merge custom additions if provided
        if custom_taxonomy:
            for cat, keywords in custom_taxonomy.items():
                if cat in self.taxonomy:
                    self.taxonomy[cat].extend([kw.lower() for kw in keywords])
                else:
                    self.taxonomy[cat] = [kw.lower() for kw in keywords]

        # Precompile regex patterns for high-speed matching
        self._compiled_patterns = self._compile_patterns()

    def _compile_patterns(self) -> Dict[str, List[re.Pattern]]:
        compiled = {}
        for cat, keywords in self.taxonomy.items():
            patterns = []
            for kw in sorted(keywords, key=len, reverse=True):  # Longest match first
                # Word boundary match
                pattern = re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
                patterns.append((kw, pattern))
            compiled[cat] = patterns
        return compiled

    def find_aspects_in_sentence(self, sentence: str) -> List[tuple]:
        """
        Scan a sentence and return all matched (category, matched_term) tuples.
        """
        matches = []
        lower_sent = sentence.lower()
        for cat, patterns in self._compiled_patterns.items():
            for kw, pattern in patterns:
                if pattern.search(lower_sent):
                    matches.append((cat, kw))
                    break  # Found primary keyword for this category in this sentence
        return matches

    def add_category(self, category_name: str, keywords: List[str]):
        """Dynamically add or extend an aspect category."""
        self.taxonomy[category_name] = [kw.lower() for kw in keywords]
        self._compiled_patterns = self._compile_patterns()
