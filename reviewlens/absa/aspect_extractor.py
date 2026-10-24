"""
Aspect-Based Sentiment Extraction Engine for ReviewLens.
Segments text into aspect-specific clauses, maps sentiment polarity, and extracts opinion contexts.
"""

from typing import List, Optional
import re
from reviewlens.absa.aspect_categories import AspectTaxonomy
from reviewlens.sentiment.vader_analyzer import SentimentAnalyzer
from reviewlens.preprocessing.tokenizer import split_sentences
from reviewlens.models import AspectMention


# Conjunctions that often flip polarity between clauses
ADVERSATIVE_CONJUNCTIONS = r"(?:[\s,;]+(?:but|however|although|whereas|yet|except(?:\s+that)?|though)\s+)"


class AspectExtractor:
    """Extracts fine-grained aspect sentiment mentions from product review texts."""

    def __init__(
        self,
        taxonomy: Optional[AspectTaxonomy] = None,
        sentiment_analyzer: Optional[SentimentAnalyzer] = None
    ):
        self.taxonomy = taxonomy or AspectTaxonomy()
        self.sentiment_analyzer = sentiment_analyzer or SentimentAnalyzer()

    def _split_clauses(self, sentence: str) -> List[str]:
        """
        Split a sentence at adversative boundaries (e.g. 'X is good, BUT Y is bad').
        Allows isolating contrasting sentiments attached to different aspects.
        """
        clauses = re.split(ADVERSATIVE_CONJUNCTIONS, sentence, flags=re.IGNORECASE)
        return [c.strip() for c in clauses if c.strip()]

    def extract_from_review(self, text: str) -> List[AspectMention]:
        """
        Analyze a full review and return all detected AspectMention instances.
        
        Args:
            text: Raw or cleaned review text.
            
        Returns:
            List of AspectMention objects.
        """
        if not text or not text.strip():
            return []

        sentences = split_sentences(text)
        detected_aspects: List[AspectMention] = []

        for sent in sentences:
            clauses = self._split_clauses(sent)

            # If there are multiple clauses, check each clause individually
            if len(clauses) > 1:
                matched_in_clauses = False
                for clause in clauses:
                    matches = self.taxonomy.find_aspects_in_sentence(clause)
                    if matches:
                        matched_in_clauses = True
                        clause_sentiment = self.sentiment_analyzer.analyze(clause)
                        for cat, term in matches:
                            detected_aspects.append(
                                AspectMention(
                                    category=cat,
                                    matched_term=term,
                                    sentence=clause,
                                    sentiment_label=clause_sentiment.label,
                                    sentiment_score=clause_sentiment.compound,
                                    opinion_words=self._extract_opinion_words(clause, term)
                                )
                            )
                if matched_in_clauses:
                    continue

            # Fallback to full sentence if no clause matched or only one clause
            matches = self.taxonomy.find_aspects_in_sentence(sent)
            if matches:
                sent_sentiment = self.sentiment_analyzer.analyze(sent)
                for cat, term in matches:
                    detected_aspects.append(
                        AspectMention(
                            category=cat,
                            matched_term=term,
                            sentence=sent,
                            sentiment_label=sent_sentiment.label,
                            sentiment_score=sent_sentiment.compound,
                            opinion_words=self._extract_opinion_words(sent, term)
                        )
                    )

        return detected_aspects

    def _extract_opinion_words(self, clause: str, aspect_term: str) -> List[str]:
        """
        Extract nearby descriptive/opinion words around the matched aspect.
        """
        # Look for descriptive adjectives or adverbs in the proximity of aspect term
        descriptors = [
            "good", "great", "excellent", "amazing", "fantastic", "stellar", "superb",
            "bad", "terrible", "awful", "horrible", "poor", "subpar", "mediocre",
            "durable", "sturdy", "flimsy", "cheap", "premium", "solid", "broken",
            "fast", "slow", "laggy", "responsive", "smooth", "comfortable", "painful",
            "tight", "loose", "clear", "crisp", "muffled", "loud", "quiet", "long",
            "short", "heavy", "light", "expensive", "affordable", "worth", "worthless"
        ]
        lower_clause = clause.lower()
        found = [d for d in descriptors if re.search(r"\b" + re.escape(d) + r"\b", lower_clause)]
        return found
