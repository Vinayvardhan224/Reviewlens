"""
Keyword and N-gram extraction using TF-IDF for ReviewLens.
Identifies defining phrases distinguishing positive vs. negative reviews.
"""

from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from reviewlens.preprocessing.tokenizer import FILTERED_STOPWORDS


class KeywordExtractor:
    """Extracts top n-grams and significant key phrases from review collections."""

    def __init__(self, max_features: int = 50, ngram_range: tuple = (1, 2)):
        # Extend stopwords with common generic e-commerce words
        domain_stopwords = list(FILTERED_STOPWORDS) + [
            "product", "item", "bought", "buy", "amazon", "use", "using", "purchased",
            "one", "get", "got", "really", "would", "also", "even", "much"
        ]
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words=domain_stopwords,
            min_df=2,
        )

    def extract_top_keywords(self, texts: List[str], top_n: int = 15) -> List[Dict[str, Any]]:
        """
        Extract the most salient key phrases from a list of texts using TF-IDF.
        """
        valid_texts = [t for t in texts if t and len(t.strip()) > 5]
        if len(valid_texts) < 2:
            # Fallback simple frequency for small sample
            return self._fallback_frequency(valid_texts, top_n)

        try:
            tfidf_matrix = self.vectorizer.fit_transform(valid_texts)
            feature_names = self.vectorizer.get_feature_names_out()
            scores = tfidf_matrix.sum(axis=0).A1

            top_indices = scores.argsort()[::-1][:top_n]
            return [
                {"term": feature_names[i], "score": round(float(scores[i]), 3)}
                for i in top_indices if scores[i] > 0
            ]
        except Exception:
            return self._fallback_frequency(valid_texts, top_n)

    def _fallback_frequency(self, texts: List[str], top_n: int) -> List[Dict[str, Any]]:
        from collections import Counter
        import re
        words = []
        for t in texts:
            tokens = re.findall(r"\b[a-zA-Z]{3,}\b", t.lower())
            words.extend([w for w in tokens if w not in FILTERED_STOPWORDS])
        counts = Counter(words).most_common(top_n)
        return [{"term": term, "score": float(count)} for term, count in counts]
