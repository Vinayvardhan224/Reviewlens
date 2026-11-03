"""
ReviewLens Keyword & Key Phrase Extraction Module.
Employs scikit-learn TfidfVectorizer (unigrams & bigrams) to surface
top positive satisfaction drivers and negative pain points.
"""

from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

DOMAIN_STOPWORDS = {
    "phone", "product", "item", "one", "also", "bought", "buy", "using", "use",
    "really", "very", "much", "even", "still", "get", "got", "device", "amazon",
    "delivery", "ordered", "order", "day", "days", "time", "months", "year"
}


def extract_keywords_by_sentiment(df_reviews: pd.DataFrame, top_n: int = 8) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract high-signal unigrams and bigrams distinguishing positive from negative feedback.
    """
    if df_reviews.empty or "review_text" not in df_reviews.columns:
        return {"positive_drivers": [], "negative_pain_points": []}

    pos_texts = []
    neg_texts = []

    for _, row in df_reviews.iterrows():
        text = str(row.get("review_text", "")).strip()
        compound = row.get("sentiment_compound", 0.0)
        rating = row.get("review_rating", 3.0)

        if compound >= 0.05 or rating >= 4.0:
            pos_texts.append(text)
        elif compound <= -0.05 or rating <= 2.0:
            neg_texts.append(text)

    pos_keywords = _extract_tfidf_terms(pos_texts, top_n)
    neg_keywords = _extract_tfidf_terms(neg_texts, top_n)

    return {
        "positive_drivers": pos_keywords,
        "negative_pain_points": neg_keywords
    }


def _extract_tfidf_terms(corpus: List[str], top_n: int) -> List[Dict[str, Any]]:
    """Helper to extract top weighted terms from a corpus."""
    if not corpus or len(corpus) == 0:
        return []

    combined_text = " ".join(corpus)
    if len(combined_text.split()) < 5:
        return []

    try:
        vec = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            max_features=25,
            min_df=1
        )
        matrix = vec.fit_transform(corpus)
        feature_names = vec.get_feature_names_out()
        scores = matrix.sum(axis=0).A1

        ranked = []
        for term, score in zip(feature_names, scores):
            parts = term.split()
            # Filter out single domain stopwords
            if any(p in DOMAIN_STOPWORDS for p in parts):
                continue
            if len(term) <= 3:
                continue
            ranked.append({"term": term, "score": round(float(score), 3)})

        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked[:top_n]
    except Exception:
        # Fallback to simple word frequency
        words = [w for w in combined_text.lower().split() if len(w) > 3 and w not in DOMAIN_STOPWORDS]
        from collections import Counter
        counts = Counter(words).most_common(top_n)
        return [{"term": term, "score": float(count)} for term, count in counts]
