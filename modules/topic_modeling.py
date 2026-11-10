"""
Topic Modeling module for ReviewLens.
Discovers latent themes and recurring conversational clusters in product feedback
using Latent Dirichlet Allocation (LDA) with customizable topic counts.
"""

from typing import List, Dict, Any, Optional
import re
import logging
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

logger = logging.getLogger("reviewlens.topic_modeling")

# Custom stopwords for electronics domain
ELECTRONICS_STOPWORDS = {
    "product", "item", "bought", "buy", "amazon", "use", "using", "purchased",
    "one", "get", "got", "really", "would", "also", "even", "much", "like", "good",
    "device", "phone", "laptop", "camera", "headphones", "headphone", "day", "days",
    "time", "first", "still", "well", "great", "very", "the", "and", "is", "it", "this"
}


def extract_topic_clusters_result(
    texts: List[str],
    n_topics: int = 3,
    top_words: int = 6
) -> Dict[str, Any]:
    """
    Fits Latent Dirichlet Allocation (LDA) over review texts and returns structured status:
      - status: 'ok', 'insufficient_text', or 'error'
      - clusters: List of validated topic dicts
      - error_message: Optional detail on failure
    """
    valid_texts = [t.strip() for t in texts if t and len(t.strip()) > 15]
    if len(valid_texts) < 4:
        return {
            "status": "insufficient_text",
            "clusters": [],
            "error_message": f"Review sample size is too small ({len(valid_texts)} valid reviews; minimum 4 required for LDA clustering)."
        }

    # Bound n_topics between 2 and 6, and not exceeding number of texts // 2
    effective_topics = min(max(2, n_topics), max(2, len(valid_texts) // 3))

    try:
        vectorizer = CountVectorizer(
            max_features=200,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1
        )
        doc_term_matrix = vectorizer.fit_transform(valid_texts)
        feature_names = vectorizer.get_feature_names_out()

        if doc_term_matrix.shape[1] < effective_topics:
            return {
                "status": "insufficient_text",
                "clusters": [],
                "error_message": "Corpus vocabulary size is too small to differentiate distinct latent topics."
            }

        lda = LatentDirichletAllocation(
            n_components=effective_topics,
            random_state=42,
            max_iter=20,
            learning_method="online"
        )
        doc_topic_dist = lda.fit_transform(doc_term_matrix)

        clusters: List[Dict[str, Any]] = []
        for topic_idx, topic in enumerate(lda.components_):
            # Sort words by weight, preferring non-generic words
            sorted_indices = topic.argsort()[::-1]
            chosen_keywords = []
            for idx in sorted_indices:
                w = feature_names[idx]
                if w not in ELECTRONICS_STOPWORDS and len(w) > 2:
                    chosen_keywords.append(w)
                if len(chosen_keywords) >= top_words:
                    break

            if not chosen_keywords:
                chosen_keywords = [feature_names[i] for i in sorted_indices[:top_words]]

            # Validate that the topic has at least 2 meaningful terms
            if len(chosen_keywords) < 2:
                continue

            label = f"Theme {topic_idx + 1}: " + ", ".join([w.title() for w in chosen_keywords[:3]])

            # Representative review snippets
            best_doc_indices = doc_topic_dist[:, topic_idx].argsort()[::-1][:3]
            sample_quotes = []
            for doc_idx in best_doc_indices:
                if doc_idx < len(valid_texts):
                    sample_quotes.append(valid_texts[doc_idx])

            clusters.append({
                "topic_id": topic_idx + 1,
                "label": label,
                "keywords": chosen_keywords,
                "sample_reviews": sample_quotes
            })

        return {
            "status": "ok" if clusters else "insufficient_text",
            "clusters": clusters,
            "error_message": None if clusters else "Could not form cohesive clusters with meaningful keywords."
        }
    except Exception as e:
        logger.warning(f"[ReviewLens TopicModeler] Topic extraction failed: {e}")
        return {
            "status": "error",
            "clusters": [],
            "error_message": f"LDA topic modeling encountered an unexpected error: {str(e)}"
        }


def extract_topic_clusters(
    texts: List[str],
    n_topics: int = 3,
    top_words: int = 6
) -> List[Dict[str, Any]]:
    """Convenience helper returning the list of topic clusters."""
    res = extract_topic_clusters_result(texts, n_topics=n_topics, top_words=top_words)
    return res["clusters"]
