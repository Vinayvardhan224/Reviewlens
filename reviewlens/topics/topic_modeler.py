"""
Unsupervised Topic Modeling engine for ReviewLens using Latent Dirichlet Allocation (LDA).
Discovers latent themes and recurring conversational clusters in product feedback.
"""

from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from reviewlens.models import TopicCluster
from reviewlens.preprocessing.tokenizer import FILTERED_STOPWORDS


class TopicModeler:
    """Discovers thematic review clusters using Latent Dirichlet Allocation (LDA)."""

    def __init__(self, n_topics: int = 4, max_features: int = 150):
        self.n_topics = n_topics
        self.max_features = max_features
        extra_stops = [
            "product", "item", "bought", "buy", "amazon", "use", "using", "purchased",
            "one", "get", "got", "really", "would", "also", "even", "much", "like", "good"
        ]
        self.stop_words = list(FILTERED_STOPWORDS) + extra_stops

    def fit_transform(self, texts: List[str], top_words_per_topic: int = 6) -> List[TopicCluster]:
        """
        Fit LDA model on input corpus and extract topic clusters.
        
        Args:
            texts: List of review texts.
            top_words_per_topic: Number of salient keywords per topic.
            
        Returns:
            List of TopicCluster objects with labels and sample reviews.
        """
        valid_texts = [t for t in texts if t and len(t.strip()) > 15]
        if len(valid_texts) < 4:
            # Not enough text to form distinct topics
            return []

        # Adjust n_topics if corpus is small
        effective_topics = min(self.n_topics, max(2, len(valid_texts) // 4))

        try:
            vectorizer = CountVectorizer(
                max_features=self.max_features,
                stop_words=self.stop_words,
                ngram_range=(1, 2),
                min_df=2
            )
            doc_term_matrix = vectorizer.fit_transform(valid_texts)
            feature_names = vectorizer.get_feature_names_out()

            if doc_term_matrix.shape[1] < effective_topics:
                return []

            lda = LatentDirichletAllocation(
                n_components=effective_topics,
                random_state=42,
                max_iter=15,
                learning_method="online"
            )
            doc_topic_dist = lda.fit_transform(doc_term_matrix)

            clusters: List[TopicCluster] = []
            for topic_idx, topic in enumerate(lda.components_):
                top_features_ind = topic.argsort()[::-1][:top_words_per_topic]
                top_keywords = [feature_names[i] for i in top_features_ind]

                # Synthesize human-readable title from top keywords
                label = f"Topic {topic_idx + 1}: " + ", ".join(top_keywords[:3]).title()

                # Find reviews most representative of this topic
                best_doc_indices = doc_topic_dist[:, topic_idx].argsort()[::-1][:3]
                sample_reviews = [valid_texts[idx] for idx in best_doc_indices if idx < len(valid_texts)]

                clusters.append(
                    TopicCluster(
                        topic_id=topic_idx + 1,
                        label=label,
                        keywords=top_keywords,
                        sample_reviews=sample_reviews
                    )
                )

            return clusters
        except Exception:
            return []
