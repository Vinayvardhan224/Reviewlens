"""
ReviewLens Review Q&A Module.
Provides extractive question-answering over customer reviews using TF-IDF retrieval,
cosine similarity ranking, sentence-level extraction, and review citations.
No external LLM required.
"""

from typing import Dict, Any, List, Optional
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from modules.data_loader import get_all_reviews, load_reviews_raw
from modules.sentiment import analyze_text_sentiment


def split_into_sentences(text: str) -> List[str]:
    """Split review text into distinct sentences."""
    if not text:
        return []
    # Split on sentence terminators (. ! ?) followed by whitespace or end of string
    raw_sents = re.split(r"(?<=[.!?])\s+", text.strip())
    clean_sents = [s.strip() for s in raw_sents if len(s.strip()) > 10]
    if not clean_sents and text.strip():
        clean_sents = [text.strip()]
    return clean_sents


def answer_product_review_question(
    product_id: str,
    question: str,
    top_k: int = 3,
    reviews_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Answers a buyer's question about a product using extractive TF-IDF retrieval
    over that product's genuine customer reviews.

    Args:
        product_id: Target product ID (e.g. 'apple-macbook-air-m3').
        question: Buyer's natural language question (e.g. 'How is the battery life?').
        top_k: Number of most relevant citations to retrieve (default 3).
        reviews_df: Optional pre-loaded reviews DataFrame.

    Returns:
        Structured dictionary containing:
          - status: 'ok', 'empty_query', 'no_reviews', or 'no_match'
          - question: Normalized question string
          - answer: Extractive synthesis and answer summary
          - consensus: Sentiment polarity consensus ('Mostly Positive', 'Mixed', etc.)
          - sentiment_summary: Counts of positive, negative, and neutral retrieved reviews
          - citations: List of top matching sentences with review_id, score, rating, and quote
    """
    clean_q = (question or "").strip()
    if not clean_q:
        return {
            "status": "empty_query",
            "product_id": product_id,
            "question": "",
            "answer": "Please enter a specific question about this product (e.g., 'How is battery life?', 'Does it overheat?').",
            "consensus": "Neutral",
            "sentiment_summary": {"positive": 0, "negative": 0, "neutral": 0},
            "citations": [],
            "top_reviews": []
        }

    # 1. Load reviews for this product
    if reviews_df is None:
        try:
            reviews_df = get_all_reviews(validate=False)
        except Exception:
            reviews_df = load_reviews_raw()

    if reviews_df is None or reviews_df.empty:
        return {
            "status": "no_reviews",
            "product_id": product_id,
            "question": clean_q,
            "answer": f"No review dataset available to answer questions for '{product_id}'.",
            "consensus": "Neutral",
            "sentiment_summary": {"positive": 0, "negative": 0, "neutral": 0},
            "citations": [],
            "top_reviews": []
        }

    p_reviews = reviews_df[reviews_df["product_id"].astype(str) == str(product_id)].copy()
    if p_reviews.empty:
        return {
            "status": "no_reviews",
            "product_id": product_id,
            "question": clean_q,
            "answer": f"No customer reviews found for product ID '{product_id}'.",
            "consensus": "Neutral",
            "sentiment_summary": {"positive": 0, "negative": 0, "neutral": 0},
            "citations": [],
            "top_reviews": []
        }

    # 2. Build corpus of sentence-level candidate snippets
    text_col = "review_text" if "review_text" in p_reviews.columns else "text"
    id_col = "review_id" if "review_id" in p_reviews.columns else p_reviews.columns[0]
    rating_col = "rating" if "rating" in p_reviews.columns else ("review_rating" if "review_rating" in p_reviews.columns else None)

    candidates: List[Dict[str, Any]] = []
    for _, row in p_reviews.iterrows():
        rev_id = str(row[id_col])
        full_text = str(row[text_col])
        rating_val = float(row[rating_col]) if rating_col and pd.notna(row[rating_col]) else 4.0
        sentences = split_into_sentences(full_text)

        for sent in sentences:
            sent_analysis = analyze_text_sentiment(sent)
            candidates.append({
                "review_id": rev_id,
                "sentence": sent,
                "full_text": full_text,
                "rating": rating_val,
                "sentiment": sent_analysis.get("label", "NEUTRAL"),
                "compound": sent_analysis.get("compound", 0.0)
            })

    if not candidates:
        return {
            "status": "no_reviews",
            "product_id": product_id,
            "question": clean_q,
            "answer": f"No readable review sentences found for product '{product_id}'.",
            "consensus": "Neutral",
            "sentiment_summary": {"positive": 0, "negative": 0, "neutral": 0},
            "citations": [],
            "top_reviews": []
        }

    # 3. Compute TF-IDF Cosine Similarity
    corpus_texts = [c["sentence"] for c in candidates]
    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
        tfidf_matrix = vectorizer.fit_transform(corpus_texts)
        q_vec = vectorizer.transform([clean_q])
        scores = cosine_similarity(q_vec, tfidf_matrix).flatten()
    except ValueError:
        # Fallback if vocabulary is empty due to stop words
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        tfidf_matrix = vectorizer.fit_transform(corpus_texts)
        q_vec = vectorizer.transform([clean_q])
        scores = cosine_similarity(q_vec, tfidf_matrix).flatten()

    # 4. Rank candidates by relevance
    ranked_indices = np.argsort(scores)[::-1]
    top_matches: List[Dict[str, Any]] = []
    seen_sentences = set()

    for idx in ranked_indices:
        sim = float(scores[idx])
        if sim <= 0.02:
            break
        cand = candidates[idx]
        norm_s = cand["sentence"].strip().lower()
        if norm_s in seen_sentences:
            continue
        seen_sentences.add(norm_s)

        top_matches.append({
            "review_id": cand["review_id"],
            "sentence": cand["sentence"],
            "full_text": cand["full_text"],
            "similarity": round(sim, 3),
            "rating": cand["rating"],
            "sentiment": cand["sentiment"],
            "compound": cand["compound"],
            "citation": f"[Review #{cand['review_id']}]"
        })
        if len(top_matches) >= top_k:
            break

    # If no relevant match found
    if not top_matches:
        return {
            "status": "no_match",
            "product_id": product_id,
            "question": clean_q,
            "answer": (
                f"Customer reviews for this product do not specifically address '{clean_q}'. "
                "Try asking about core hardware attributes such as battery life, display quality, audio, camera, or thermal performance."
            ),
            "consensus": "Neutral",
            "sentiment_summary": {"positive": 0, "negative": 0, "neutral": 0},
            "citations": [],
            "top_reviews": []
        }

    # 5. Compute Sentiment Consensus
    pos_count = sum(1 for m in top_matches if m["sentiment"] == "POSITIVE")
    neg_count = sum(1 for m in top_matches if m["sentiment"] == "NEGATIVE")
    neu_count = sum(1 for m in top_matches if m["sentiment"] == "NEUTRAL")

    if pos_count > neg_count:
        consensus = "Mostly Positive"
    elif neg_count > pos_count:
        consensus = "Mostly Negative"
    elif pos_count == neg_count and pos_count > 0:
        consensus = "Mixed / Polarized"
    else:
        consensus = "Neutral / Descriptive"

    # 6. Synthesize Extractive Answer
    quotes = [f"\"{m['sentence']}\" {m['citation']}" for m in top_matches]
    summary_text = (
        f"**Consensus: {consensus}** ({pos_count} positive, {neg_count} negative, {neu_count} neutral). "
        f"Based on customer feedback:\n- " + "\n- ".join(quotes)
    )

    return {
        "status": "ok",
        "product_id": product_id,
        "question": clean_q,
        "answer": summary_text,
        "consensus": consensus,
        "sentiment_summary": {
            "positive": pos_count,
            "negative": neg_count,
            "neutral": neu_count
        },
        "citations": top_matches,
        "top_reviews": list({m["review_id"] for m in top_matches})
    }
