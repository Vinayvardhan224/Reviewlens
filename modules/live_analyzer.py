"""
ReviewLens Live Review Analyzer Module.
Provides interactive, explainable, real-time diagnostic evaluation of arbitrary review text:
  1. Clause-by-clause Aspect Extraction & Polarity Assignment
  2. VADER Sentiment Scoring & Valence Breakdown
  3. Star-Rating Anomaly & Sentiment Discrepancy Detection
  4. Spam, Bot, and Synthetic Manipulation Risk Scoring
  5. Negation Scope & Syntactic Modifier Detection
"""

from typing import Dict, Any, List, Optional
import html
from modules.sentiment import analyze_text_sentiment
from modules.preprocessing import split_into_aspect_clauses, detect_negations, tokenize
from modules.aspect_analysis import extract_aspects_from_sentence
from reviewlens.spam_detector.detector import SpamDetector

# Cached spam detector instance
_spam_detector = SpamDetector()


def analyze_custom_review(
    text: str,
    rating: Optional[float] = None,
    category: Optional[str] = "Smartphones"
) -> Dict[str, Any]:
    """
    Runs complete real-time NLP diagnostic on arbitrary review text.
    Returns structured results for sentiment, clauses, aspects, anomalies, and spam risk.
    """
    clean_input = text.strip() if text else ""
    if not clean_input:
        return {
            "status": "empty",
            "message": "Please enter review text to analyze."
        }

    # 1. Overall Text Sentiment
    sentiment = analyze_text_sentiment(clean_input)

    # 2. Clause-by-Clause Aspect & Polarity Segmentation
    clauses_raw = split_into_aspect_clauses(clean_input)
    clause_results = []
    aspects_detected = set()

    for idx, clause in enumerate(clauses_raw):
        c_sentiment = analyze_text_sentiment(clause)
        c_aspects = extract_aspects_from_sentence(clause, category=category)
        aspects_detected.update(c_aspects)

        clause_results.append({
            "clause_index": idx + 1,
            "text": clause,
            "sentiment_label": c_sentiment["label"],
            "compound": c_sentiment["compound"],
            "aspects": c_aspects
        })

    # 3. Negation Scope Analysis
    tokens = tokenize(clean_input)
    negation_pairs = detect_negations(tokens)
    negated_words = [tok for tok, is_neg in negation_pairs if is_neg]

    # 4. Spam, Bot & Synthetic Review Detection
    spam_res = _spam_detector.analyze(
        text=clean_input,
        rating=rating,
        sentiment_compound=sentiment["compound"],
        check_duplicate=False
    )

    # 5. Rating vs Sentiment Anomaly Check
    anomaly_flag = False
    anomaly_desc = "No rating provided for contradiction check."
    if rating is not None:
        compound = sentiment["compound"]
        # Star rating 4-5 with strong negative sentiment
        if rating >= 4.0 and compound <= -0.2:
            anomaly_flag = True
            anomaly_desc = f"Contradiction flagged: High star rating ({rating}★) paired with strongly negative text sentiment (compound {compound:+.2f})."
        # Star rating 1-2 with strong positive sentiment
        elif rating <= 2.0 and compound >= 0.2:
            anomaly_flag = True
            anomaly_desc = f"Contradiction flagged: Low star rating ({rating}★) paired with strongly positive text sentiment (compound {compound:+.2f})."
        else:
            anomaly_desc = f"Rating ({rating}★) aligns normally with computed text sentiment ({sentiment['label']}, compound {compound:+.2f})."

    return {
        "status": "ok",
        "text": clean_input,
        "rating": rating,
        "category": category,
        "sentiment": sentiment,
        "clause_analysis": clause_results,
        "aspects_found": sorted(list(aspects_detected)),
        "negated_words": negated_words,
        "spam_analysis": {
            "is_spam": spam_res.is_spam,
            "spam_score": round(spam_res.spam_score, 2),
            "reasons": spam_res.reasons
        },
        "anomaly": {
            "is_anomaly": anomaly_flag,
            "description": anomaly_desc
        }
    }
