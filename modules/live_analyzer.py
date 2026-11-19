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
from reviewlens.pipeline import ReviewLensPipeline
from reviewlens.sentiment.vader_analyzer import SentimentAnalyzer
from reviewlens.spam_detector.detector import SpamDetector
from modules.sentiment import detect_rating_anomaly
from modules.preprocessing import split_into_aspect_clauses, detect_negations, tokenize

# Authoritative ReviewLens NLP Pipeline singleton for interactive diagnostics
_pipeline = ReviewLensPipeline()


def analyze_custom_review(
    text: str,
    rating: Optional[float] = None,
    category: Optional[str] = "Smartphones"
) -> Dict[str, Any]:
    """
    Runs complete real-time NLP diagnostic on arbitrary review text using the authoritative ReviewLensPipeline.
    Ensures Streamlit, FastAPI, and CLI return 100% consistent scores and classifications.
    """
    clean_input = text.strip() if text else ""
    if not clean_input:
        return {
            "status": "empty",
            "message": "Please enter review text to analyze."
        }

    # 1. Execute authoritative ReviewLens Pipeline
    analysis = _pipeline.analyze_single(
        text=clean_input,
        rating=rating,
        check_duplicate=False
    )

    # 2. Extract standardized sentiment dictionary
    sentiment = {
        "compound": analysis.sentiment.compound,
        "positive": analysis.sentiment.pos,
        "pos": analysis.sentiment.pos,
        "neutral": analysis.sentiment.neu,
        "neu": analysis.sentiment.neu,
        "negative": analysis.sentiment.neg,
        "neg": analysis.sentiment.neg,
        "label": analysis.sentiment.label,
        "confidence": analysis.sentiment.confidence,
        "primary_emotion": analysis.sentiment.primary_emotion
    }

    # 3. Clause-by-Clause Aspect & Polarity Segmentation
    clauses_raw = split_into_aspect_clauses(clean_input)
    clause_results = []
    aspects_detected = set(a.category for a in analysis.aspects)

    for idx, clause in enumerate(clauses_raw):
        c_sent = _pipeline.sentiment_analyzer.analyze(clause)
        c_aspects = [a.category for a in analysis.aspects if a.sentence.strip() == clause.strip() or a.matched_term.lower() in clause.lower()]
        clause_results.append({
            "clause_index": idx + 1,
            "text": clause,
            "sentiment_label": c_sent.label,
            "compound": c_sent.compound,
            "aspects": list(set(c_aspects))
        })

    # 4. Negation Scope Analysis
    tokens = tokenize(clean_input)
    negation_pairs = detect_negations(tokens)
    negated_words = [tok for tok, is_neg in negation_pairs if is_neg]

    # 5. Rating vs Sentiment Anomaly Check
    anomaly_flag = False
    anomaly_desc = "No rating provided for contradiction check."
    if rating is not None:
        anomaly_info = detect_rating_anomaly(rating, analysis.sentiment.compound)
        anomaly_flag = anomaly_info.get("is_anomaly", False)
        if anomaly_flag:
            anomaly_desc = f"Contradiction flagged: {anomaly_info.get('anomaly_type')} (Rating {rating}★ paired with compound score {analysis.sentiment.compound:+.2f})."
        else:
            anomaly_desc = f"Rating ({rating}★) aligns normally with computed text sentiment ({analysis.sentiment.label}, compound {analysis.sentiment.compound:+.2f})."

    return {
        "status": "ok",
        "text": clean_input,
        "rating": rating,
        "category": category,
        "sentiment": sentiment,
        "predicted_rating": analysis.predicted_rating,
        "clause_analysis": clause_results,
        "aspects_found": sorted(list(aspects_detected)),
        "aspect_mentions": [a.model_dump() for a in analysis.aspects],
        "negated_words": negated_words,
        "spam_analysis": {
            "is_spam": analysis.spam.is_spam,
            "spam_score": round(analysis.spam.spam_score, 2),
            "reasons": analysis.spam.reasons,
            "has_mismatch": analysis.spam.has_mismatch
        },
        "anomaly": {
            "is_anomaly": anomaly_flag,
            "description": anomaly_desc
        }
    }
