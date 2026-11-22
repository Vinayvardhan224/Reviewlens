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
from pathlib import Path
import html
import joblib
from reviewlens.pipeline import ReviewLensPipeline
from reviewlens.sentiment.vader_analyzer import SentimentAnalyzer
from reviewlens.spam_detector.detector import SpamDetector
from modules.sentiment import detect_rating_anomaly
from modules.preprocessing import split_into_aspect_clauses, detect_negations, tokenize
from modules.action_brief import evaluate_review_actionability

# Authoritative ReviewLens NLP Pipeline singleton for interactive diagnostics
_pipeline = ReviewLensPipeline()
_ml_sentiment_model = None


def get_ml_sentiment_model():
    """Lazy loader for trained TF-IDF + Logistic Regression sentiment model."""
    global _ml_sentiment_model
    if _ml_sentiment_model is None:
        model_path = Path(__file__).resolve().parent.parent / "models" / "sentiment_model.joblib"
        if model_path.exists():
            try:
                _ml_sentiment_model = joblib.load(model_path)
            except Exception as e:
                print(f"[!] Warning: Could not load ML model: {e}")
    return _ml_sentiment_model


def analyze_custom_review(
    text: str,
    rating: Optional[float] = None,
    category: Optional[str] = "Smartphones",
    model_type: str = "VADER"
) -> Dict[str, Any]:
    """
    Runs complete real-time NLP diagnostic on arbitrary review text using the authoritative ReviewLensPipeline.
    Supports switching between rule-based VADER and trained ML (TF-IDF + Logistic Regression).
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
        "primary_emotion": analysis.sentiment.primary_emotion,
        "engine": "VADER (Lexicon)"
    }

    # Check if ML Model is requested
    is_ml_mode = "ML" in model_type or "LogReg" in model_type or "TF-IDF" in model_type
    if is_ml_mode:
        ml_model = get_ml_sentiment_model()
        if ml_model is not None:
            try:
                pred_label = ml_model.predict([clean_input])[0].lower()
                classes = list(ml_model.classes_)
                probs = ml_model.predict_proba([clean_input])[0]
                prob_dict = dict(zip(classes, probs))

                p_pos = float(prob_dict.get("positive", 0.0))
                p_neg = float(prob_dict.get("negative", 0.0))
                p_neu = float(prob_dict.get("neutral", 0.0))
                ml_compound = round(p_pos - p_neg, 4)

                sentiment = {
                    "compound": ml_compound,
                    "positive": round(p_pos, 4),
                    "pos": round(p_pos, 4),
                    "neutral": round(p_neu, 4),
                    "neu": round(p_neu, 4),
                    "negative": round(p_neg, 4),
                    "neg": round(p_neg, 4),
                    "label": pred_label.upper(),
                    "confidence": round(float(max(probs)), 2),
                    "primary_emotion": analysis.sentiment.primary_emotion,
                    "engine": "TF-IDF + Logistic Regression (ML)"
                }
            except Exception as e:
                print(f"[!] Warning: ML prediction fallback to VADER: {e}")

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

    # 6. Engineering Actionability & Gap Evaluation (Vector-Inspired Analysis)
    actionability_res = evaluate_review_actionability(
        text=clean_input,
        aspects=list(aspects_detected),
        sentiment_compound=analysis.sentiment.compound
    )

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
        "actionability": actionability_res,
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
