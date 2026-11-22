"""
ReviewLens Custom Review CSV Batch Analyzer & Explainability Engine.
Processes uploaded review CSV datasets (up to 5,000 rows):
  1. Validates columns and data integrity.
  2. Runs sentiment polarity and aspect-based decomposition.
  3. Audits spam, temporal velocity bursts, and near-duplicate text patterns.
  4. Computes word-level lexical importance drivers and explainability tokens.
"""

from typing import Dict, Any, List, Optional
import re
from collections import Counter
import pandas as pd
import numpy as np

from reviewlens.sentiment.vader_analyzer import SentimentAnalyzer
from reviewlens.spam_detector.detector import SpamDetector, detect_near_duplicates, detect_review_bursts
from modules.aspect_analysis import analyze_reviews_aspects
from modules.sentiment import compute_nss

_shared_sia = SentimentAnalyzer()
_shared_spam = SpamDetector()


def decompose_review_words(text: str) -> List[Dict[str, Any]]:
    """
    Decompose a single review into tokens with word-level VADER valence scores
    for glass-box explainability.
    """
    if not text:
        return []
    words = re.findall(r"\b[\w'-]+\b", text)
    tokens: List[Dict[str, Any]] = []
    for w in words:
        val = _shared_sia.sia.lexicon.get(w.lower(), 0.0)
        tokens.append({
            "word": w,
            "valence": round(float(val), 2),
            "is_positive": val > 0.05,
            "is_negative": val < -0.05,
            "impact": "positive" if val > 0.05 else ("negative" if val < -0.05 else "neutral")
        })
    return tokens


def analyze_custom_reviews_csv(
    df: pd.DataFrame,
    category: str = "Electronics",
    max_rows: int = 5000
) -> Dict[str, Any]:
    """
    Run full batch analysis and word-level explainability on a user-uploaded reviews DataFrame.

    Args:
        df: Input pandas DataFrame (must contain review text).
        category: E-commerce category for ABSA taxonomy (default 'Electronics').
        max_rows: Maximum rows to process (default 5,000).

    Returns:
        Structured dictionary with KPIs, aspect breakdown, spam audit,
        word drivers, and annotated DataFrame.
    """
    if df is None or df.empty:
        raise ValueError("Uploaded CSV is empty or could not be parsed.")

    # 1. Enforce row cap
    is_capped = len(df) > max_rows
    working_df = df.iloc[:max_rows].copy()

    # 2. Normalize text column
    text_col = None
    for cand in ["review_text", "text", "Review", "review", "content", "comments", "body"]:
        if cand in working_df.columns:
            text_col = cand
            break

    if text_col is None:
        raise ValueError(
            "Could not identify a review text column. "
            "Please ensure your CSV has a column named 'review_text', 'text', or 'review'."
        )

    working_df["clean_text"] = working_df[text_col].astype(str)

    # 3. Normalize optional rating column
    rating_col = None
    for cand in ["rating", "review_rating", "stars", "score", "Rating"]:
        if cand in working_df.columns:
            rating_col = cand
            break

    ratings = None
    if rating_col:
        ratings = pd.to_numeric(working_df[rating_col], errors="coerce")

    # 4. Normalize optional ID and date
    id_col = None
    for cand in ["review_id", "id", "ID", "Review_ID"]:
        if cand in working_df.columns:
            id_col = cand
            break
    if not id_col:
        working_df["review_id"] = [f"UPLOAD_{i+1:04d}" for i in range(len(working_df))]
        id_col = "review_id"

    # 5. Pipeline execution over each row
    sentiments = []
    compounds = []
    is_spam_list = []
    spam_scores = []
    reasons_list = []

    pos_drivers = Counter()
    neg_drivers = Counter()

    for idx, row in working_df.iterrows():
        t = row["clean_text"]
        r_val = float(ratings.iloc[idx]) if ratings is not None and pd.notna(ratings.iloc[idx]) else None

        # Sentiment
        s_res = _shared_sia.analyze(t)
        sentiments.append(s_res.label)
        compounds.append(s_res.compound)

        # Spam
        sp_res = _shared_spam.analyze(t, rating=r_val, sentiment_compound=s_res.compound, check_duplicate=False)
        is_spam_list.append(sp_res.is_spam)
        spam_scores.append(sp_res.spam_score)
        reasons_list.append("; ".join(sp_res.reasons) if sp_res.reasons else "Clean")

        # Word-level explainability counter
        words = re.findall(r"\b[a-zA-Z]{3,}\b", t.lower())
        for w in words:
            valence = _shared_sia.sia.lexicon.get(w, 0.0)
            if valence >= 1.0:
                pos_drivers[w] += 1
            elif valence <= -1.0:
                neg_drivers[w] += 1

    working_df["predicted_sentiment"] = sentiments
    working_df["sentiment_compound"] = compounds
    working_df["is_spam"] = is_spam_list
    working_df["spam_score"] = spam_scores
    working_df["audit_flags"] = reasons_list

    # 6. Compute KPIs
    total = len(working_df)
    pos_c = sum(1 for s in sentiments if s == "POSITIVE")
    neg_c = sum(1 for s in sentiments if s == "NEGATIVE")
    neu_c = sum(1 for s in sentiments if s == "NEUTRAL")
    nss = compute_nss(pos_c, neg_c, total)
    spam_c = sum(1 for sp in is_spam_list if sp)
    spam_rate = round((spam_c / max(1, total)) * 100, 1)

    mean_rating = round(float(ratings.mean()), 2) if ratings is not None and not ratings.isna().all() else None

    # 7. Aspect Analysis
    aspect_df = working_df.rename(columns={text_col: "review_text"})
    aspect_res = analyze_reviews_aspects(aspect_df, category=category)

    # 8. Spam Bursts and Duplicates
    review_dicts = working_df[[id_col, "clean_text"]].rename(columns={id_col: "review_id", "clean_text": "review_text"}).to_dict("records")
    if "date" in working_df.columns:
        for i, rd in enumerate(review_dicts):
            rd["date"] = str(working_df["date"].iloc[i])

    bursts = detect_review_bursts(review_dicts) if "date" in working_df.columns else []
    near_dups = detect_near_duplicates(review_dicts, similarity_threshold=0.85)

    # 9. Top Word Drivers
    top_pos = [{"word": w, "count": c, "valence": _shared_sia.sia.lexicon.get(w, 0.0)} for w, c in pos_drivers.most_common(10)]
    top_neg = [{"word": w, "count": c, "valence": _shared_sia.sia.lexicon.get(w, 0.0)} for w, c in neg_drivers.most_common(10)]

    return {
        "status": "ok",
        "total_rows": total,
        "is_capped": is_capped,
        "kpis": {
            "total_reviews": total,
            "positive_count": pos_c,
            "negative_count": neg_c,
            "neutral_count": neu_c,
            "nss": nss,
            "mean_rating": mean_rating,
            "spam_count": spam_c,
            "spam_rate_pct": spam_rate
        },
        "aspects": aspect_res.get("aspects", []),
        "bursts": bursts,
        "near_duplicates": near_dups,
        "word_drivers": {
            "positive": top_pos,
            "negative": top_neg
        },
        "annotated_df": working_df
    }
