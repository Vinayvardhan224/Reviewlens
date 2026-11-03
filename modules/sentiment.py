"""
ReviewLens Sentiment Analysis Module.
Powered by VADER (Valence Aware Dictionary and sEntiment Reasoner) with
Net Sentiment Score (NSS) calculation, confidence thresholds, and rating anomaly detection.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure VADER lexicon is cached and domain-adapted
try:
    _vader = SentimentIntensityAnalyzer()
except Exception:
    nltk.download("vader_lexicon", quiet=True)
    _vader = SentimentIntensityAnalyzer()

# Domain-specific sentiment modifiers for electronics / product reviews
DOMAIN_LEXICON = {
    "durable": 2.2, "crisp": 1.9, "punchy": 1.7, "ergonomic": 2.0,
    "sleek": 1.8, "seamless": 2.1, "premium": 2.4, "sturdy": 2.0,
    "intuitive": 2.0, "responsive": 2.1, "bargain": 2.0,
    "worth every penny": 3.2, "long-lasting": 2.3, "flawless": 2.8,
    "overpriced": -2.3, "flimsy": -2.4, "laggy": -2.3, "lags": -2.0,
    "bloatware": -2.2, "unresponsive": -2.4, "bricked": -3.5,
    "drain": -2.0, "drains": -2.0, "draining": -2.0, "defective": -2.8,
    "subpar": -2.1, "mediocre": -1.5, "overheats": -2.5, "overheating": -2.6,
    "gimmick": -2.0, "waste of money": -3.4, "uncomfortable": -2.4,
    "stutter": -1.8, "stutters": -1.8, "buggy": -2.2
}
_vader.lexicon.update(DOMAIN_LEXICON)


def analyze_text_sentiment(text: str) -> Dict[str, Any]:
    """
    Compute sentiment polarity scores for a given text.
    Returns:
        compound: float (-1.0 to 1.0)
        pos: float (0.0 to 1.0)
        neu: float (0.0 to 1.0)
        neg: float (0.0 to 1.0)
        label: 'POSITIVE', 'NEUTRAL', or 'NEGATIVE'
    """
    if not text or not isinstance(text, str):
        return {"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0, "label": "NEUTRAL"}

    scores = _vader.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        label = "POSITIVE"
    elif compound <= -0.05:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"

    return {
        "compound": round(compound, 4),
        "pos": round(scores["pos"], 4),
        "neu": round(scores["neu"], 4),
        "neg": round(scores["neg"], 4),
        "label": label
    }


def compute_nss(pos_count: int, neg_count: int, total_count: int) -> float:
    """
    Calculate Net Sentiment Score (NSS).
    Formula: ((pos_count - neg_count) / total_count) * 100
    Range: -100 to +100
    """
    if total_count <= 0:
        return 0.0
    return round(((pos_count - neg_count) / float(total_count)) * 100.0, 1)


def detect_rating_anomaly(rating: Optional[float], compound: float) -> Dict[str, Any]:
    """
    Flags discrepancies between numeric star rating and textual sentiment.
    Example: 5-star rating with strongly negative text (sarcasm/bot error),
    or 1-star rating with glowing positive text.
    """
    if rating is None or pd.isna(rating):
        return {"is_anomaly": False, "anomaly_type": None, "confidence": 0.0}

    r = float(rating)
    if r >= 4.0 and compound <= -0.30:
        return {
            "is_anomaly": True,
            "anomaly_type": "High Rating / Negative Review (Potential Sarcasm or Defect)",
            "severity": "HIGH" if r == 5.0 and compound <= -0.5 else "MEDIUM"
        }
    elif r <= 2.0 and compound >= 0.30:
        return {
            "is_anomaly": True,
            "anomaly_type": "Low Rating / Positive Review (Misclicked Star or Inversion)",
            "severity": "HIGH" if r == 1.0 and compound >= 0.5 else "MEDIUM"
        }

    return {"is_anomaly": False, "anomaly_type": None, "severity": "NONE"}


def analyze_reviews_sentiment(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run complete sentiment pipeline on a DataFrame of reviews.
    Returns aggregated metrics, distribution, and enriched rows.
    """
    if df.empty or "review_text" not in df.columns:
        return {
            "total_reviews": 0,
            "nss": 0.0,
            "csat_estimate": 0.0,
            "distribution": {"POSITIVE": 0, "NEUTRAL": 0, "NEGATIVE": 0},
            "anomalies_count": 0,
            "records": []
        }

    records = []
    pos_count = 0
    neu_count = 0
    neg_count = 0
    anomalies = 0

    for _, row in df.iterrows():
        text = str(row.get("review_text", ""))
        sent = analyze_text_sentiment(text)
        rating = row.get("review_rating")
        anomaly = detect_rating_anomaly(rating, sent["compound"])

        if sent["label"] == "POSITIVE":
            pos_count += 1
        elif sent["label"] == "NEGATIVE":
            neg_count += 1
        else:
            neu_count += 1

        if anomaly["is_anomaly"]:
            anomalies += 1

        rec = dict(row)
        rec.update({
            "sentiment_compound": sent["compound"],
            "sentiment_label": sent["label"],
            "sentiment_pos": sent["pos"],
            "sentiment_neu": sent["neu"],
            "sentiment_neg": sent["neg"],
            "is_anomaly": anomaly["is_anomaly"],
            "anomaly_type": anomaly.get("anomaly_type")
        })
        records.append(rec)

    total = len(records)
    nss = compute_nss(pos_count, neg_count, total)
    csat = round((pos_count / float(total)) * 100.0, 1) if total > 0 else 0.0

    return {
        "total_reviews": total,
        "nss": nss,
        "csat_estimate": csat,
        "distribution": {
            "POSITIVE": pos_count,
            "NEUTRAL": neu_count,
            "NEGATIVE": neg_count
        },
        "anomalies_count": anomalies,
        "records": records
    }
