"""
Spam, Bot, and Fake Review Detection Engine for ReviewLens.
Analyzes reviews for synthetic patterns, promotional manipulation, rating mismatches, and copy-paste duplicates.
"""

from typing import List, Optional, Set, Dict, Any
import re
import datetime
import threading
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reviewlens.config import SPAM_DETECTION_CONFIG
from reviewlens.models import SpamResult


def detect_review_bursts(
    reviews: List[dict],
    window_days: int = 2,
    threshold_count: int = 5
) -> List[Dict[str, Any]]:
    """
    Detect unnatural temporal velocity bursts (sudden spikes in review submission
    within a short window), typical of coordinated astroturfing campaigns.

    Args:
        reviews: List of review dicts containing 'date' or 'timestamp', and optionally 'product_id', 'review_id'.
        window_days: Sliding window duration in days (default 2).
        threshold_count: Minimum review count within the window to flag as a burst (default 5).

    Returns:
        List of detected burst clusters with start_date, end_date, count, review_ids, product_id.
    """
    if not reviews or len(reviews) < threshold_count:
        return []

    # Parse timestamps
    parsed_reviews = []
    for idx, r in enumerate(reviews):
        raw_date = r.get("date") or r.get("timestamp")
        if not raw_date:
            continue
        try:
            ts = pd.to_datetime(raw_date)
            if pd.notna(ts):
                rid = str(r.get("review_id", f"REV_{idx}"))
                pid = str(r.get("product_id", "default_product"))
                parsed_reviews.append({
                    "timestamp": ts,
                    "review_id": rid,
                    "product_id": pid
                })
        except Exception:
            continue

    if len(parsed_reviews) < threshold_count:
        return []

    # Group by product_id
    by_product: Dict[str, List[dict]] = {}
    for pr in parsed_reviews:
        by_product.setdefault(pr["product_id"], []).append(pr)

    bursts: List[Dict[str, Any]] = []
    for pid, p_revs in by_product.items():
        if len(p_revs) < threshold_count:
            continue
        p_revs.sort(key=lambda x: x["timestamp"])

        i = 0
        while i < len(p_revs):
            start_ts = p_revs[i]["timestamp"]
            end_ts = start_ts + datetime.timedelta(days=window_days)
            window_revs = [r for r in p_revs if start_ts <= r["timestamp"] <= end_ts]

            if len(window_revs) >= threshold_count:
                bursts.append({
                    "product_id": pid,
                    "start_date": start_ts.strftime("%Y-%m-%d"),
                    "end_date": end_ts.strftime("%Y-%m-%d"),
                    "count": len(window_revs),
                    "review_ids": [r["review_id"] for r in window_revs]
                })
                # Skip forward past this window to avoid duplicate sub-bursts
                i += len(window_revs)
            else:
                i += 1

    return bursts


def detect_near_duplicates(
    reviews: List[dict],
    similarity_threshold: float = 0.85
) -> List[Dict[str, Any]]:
    """
    Detect copy-pasted or template-generated near-duplicate reviews across distinct IDs
    using TF-IDF n-gram vectorization and cosine similarity.

    Args:
        reviews: List of review dicts with 'review_text' or 'text', and optionally 'review_id'.
        similarity_threshold: Cosine similarity cutoff to classify as near-duplicate (default 0.85).

    Returns:
        List of duplicate pairs with review_id_a, review_id_b, similarity, text_a, text_b.
    """
    if not reviews or len(reviews) < 2:
        return []

    items = []
    for idx, r in enumerate(reviews):
        text = str(r.get("review_text") or r.get("text") or "").strip()
        rid = str(r.get("review_id", f"REV_{idx}"))
        if len(text) >= 15:
            items.append({"id": rid, "text": text})

    if len(items) < 2:
        return []

    corpus = [item["text"] for item in items]
    try:
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        tfidf_mat = vec.fit_transform(corpus)
        sim_mat = cosine_similarity(tfidf_mat)
    except Exception:
        return []

    duplicates = []
    n = len(items)
    for i in range(n):
        for j in range(i + 1, n):
            # Exclude self-comparison if same review_id
            if items[i]["id"] == items[j]["id"]:
                continue
            sim = float(sim_mat[i, j])
            if sim >= similarity_threshold:
                duplicates.append({
                    "review_id_a": items[i]["id"],
                    "review_id_b": items[j]["id"],
                    "similarity": round(sim, 3),
                    "text_a": items[i]["text"][:90],
                    "text_b": items[j]["text"][:90]
                })

    return duplicates


_ML_SPAM_MODEL_CACHE = None
_ML_SPAM_MODEL_LOADED = False
_ML_SPAM_LOCK = threading.Lock()


def _get_ml_spam_model():
    """Lazy loader for persisted statistical ML fake-review model."""
    global _ML_SPAM_MODEL_CACHE, _ML_SPAM_MODEL_LOADED
    if not _ML_SPAM_MODEL_LOADED:
        with _ML_SPAM_LOCK:
            if not _ML_SPAM_MODEL_LOADED:
                try:
                    import joblib
                    from pathlib import Path
                    model_path = Path(__file__).resolve().parent.parent.parent / "data" / "fake_review_model.joblib"
                    if model_path.exists():
                        _ML_SPAM_MODEL_CACHE = joblib.load(model_path)
                except Exception:
                    _ML_SPAM_MODEL_CACHE = None
                _ML_SPAM_MODEL_LOADED = True
    return _ML_SPAM_MODEL_CACHE


class SpamDetector:
    """Detects spam, incentivized, or untrustworthy product reviews."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or SPAM_DETECTION_CONFIG
        self.seen_signatures: Set[str] = set()
        self._lock = threading.Lock()

    def _normalize_signature(self, text: str) -> str:
        """Create a simplified hashable signature for duplicate checking."""
        return re.sub(r"[^a-zA-Z0-9]", "", text.lower())

    def analyze(
        self,
        text: str,
        rating: Optional[float] = None,
        sentiment_compound: Optional[float] = None,
        check_duplicate: bool = True
    ) -> SpamResult:
        """
        Evaluate a single review against multi-point spam heuristics.
        
        Args:
            text: Raw review content.
            rating: Optional user star rating (1-5).
            sentiment_compound: Optional computed sentiment compound score (-1.0 to 1.0).
            check_duplicate: Whether to compare against previously seen reviews in this session.
            
        Returns:
            SpamResult with boolean decision, continuous score, and reasons.
        """
        if not text or not text.strip():
            return SpamResult(is_spam=True, spam_score=1.0, reasons=["Empty review"])

        reasons: List[str] = []
        spam_points = 0.0
        text_len = len(text.strip())

        # 1. Extremely short review
        if text_len < self.config["min_review_length_chars"]:
            spam_points += 0.25
            reasons.append(f"Excessively brief review (<{self.config['min_review_length_chars']} chars)")

        # 2. Uppercase ratio (Shouting)
        alphabetic_chars = [c for c in text if c.isalpha()]
        if alphabetic_chars:
            upper_ratio = sum(1 for c in alphabetic_chars if c.isupper()) / len(alphabetic_chars)
            if upper_ratio > self.config["max_uppercase_ratio"] and len(alphabetic_chars) > 20:
                spam_points += 0.35
                reasons.append(f"Excessive capitalization ({upper_ratio:.0%} uppercase)")

        # 3. Punctuation repetition (e.g. !!!!! or ?????)
        punct_match = re.search(r"([!?,.]){" + str(self.config["max_punctuation_repetition"]) + r",}", text)
        if punct_match:
            spam_points += 0.25
            reasons.append(f"Repetitive punctuation pattern: '{punct_match.group(0)}'")

        # 4. Known promotional / spam triggers
        lower_text = text.lower()
        for phrase in self.config["spam_phrases"]:
            if phrase in lower_text:
                spam_points += 0.50
                reasons.append(f"Contains promotional/spam phrase: '{phrase}'")
                break

        has_mismatch = False

        # 5. Rating vs Sentiment Inconsistency (Data quality flag & moderate suspicion)
        if rating is not None and sentiment_compound is not None:
            # High rating (>=4.0) but deeply negative sentiment (<=-0.30)
            if rating >= 4.0 and sentiment_compound <= -0.30:
                spam_points += 0.30
                has_mismatch = True
                reasons.append(f"Rating-sentiment mismatch ({rating}★ rating paired with negative text sentiment compound {sentiment_compound:.2f})")
            # Low rating (<=2.0) but deeply positive sentiment (>=0.30)
            elif rating <= 2.0 and sentiment_compound >= 0.30:
                spam_points += 0.30
                has_mismatch = True
                reasons.append(f"Rating-sentiment mismatch ({rating}★ rating paired with positive text sentiment compound {sentiment_compound:.2f})")

        # 6. Repetitive character loops (e.g. "sooooo goooood" or "aaaaaaa")
        char_repetition = re.search(r"(.)\1{5,}", text)
        if char_repetition:
            spam_points += 0.20
            reasons.append("Contains unnatural repeated character sequences")

        # 7. Duplicate detection across session/batch
        if check_duplicate:
            sig = self._normalize_signature(text)
            if len(sig) > 20:
                with self._lock:
                    if sig in self.seen_signatures:
                        spam_points += 0.70
                        reasons.append("Exact or near-identical duplicate of an earlier review")
                    else:
                        self.seen_signatures.add(sig)

        # 8. ML Fake-Review Model Signal (Trained on Ott et al. benchmark)
        ml_prob: Optional[float] = None
        ml_caveat: Optional[str] = None
        ml_model = _get_ml_spam_model()
        if ml_model is not None and text.strip():
            try:
                vec = ml_model.get("vectorizer")
                clf = ml_model.get("clf")
                if vec is not None and clf is not None:
                    X_vec = vec.transform([text])
                    if hasattr(clf, "predict_proba"):
                        probs = clf.predict_proba(X_vec)[0]
                        deceptive_idx = 1 if len(probs) > 1 else 0
                        ml_prob = round(float(probs[deceptive_idx]), 4)
                        ml_caveat = (
                            "Trained on Ott et al. (ACL 2011) deceptive hotel reviews; "
                            "Amazon electronics review accuracy may vary due to domain shift."
                        )
                        if ml_prob >= 0.75:
                            reasons.append(
                                f"Statistical ML Fake-Review Model flagged deceptive patterns (probability: {ml_prob:.1%}) [Ott et al. benchmark]"
                            )
            except Exception:
                pass

        normalized_score = min(1.0, round(spam_points, 2))
        is_spam = normalized_score >= 0.50

        return SpamResult(
            is_spam=is_spam,
            spam_score=normalized_score,
            has_mismatch=has_mismatch,
            reasons=reasons,
            ml_spam_prob=ml_prob,
            ml_domain_caveat=ml_caveat
        )

    def reset_session(self):
        """Clear duplicate signature memory safely."""
        with self._lock:
            self.seen_signatures.clear()

    def detect_review_bursts(
        self,
        reviews: List[dict],
        window_days: int = 2,
        threshold_count: int = 5
    ) -> List[Dict[str, Any]]:
        """Instance wrapper for temporal velocity burst detection."""
        return detect_review_bursts(reviews, window_days=window_days, threshold_count=threshold_count)

    def detect_near_duplicates(
        self,
        reviews: List[dict],
        similarity_threshold: float = 0.85
    ) -> List[Dict[str, Any]]:
        """Instance wrapper for TF-IDF near-duplicate detection."""
        return detect_near_duplicates(reviews, similarity_threshold=similarity_threshold)
