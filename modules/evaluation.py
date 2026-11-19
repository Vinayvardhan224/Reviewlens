"""
ReviewLens Model Evaluation & Benchmark Module.
Evaluates VADER sentiment classification and Spam Detection against a small hand-labelled
sanity-check test set, reporting Accuracy, Precision, Recall, F1-score, and Confusion Matrices.
"""

from typing import Dict, Any, List
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from modules.sentiment import analyze_text_sentiment
from reviewlens.spam_detector.detector import SpamDetector

BENCHMARK_CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "benchmark_sentiment.csv"
SST2_CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "public_sst2_benchmark.csv"

# Fallback curated benchmark test set (50 reviews with ground-truth sentiment and spam labels)
BENCHMARK_DATASET = [
    # Positive reviews
    ("The battery life on this laptop easily lasts 15 hours and the M3 chip is blazing fast.", "POSITIVE", False),
    ("Superb image quality with breathtaking dynamic range and sharp optics.", "POSITIVE", False),
    ("Active noise cancellation silences all airplane cabin hum completely. 10/10 sound.", "POSITIVE", False),
    ("Gorgeous 120Hz AMOLED display with punchy colors and high outdoor peak brightness.", "POSITIVE", False),
    ("Fast 100W charging is revolutionary. 0 to 100% in 25 minutes flat.", "POSITIVE", False),
    ("Extremely comfortable ear cushions, zero fatigue after an entire 8-hour shift.", "POSITIVE", False),
    ("Autofocus locks onto moving birds and pets instantaneously. Unbelievable camera.", "POSITIVE", False),
    ("The keyboard tactile feedback is crisp and satisfying for programming.", "POSITIVE", False),
    ("Seamless Bluetooth multipoint connection switches between phone and laptop smoothly.", "POSITIVE", False),
    ("Great value for money. Premium build with titanium rails at an affordable price.", "POSITIVE", False),
    ("Audio clarity on calls is pristine even when walking down noisy city streets.", "POSITIVE", False),
    ("Low-light performance at ISO 12800 is clean with remarkably low noise.", "POSITIVE", False),
    ("One UI 6.1 software is smooth with useful AI translation and search features.", "POSITIVE", False),
    ("The cooling system keeps GPU temps under 70C during intense 4K gaming.", "POSITIVE", False),
    ("Solid construction with Gorilla Glass Victus+ back that survives drops.", "POSITIVE", False),
    ("Exceeded expectations in every dimension. Truly a fantastic everyday workhorse.", "POSITIVE", False),
    ("Brilliant stereo speakers with wide soundstage and rich low-end bass.", "POSITIVE", False),
    ("Fast delivery and packaging was secure with factory seal intact.", "POSITIVE", False),

    # Negative reviews
    ("Terrible battery drain, drops 20% in an hour of basic web browsing. Total waste of money.", "NEGATIVE", False),
    ("The phone overheats constantly and lags badly whenever the camera app is opened.", "NEGATIVE", False),
    ("Earcups feel like a vice clamp on my head. Uncomfortable and headache inducing.", "NEGATIVE", False),
    ("Horrible 20W charging takes nearly 2 hours to charge completely. Completely unacceptable.", "NEGATIVE", False),
    ("Screen has terrible viewing angles with noticeable color shifting and glare.", "NEGATIVE", False),
    ("Fan noise is deafening like a vacuum cleaner under moderate CPU loads.", "NEGATIVE", False),
    ("Autofocus hunts endlessly in low light and misses focus on stationary subjects.", "NEGATIVE", False),
    ("The flat keyboard layout causes constant typos. Terrible typing ergonomics.", "NEGATIVE", False),
    ("Cheap flimsy plastic chassis that creaks and scratches upon touching.", "NEGATIVE", False),
    ("Bluetooth disconnects every 5 minutes and drops audio constantly.", "NEGATIVE", False),
    ("Severely overpriced for what it offers. Competitors offer double the performance.", "NEGATIVE", False),
    ("Customer support was rude and refused warranty replacement for a defective hinge.", "NEGATIVE", False),
    ("Camera photos are blurry, grainy, and washed out with muted colors.", "NEGATIVE", False),
    ("The device bricked and became completely unresponsive after the latest firmware update.", "NEGATIVE", False),
    ("Extremely heavy and bulky in pocket. Disappointed with the build ergonomics.", "NEGATIVE", False),
    ("Microphone sounds muffled as if speaking underwater during Zoom calls.", "NEGATIVE", False),

    # Neutral / Informational reviews
    ("The phone weighs 213 grams and includes a USB-C to USB-C cable in the box.", "NEUTRAL", False),
    ("It comes in silver, space black, and starlight color options.", "NEUTRAL", False),
    ("The laptop measures 11.5mm in thickness and has two Thunderbolt 4 ports.", "NEUTRAL", False),
    ("Standard 3.5mm audio jack is located on the right side of the device.", "NEUTRAL", False),
    ("Camera supports 4K at 30fps and 60fps recording modes.", "NEUTRAL", False),
    ("Battery capacity is rated at 4700 mAh with standard lithium-ion chemistry.", "NEUTRAL", False),
    ("It uses Bluetooth 5.3 protocol and supports AAC and SBC codecs.", "NEUTRAL", False),
    ("The package dimensions are 15 by 10 by 3 centimeters.", "NEUTRAL", False),
    ("The box contains the laptop, a 70W USB-C power adapter, and a 2m braided charging cable.", "NEUTRAL", False),

    # Spam / Fake reviews (naturalistic evaluation cases: incentivized reviews, promotional links, character loops, shouting)
    ("The merchant offered a free gift card if I posted a positive rating for this listing.", "POSITIVE", True),
    ("You can find a 30% discount code for this model online at promo-savings-hub.com today.", "POSITIVE", True),
    ("This phone is sooooo good and fast, battery lasts foreverrrrr!", "POSITIVE", True),
    ("DO NOT PURCHASE THIS ITEM AS IT ARRIVED DAMAGED AND CUSTOMER SERVICE WAS UNRESPONSIVE!", "NEGATIVE", True),
    ("Contact seller on WhatsApp at +1-800-555-0199 for free replacement parts.", "NEUTRAL", True),
    ("The seller sent an email promising a cash back reward for a five star rating.", "POSITIVE", True),
    ("FIVE STARS FOR FREE ACCESSORIES VISIT OUR STORE PAGE FOR SPECIAL BONUS DEALS!!!!", "POSITIVE", True)
]


def load_benchmark_dataset() -> List[tuple]:
    """
    Loads small hand-labelled benchmark dataset from data/benchmark_sentiment.csv.
    Falls back to curated dataset if file is absent or fails to load.
    """
    if BENCHMARK_CSV_PATH.exists():
        try:
            df = pd.read_csv(BENCHMARK_CSV_PATH)
            records = []
            for _, row in df.iterrows():
                is_spam = str(row["is_spam"]).strip().lower() in ("true", "1", "yes")
                records.append((str(row["text"]), str(row["ground_truth_sentiment"]).strip().upper(), is_spam))
            if len(records) > 0:
                return records
        except Exception:
            pass
    return BENCHMARK_DATASET


def evaluate_sentiment_model() -> Dict[str, Any]:
    """
    Evaluates VADER sentiment against the small hand-labelled benchmark dataset.
    Returns accuracy, precision, recall, F1, and a 3x3 confusion matrix.
    """
    dataset = load_benchmark_dataset()
    y_true = []
    y_pred = []

    for text, true_label, _ in dataset:
        y_true.append(true_label)
        res = analyze_text_sentiment(text)
        y_pred.append(res["label"])

    labels = ["POSITIVE", "NEUTRAL", "NEGATIVE"]

    acc = round(float(accuracy_score(y_true, y_pred)), 4)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    p_class, r_class, f1_class, support = precision_recall_fscore_support(y_true, y_pred, average=None, labels=labels, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_class_metrics = {}
    for idx, l in enumerate(labels):
        per_class_metrics[l] = {
            "precision": round(float(p_class[idx]), 3),
            "recall": round(float(r_class[idx]), 3),
            "f1": round(float(f1_class[idx]), 3),
            "support": int(support[idx])
        }

    return {
        "total_test_samples": len(dataset),
        "accuracy": acc,
        "precision_macro": round(float(p_macro), 3),
        "recall_macro": round(float(r_macro), 3),
        "f1_macro": round(float(f1_macro), 3),
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class_metrics
    }


def evaluate_spam_model() -> Dict[str, Any]:
    """
    Evaluates the SpamDetector against the benchmark dataset, measuring
    precision, recall, false positives, and false negatives.
    """
    detector = SpamDetector()
    dataset = load_benchmark_dataset()
    y_true = []
    y_pred = []
    false_positives = []
    false_negatives = []

    for text, sent_label, is_spam_true in dataset:
        y_true.append(1 if is_spam_true else 0)
        sent_res = analyze_text_sentiment(text)
        spam_res = detector.analyze(text, sentiment_compound=sent_res["compound"])
        pred_spam = 1 if spam_res.is_spam else 0
        y_pred.append(pred_spam)

        if pred_spam == 1 and not is_spam_true:
            false_positives.append({"text": text, "reasons": spam_res.reasons})
        elif pred_spam == 0 and is_spam_true:
            false_negatives.append({"text": text, "score": spam_res.spam_score})

    acc = round(float(accuracy_score(y_true, y_pred)), 4)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

    return {
        "total_test_samples": len(dataset),
        "accuracy": acc,
        "precision": round(float(p), 3),
        "recall": round(float(r), 3),
        "f1_score": round(float(f1), 3),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "false_positive_count": len(false_positives),
        "false_negative_count": len(false_negatives)
    }


REVIEWS_CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "reviews.csv"

EVALUATION_METHODOLOGY_DISCLOSURE = """
**Evaluation Methodology & Caveats:**
- **Small Hand-Labelled Sanity Check Corpus:** The evaluation suite benchmarks VADER against a small hand-labelled sanity check corpus (n=50) in `data/benchmark_sentiment.csv`, with 23 positive, 17 negative, and 10 neutral sentences (including 7 containing spam indicators).
- **Small Sample Disclaimer:** These evaluation results are computed on a small hand-crafted sample of 50 benchmark sentences and 101 catalogue demonstration reviews. They serve as regression validation checks and a demonstration of the evaluation methodology rather than empirical proof of generalization on large-scale production distributions (e.g. Stanford SST-2 or Amazon Reviews Polarity with 100,000+ samples).
- **Supervised Baseline:** A TF-IDF (1-2 ngrams, 200 features) + Logistic Regression baseline is evaluated via 5-Fold Stratified Cross-Validation on the 50-sentence benchmark corpus as an illustrative comparative exercise against rule-based VADER.
"""


def evaluate_curated_reviews_corpus() -> Dict[str, Any]:
    """
    Evaluates rule-based VADER sentiment against star ratings in the 101-review dataset (data/reviews.csv).
    Ratings >= 4.0 are mapped to POSITIVE, <= 2.0 to NEGATIVE, and 3.0 to NEUTRAL.
    Returns accuracy, agreement count, and total samples.
    """
    if not REVIEWS_CSV_PATH.exists():
        return {"status": "error", "message": "reviews.csv not found", "sample_size": 0, "agreement_rate": 0.0}

    try:
        df = pd.read_csv(REVIEWS_CSV_PATH)
        text_col = "review_text" if "review_text" in df.columns else "text"
        rating_col = "rating" if "rating" in df.columns else "review_rating"

        y_true = []
        y_pred = []

        for _, row in df.iterrows():
            r = float(row[rating_col])
            t = str(row[text_col])
            if r >= 4.0:
                expected = "POSITIVE"
            elif r <= 2.0:
                expected = "NEGATIVE"
            else:
                expected = "NEUTRAL"

            pred = analyze_text_sentiment(t)["label"]
            y_true.append(expected)
            y_pred.append(pred)

        acc = round(float(accuracy_score(y_true, y_pred)), 4)
        return {
            "status": "ok",
            "sample_size": len(df),
            "agreement_rate": acc,
            "positive_count": sum(1 for y in y_true if y == "POSITIVE"),
            "neutral_count": sum(1 for y in y_true if y == "NEUTRAL"),
            "negative_count": sum(1 for y in y_true if y == "NEGATIVE"),
            "disclaimer": "Evaluated on curated demonstration reviews (n=101 across 12 products). User star ratings used as proxy ground truth."
        }
    except Exception as e:
        return {"status": "error", "message": str(e), "sample_size": 0, "agreement_rate": 0.0}


def evaluate_baseline_comparison() -> Dict[str, Any]:
    """
    Evaluates a supervised TF-IDF + Logistic Regression baseline model on the benchmark corpus
    using 5-fold Stratified Cross-Validation for comparison against VADER rule-based sentiment.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    dataset = load_benchmark_dataset()
    texts = [t for t, l, _ in dataset]
    labels = [l for t, l, _ in dataset]

    vec = TfidfVectorizer(max_features=200, ngram_range=(1, 2))
    X = vec.fit_transform(texts)
    y = np.array(labels)

    clf = LogisticRegression(max_iter=200, random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    acc_scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy")
    f1_scores = cross_val_score(clf, X, y, cv=skf, scoring="f1_macro")

    vader_res = evaluate_sentiment_model()

    return {
        "baseline_model": "TF-IDF (1-2 ngrams) + Logistic Regression (5-Fold Stratified CV)",
        "baseline_accuracy": round(float(np.mean(acc_scores)), 3),
        "baseline_f1_macro": round(float(np.mean(f1_scores)), 3),
        "vader_accuracy": vader_res["accuracy"],
        "vader_f1_macro": vader_res["f1_macro"],
        "sample_size": len(dataset),
        "methodology": EVALUATION_METHODOLOGY_DISCLOSURE
    }


def evaluate_sst2_public_benchmark() -> Dict[str, Any]:
    """
    Evaluates ReviewLens sentiment models on the official Stanford Sentiment Treebank
    (SST-2 / GLUE development set, n=872 samples).
    Provides genuine empirical accuracy against an independent, public NLP benchmark.
    """
    if not SST2_CSV_PATH.exists():
        return {
            "status": "not_found",
            "message": "SST-2 benchmark file data/public_sst2_benchmark.csv not found.",
            "sample_size": 0
        }

    try:
        df = pd.read_csv(SST2_CSV_PATH)
        y_true = df["ground_truth_sentiment"].tolist()
        texts = df["text"].tolist()

        y_pred_all = []
        for t in texts:
            y_pred_all.append(analyze_text_sentiment(t)["label"])

        acc_all = round(float(accuracy_score(y_true, y_pred_all)), 4)

        # Binary directional evaluation (excluding neutral predictions)
        binary_pairs = [(t, p) for t, p in zip(y_true, y_pred_all) if p in ("POSITIVE", "NEGATIVE")]
        acc_binary = round(float(accuracy_score([t for t, p in binary_pairs], [p for t, p in binary_pairs])), 4) if binary_pairs else 0.0

        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred_all, average="macro", labels=["POSITIVE", "NEGATIVE"], zero_division=0
        )

        # Supervised baseline on SST-2 via 5-fold Stratified CV
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score

        vec = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        X = vec.fit_transform(texts)
        y_arr = np.array(y_true)
        clf = LogisticRegression(max_iter=300, random_state=42)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        baseline_acc_scores = cross_val_score(clf, X, y_arr, cv=skf, scoring="accuracy")
        baseline_f1_scores = cross_val_score(clf, X, y_arr, cv=skf, scoring="f1_macro")

        return {
            "status": "ok",
            "benchmark_name": "Stanford Sentiment Treebank (SST-2 / GLUE Dev Set)",
            "sample_size": len(df),
            "positive_count": sum(1 for y in y_true if y == "POSITIVE"),
            "negative_count": sum(1 for y in y_true if y == "NEGATIVE"),
            "vader_accuracy_all": acc_all,
            "vader_accuracy_binary": acc_binary,
            "vader_neutral_count": sum(1 for p in y_pred_all if p == "NEUTRAL"),
            "vader_precision_macro": round(float(p_macro), 3),
            "vader_recall_macro": round(float(r_macro), 3),
            "vader_f1_macro": round(float(f1_macro), 3),
            "baseline_accuracy": round(float(np.mean(baseline_acc_scores)), 4),
            "baseline_f1_macro": round(float(np.mean(baseline_f1_scores)), 4),
            "data_provenance": "Official Stanford Sentiment Treebank (SST-2 / GLUE development set). Genuine public NLP research benchmark."
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "sample_size": 0
        }
