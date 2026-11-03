"""
ReviewLens Model Evaluation & Benchmark Module.
Evaluates VADER sentiment classification and Spam Detection against a manually labelled
gold-standard test set, reporting Accuracy, Precision, Recall, F1-score, and Confusion Matrices.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from modules.sentiment import analyze_text_sentiment
from reviewlens.spam_detector.detector import SpamDetector

# Manually labelled benchmark test set (50 reviews with ground-truth sentiment and spam labels)
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

    # Spam / Fake reviews
    ("BUY NOW FREE GIFT ON WHATSAPP FIVE STARS CLICK HERE FOR 100 PERCENT CASH BACK BONUS!!!!", "POSITIVE", True),
    ("VISIT WWW.FREE-COUPONS-DEAL.COM GET FREE IPHONE 15 NOW DISCOUNT CODE WHATSAPP!!", "POSITIVE", True),
    ("BEST BEST BEST BEST BEST BEST BEST BEST BEST SOOOOO GOOOOOD AAAAAAA", "POSITIVE", True),
    ("I love this product it is amazing. I love this product it is amazing.", "POSITIVE", True),
    ("CHECK MY TELEGRAM @FREEGIFTS FOR FREE REPLACEMENT AND 500 RUPEES CASHBACK GUARANTEED!!!", "POSITIVE", True),
    ("FIVE STARS FIVE STARS CALL 9876543210 FOR FREE ACCESSORY GIFT BUNDLE DISCOUNT", "POSITIVE", True),
    ("CLAIM 5000 RUPEES DISCOUNT COUPON CODE NOW ON OUR TELEGRAM LINK IN BIO 100 PERCENT REAL", "POSITIVE", True)
]


def evaluate_sentiment_model() -> Dict[str, Any]:
    """
    Evaluates VADER sentiment against the manually labelled ground-truth dataset.
    Returns accuracy, precision, recall, F1, and a 3x3 confusion matrix.
    """
    y_true = []
    y_pred = []

    for text, true_label, _ in BENCHMARK_DATASET:
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
        "total_test_samples": len(BENCHMARK_DATASET),
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
    y_true = []
    y_pred = []
    false_positives = []
    false_negatives = []

    for text, sent_label, is_spam_true in BENCHMARK_DATASET:
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
        "total_test_samples": len(BENCHMARK_DATASET),
        "accuracy": acc,
        "precision": round(float(p), 3),
        "recall": round(float(r), 3),
        "f1_score": round(float(f1), 3),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "false_positive_count": len(false_positives),
        "false_negative_count": len(false_negatives)
    }
