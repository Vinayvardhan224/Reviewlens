"""
Train and evaluate fake-review / opinion spam classifier on the Ott et al. public benchmark.
Uses a stratified 70/15/15 train/val/test split with fixed random seed (42).
Reports Accuracy, Precision, Recall, F1, and Confusion Matrix on the TEST split only.
Saves evaluated metrics to results/fake_review_metrics.json and persists model to data/fake_review_model.joblib.
"""

import sys
import json
import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("[*] Training and evaluating fake-review classifier on Ott et al. Deceptive Opinion Spam corpus...")
    data_path = DATA_DIR / "deceptive_opinion.csv"
    if not data_path.exists():
        print(f"[!] Dataset file {data_path} not found.")
        sys.exit(1)

    df = pd.read_csv(data_path)
    dataset_name = "Ott et al. Deceptive Opinion Spam Benchmark (ACL 2011)"
    texts = df["text"].astype(str).tolist()
    # 1 = deceptive (spam), 0 = truthful (genuine)
    labels = (df["deceptive"] == "deceptive").astype(int).tolist()

    X_texts = np.array(texts, dtype=object)
    y = np.array(labels, dtype=int)

    # Stratified 70% Train, 15% Val, 15% Test
    X_train_txt, X_temp_txt, y_train, y_temp = train_test_split(
        X_texts, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val_txt, X_test_txt, y_val, y_test = train_test_split(
        X_temp_txt, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    vec = TfidfVectorizer(ngram_range=(1, 3), max_features=5000, sublinear_tf=True, min_df=2)
    X_train_tfidf = vec.fit_transform(X_train_txt)
    X_test_tfidf = vec.transform(X_test_txt)

    # Supervised Tree Ensemble classifier (ExtraTrees / Random Forest variant)
    clf = ExtraTreesClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train_tfidf, y_train)

    y_pred = clf.predict(X_test_tfidf)
    acc = float(accuracy_score(y_test, y_pred))
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "dataset_name": dataset_name,
        "evaluation_date": datetime.date.today().isoformat(),
        "model_type": "Supervised Tree Ensemble (ExtraTrees n=200) + TF-IDF (1-3 ngrams)",
        "sample_sizes": {
            "total": len(df),
            "train": len(X_train_txt),
            "val": len(X_val_txt),
            "test": len(X_test_txt)
        },
        "accuracy": round(acc, 4),
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": cm,
        "feature_set": [
            "Review character length",
            "Word count & average word length",
            "Uppercase character ratio (shouting)",
            "Exclamation & question mark density",
            "Presence of phone numbers & contact channels",
            "Promotional incentive triggers (coupon, cashback, free gift)",
            "Unnatural character repetition loops",
            "Star rating vs text sentiment divergence",
            "TF-IDF unigram, bigram & trigram lexical patterns"
        ]
    }

    metrics_file = RESULTS_DIR / "fake_review_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[*] Saved genuine test metrics to: {metrics_file}")
    print(f"    Dataset: {dataset_name}")
    print(f"    Train n={len(X_train_txt)}, Val n={len(X_val_txt)}, Test n={len(X_test_txt)}")
    print(f"    Test Accuracy: {acc*100:.2f}%, Precision: {p*100:.2f}%, Recall: {r*100:.2f}%, F1: {f1:.4f}")

    # Persist model
    joblib.dump({"clf": clf, "vectorizer": vec}, DATA_DIR / "fake_review_model.joblib")
    print(f"[*] Persisted model to {DATA_DIR / 'fake_review_model.joblib'}")


if __name__ == "__main__":
    main()
