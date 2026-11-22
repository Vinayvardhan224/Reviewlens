"""
ReviewLens Real Data Training & Empirical Model Comparison Script
Loads a real review dataset from data/, trains TF-IDF + Logistic Regression,
tunes hyperparameter C on the validation split, evaluates against VADER on the held-out test split,
saves comparison metrics to results/model_comparison.json & results/model_comparison.md,
generates confusion matrix plots, exports error analysis cases, and persists the model to models/.
"""

import os
import sys
import json
import re
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import ReviewLens core VADER analyzer
from reviewlens.sentiment.vader_analyzer import SentimentAnalyzer

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_LABELS = ["negative", "neutral", "positive"]

NEGATION_PATTERNS = [
    r"\bnot\b", r"\bnever\b", r"\bno\b", r"\bhardly\b", r"\bbarely\b",
    r"\bscarcely\b", r"n't\b", r"\bcannot\b", r"\bneither\b", r"\bnor\b"
]
CONTRASTIVE_PATTERNS = [
    r"\bbut\b", r"\bhowever\b", r"\balthough\b", r"\bthough\b",
    r"\byet\b", r"\bexcept\b", r"\bdespite\b", r"\binstead\b"
]


def load_dataset():
    """
    Loads real review dataset from data/.
    Prefers real_amazon_reviews.csv, then reviews.csv, falling back to public_sst2_benchmark.csv.
    """
    amz_path = DATA_DIR / "real_amazon_reviews.csv"
    if amz_path.exists():
        print(f"[*] Loading real customer review dataset: {amz_path.name}")
        df = pd.read_csv(amz_path)
        text_col = "review_text" if "review_text" in df.columns else "text"
        rating_col = "rating"
        dataset_name = "UCSD Amazon Electronics Review Corpus"
        is_sst2 = False
    else:
        sst2_path = DATA_DIR / "public_sst2_benchmark.csv"
        if sst2_path.exists():
            print(f"[!] Falling back to public SST-2 benchmark: {sst2_path.name}")
            df = pd.read_csv(sst2_path)
            text_col = "text"
            rating_col = "numeric_label"
            dataset_name = "Stanford SST-2 Sentiment Benchmark (Dev Split)"
            is_sst2 = True
        else:
            raise FileNotFoundError("No benchmark or real review dataset found in data/")

    # Drop nulls
    df = df.dropna(subset=[text_col]).copy()
    if not is_sst2 and rating_col in df.columns:
        df = df.dropna(subset=[rating_col]).copy()
        def map_rating(r):
            try:
                val = float(r)
                if val <= 2.0:
                    return "negative"
                elif val == 3.0:
                    return "neutral"
                else:
                    return "positive"
            except (ValueError, TypeError):
                return "neutral"
        df["target_label"] = df[rating_col].apply(map_rating)
        df["clean_rating"] = df[rating_col].astype(float)
    elif is_sst2:
        # Binary SST-2 maps 0 -> negative, 1 -> positive
        def map_sst2(v):
            return "negative" if int(v) == 0 else "positive"
        df["target_label"] = df["numeric_label"].apply(map_sst2)
        df["clean_rating"] = df["numeric_label"].apply(lambda v: 1.0 if int(v) == 0 else 5.0)

    df["clean_text"] = df[text_col].astype(str).str.strip()
    df = df[df["clean_text"].str.len() > 3].reset_index(drop=True)

    print(f"[*] Dataset loaded: {dataset_name} (Total rows: {len(df)})")
    print(f"[*] Class distribution:\n{df['target_label'].value_counts().to_string()}\n")
    return df, dataset_name


def guess_misclassification_cause(text: str, true_label: str, pred_label: str, rating: float) -> str:
    """Classifies the linguistic or contextual reason for misclassification."""
    lower = text.lower()
    
    # 1. Short text
    if len(text) < 35 or len(text.split()) < 6:
        return "short_text"
        
    # 2. Ambiguous neutral boundary
    if rating == 3.0 or true_label == "neutral" or pred_label == "neutral":
        return "boundary_neutral"

    # 3. Negation handling failure
    if any(re.search(pat, lower) for pat in NEGATION_PATTERNS):
        return "negation"

    # 4. Mixed sentiment / Contrastive clauses
    if any(re.search(pat, lower) for pat in CONTRASTIVE_PATTERNS):
        return "mixed_sentiment"

    # 5. Sarcasm / Irony signals
    if ('"' in text or "!" in text) and ((true_label == "negative" and "great" in lower) or (true_label == "positive" and "bad" in lower)):
        return "sarcasm"

    return "lexical_ambiguity"


def plot_and_save_confusion_matrix(cm, labels, title, save_path):
    """Plots and saves high-contrast dark-themed confusion matrix."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5), facecolor="#07051A")
    ax.set_facecolor("#120D2A")

    im = ax.imshow(cm, interpolation="nearest", cmap="Purples")
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.yaxis.set_tick_params(color="#F1F5F9")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#F1F5F9")

    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels([l.capitalize() for l in labels], color="#F1F5F9", fontweight="bold", fontsize=10)
    ax.set_yticklabels([l.capitalize() for l in labels], color="#F1F5F9", fontweight="bold", fontsize=10)

    # Annotate numbers
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            ax.text(
                j, i, f"{val}",
                ha="center", va="center",
                color="#07051A" if val > thresh else "#FFFFFF",
                fontsize=12, fontweight="bold"
            )

    ax.set_title(title, color="#FFFFFF", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", color="#C4B5FD", fontsize=10, labelpad=8)
    ax.set_ylabel("True Rating Label", color="#C4B5FD", fontsize=10, labelpad=8)

    # Style spines
    for spine in ax.spines.values():
        spine.set_color("#9B5CFF")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"[*] Confusion matrix saved: {save_path.name}")


def main():
    print("=" * 70)
    print("ReviewLens: Empirical Model Training & Benchmark Comparison")
    print("=" * 70)

    df, dataset_name = load_dataset()
    X = np.array(df["clean_text"].tolist(), dtype=object)
    y = np.array(df["target_label"].tolist(), dtype=object)
    ratings = np.array(df["clean_rating"].tolist(), dtype=float)

    # Fixed 70% Train, 15% Validation, 15% Test split with fixed seed
    # Step 1: Split into Train (70%) and Temp (30%)
    X_train, X_temp, y_train, y_temp, r_train, r_temp = train_test_split(
        X, y, ratings, test_size=0.30, random_state=42, stratify=y
    )
    # Step 2: Split Temp into Validation (15% overall) and Test (15% overall)
    X_val, X_test, y_val, y_test, r_val, r_test = train_test_split(
        X_temp, y_temp, r_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"[*] Split summary: Train={len(X_train)} (70%), Val={len(X_val)} (15%), Test={len(X_test)} (15%)")

    # Tune C on Validation Split
    print("[*] Tuning Logistic Regression regularizer C on validation set...")
    c_candidates = [0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    best_c = 1.0
    best_val_f1 = -1.0

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=12000,
        sublinear_tf=True,
        min_df=2
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)

    for c in c_candidates:
        clf = LogisticRegression(C=c, max_iter=1000, random_state=42, class_weight="balanced")
        clf.fit(X_train_vec, y_train)
        val_preds = clf.predict(X_val_vec)
        _, _, f1_macro, _ = precision_recall_fscore_support(y_val, val_preds, average="macro", zero_division=0)
        print(f"    C={c:<5} -> Val Macro F1: {f1_macro:.4f}")
        if f1_macro > best_val_f1:
            best_val_f1 = f1_macro
            best_c = c

    print(f"[*] Selected optimal C={best_c} (Validation Macro F1: {best_val_f1:.4f})")

    # Fit best model on Train split
    final_clf = LogisticRegression(C=best_c, max_iter=1000, random_state=42, class_weight="balanced")
    final_clf.fit(X_train_vec, y_train)

    # Full scikit-learn Pipeline for persistence & live inference
    full_pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("clf", final_clf)
    ])
    model_save_path = MODELS_DIR / "sentiment_model.joblib"
    joblib.dump(full_pipeline, model_save_path)
    print(f"[*] Trained ML Pipeline saved to: {model_save_path}")

    # Evaluate Logistic Regression on Test Split
    lr_preds = final_clf.predict(X_test_vec)
    lr_acc = float(accuracy_score(y_test, lr_preds))
    lr_prec, lr_rec, lr_f1, _ = precision_recall_fscore_support(y_test, lr_preds, average="macro", zero_division=0)
    lr_p_class, lr_r_class, lr_f_class, _ = precision_recall_fscore_support(y_test, lr_preds, labels=CLASS_LABELS, average=None, zero_division=0)
    lr_cm = confusion_matrix(y_test, lr_preds, labels=CLASS_LABELS)

    # Evaluate VADER Rule Engine on the exact same Test Split
    print("[*] Running VADER rule-based sentiment engine on test split...")
    vader_analyzer = SentimentAnalyzer()
    vader_preds = []
    for text in X_test:
        v_res = vader_analyzer.analyze(text)
        vader_preds.append(v_res.label.lower())
    vader_preds = np.array(vader_preds)

    v_acc = float(accuracy_score(y_test, vader_preds))
    v_prec, v_rec, v_f1, _ = precision_recall_fscore_support(y_test, vader_preds, average="macro", zero_division=0)
    v_p_class, v_r_class, v_f_class, _ = precision_recall_fscore_support(y_test, vader_preds, labels=CLASS_LABELS, average=None, zero_division=0)
    v_cm = confusion_matrix(y_test, vader_preds, labels=CLASS_LABELS)

    print("\n" + "=" * 70)
    print(f"TEST SPLIT EMPIRICAL RESULTS (N={len(y_test)})")
    print("=" * 70)
    print(f"{'Metric':<25} | {'VADER (Rule-Based)':<20} | {'TF-IDF + LogReg (ML)':<20}")
    print("-" * 70)
    print(f"{'Accuracy':<25} | {v_acc * 100:>18.2f}% | {lr_acc * 100:>18.2f}%")
    print(f"{'Macro Precision':<25} | {v_prec * 100:>18.2f}% | {lr_prec * 100:>18.2f}%")
    print(f"{'Macro Recall':<25} | {v_rec * 100:>18.2f}% | {lr_rec * 100:>18.2f}%")
    print(f"{'Macro F1-Score':<25} | {v_f1:>19.4f}  | {lr_f1:>19.4f} ")
    print("-" * 70)
    for idx, c_name in enumerate(CLASS_LABELS):
        print(f"{c_name.capitalize() + ' F1-Score':<25} | {v_f_class[idx]:>19.4f}  | {lr_f_class[idx]:>19.4f} ")
    print("=" * 70)

    # Plot & Save Confusion Matrices
    plot_and_save_confusion_matrix(
        lr_cm, CLASS_LABELS,
        f"TF-IDF + Logistic Regression (Acc: {lr_acc*100:.1f}%)",
        RESULTS_DIR / "cm_logistic_regression.png"
    )
    plot_and_save_confusion_matrix(
        v_cm, CLASS_LABELS,
        f"VADER Lexicon Engine (Acc: {v_acc*100:.1f}%)",
        RESULTS_DIR / "cm_vader.png"
    )

    # Error Analysis: 15 misclassified examples per model
    error_rows = []
    # Logistic Regression Errors
    lr_err_indices = np.where(lr_preds != y_test)[0]
    for idx in lr_err_indices[:15]:
        cause = guess_misclassification_cause(X_test[idx], y_test[idx], lr_preds[idx], r_test[idx])
        error_rows.append({
            "model": "TF-IDF + Logistic Regression",
            "text": X_test[idx],
            "true_rating": float(r_test[idx]),
            "true_label": y_test[idx],
            "predicted_label": lr_preds[idx],
            "suspected_cause": cause
        })

    # VADER Errors
    vader_err_indices = np.where(vader_preds != y_test)[0]
    for idx in vader_err_indices[:15]:
        cause = guess_misclassification_cause(X_test[idx], y_test[idx], vader_preds[idx], r_test[idx])
        error_rows.append({
            "model": "VADER Lexicon",
            "text": X_test[idx],
            "true_rating": float(r_test[idx]),
            "true_label": y_test[idx],
            "predicted_label": vader_preds[idx],
            "suspected_cause": cause
        })

    err_df = pd.DataFrame(error_rows)
    err_df.to_csv(RESULTS_DIR / "error_analysis.csv", index=False)
    print(f"[*] Error analysis (30 cases) saved: results/error_analysis.csv")

    # Export structured JSON metrics
    metrics_summary = {
        "dataset_name": dataset_name,
        "total_samples": int(len(df)),
        "splits": {
            "train": int(len(X_train)),
            "val": int(len(X_val)),
            "test": int(len(X_test))
        },
        "hyperparameters": {
            "best_c": best_c,
            "ngram_range": [1, 2],
            "max_features": 12000,
            "class_weight": "balanced"
        },
        "vader": {
            "accuracy": round(v_acc, 4),
            "macro_precision": round(float(v_prec), 4),
            "macro_recall": round(float(v_rec), 4),
            "macro_f1": round(float(v_f1), 4),
            "class_f1": {
                c: round(float(f), 4) for c, f in zip(CLASS_LABELS, v_f_class)
            },
            "confusion_matrix": v_cm.tolist()
        },
        "logistic_regression": {
            "accuracy": round(lr_acc, 4),
            "macro_precision": round(float(lr_prec), 4),
            "macro_recall": round(float(lr_rec), 4),
            "macro_f1": round(float(lr_f1), 4),
            "class_f1": {
                c: round(float(f), 4) for c, f in zip(CLASS_LABELS, lr_f_class)
            },
            "confusion_matrix": lr_cm.tolist()
        },
        "delta": {
            "accuracy_gain_pct": round((lr_acc - v_acc) * 100, 2),
            "f1_gain": round(lr_f1 - v_f1, 4)
        }
    }

    with open(RESULTS_DIR / "model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    print("[*] JSON summary saved: results/model_comparison.json")

    # Export Markdown comparison table
    md_table = f"""# Empirical Sentiment Model Evaluation Benchmark

**Dataset:** {dataset_name}  
**Split Protocol:** 70% Train ($N={len(X_train)}$), 15% Validation ($N={len(X_val)}$), 15% Held-Out Test ($N={len(X_test)}$)  
**Evaluation Standard:** Independent test split evaluation with fixed random seed (42).

| Metric | VADER Lexicon Engine | TF-IDF + Logistic Regression | Empirical Delta |
| :--- | :---: | :---: | :---: |
| **Test Accuracy** | **{v_acc*100:.2f}%** | **{lr_acc*100:.2f}%** | `{lr_acc*100 - v_acc*100:+.2f}%` |
| **Macro F1-Score** | **{v_f1:.4f}** | **{lr_f1:.4f}** | `{lr_f1 - v_f1:+.4f}` |
| **Macro Precision** | {v_prec*100:.2f}% | {lr_prec*100:.2f}% | `{lr_prec*100 - v_prec*100:+.2f}%` |
| **Macro Recall** | {v_rec*100:.2f}% | {lr_rec*100:.2f}% | `{lr_rec*100 - v_rec*100:+.2f}%` |
| **Negative F1** | {v_f_class[0]:.4f} | {lr_f_class[0]:.4f} | `{lr_f_class[0] - v_f_class[0]:+.4f}` |
| **Neutral F1** | {v_f_class[1]:.4f} | {lr_f_class[1]:.4f} | `{lr_f_class[1] - v_f_class[1]:+.4f}` |
| **Positive F1** | {v_f_class[2]:.4f} | {lr_f_class[2]:.4f} | `{lr_f_class[2] - v_f_class[2]:+.4f}` |

### Hyperparameters (Tuned on Validation Set)
- **Optimal C:** `{best_c}`
- **N-Gram Range:** `(1, 2)` unigrams and bigrams
- **Sublinear TF Scaling:** `True`
- **Class Weighting:** `balanced` (mitigating majority positive class bias)

### Visual Confusion Matrices
- `results/cm_logistic_regression.png`
- `results/cm_vader.png`

### Qualitative Error Analysis
A representative sample of 30 misclassified test cases (15 per model) is exported to `results/error_analysis.csv` covering negation scope, sarcasm, mixed sentiment, and neutral star-boundary transitions.
"""

    with open(RESULTS_DIR / "model_comparison.md", "w", encoding="utf-8") as f:
        f.write(md_table)
    print("[*] Markdown report saved: results/model_comparison.md")
    print("\n[✓] Real Data Training & Model Comparison Complete.")


if __name__ == "__main__":
    main()
