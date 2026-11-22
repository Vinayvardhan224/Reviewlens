# Empirical Sentiment Model Evaluation Benchmark

**Dataset:** UCSD Amazon Electronics Review Corpus  
**Split Protocol:** 70% Train ($N=697$), 15% Validation ($N=150$), 15% Held-Out Test ($N=150$)  
**Evaluation Standard:** Independent test split evaluation with fixed random seed (42).

| Metric | VADER Lexicon Engine | TF-IDF + Logistic Regression | Empirical Delta |
| :--- | :---: | :---: | :---: |
| **Test Accuracy** | **75.33%** | **73.33%** | `-2.00%` |
| **Macro F1-Score** | **0.4323** | **0.4578** | `+0.0255` |
| **Macro Precision** | 41.20% | 46.11% | `+4.91%` |
| **Macro Recall** | 45.73% | 45.65% | `-0.08%` |
| **Negative F1** | 0.4167 | 0.4390 | `+0.0224` |
| **Neutral F1** | 0.0000 | 0.0833 | `+0.0833` |
| **Positive F1** | 0.8803 | 0.8511 | `-0.0293` |

### Hyperparameters (Tuned on Validation Set)
- **Optimal C:** `0.1`
- **N-Gram Range:** `(1, 2)` unigrams and bigrams
- **Sublinear TF Scaling:** `True`
- **Class Weighting:** `balanced` (mitigating majority positive class bias)

### Visual Confusion Matrices
- `results/cm_logistic_regression.png`
- `results/cm_vader.png`

### Qualitative Error Analysis
A representative sample of 30 misclassified test cases (15 per model) is exported to `results/error_analysis.csv` covering negation scope, sarcasm, mixed sentiment, and neutral star-boundary transitions.
