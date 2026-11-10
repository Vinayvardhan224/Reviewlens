"""
NLTK Corpus Setup Script for ReviewLens.
Downloads all lexical resources and tokenizers required for sentiment analysis,
stopword removal, and Aspect-Based Sentiment Analysis (ABSA).
"""

import nltk


def setup_nltk_corpora():
    """Download required NLTK corpora."""
    corpora = [
        "vader_lexicon",
        "punkt",
        "punkt_tab",
        "stopwords",
        "wordnet"
    ]
    print("[ReviewLens] Setting up NLTK lexical resources...")
    for c in corpora:
        try:
            nltk.download(c, quiet=True)
            print(f"  ✓ {c} ready")
        except Exception as e:
            print(f"  ⚠ Could not download {c}: {e}")
    print("[ReviewLens] NLTK setup complete.")


if __name__ == "__main__":
    setup_nltk_corpora()
