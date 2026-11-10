"""
NLTK Corpus Setup Script for ReviewLens.
Downloads and rigorously verifies all lexical resources and tokenizers required
for sentiment analysis, stopword removal, and Aspect-Based Sentiment Analysis (ABSA).
Exits with non-zero status code if any required core corpus fails to download or verify.
"""

import sys
import nltk


def setup_nltk_corpora() -> None:
    """Download and verify required NLTK corpora."""
    required_corpora = [
        ("vader_lexicon", "sentiment/vader_lexicon.zip"),
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("stopwords", "corpora/stopwords")
    ]
    optional_corpora = [
        ("wordnet", "corpora/wordnet")
    ]

    print("[ReviewLens] Setting up and verifying NLTK lexical resources...")
    failed_required = []

    for name, find_path in required_corpora:
        try:
            nltk.data.find(find_path)
            print(f"  ✓ {name} is already available.")
            continue
        except LookupError:
            pass

        print(f"  ⬇ Downloading required resource: {name}...")
        success = nltk.download(name, quiet=True)
        try:
            nltk.data.find(find_path)
            print(f"  ✓ {name} verified successfully.")
        except LookupError:
            print(f"  ❌ Failed to verify required resource {name} (download status: {success})")
            failed_required.append(name)

    for name, find_path in optional_corpora:
        try:
            nltk.data.find(find_path)
            print(f"  ✓ {name} (optional) is already available.")
            continue
        except LookupError:
            pass

        print(f"  ⬇ Downloading optional resource: {name}...")
        try:
            nltk.download(name, quiet=True)
            nltk.data.find(find_path)
            print(f"  ✓ {name} (optional) verified successfully.")
        except Exception:
            print(f"  ⚠ Optional resource {name} could not be fetched (non-critical fallback active).")

    if failed_required:
        print(f"\n[ReviewLens Error] Missing required NLTK resources: {failed_required}")
        sys.exit(1)

    print("[ReviewLens] All required NLTK resources verified successfully.\n")
    sys.exit(0)


if __name__ == "__main__":
    setup_nltk_corpora()
