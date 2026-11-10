"""
ReviewLens Text Preprocessing Module.
Handles cleaning, sentence splitting, tokenization, and negation phrase detection.
"""

import re
from typing import List, Tuple

NEGATION_TERMS = {
    "not", "no", "never", "neither", "nor", "none", "hardly", "scarcely",
    "barely", "without", "don't", "dont", "doesn't", "doesnt", "didn't",
    "didnt", "won't", "wont", "wouldn't", "wouldnt", "can't", "cant",
    "cannot", "couldn't", "couldnt", "isn't", "isnt", "aren't", "arent",
    "wasn't", "wasnt", "weren't", "werent"
}

HTML_CLEANER = re.compile(r"<[^>]+>")
URL_CLEANER = re.compile(r"https?://\S+|www\.\S+")
WHITESPACE_CLEANER = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Clean raw review text by stripping HTML, URLs, and standardizing whitespace."""
    if not text or not isinstance(text, str):
        return ""
    # Strip HTML tags
    text = HTML_CLEANER.sub(" ", text)
    # Strip URLs
    text = URL_CLEANER.sub(" ", text)
    # Normalize whitespace
    text = WHITESPACE_CLEANER.sub(" ", text).strip()
    return text


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into distinct sentences while guarding common abbreviations
    like 'e.g.', 'i.e.', 'vs.', '3.5mm', and numbers.
    """
    if not text:
        return []

    # Protect common patterns
    protected = text
    protected = re.sub(r"\b(e\.g\.|i\.e\.|vs\.|approx\.)", lambda m: m.group(0).replace(".", "<DOT>"), protected, flags=re.I)
    protected = re.sub(r"(\d+)\.(\d+)", r"\1<DOT>\2", protected)

    # Split by punctuation (. ! ?) followed by whitespace or end of string
    raw_sentences = re.split(r"[.!?]+(?:\s+|$)", protected)

    clean_sentences = []
    for s in raw_sentences:
        s = s.replace("<DOT>", ".").strip()
        if len(s) > 3:  # Discard trivial fragments
            clean_sentences.append(s)

    return clean_sentences if clean_sentences else [text.strip()]


def split_into_aspect_clauses(sentence: str) -> List[str]:
    """
    Split complex sentences containing contrastive conjunctions (e.g. 'but', 'however',
    'although', 'while', 'yet') into sub-clauses for accurate aspect-level polarity assignment.
    Example:
      'The battery is great, but the camera is not good' ->
      ['The battery is great', 'the camera is not good']
    """
    if not sentence:
        return []
    clauses = re.split(r",?\s+\b(?:but|however|although|though|yet|whereas|while)\b\s*", sentence, flags=re.I)
    cleaned = [c.strip() for c in clauses if c.strip() and len(c.strip()) > 3]
    return cleaned if cleaned else [sentence.strip()]


def tokenize(text: str) -> List[str]:
    """Simple alphanumeric tokenizer."""
    if not text:
        return []
    return re.findall(r"\b[a-zA-Z0-9_\-']+\b", text.lower())


def detect_negations(tokens: List[str], window_size: int = 3) -> List[Tuple[str, bool]]:
    """
    Identifies tokens that fall under the grammatical scope of a negation word.
    Returns list of (token, is_negated).
    """
    result = []
    negation_counter = 0

    for token in tokens:
        clean_tok = token.replace("'", "")
        if clean_tok in NEGATION_TERMS or token in NEGATION_TERMS:
            negation_counter = window_size
            result.append((token, False))
        elif negation_counter > 0:
            result.append((token, True))
            negation_counter -= 1
        else:
            result.append((token, False))

    return result
