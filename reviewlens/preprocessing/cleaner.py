"""
Text cleaning and normalization module for ReviewLens.
Cleans raw product reviews while preserving critical sentiment markers and negations.
"""

import re
from bs4 import BeautifulSoup

# Common English contraction mappings (ordered to avoid greedy substring collisions)
# Handles won't/can't first, then pronoun+'s (preserving noun possessives), then general n't
CONTRACTIONS = [
    (r"\bwon\'t\b", "will not"),
    (r"\bcan\'t\b", "cannot"),
    (r"\bshan\'t\b", "shall not"),
    (r"\b(it|that|there|what|he|she)\'s\b", r"\1 is"),
    (r"n\'t\b", " not"),
    (r"\'re\b", " are"),
    (r"\'d\b", " would"),
    (r"\'ll\b", " will"),
    (r"\'ve\b", " have"),
    (r"\'m\b", " am"),
]

# Key emojis mapped to single sentiment-carrying words (prevents sentiment double-counting)
EMOJI_SENTIMENT_MAP = {
    "❤️": " love ",
    "😍": " love ",
    "😊": " happy ",
    "👍": " good ",
    "🔥": " great ",
    "⭐": " good ",
    "🌟": " excellent ",
    "💯": " perfect ",
    "👏": " good ",
    "👌": " perfect ",
    "👎": " dislike ",
    "😡": " angry ",
    "😠": " annoyed ",
    "🤬": " terrible ",
    "💔": " broken ",
    "🤮": " disgusting ",
    "💩": " terrible ",
    "🤦": " frustrating ",
    "🙄": " annoyed ",
    "😢": " sad ",
    "😭": " terrible ",
}


def expand_contractions(text: str) -> str:
    """Expand standard English contractions into full words while preserving possessives."""
    for pattern, replacement in CONTRACTIONS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def convert_emojis(text: str) -> str:
    """Replace sentiment emojis with equivalent English tokens."""
    for emoji_char, word in EMOJI_SENTIMENT_MAP.items():
        text = text.replace(emoji_char, word)
    return text


def clean_html(text: str) -> str:
    """Remove HTML tags and entities."""
    if "<" in text and ">" in text:
        try:
            return BeautifulSoup(text, "html.parser").get_text(separator=" ")
        except Exception:
            return re.sub(r"<[^>]+>", " ", text)
    return text


def clean_text(
    text: str,
    expand_contract: bool = True,
    process_emojis: bool = True,
    remove_urls: bool = True,
    lowercase: bool = False,
    keep_punct_for_vader: bool = True,
) -> str:
    """
    Primary text cleaning function for review processing.
    
    Args:
        text: Raw review string.
        expand_contract: Whether to expand contractions like don't -> do not.
        process_emojis: Whether to map common emojis to semantic words.
        remove_urls: Whether to strip web links.
        lowercase: Whether to convert text to lowercase.
        keep_punct_for_vader: If True, keep punctuation like ! and ? which VADER utilizes.
        
    Returns:
        Cleaned, normalized review text.
    """
    if not isinstance(text, str):
        return ""

    # Strip HTML
    cleaned = clean_html(text)

    # Process emojis
    if process_emojis:
        cleaned = convert_emojis(cleaned)

    # Expand contractions
    if expand_contract:
        cleaned = expand_contractions(cleaned)

    # Remove URLs
    if remove_urls:
        cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)

    # Lowercase if requested
    if lowercase:
        cleaned = cleaned.lower()

    if not keep_punct_for_vader:
        # Strip non-alphanumeric except spaces
        cleaned = re.sub(r"[^\w\s]", " ", cleaned)

    # Normalize excessive whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned
