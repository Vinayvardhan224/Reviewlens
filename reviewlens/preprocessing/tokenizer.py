"""
Tokenization, POS tagging, lemmatization, and sentence segmentation for ReviewLens.
"""

from typing import List, Tuple
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Ensure resources are downloaded
try:
    _lemmatizer = WordNetLemmatizer()
    _stopwords_set = set(stopwords.words("english"))
except Exception:
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("punkt", quiet=True)
    _lemmatizer = WordNetLemmatizer()
    _stopwords_set = set(stopwords.words("english"))

# Sentiment-critical words that must NEVER be removed as stopwords
CRITICAL_WORDS = {
    "not", "no", "nor", "never", "none", "neither", "without",
    "hardly", "barely", "scarcely", "cannot", "against", "but", "however"
}
FILTERED_STOPWORDS = _stopwords_set - CRITICAL_WORDS

# Attempt to load spaCy model for high-precision POS and dependency parsing
_spacy_nlp = None
try:
    import spacy
    _spacy_nlp = spacy.load("en_core_web_sm")
except Exception:
    _spacy_nlp = None


def get_spacy_nlp():
    """Retrieve initialized spaCy instance or None."""
    return _spacy_nlp


def split_sentences(text: str) -> List[str]:
    """
    Split a review into discrete sentences using NLTK or regex fallback.
    Essential for Aspect-Based Sentiment Analysis.
    """
    if not text:
        return []

    try:
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
        if sentences:
            return [s.strip() for s in sentences if s.strip()]
    except Exception:
        pass

    # Regex sentence boundary splitter fallback
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def tokenize_words(text: str, remove_stop: bool = True, lemmatize: bool = True) -> List[str]:
    """
    Tokenize text into cleaned, lemmatized tokens.
    """
    if not text:
        return []

    # Simple word boundary tokenization
    tokens = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())

    if remove_stop:
        tokens = [t for t in tokens if t not in FILTERED_STOPWORDS]

    if lemmatize:
        tokens = [_lemmatizer.lemmatize(t) for t in tokens]

    return tokens


def extract_noun_phrases(text: str) -> List[str]:
    """
    Extract noun chunks/phrases representing potential product features or aspects.
    Utilizes spaCy when available, falling back to regex patterns.
    """
    if not text:
        return []

    if _spacy_nlp is not None:
        try:
            doc = _spacy_nlp(text)
            chunks = [chunk.text.lower().strip() for chunk in doc.noun_chunks if len(chunk.text.strip()) > 2]
            return list(dict.fromkeys(chunks))  # Preserve order, unique
        except Exception:
            pass

    # Fallback noun phrase regex: (adjective)* (noun)+
    pattern = r"\b(?:[a-zA-Z]+-)?(?:[a-zA-Z]+\s+)?(?:battery|sound|audio|screen|display|build|price|cord|mic|bass|cushion|earcup|case|charge|speed|app|feature|quality)\b"
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    return list(dict.fromkeys([m.lower().strip() for m in matches]))
