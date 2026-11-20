"""
ReviewLens spaCy-Powered Aspect & Opinion Term Extraction Engine.
Uses English dependency parsing and noun chunk segmentation to isolate
candidate aspect targets and their syntactically bound opinion modifiers (amod, acomp, advmod).
Evaluated against SemEval-2014 Task 4 (Laptops).
"""

from typing import List, Dict, Any, Optional, Set
import re

_SPACY_NLP = None
_SPACY_AVAILABLE = False

try:
    import spacy
    try:
        _SPACY_NLP = spacy.load("en_core_web_sm")
        _SPACY_AVAILABLE = True
    except Exception:
        _SPACY_AVAILABLE = False
except ImportError:
    _SPACY_AVAILABLE = False


def is_spacy_available() -> bool:
    """Returns True if spaCy and the en_core_web_sm model are ready."""
    return _SPACY_AVAILABLE


def extract_aspect_opinion_pairs_spacy(
    text: str,
    known_aspects: Optional[Set[str]] = None
) -> List[Dict[str, Any]]:
    """
    Extracts fine-grained aspect targets and their associated opinion descriptors
    using syntactic dependency relations and noun chunk boundaries.
    
    Args:
        text: Input review sentence or document.
        known_aspects: Optional set of category taxonomy terms to match/filter.
        
    Returns:
        List of dicts: {'aspect': str, 'opinion_words': List[str], 'head': str, 'has_taxonomy_match': bool}
    """
    if not text or not text.strip():
        return []

    if not _SPACY_AVAILABLE or _SPACY_NLP is None:
        # Fallback to regex-based extraction if spaCy is unavailable
        words = re.findall(r"\b[A-Za-z0-9_-]+\b", text)
        results = []
        if known_aspects:
            for w in words:
                if w.lower() in known_aspects:
                    results.append({
                        "aspect": w,
                        "opinion_words": [],
                        "head": w,
                        "has_taxonomy_match": True,
                        "method": "regex_fallback"
                    })
        return results

    doc = _SPACY_NLP(text)
    pairs = []
    seen_aspects = set()

    for chunk in doc.noun_chunks:
        # Filter out pure pronouns like 'it', 'they', 'this'
        if all(token.pos_ == "PRON" for token in chunk):
            continue

        aspect_text = chunk.text.strip()
        root = chunk.root
        opinions: List[str] = []

        # 1. Attributive adjective modifiers within the noun phrase (e.g. 'vibrant display')
        for child in chunk:
            if child.pos_ == "ADJ" and child != root:
                opinions.append(child.text.lower())

        # 2. Copular and predicative adjectives (e.g. 'display is gorgeous', 'trackpad feels sticky')
        if root.dep_ == "nsubj" and root.head.pos_ in ("AUX", "VERB"):
            verb = root.head
            for child in verb.children:
                if child.dep_ in ("acomp", "attr", "advmod", "oprd") and child.pos_ in ("ADJ", "ADV"):
                    opinions.append(child.text.lower())
                # Handle negated opinions (e.g. 'is not good')
                elif child.dep_ == "neg":
                    opinions.append(f"not_{child.head.text.lower()}")

        # 3. Direct objects modified by adjectives (e.g. 'has a fast processor')
        if root.dep_ in ("dobj", "pobj"):
            for child in root.children:
                if child.pos_ == "ADJ":
                    opinions.append(child.text.lower())

        clean_opinions = list(dict.fromkeys(opinions))
        has_tax = False
        if known_aspects:
            chunk_lower = aspect_text.lower()
            has_tax = any(k in chunk_lower or chunk_lower in k for k in known_aspects)

        # Avoid redundant duplicates
        asp_key = aspect_text.lower()
        if asp_key not in seen_aspects:
            seen_aspects.add(asp_key)
            pairs.append({
                "aspect": aspect_text,
                "opinion_words": clean_opinions,
                "head": root.text,
                "has_taxonomy_match": has_tax,
                "method": "spacy_dependency_parser"
            })

    return pairs
