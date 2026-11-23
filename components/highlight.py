"""Word-level sentiment highlighting for review text (explainability view).

Pure Python (html + re only). The sentiment lexicon is passed in, so this works with
the VADER lexicon used by ReviewLens:  analyzer.sia.lexicon  (word -> valence, about -4..+4).

Rules (a light version of VADER's own logic, for display only):
  * A word is coloured if |valence| >= threshold.
  * If a negator appears within the previous 3 words, the colour flips and the word gets a
    dotted underline plus a tooltip ("negated").
  * All user text is HTML-escaped.
"""

import html
import re
from typing import Dict, List, Tuple

NEGATORS = {
    "not", "no", "never", "none", "nobody", "nothing", "neither", "nowhere", "hardly", "barely",
    "scarcely", "without", "cannot", "cant", "dont", "doesnt", "didnt", "isnt", "wasnt", "wont",
    "wouldnt", "couldnt", "shouldnt", "arent", "werent", "havent", "hasnt", "hadnt", "aint",
}
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*|\d+(?:\.\d+)?|[^\sA-Za-z\d]|\s+")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")

POS_COLOR = "#34D399"
NEG_COLOR = "#F87171"


def _is_negator(word: str) -> bool:
    w = word.lower().replace("'", "")
    return w in NEGATORS or w.endswith("nt")


def score_words(text: str, lexicon: Dict[str, float], threshold: float = 0.8) -> List[Tuple[str, float, bool]]:
    """Return [(token, display_valence, negated)] for every token in text.

    display_valence is 0.0 for tokens that should not be coloured.
    """
    tokens = _TOKEN_RE.findall(text or "")
    out: List[Tuple[str, float, bool]] = []
    recent_words: List[str] = []  # last words seen, for negation scope
    for tok in tokens:
        if _WORD_RE.fullmatch(tok):
            low = tok.lower()
            val = float(lexicon.get(low, 0.0))
            negated = False
            if abs(val) >= threshold:
                if any(_is_negator(w) for w in recent_words[-3:]):
                    val = -val
                    negated = True
            else:
                val = 0.0
            out.append((tok, val, negated))
            recent_words.append(tok)
        else:
            out.append((tok, 0.0, False))
    return out


def highlight_review_html(text: str, lexicon: Dict[str, float], threshold: float = 0.8) -> str:
    """HTML for the review with sentiment-carrying words highlighted. Safe to render with unsafe_allow_html."""
    parts: List[str] = []
    for tok, val, negated in score_words(text, lexicon, threshold):
        esc = html.escape(tok)
        if val == 0.0:
            parts.append(esc)
            continue
        color = POS_COLOR if val > 0 else NEG_COLOR
        alpha = min(0.45, 0.12 + abs(val) / 4 * 0.33)
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        style = f"background:rgba({r},{g},{b},{alpha:.2f});color:{color};padding:1px 5px;border-radius:5px;font-weight:600;"
        if negated:
            style += f"border-bottom:2px dotted {color};"
        title = f"{'negated, ' if negated else ''}{'positive' if val > 0 else 'negative'} ({val:+.1f})"
        parts.append(f'<span style="{style}" title="{html.escape(title)}">{esc}</span>')
    body = "".join(parts)
    return (
        '<div style="line-height:2;font-size:1.02rem;color:#E2E8F0;background:rgba(22,17,46,.7);'
        'border:1px solid rgba(155,92,255,.3);border-radius:14px;padding:16px 18px;white-space:pre-wrap;">'
        f"{body}</div>"
        '<div style="font-size:.76rem;color:#94A3B8;margin-top:6px;">'
        f'<span style="color:{POS_COLOR}">&#9632;</span> positive &nbsp; '
        f'<span style="color:{NEG_COLOR}">&#9632;</span> negative &nbsp; '
        "dotted underline = flipped by a negation word</div>"
    )


def top_drivers(text: str, lexicon: Dict[str, float], threshold: float = 0.8, k: int = 5):
    """(top_positive, top_negative) as lists of (word, valence), strongest first, for a summary line."""
    scored = [(t, v) for t, v, _ in score_words(text, lexicon, threshold) if v != 0.0]
    pos = sorted([x for x in scored if x[1] > 0], key=lambda x: -x[1])[:k]
    neg = sorted([x for x in scored if x[1] < 0], key=lambda x: x[1])[:k]
    return pos, neg
