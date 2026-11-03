"""
ReviewLens Aspect-Based Sentiment Analysis (ABSA) Module.
Deconstructs consumer feedback into 8 primary electronics dimensions with
sentence-level keyword extraction, sentiment polarity, and evidence retrieval.
"""

from typing import Dict, Any, List, Set
import re
from modules.preprocessing import split_into_sentences
from modules.sentiment import analyze_text_sentiment, compute_nss

ASPECT_LEXICON: Dict[str, Set[str]] = {
    "Performance & Speed": {
        "processor", "cpu", "gpu", "chip", "chipset", "speed", "fast", "slow",
        "lag", "laggy", "frame", "fps", "gaming", "multitask", "multitasking",
        "snapdragon", "exynos", "bionic", "tensor", "benchmark", "antutu", "geekbench",
        "performance", "smooth", "fluid", "stutter", "heat", "heating", "warm", "thermal"
    },
    "Display & Screen": {
        "display", "screen", "amoled", "oled", "retina", "120hz", "refresh",
        "brightness", "bright", "nits", "glare", "anti-reflective", "resolution",
        "fhd", "qhd", "colors", "viewing", "bezel", "bezels", "panel"
    },
    "Camera & Photography": {
        "camera", "cameras", "photo", "photos", "picture", "pictures", "lens",
        "portrait", "portraits", "zoom", "telephoto", "periscope", "optical",
        "night", "nightography", "video", "videos", "4k", "8k", "selfie", "selfies",
        "sensor", "autofocus", "ois", "shutter", "bokeh", "prores"
    },
    "Battery & Charging": {
        "battery", "charge", "charger", "charging", "watt", "25w", "45w", "100w",
        "fast charging", "drain", "draining", "backup", "endurance", "screen-on-time",
        "sot", "mah", "wireless charging", "magsafe", "power"
    },
    "Build Quality & Design": {
        "build", "design", "titanium", "aluminum", "glass", "plastic", "durability",
        "durable", "finish", "premium", "weight", "heavy", "light", "lightweight",
        "ergonomic", "scratch", "waterproof", "ip68", "ip65", "in-hand", "chassis"
    },
    "Value for Money & Price": {
        "price", "pricing", "value", "expensive", "overpriced", "cheap", "cost",
        "costly", "worth", "deal", "budget", "bargain", "rupees", "lakh", "lakhs",
        "affordable", "investment"
    },
    "Audio & Sound": {
        "audio", "sound", "speaker", "speakers", "bass", "treble", "volume",
        "loud", "microphone", "mic", "anc", "noise cancellation", "cancelling",
        "dolby", "stereo", "headphone", "spatial audio", "clarity", "calls"
    },
    "Software & Features": {
        "software", "os", "android", "ios", "update", "updates", "one ui",
        "oxygenos", "bloatware", "bugs", "s-pen", "pen", "stylus", "gesture",
        "action button", "island", "ecosystem", "features", "assistant", "siri"
    }
}

ASPECT_CAVEAT_DISCLOSURE = (
    "Methodology Note: Aspect sentiment is computed using rule-based keyword matching "
    "combined with VADER sentence polarity. This transparent, explainable prototype "
    "provides immediate clarity into specific product attributes without black-box neural ambiguity."
)


def extract_aspects_from_sentence(sentence: str) -> List[str]:
    """Identify which aspect categories are referenced in a given sentence."""
    tokens = re.findall(r"\b[a-zA-Z0-9_\-']+\b", sentence.lower())
    token_set = set(tokens)
    matched = []

    for aspect_name, keywords in ASPECT_LEXICON.items():
        if token_set.intersection(keywords):
            matched.append(aspect_name)
        else:
            # Check 2-word phrase matches (e.g. "battery life", "night mode")
            for kw in keywords:
                if " " in kw and kw in sentence.lower():
                    matched.append(aspect_name)
                    break

    return matched


def analyze_reviews_aspects(df_reviews) -> Dict[str, Any]:
    """
    Run sentence-level Aspect-Based Sentiment Analysis on all reviews for a product.
    Returns:
        aspect_summaries: List of aspect metric dictionaries
        evidence_quotes: Dictionary of positive & negative snippets per aspect
        methodology_note: String disclosure
    """
    aspect_data = {
        name: {
            "name": name,
            "mention_count": 0,
            "pos_count": 0,
            "neu_count": 0,
            "neg_count": 0,
            "positive_quotes": [],
            "negative_quotes": []
        }
        for name in ASPECT_LEXICON.keys()
    }

    if df_reviews is None or len(df_reviews) == 0:
        return {
            "aspects": [],
            "methodology_note": ASPECT_CAVEAT_DISCLOSURE
        }

    for _, row in df_reviews.iterrows():
        review_text = str(row.get("review_text", ""))
        sentences = split_into_sentences(review_text)

        for sent in sentences:
            aspects_found = extract_aspects_from_sentence(sent)
            if not aspects_found:
                continue

            sentiment = analyze_text_sentiment(sent)
            label = sentiment["label"]

            for asp in aspects_found:
                data = aspect_data[asp]
                data["mention_count"] += 1

                if label == "POSITIVE":
                    data["pos_count"] += 1
                    if len(data["positive_quotes"]) < 4:
                        data["positive_quotes"].append(sent)
                elif label == "NEGATIVE":
                    data["neg_count"] += 1
                    if len(data["negative_quotes"]) < 4:
                        data["negative_quotes"].append(sent)
                else:
                    data["neu_count"] += 1

    # Format summaries and compute percentages & NSS
    summaries = []
    for name, data in aspect_data.items():
        total_m = data["mention_count"]
        if total_m > 0:
            pos_pct = round((data["pos_count"] / float(total_m)) * 100.0, 1)
            neu_pct = round((data["neu_count"] / float(total_m)) * 100.0, 1)
            neg_pct = round((data["neg_count"] / float(total_m)) * 100.0, 1)
            nss = compute_nss(data["pos_count"], data["neg_count"], total_m)
        else:
            pos_pct, neu_pct, neg_pct, nss = 0.0, 0.0, 0.0, 0.0

        summaries.append({
            "category": name,
            "mentions": total_m,
            "positive_count": data["pos_count"],
            "neutral_count": data["neu_count"],
            "negative_count": data["neg_count"],
            "pos_pct": pos_pct,
            "neu_pct": neu_pct,
            "neg_pct": neg_pct,
            "net_sentiment_score": nss,
            "positive_quotes": data["positive_quotes"],
            "negative_quotes": data["negative_quotes"]
        })

    # Sort aspects by mentions descending
    summaries.sort(key=lambda x: x["mentions"], reverse=True)

    return {
        "aspects": summaries,
        "methodology_note": ASPECT_CAVEAT_DISCLOSURE
    }
