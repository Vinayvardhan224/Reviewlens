"""
ReviewLens Aspect-Based Sentiment Analysis (ABSA) Module.
Features category-specific aspect taxonomies for Smartphones, Laptops, Cameras, and Headphones.
Handles missing evidence transparently without assigning false neutral scores.
"""

from typing import Dict, Any, List, Set, Optional
import re
from modules.preprocessing import split_into_sentences
from modules.sentiment import analyze_text_sentiment, compute_nss

# Section 4.1 Category-Specific Aspect Taxonomy
CATEGORY_TAXONOMY: Dict[str, Dict[str, Set[str]]] = {
    "Smartphones": {
        "Camera": {
            "camera", "cameras", "photo", "photos", "picture", "pictures", "lens",
            "portrait", "portraits", "zoom", "telephoto", "periscope", "optical",
            "night", "nightography", "video", "videos", "4k", "8k", "selfie", "selfies",
            "sensor", "autofocus", "ois", "shutter", "bokeh", "megapixels", "mp"
        },
        "Battery": {
            "battery", "battery life", "backup", "endurance", "screen-on-time",
            "sot", "mah", "drain", "draining", "power", "standby"
        },
        "Display": {
            "display", "screen", "amoled", "oled", "retina", "120hz", "refresh",
            "brightness", "bright", "nits", "glare", "anti-reflective", "resolution",
            "fhd", "qhd", "colors", "viewing", "bezel", "bezels", "panel"
        },
        "Charging": {
            "charging", "charge", "charger", "fast charging", "watt", "25w", "45w",
            "100w", "supervooc", "wireless charging", "magsafe", "wired charging", "top up"
        },
        "Performance": {
            "performance", "speed", "fast", "slow", "lag", "laggy", "frame", "fps",
            "gaming", "multitask", "multitasking", "snapdragon", "exynos", "bionic",
            "tensor", "benchmark", "antutu", "geekbench", "smooth", "fluid", "stutter",
            "heat", "heating", "warm", "thermal", "chip", "chipset", "processor"
        }
    },
    "Laptops": {
        "Processor": {
            "processor", "cpu", "m3", "ryzen", "intel", "core i7", "speed", "fast",
            "multitasking", "compilation", "render", "clock", "benchmarks", "threads",
            "cores", "chip", "silicon"
        },
        "RAM": {
            "ram", "memory", "unified memory", "lpddr5", "16gb", "32gb", "swapping",
            "multitasking", "memory speed", "memory bandwidth"
        },
        "GPU": {
            "gpu", "graphics", "rtx", "4070", "iris xe", "dedicated graphics", "vram",
            "ray tracing", "3d rendering", "gaming framerates", "fps", "video editing"
        },
        "Keyboard": {
            "keyboard", "keys", "key travel", "typing", "tactile", "magic keyboard",
            "backlighting", "rgb", "layout", "trackpad", "touchpad", "touch function"
        },
        "Cooling": {
            "cooling", "fans", "fan noise", "thermal", "heat", "temperatures",
            "heat pipes", "liquid metal", "vapor chamber", "loud fans", "quiet",
            "fanless", "passive cooling", "hot"
        }
    },
    "Cameras": {
        "Image quality": {
            "image quality", "sensor", "33mp", "24mp", "dynamic range", "colors",
            "color science", "raw", "megapixels", "sharpness", "resolution", "detail",
            "exmor", "cmos", "full-frame"
        },
        "Autofocus": {
            "autofocus", "af", "eye af", "tracking", "phase detection", "dual pixel",
            "focus hunting", "focus lock", "subject detection", "burst"
        },
        "Lens quality": {
            "lens quality", "lens", "lenses", "optics", "e-mount", "rf mount",
            "sharpness", "prime lens", "zoom lens", "g master", "l-series", "glass",
            "aperture", "optical"
        },
        "Low-light performance": {
            "low-light", "low light", "high iso", "iso", "noise", "grain", "nighttime",
            "dark", "darkness", "-6.5 ev", "shadow recovery", "clean files"
        },
        "Video recording": {
            "video", "video recording", "4k", "60p", "10-bit", "4:2:2", "s-log",
            "s-cinetone", "oversampling", "crop", "rolling shutter", "cinematic", "6k"
        }
    },
    "Headphones": {
        "Sound quality": {
            "sound quality", "audio", "sound", "bass", "treble", "vocals",
            "soundstage", "clarity", "ldac", "drivers", "spatial audio", "acoustics",
            "immersive", "high-res"
        },
        "Battery": {
            "battery", "battery life", "playback", "runtime", "30 hours", "24 hours",
            "quick charge", "charging case", "hours listening"
        },
        "Comfort": {
            "comfort", "comfortable", "ear cushions", "earcups", "headband",
            "clamping force", "fatigue", "lightweight", "soft", "fit", "silicone tips",
            "ear tips", "wear"
        },
        "Noise cancellation": {
            "noise cancellation", "anc", "active noise cancellation", "ambient",
            "transparency", "quiet mode", "block out", "cabin noise", "airplane",
            "hum", "chatter"
        },
        "Connectivity": {
            "connectivity", "bluetooth", "pairing", "connection", "multipoint",
            "drops", "range", "disconnect", "codecs", "aptx", "fast pair",
            "device switching"
        }
    }
}

# General fallback taxonomy if category is unspecified
GENERAL_TAXONOMY: Dict[str, Set[str]] = {
    "Performance": {"performance", "speed", "fast", "processor", "chip", "lag"},
    "Display": {"display", "screen", "resolution", "brightness", "colors"},
    "Battery & Power": {"battery", "charging", "charge", "power", "backup"},
    "Build & Design": {"build", "design", "durability", "premium", "weight", "materials"},
    "Value for Money": {"price", "value", "cost", "worth", "expensive", "affordable"}
}

ASPECT_CAVEAT_DISCLOSURE = (
    "Methodology Note: Aspect sentiment is computed using category-specific keyword lexicons "
    "combined with VADER sentence polarity. Missing evidence is explicitly reported as "
    "'Insufficient review data' rather than assigned an artificial neutral score."
)


def get_taxonomy_for_category(category: Optional[str]) -> Dict[str, Set[str]]:
    """Retrieve category-specific aspect taxonomy per Section 4.1."""
    if not category:
        return GENERAL_TAXONOMY

    cat_norm = category.strip().capitalize()
    for k, tax in CATEGORY_TAXONOMY.items():
        if k.lower() in cat_norm.lower() or cat_norm.lower() in k.lower():
            return tax

    # Special aliases
    if "audio" in cat_norm.lower():
        return CATEGORY_TAXONOMY["Headphones"]

    return GENERAL_TAXONOMY


def extract_aspects_from_sentence(sentence: str, category: Optional[str] = None) -> List[str]:
    """Identify which category-specific aspect dimensions are referenced in a sentence."""
    taxonomy = get_taxonomy_for_category(category)
    tokens = re.findall(r"\b[a-zA-Z0-9_\-']+\b", sentence.lower())
    token_set = set(tokens)
    matched = []

    for aspect_name, keywords in taxonomy.items():
        if token_set.intersection(keywords):
            matched.append(aspect_name)
        else:
            for kw in keywords:
                if " " in kw and kw in sentence.lower():
                    matched.append(aspect_name)
                    break

    return matched


def analyze_reviews_aspects(df_reviews, category: Optional[str] = None) -> Dict[str, Any]:
    """
    Run sentence-level Aspect-Based Sentiment Analysis on product reviews using category-specific taxonomy.
    Missing evidence is explicitly flagged with has_evidence=False and score_display='Insufficient review data'.
    """
    taxonomy = get_taxonomy_for_category(category)

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
        for name in taxonomy.keys()
    }

    if df_reviews is not None and len(df_reviews) > 0:
        for _, row in df_reviews.iterrows():
            review_text = str(row.get("review_text", ""))
            sentences = split_into_sentences(review_text)

            for sent in sentences:
                aspects_found = extract_aspects_from_sentence(sent, category=category)
                if not aspects_found:
                    continue

                sentiment = analyze_text_sentiment(sent)
                label = sentiment["label"]

                for asp in aspects_found:
                    if asp in aspect_data:
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

    # Format summaries and preserve honest 'Insufficient review data' state
    summaries = []
    for name, data in aspect_data.items():
        total_m = data["mention_count"]
        if total_m > 0:
            pos_pct = round((data["pos_count"] / float(total_m)) * 100.0, 1)
            neu_pct = round((data["neu_count"] / float(total_m)) * 100.0, 1)
            neg_pct = round((data["neg_count"] / float(total_m)) * 100.0, 1)
            nss = compute_nss(data["pos_count"], data["neg_count"], total_m)
            has_evidence = True
            score_display = f"{nss:+.1f} NSS ({total_m} mentions)"
        else:
            pos_pct, neu_pct, neg_pct = 0.0, 0.0, 0.0
            nss = None  # Crucial: NOT 0.0!
            has_evidence = False
            score_display = "Insufficient review data"

        summaries.append({
            "category": name,
            "mentions": total_m,
            "has_evidence": has_evidence,
            "positive_count": data["pos_count"],
            "neutral_count": data["neu_count"],
            "negative_count": data["neg_count"],
            "pos_pct": pos_pct,
            "neu_pct": neu_pct,
            "neg_pct": neg_pct,
            "net_sentiment_score": nss,
            "score_display": score_display,
            "positive_quotes": data["positive_quotes"],
            "negative_quotes": data["negative_quotes"]
        })

    return {
        "aspects": summaries,
        "category_used": category or "General",
        "methodology_note": ASPECT_CAVEAT_DISCLOSURE
    }
