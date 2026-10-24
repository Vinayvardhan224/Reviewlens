"""
Configuration module for ReviewLens.
Contains aspect taxonomies, sentiment thresholds, spam detection rules, and model constants.
"""

from typing import Dict, List

# Sentiment Score Thresholds (VADER Compound score ranges between -1.0 and 1.0)
SENTIMENT_THRESHOLDS = {
    "positive": 0.05,
    "negative": -0.05,
}

# Star rating conversion thresholds from compound score
RATING_MAPPING = [
    (-0.6, 1.0),
    (-0.1, 2.0),
    (0.1, 3.0),
    (0.6, 4.0),
    (1.0, 5.0),
]

# Aspect Categories and Seed Keywords for Product Reviews
DEFAULT_ASPECT_TAXONOMY: Dict[str, List[str]] = {
    "Battery & Power": [
        "battery", "battery life", "charge", "charging", "charger", "power", 
        "cord", "drain", "draining", "recharge", "standby", "runtime", "mah"
    ],
    "Sound & Audio": [
        "sound", "sound quality", "audio", "bass", "treble", "volume", 
        "clarity", "noise", "noise cancellation", "anc", "mic", "microphone",
        "voice", "speaker", "muffled", "distortion", "hiss", "loudness", "echo"
    ],
    "Build Quality & Durability": [
        "build", "build quality", "durability", "material", "sturdy", "flimsy",
        "plastic", "metal", "hinge", "broken", "scratch", "crack", "heavy",
        "lightweight", "premium", "cheap", "durable", "tough", "fragile"
    ],
    "Comfort & Ergonomics": [
        "comfort", "comfortable", "fit", "earcups", "cushion", "headband",
        "ergonomic", "tight", "loose", "pain", "hurt", "ear", "ears", "weight",
        "grip", "soft", "padding", "wearable"
    ],
    "Display & Visuals": [
        "display", "screen", "resolution", "brightness", "color", "oled",
        "amoled", "refresh rate", "hz", "pixels", "viewing angle", "hdr",
        "glare", "contrast"
    ],
    "Performance & Speed": [
        "performance", "speed", "fast", "slow", "lag", "laggy", "smooth",
        "processor", "ram", "app", "crash", "bug", "software", "update",
        "responsiveness", "freeze", "stutter", "heat", "overheating"
    ],
    "Price & Value for Money": [
        "price", "value", "cost", "worth", "expensive", "cheap", "affordable",
        "bang for the buck", "deal", "overpriced", "bargain", "money", "investment"
    ],
    "Ease of Use & Connectivity": [
        "bluetooth", "connection", "connect", "pairing", "range", "disconnect",
        "easy to use", "setup", "controls", "button", "app", "interface",
        "user friendly", "intuitive", "manual", "sync"
    ],
    "Customer Support & Warranty": [
        "support", "customer service", "warranty", "replacement", "return",
        "refund", "agent", "representative", "help", "contact", "policy", "repair"
    ],
    "Packaging & Delivery": [
        "package", "packaging", "box", "shipping", "delivery", "arrived",
        "damaged", "sealed", "on time", "late", "courier", "missing"
    ],
}

# Domain-specific sentiment modifier lexicon additions
DOMAIN_SENTIMENT_LEXICON: Dict[str, float] = {
    # Positive product review terms
    "durable": 2.2,
    "crisp": 1.9,
    "punchy": 1.7,
    "ergonomic": 2.0,
    "sleek": 1.8,
    "seamless": 2.1,
    "premium": 2.4,
    "sturdy": 2.0,
    "intuitive": 2.0,
    "responsive": 2.1,
    "game changer": 2.9,
    "bargain": 2.0,
    "worth every penny": 3.2,
    "well-built": 2.2,
    "long-lasting": 2.3,
    "exceeded expectations": 3.0,
    "crystal clear": 2.7,
    "lightweight": 1.5,
    "flawless": 2.8,

    # Negative product review terms
    "overpriced": -2.3,
    "flimsy": -2.4,
    "laggy": -2.3,
    "bloatware": -2.2,
    "unresponsive": -2.4,
    "bricked": -3.5,
    "drain": -1.8,
    "defective": -2.8,
    "subpar": -2.1,
    "mediocre": -1.5,
    "overheats": -2.5,
    "overheating": -2.6,
    "gimmick": -2.0,
    "stopped working": -3.2,
    "waste of money": -3.4,
    "lemon": -2.5,
    "uncomfortable": -2.4,
    "scratch": -1.2,
    "crack": -2.0,
    "buyer beware": -3.1,
    "disconnects": -2.3,
    "distorted": -2.2,
    "muffled": -2.0,
    "rip-off": -3.4,
    "ripoff": -3.4,
}

# Emotion Lexicon mapping
EMOTION_KEYWORDS: Dict[str, List[str]] = {
    "Joy & Delight": [
        "love", "amazing", "awesome", "fantastic", "superb", "delighted", "thrilled",
        "pleased", "great", "excellent", "wonderful", "happy", "favorite", "impressed",
        "best", "perfect", "adore", "enjoy"
    ],
    "Trust & Confidence": [
        "reliable", "dependable", "solid", "authentic", "genuine", "recommend",
        "trust", "durable", "sturdy", "accurate", "safe", "consistent", "verified"
    ],
    "Frustration & Anger": [
        "terrible", "horrible", "awful", "scam", "cheat", "furious", "angry",
        "annoying", "disaster", "garbage", "trash", "hate", "unacceptable",
        "waste of money", "rip-off", "useless", "pathetic", "nightmare"
    ],
    "Disappointment": [
        "disappointed", "disappointing", "underwhelmed", "expected more", "mediocre",
        "subpar", "letdown", "sadly", "unfortunate", "flimsy", "average", "bland",
        "regret", "not worth"
    ],
    "Anticipation & Surprise": [
        "surprised", "unexpected", "curious", "hopeful", "shocked", "wonder",
        "astonished", "pleasantly surprised"
    ]
}

# Spam & Fake Review Detection Parameters
SPAM_DETECTION_CONFIG = {
    "max_uppercase_ratio": 0.35,          # e.g. "BUY THIS NOW SO AMAZING!!!!"
    "max_punctuation_repetition": 4,       # e.g. "!!!!!" or "?????"
    "min_review_length_chars": 12,        # reviews like "ok", "good"
    "spam_phrases": [
        "free gift", "discount code", "contact on whatsapp", "buy now",
        "click here", "cash back", "five stars for free", "paid review",
        "visit website", "check out my channel", "guaranteed money"
    ],
    "extreme_sentiment_threshold": 0.75,   # for rating-sentiment mismatch check
}
