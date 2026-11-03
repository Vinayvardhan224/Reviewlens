"""
Unit tests for sentiment analysis, emotion classification, and ReviewLens sentiment module.
"""

import unittest
import pandas as pd
from reviewlens.sentiment.vader_analyzer import SentimentAnalyzer
from reviewlens.sentiment.emotion_analyzer import EmotionAnalyzer
from modules.sentiment import (
    analyze_text_sentiment,
    compute_nss,
    detect_rating_anomaly,
    analyze_reviews_sentiment
)


class TestSentiment(unittest.TestCase):

    def setUp(self):
        self.analyzer = SentimentAnalyzer()
        self.emotion_analyzer = EmotionAnalyzer()

    def test_positive_sentiment(self):
        text = "This headphone has incredible sound quality and punchy bass. Truly exceeded expectations!"
        res = self.analyzer.analyze(text)
        self.assertEqual(res.label, "POSITIVE")
        self.assertGreater(res.compound, 0.4)
        self.assertGreaterEqual(res.predicted_rating, 4.0)

    def test_negative_sentiment(self):
        text = "The product arrived broken and bricked on first use. Complete waste of money!"
        res = self.analyzer.analyze(text)
        self.assertEqual(res.label, "NEGATIVE")
        self.assertLess(res.compound, -0.4)
        self.assertLessEqual(res.predicted_rating, 2.5)

    def test_domain_lexicon(self):
        text = "My phone bricked after updating."
        res = self.analyzer.analyze(text)
        self.assertEqual(res.label, "NEGATIVE")

    def test_emotion_detection(self):
        pos_text = "I absolutely love and adore this phone, it is a wonderful delight!"
        dominant, scores = self.emotion_analyzer.analyze(pos_text, sentiment_compound=0.8)
        self.assertEqual(dominant, "Joy & Delight")

        neg_text = "Terrible horrible awful scam! I hate this garbage product!"
        dominant_neg, scores_neg = self.emotion_analyzer.analyze(neg_text, sentiment_compound=-0.8)
        self.assertEqual(dominant_neg, "Frustration & Anger")

    def test_module_sentiment_and_nss(self):
        text = "Superb battery life and blazing fast speed!"
        sent = analyze_text_sentiment(text)
        self.assertEqual(sent["label"], "POSITIVE")
        self.assertGreater(sent["compound"], 0.4)

        nss = compute_nss(80, 20, 100)
        self.assertEqual(nss, 60.0)

    def test_anomaly_detection(self):
        res = detect_rating_anomaly(5.0, -0.6)
        self.assertTrue(res["is_anomaly"])

        res2 = detect_rating_anomaly(5.0, 0.6)
        self.assertFalse(res2["is_anomaly"])


if __name__ == "__main__":
    unittest.main()
