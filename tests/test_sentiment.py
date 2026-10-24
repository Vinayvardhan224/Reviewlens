"""Unit tests for sentiment analysis and emotion classification."""

import unittest
from reviewlens.sentiment.vader_analyzer import SentimentAnalyzer
from reviewlens.sentiment.emotion_analyzer import EmotionAnalyzer


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
        # "bricked" and "waste of money" are domain negative additions
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


if __name__ == "__main__":
    unittest.main()
