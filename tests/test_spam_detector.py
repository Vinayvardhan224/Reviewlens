"""Unit tests for spam and fake review detector."""

import unittest
from reviewlens.spam_detector.detector import SpamDetector


class TestSpamDetector(unittest.TestCase):

    def setUp(self):
        self.detector = SpamDetector()

    def test_promotional_spam(self):
        text = "BUY NOW FREE GIFT ON WHATSAPP VISIT WEBSITE FOR DISCOUNT CODE!!!!"
        res = self.detector.analyze(text)
        self.assertTrue(res.is_spam)
        self.assertGreaterEqual(res.spam_score, 0.5)

    def test_rating_sentiment_mismatch(self):
        # 5 star rating with deeply negative text
        text = "The product broke on day one, completely defective and horrible."
        res = self.detector.analyze(text, rating=5.0, sentiment_compound=-0.6)
        self.assertTrue(res.is_spam)
        self.assertTrue(any("mismatch" in r.lower() for r in res.reasons))

    def test_clean_review(self):
        text = "The headphones arrived in good condition. Sound is clear and battery backup lasts all day."
        res = self.detector.analyze(text, rating=4.0, sentiment_compound=0.5)
        self.assertFalse(res.is_spam)
        self.assertEqual(res.spam_score, 0.0)

    def test_duplicate_detection(self):
        self.detector.reset_session()
        review = "This is a great product with very good audio quality and reliable battery."
        res1 = self.detector.analyze(review, check_duplicate=True)
        self.assertFalse(res1.is_spam)

        res2 = self.detector.analyze(review, check_duplicate=True)
        self.assertTrue(res2.is_spam)
        self.assertTrue(any("duplicate" in r.lower() for r in res2.reasons))


if __name__ == "__main__":
    unittest.main()
