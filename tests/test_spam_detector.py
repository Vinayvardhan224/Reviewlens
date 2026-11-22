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
        self.assertTrue(res.has_mismatch)
        self.assertFalse(res.is_spam)  # Decoupled: mismatch alone is a data quality signal, not an automatic spam verdict
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

    def test_detect_review_bursts(self):
        # 6 reviews submitted on the same day for a single product
        reviews = [
            {"review_id": f"R{i}", "product_id": "phone-1", "date": "2024-05-10 10:00:00", "review_text": "Good phone."}
            for i in range(6)
        ]
        bursts = self.detector.detect_review_bursts(reviews, window_days=1, threshold_count=5)
        self.assertEqual(len(bursts), 1)
        self.assertEqual(bursts[0]["count"], 6)
        self.assertEqual(bursts[0]["product_id"], "phone-1")

    def test_detect_near_duplicates(self):
        reviews = [
            {"review_id": "R1", "review_text": "This phone has extraordinary battery life and a stunning display screen."},
            {"review_id": "R2", "review_text": "This phone has extraordinary battery life and a stunning display screen!"},
            {"review_id": "R3", "review_text": "Completely different text talking about camera lens quality and tripod."}
        ]
        dups = self.detector.detect_near_duplicates(reviews, similarity_threshold=0.85)
        self.assertEqual(len(dups), 1)
        self.assertEqual(dups[0]["review_id_a"], "R1")
        self.assertEqual(dups[0]["review_id_b"], "R2")
        self.assertGreaterEqual(dups[0]["similarity"], 0.85)


if __name__ == "__main__":
    unittest.main()
