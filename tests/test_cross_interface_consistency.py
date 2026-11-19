"""
Integration & Regression Tests: Cross-Interface NLP Consistency.
Verifies that:
  1. Streamlit Live Review Analyzer (modules.live_analyzer)
  2. FastAPI Microservice Pipeline (reviewlens.pipeline)
  3. Terminal CLI Pipeline (cli)
return identical sentiment scores, classification labels, aspect categories, and spam verdicts.
"""

import unittest
from reviewlens.pipeline import ReviewLensPipeline
from modules.live_analyzer import analyze_custom_review
from modules.sentiment import analyze_text_sentiment


class TestCrossInterfaceConsistency(unittest.TestCase):

    def setUp(self):
        self.pipeline = ReviewLensPipeline()

    def test_positive_review_consistency(self):
        text = "The battery easily lasts two full days and the camera takes breathtaking photos."
        rating = 5.0

        # 1. Direct Pipeline (FastAPI / CLI)
        pipe_res = self.pipeline.analyze_single(text=text, rating=rating)

        # 2. Streamlit Live Analyzer
        st_res = analyze_custom_review(text=text, rating=rating)

        # 3. Core module sentiment
        mod_sent = analyze_text_sentiment(text)

        # Assert identical sentiment scores
        self.assertAlmostEqual(pipe_res.sentiment.compound, st_res["sentiment"]["compound"], places=4)
        self.assertAlmostEqual(pipe_res.sentiment.compound, mod_sent["compound"], places=4)
        self.assertEqual(pipe_res.sentiment.label, st_res["sentiment"]["label"])
        self.assertEqual(pipe_res.sentiment.label, mod_sent["label"])
        self.assertEqual(pipe_res.sentiment.label, "POSITIVE")

        # Assert identical spam decisions
        self.assertEqual(pipe_res.spam.is_spam, st_res["spam_analysis"]["is_spam"])
        self.assertEqual(pipe_res.spam.is_spam, False)
        self.assertAlmostEqual(pipe_res.spam.spam_score, st_res["spam_analysis"]["spam_score"], places=2)

        # Assert aspects identified
        pipe_aspects = sorted(list({a.category for a in pipe_res.aspects}))
        self.assertEqual(pipe_aspects, st_res["aspects_found"])
        self.assertTrue(any("Battery" in a for a in pipe_aspects))
        self.assertTrue(any("Camera" in a for a in pipe_aspects))

    def test_negative_review_consistency(self):
        text = "Terrible build quality. The screen cracked after one day and battery drain is appalling."
        rating = 1.0

        pipe_res = self.pipeline.analyze_single(text=text, rating=rating)
        st_res = analyze_custom_review(text=text, rating=rating)

        self.assertAlmostEqual(pipe_res.sentiment.compound, st_res["sentiment"]["compound"], places=4)
        self.assertEqual(pipe_res.sentiment.label, st_res["sentiment"]["label"])
        self.assertEqual(pipe_res.sentiment.label, "NEGATIVE")
        self.assertEqual(pipe_res.spam.is_spam, st_res["spam_analysis"]["is_spam"])

    def test_spam_detection_consistency(self):
        # Suspicious spam with repeated punctuation and shouting
        text = "BUY NOW AT WWW.SCAMDEAL.COM !!! BEST PHONE EVER AAAAAAAAAA"
        rating = 5.0

        pipe_res = self.pipeline.analyze_single(text=text, rating=rating)
        st_res = analyze_custom_review(text=text, rating=rating)

        self.assertEqual(pipe_res.spam.is_spam, True)
        self.assertEqual(st_res["spam_analysis"]["is_spam"], True)
        self.assertAlmostEqual(pipe_res.spam.spam_score, st_res["spam_analysis"]["spam_score"], places=2)


if __name__ == "__main__":
    unittest.main()
