"""Integration tests for end-to-end ReviewLens pipeline."""

import unittest
from reviewlens.pipeline import ReviewLensPipeline


class TestPipeline(unittest.TestCase):

    def setUp(self):
        self.pipeline = ReviewLensPipeline()

    def test_single_review_pipeline(self):
        text = "The screen and display are bright and sharp, but the battery drains fast."
        analysis = self.pipeline.analyze_single(text, rating=3.5)

        self.assertIsNotNone(analysis.sentiment)
        self.assertIsNotNone(analysis.predicted_rating)
        self.assertFalse(analysis.spam.is_spam)
        self.assertTrue(len(analysis.aspects) >= 2)

    def test_batch_review_pipeline(self):
        reviews = [
            {"id": "1", "text": "Amazing sound and great battery life! Highly recommend.", "rating": 5.0},
            {"id": "2", "text": "Terrible customer service and broken plastic hinge.", "rating": 1.0},
            {"id": "3", "text": "Decent performance for the price point.", "rating": 3.0},
            {"id": "4", "text": "Super comfortable memory foam ear cushions.", "rating": 5.0},
            {"id": "5", "text": "BUY NOW VISIT WEBSITE FREE GIFT DISCOUNT CODE!!!!", "rating": 5.0},
        ]
        batch_result = self.pipeline.analyze_dataset(reviews)

        self.assertEqual(batch_result.total_reviews, 5)
        self.assertEqual(batch_result.spam_reviews, 1)
        self.assertGreaterEqual(batch_result.valid_reviews, 4)
        self.assertIsNotNone(batch_result.executive_summary)
        self.assertIsNotNone(batch_result.executive_summary.net_sentiment_score)
        self.assertTrue(len(batch_result.aspect_summaries) > 0)


if __name__ == "__main__":
    unittest.main()
