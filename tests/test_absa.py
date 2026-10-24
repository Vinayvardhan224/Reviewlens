"""Unit tests for Aspect-Based Sentiment Analysis (ABSA)."""

import unittest
from reviewlens.absa.aspect_categories import AspectTaxonomy
from reviewlens.absa.aspect_extractor import AspectExtractor
from reviewlens.absa.aspect_scorer import AspectScorer


class TestABSA(unittest.TestCase):

    def setUp(self):
        self.taxonomy = AspectTaxonomy()
        self.extractor = AspectExtractor(taxonomy=self.taxonomy)

    def test_aspect_mention_extraction(self):
        text = "The battery lasts for days and the sound clarity is amazing."
        mentions = self.extractor.extract_from_review(text)
        categories = [m.category for m in mentions]
        self.assertIn("Battery & Power", categories)
        self.assertIn("Sound & Audio", categories)

    def test_adversative_clause_splitting(self):
        text = "The audio quality is fantastic, but the earcups are extremely painful and uncomfortable."
        mentions = self.extractor.extract_from_review(text)

        audio_mentions = [m for m in mentions if m.category == "Sound & Audio"]
        comfort_mentions = [m for m in mentions if m.category == "Comfort & Ergonomics"]

        self.assertTrue(len(audio_mentions) > 0)
        self.assertEqual(audio_mentions[0].sentiment_label, "POSITIVE")

        self.assertTrue(len(comfort_mentions) > 0)
        self.assertEqual(comfort_mentions[0].sentiment_label, "NEGATIVE")

    def test_aspect_scorer_aggregation(self):
        text1 = "Great battery life."
        text2 = "Terrible battery life, drains too fast."
        mentions1 = self.extractor.extract_from_review(text1)
        mentions2 = self.extractor.extract_from_review(text2)

        summaries = AspectScorer.aggregate(mentions1 + mentions2)
        battery_summary = next((s for s in summaries if s.category == "Battery & Power"), None)

        self.assertIsNotNone(battery_summary)
        self.assertEqual(battery_summary.total_mentions, 2)
        self.assertEqual(battery_summary.positive_mentions, 1)
        self.assertEqual(battery_summary.negative_mentions, 1)
        self.assertEqual(battery_summary.net_sentiment_score, 0.0)


if __name__ == "__main__":
    unittest.main()
