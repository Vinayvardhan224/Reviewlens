"""
Unit tests for ReviewLens Category-Specific Aspect-Based Sentiment Analysis Module.
"""

import unittest
import pandas as pd
from modules.aspect_analysis import (
    extract_aspects_from_sentence,
    analyze_reviews_aspects
)


class TestAspects(unittest.TestCase):

    def test_extract_aspects_smartphones(self):
        s1 = "The camera takes stunning portrait shots and 4K video."
        asp1 = extract_aspects_from_sentence(s1, category="Smartphones")
        self.assertIn("Camera", asp1)

        s2 = "Charging takes 26 minutes with the 100W fast charger."
        asp2 = extract_aspects_from_sentence(s2, category="Smartphones")
        self.assertIn("Charging", asp2)

        s3 = "Dynamic AMOLED display is super bright with 120Hz refresh."
        asp3 = extract_aspects_from_sentence(s3, category="Smartphones")
        self.assertIn("Display", asp3)

    def test_extract_aspects_laptops(self):
        s1 = "The M3 processor compiles large codebases in seconds."
        asp1 = extract_aspects_from_sentence(s1, category="Laptops")
        self.assertIn("Processor", asp1)

        s2 = "Cooling fan noise is very loud under gaming loads."
        asp2 = extract_aspects_from_sentence(s2, category="Laptops")
        self.assertIn("Cooling", asp2)

        s3 = "The Magic Keyboard typing feel is tactile and comfortable."
        asp3 = extract_aspects_from_sentence(s3, category="Laptops")
        self.assertIn("Keyboard", asp3)

    def test_extract_aspects_headphones(self):
        s1 = "Active noise cancellation completely silences airplane hum."
        asp1 = extract_aspects_from_sentence(s1, category="Headphones")
        self.assertIn("Noise cancellation", asp1)

        s2 = "Extremely comfortable ear cushions with zero clamping fatigue."
        asp2 = extract_aspects_from_sentence(s2, category="Headphones")
        self.assertIn("Comfort", asp2)

        s3 = "Bluetooth connectivity switches seamlessly across devices."
        asp3 = extract_aspects_from_sentence(s3, category="Headphones")
        self.assertIn("Connectivity", asp3)

    def test_extract_aspects_cameras(self):
        s1 = "Eye autofocus tracking locks onto flying birds instantaneously."
        asp1 = extract_aspects_from_sentence(s1, category="Cameras")
        self.assertIn("Autofocus", asp1)

        s2 = "Superb low-light performance with clean shadow recovery at ISO 12800."
        asp2 = extract_aspects_from_sentence(s2, category="Cameras")
        self.assertIn("Low-light performance", asp2)

    def test_missing_aspect_evidence_handling(self):
        # When an aspect is not mentioned in any reviews
        reviews = pd.DataFrame([
            {"review_id": "r1", "review_text": "Great battery life that lasts all day.", "review_rating": 5.0}
        ])

        res = analyze_reviews_aspects(reviews, category="Smartphones")
        aspects = res["aspects"]

        # Battery should have evidence
        battery_asp = next((a for a in aspects if a["category"] == "Battery"), None)
        self.assertIsNotNone(battery_asp)
        self.assertTrue(battery_asp["has_evidence"])
        self.assertEqual(battery_asp["net_sentiment_score"], 100.0)

        # Charging should have NO evidence
        charging_asp = next((a for a in aspects if a["category"] == "Charging"), None)
        self.assertIsNotNone(charging_asp)
        self.assertFalse(charging_asp["has_evidence"])
        self.assertIsNone(charging_asp["net_sentiment_score"])
        self.assertEqual(charging_asp["score_display"], "Insufficient review data")


if __name__ == "__main__":
    unittest.main()
