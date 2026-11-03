"""
Unit tests for ReviewLens Aspect-Based Sentiment Analysis Module.
"""

import unittest
import pandas as pd
from modules.aspect_analysis import (
    extract_aspects_from_sentence,
    analyze_reviews_aspects
)


class TestAspects(unittest.TestCase):

    def test_extract_aspects(self):
        s1 = "The camera takes stunning portrait shots and 4K video."
        asp1 = extract_aspects_from_sentence(s1)
        self.assertIn("Camera & Photography", asp1)

        s2 = "Battery life easily lasts 2 days and the 100W fast charger is incredible."
        asp2 = extract_aspects_from_sentence(s2)
        self.assertIn("Battery & Charging", asp2)

        s3 = "Dynamic AMOLED screen with 120Hz refresh is bright and buttery smooth."
        asp3 = extract_aspects_from_sentence(s3)
        self.assertIn("Display & Screen", asp3)

    def test_analyze_reviews_aspects_aggregation(self):
        reviews = pd.DataFrame([
            {"review_id": "r1", "review_text": "Great battery life that lasts all day.", "review_rating": 5.0},
            {"review_id": "r2", "review_text": "Battery drains quickly while gaming.", "review_rating": 2.0},
            {"review_id": "r3", "review_text": "Awesome display with vibrant colors.", "review_rating": 5.0}
        ])

        res = analyze_reviews_aspects(reviews)
        aspects = res["aspects"]
        self.assertGreater(len(aspects), 0)

        battery_asp = next((a for a in aspects if a["category"] == "Battery & Charging"), None)
        self.assertIsNotNone(battery_asp)
        self.assertEqual(battery_asp["positive_count"], 1)
        self.assertEqual(battery_asp["negative_count"], 1)
        self.assertEqual(battery_asp["mentions"], 2)
        self.assertEqual(battery_asp["net_sentiment_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
