"""
Unit tests for Custom Review CSV Analyzer (modules/csv_analyzer.py).
Tests:
  - Valid review batch analysis with KPIs and aspect decomposition.
  - Word decomposition for explainability.
  - Missing text column raises ValueError.
  - Row cap enforcement.
"""

import unittest
import pandas as pd
from modules.csv_analyzer import analyze_custom_reviews_csv, decompose_review_words


class TestCSVAnalyzer(unittest.TestCase):

    def setUp(self):
        self.mock_df = pd.DataFrame([
            {"review_id": "U1", "review_text": "Super fast processor and beautiful vibrant display.", "rating": 5.0},
            {"review_id": "U2", "review_text": "Battery drains within 2 hours. Terribly defective and slow.", "rating": 1.0},
            {"review_id": "U3", "review_text": "Standard packaging, arrived on time.", "rating": 3.0},
            {"review_id": "U4", "review_text": "BUY NOW FREE DISCOUNT CODE ON TELEGRAM VISIT NOW!!!!", "rating": 5.0}
        ])

    def test_analyze_custom_reviews_csv(self):
        res = analyze_custom_reviews_csv(self.mock_df, category="Smartphones")
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["total_rows"], 4)
        self.assertFalse(res["is_capped"])

        kpis = res["kpis"]
        self.assertGreater(kpis["positive_count"], 0)
        self.assertGreater(kpis["negative_count"], 0)
        self.assertGreater(kpis["spam_count"], 0)
        self.assertIn("word_drivers", res)
        self.assertIn("positive", res["word_drivers"])
        self.assertIn("negative", res["word_drivers"])

    def test_word_decomposition(self):
        tokens = decompose_review_words("Amazing crystal clear screen but terrible battery.")
        self.assertGreater(len(tokens), 0)
        words = [t["word"].lower() for t in tokens]
        self.assertIn("amazing", words)
        self.assertIn("terrible", words)

        amazing_token = next(t for t in tokens if t["word"].lower() == "amazing")
        self.assertTrue(amazing_token["is_positive"])
        self.assertGreater(amazing_token["valence"], 0)

        terrible_token = next(t for t in tokens if t["word"].lower() == "terrible")
        self.assertTrue(terrible_token["is_negative"])
        self.assertLess(terrible_token["valence"], 0)

    def test_missing_text_column(self):
        bad_df = pd.DataFrame({"random_col": [1, 2, 3]})
        with self.assertRaises(ValueError):
            analyze_custom_reviews_csv(bad_df)

    def test_row_capping(self):
        large_df = pd.DataFrame({"text": ["Good product."] * 20})
        res = analyze_custom_reviews_csv(large_df, max_rows=10)
        self.assertEqual(res["total_rows"], 10)
        self.assertTrue(res["is_capped"])


if __name__ == "__main__":
    unittest.main()
