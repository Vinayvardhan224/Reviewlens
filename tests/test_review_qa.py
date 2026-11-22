"""
Unit tests for the ReviewLens Review Q&A Module (modules/review_qa.py).
Tests:
  - Empty or whitespace query handling.
  - Non-existent product ID handling.
  - Extractive retrieval with valid question.
  - Citation format verification ([Review #...]).
  - Sentiment consensus calculation.
  - Top-k parameter limit.
"""

import unittest
import pandas as pd
from modules.review_qa import answer_product_review_question, split_into_sentences


class TestReviewQA(unittest.TestCase):

    def setUp(self):
        # Create a small mock reviews DataFrame for deterministic unit testing
        self.mock_reviews = pd.DataFrame([
            {
                "review_id": "REV-001",
                "product_id": "test-phone-x",
                "review_text": "The battery life easily lasts two full days. However, the camera struggles in low light.",
                "rating": 4.0
            },
            {
                "review_id": "REV-002",
                "product_id": "test-phone-x",
                "review_text": "Super fast 65W charging gets from zero to 100% in 30 minutes flat.",
                "rating": 5.0
            },
            {
                "review_id": "REV-003",
                "product_id": "test-phone-x",
                "review_text": "The display is vibrant with smooth 120Hz scrolling. The speaker sounds tinny.",
                "rating": 3.0
            }
        ])

    def test_split_into_sentences(self):
        text = "Great battery life! The screen is bright. Audio could be better."
        sents = split_into_sentences(text)
        self.assertEqual(len(sents), 3)
        self.assertIn("Great battery life!", sents)

    def test_empty_query(self):
        res = answer_product_review_question("test-phone-x", "", reviews_df=self.mock_reviews)
        self.assertEqual(res["status"], "empty_query")
        self.assertEqual(len(res["citations"]), 0)

        res_spaces = answer_product_review_question("test-phone-x", "   ", reviews_df=self.mock_reviews)
        self.assertEqual(res_spaces["status"], "empty_query")

    def test_non_existent_product(self):
        res = answer_product_review_question("non-existent-device-xyz", "How is the battery?", reviews_df=self.mock_reviews)
        self.assertEqual(res["status"], "no_reviews")
        self.assertEqual(len(res["citations"]), 0)

    def test_retrieval_with_citations(self):
        res = answer_product_review_question(
            "test-phone-x",
            "How long does the battery last?",
            top_k=2,
            reviews_df=self.mock_reviews
        )
        self.assertEqual(res["status"], "ok")
        self.assertGreater(len(res["citations"]), 0)

        # Check citation structure
        top_cit = res["citations"][0]
        self.assertIn("review_id", top_cit)
        self.assertIn("citation", top_cit)
        self.assertTrue(top_cit["citation"].startswith("[Review #"))
        self.assertIn("REV-001", top_cit["citation"])
        self.assertIn("battery", top_cit["sentence"].lower())
        self.assertIn("sentiment", top_cit)

        # Check consensus and sentiment summary
        self.assertIn("sentiment_summary", res)
        self.assertIn("positive", res["sentiment_summary"])

    def test_unrelated_query(self):
        res = answer_product_review_question(
            "test-phone-x",
            "quantum teleportation warp drive mechanics",
            reviews_df=self.mock_reviews
        )
        self.assertEqual(res["status"], "no_match")
        self.assertEqual(len(res["citations"]), 0)

    def test_top_k_limit(self):
        res = answer_product_review_question(
            "test-phone-x",
            "battery charging screen display",
            top_k=1,
            reviews_df=self.mock_reviews
        )
        self.assertEqual(res["status"], "ok")
        self.assertEqual(len(res["citations"]), 1)

    def test_real_catalogue_qa(self):
        """Test against real catalogue data loaded from disk."""
        res = answer_product_review_question(
            product_id="apple-macbook-air-m3",
            question="How is the M3 performance and cooling?"
        )
        self.assertEqual(res["status"], "ok")
        self.assertGreater(len(res["citations"]), 0)
        self.assertIn("apple-macbook-air-m3", res["product_id"])


if __name__ == "__main__":
    unittest.main()
