"""
Unit tests for ReviewLens new advanced features:
  1. Live Review Analyzer (modules/live_analyzer.py)
  2. Custom Preference Matchmaker (modules/recommendation.py)
  3. Temporal Sentiment Timeline Drift (modules/sentiment.py)
  4. Formatted Intelligence Dossier Export (modules/review_summary.py)
"""

import unittest
import pandas as pd
from modules.live_analyzer import analyze_custom_review
from modules.recommendation import rank_products_by_preferences
from modules.sentiment import compute_review_timeline_metrics
from modules.review_summary import generate_markdown_dossier


class TestLiveReviewAnalyzer(unittest.TestCase):

    def test_empty_review_input(self):
        res = analyze_custom_review("")
        self.assertEqual(res["status"], "empty")

    def test_positive_smartphone_review(self):
        text = "The battery life is phenomenal and lasts two full days. Camera produces crystal clear shots."
        res = analyze_custom_review(text, rating=5.0, category="Smartphones")
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["sentiment"]["label"], "POSITIVE")
        self.assertGreaterEqual(res["sentiment"]["compound"], 0.2)
        self.assertFalse(res["anomaly"]["is_anomaly"])
        self.assertFalse(res["spam_analysis"]["is_spam"])
        self.assertIn("Battery", res["aspects_found"])
        self.assertIn("Camera", res["aspects_found"])

    def test_mixed_review_clauses(self):
        text = "The display is stunning, but the battery drains ridiculously fast."
        res = analyze_custom_review(text, rating=3.0, category="Smartphones")
        self.assertEqual(res["status"], "ok")
        self.assertGreaterEqual(len(res["clause_analysis"]), 2)
        # Verify first clause positive, second negative
        self.assertEqual(res["clause_analysis"][0]["sentiment_label"], "POSITIVE")
        self.assertEqual(res["clause_analysis"][1]["sentiment_label"], "NEGATIVE")

    def test_spam_detection(self):
        text = "BUY NOW DISCOUNT FREE GIFT CLICK LINK http://spam.example.com BEST DEAL EVER BUY NOW"
        res = analyze_custom_review(text, rating=5.0, category="Smartphones")
        self.assertEqual(res["status"], "ok")
        self.assertTrue(res["spam_analysis"]["is_spam"])
        self.assertGreater(res["spam_analysis"]["spam_score"], 0.4)

    def test_rating_anomaly_contradiction(self):
        # 5 star rating with terrible negative text
        text = "Horrible device. Overheats constantly, terrible lag, and broken within a week."
        res = analyze_custom_review(text, rating=5.0, category="Smartphones")
        self.assertEqual(res["status"], "ok")
        self.assertTrue(res["anomaly"]["is_anomaly"])
        self.assertIn("Contradiction flagged", res["anomaly"]["description"])


class TestPreferenceMatchmaker(unittest.TestCase):

    def setUp(self):
        self.sample_items = [
            {
                "product": {
                    "product_id": "phone-battery-champ",
                    "product_name": "Battery Monster Max",
                    "category": "Smartphones",
                    "price": 30000,
                    "rating": 4.5
                },
                "summary": {
                    "aspects": [
                        {"category": "Battery", "has_evidence": True, "net_sentiment_score": 85.0},
                        {"category": "Camera", "has_evidence": True, "net_sentiment_score": 20.0}
                    ]
                }
            },
            {
                "product": {
                    "product_id": "phone-camera-champ",
                    "product_name": "Photo Pro Ultra",
                    "category": "Smartphones",
                    "price": 80000,
                    "rating": 4.8
                },
                "summary": {
                    "aspects": [
                        {"category": "Battery", "has_evidence": True, "net_sentiment_score": 30.0},
                        {"category": "Camera", "has_evidence": True, "net_sentiment_score": 90.0}
                    ]
                }
            }
        ]

    def test_battery_priority_ranking(self):
        weights = {"battery": 0.8, "camera": 0.1, "value_for_money": 0.1}
        ranked = rank_products_by_preferences(self.sample_items, weights)
        self.assertEqual(len(ranked), 2)
        # Battery Monster should rank #1 when battery is heavily weighted
        self.assertEqual(ranked[0]["product"]["product_id"], "phone-battery-champ")
        self.assertGreater(ranked[0]["match_score"], ranked[1]["match_score"])

    def test_budget_cutoff(self):
        weights = {"camera": 0.9}
        # Budget 50,000 should filter out the 80,000 Photo Pro Ultra
        ranked = rank_products_by_preferences(self.sample_items, weights, max_budget=50000)
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0]["product"]["product_id"], "phone-battery-champ")


class TestTemporalSentimentDrift(unittest.TestCase):

    def test_compute_review_timeline_metrics(self):
        data = {
            "review_date": [
                "2024-01-10", "2024-01-15",
                "2024-02-05", "2024-02-20",
                "2024-03-01"
            ],
            "review_rating": [5.0, 5.0, 1.0, 2.0, 4.0],
            "review_text": [
                "Amazing love this device so much!",
                "Great experience super fast and sleek.",
                "Terrible broken junk worst purchase ever.",
                "Poor performance overheats immediately.",
                "Decent product works fine for daily use."
            ]
        }
        df = pd.DataFrame(data)
        timeline = compute_review_timeline_metrics(df)
        self.assertEqual(len(timeline), 3)  # Jan, Feb, Mar
        self.assertEqual(timeline[0]["period"], "2024-01")
        self.assertEqual(timeline[0]["review_count"], 2)
        self.assertGreater(timeline[0]["net_sentiment_score"], 50.0)

        # Feb should be negative NSS
        self.assertEqual(timeline[1]["period"], "2024-02")
        self.assertLess(timeline[1]["net_sentiment_score"], -50.0)


class TestMarkdownDossierExport(unittest.TestCase):

    def test_generate_markdown_dossier(self):
        prod = {
            "product_id": "test-phone",
            "product_name": "Test Phone Pro",
            "category": "Smartphones",
            "brand": "TechCorp",
            "price": 45000,
            "rating": 4.4
        }
        aspects = [
            {"category": "Battery", "mentions": 12, "net_sentiment_score": 50.0, "has_evidence": True},
            {"category": "Camera", "mentions": 8, "net_sentiment_score": -20.0, "has_evidence": True}
        ]
        sentiment_res = {
            "nss": 35.0,
            "csat_estimate": 78.5,
            "total_reviews": 25
        }
        exec_summary = {
            "verdict": "Great device overall with high battery endurance.",
            "persona": "Power users who value battery over camera performance.",
            "pros": [{"aspect": "Battery", "nss": 50.0, "quote": "Lasts all day."}],
            "cons": [{"aspect": "Camera", "nss": -20.0, "quote": "Blurry in low light."}]
        }
        md = generate_markdown_dossier(prod, aspects, sentiment_res, exec_summary)
        self.assertIn("# ReviewLens Intelligence Dossier", md)
        self.assertIn("Test Phone Pro", md)
        self.assertIn("₹45,000", md)
        self.assertIn("Battery", md)
        self.assertIn("(ABSA) Scorecard", md)


if __name__ == "__main__":
    unittest.main()
