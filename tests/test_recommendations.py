"""
Unit tests for ReviewLens Recommendation Engine.
"""

import unittest
from modules.recommendation import (
    compute_product_recommendation_score,
    generate_category_recommendations
)


class TestRecommendations(unittest.TestCase):

    def test_composite_score_calculation(self):
        prod = {"rating": 5.0}
        summary = {"nss": 100.0, "csat_estimate": 100.0}
        # 0.30*(100) + 0.45*(100) + 0.25*(100) = 100.0
        score = compute_product_recommendation_score(prod, summary)
        self.assertEqual(score, 100.0)

        prod_mid = {"rating": 4.0}
        summary_mid = {"nss": 0.0, "csat_estimate": 50.0}
        # 0.30*(80) + 0.45*(50) + 0.25*(50) = 24 + 22.5 + 12.5 = 59.0
        score_mid = compute_product_recommendation_score(prod_mid, summary_mid)
        self.assertEqual(score_mid, 59.0)

    def test_generate_category_recommendations(self):
        items = [
            {
                "product": {
                    "product_id": "flagship-1",
                    "product_name": "Flagship Pro",
                    "category": "Smartphones",
                    "price": 120000.0,
                    "rating": 4.8
                },
                "summary": {
                    "nss": 85.0,
                    "csat_estimate": 92.0,
                    "aspects": [
                        {"category": "Camera & Photography", "net_sentiment_score": 95.0, "mentions": 10},
                        {"category": "Battery & Charging", "net_sentiment_score": 70.0, "mentions": 10},
                        {"category": "Performance & Speed", "net_sentiment_score": 90.0, "mentions": 10}
                    ]
                }
            },
            {
                "product": {
                    "product_id": "budget-1",
                    "product_name": "Budget Champ",
                    "category": "Smartphones",
                    "price": 25000.0,
                    "rating": 4.4
                },
                "summary": {
                    "nss": 65.0,
                    "csat_estimate": 80.0,
                    "aspects": [
                        {"category": "Camera & Photography", "net_sentiment_score": 50.0, "mentions": 10},
                        {"category": "Battery & Charging", "net_sentiment_score": 92.0, "mentions": 10},
                        {"category": "Performance & Speed", "net_sentiment_score": 60.0, "mentions": 10}
                    ]
                }
            }
        ]

        recs = generate_category_recommendations(items, category="Smartphones")
        self.assertIn("best_overall", recs)
        self.assertEqual(recs["best_overall"]["product"]["product_id"], "flagship-1")

        self.assertIn("best_budget", recs)
        self.assertEqual(recs["best_budget"]["product"]["product_id"], "budget-1")

        self.assertIn("best_camera", recs)
        self.assertEqual(recs["best_camera"]["product"]["product_id"], "flagship-1")

        self.assertIn("best_battery", recs)
        self.assertEqual(recs["best_battery"]["product"]["product_id"], "budget-1")


if __name__ == "__main__":
    unittest.main()
