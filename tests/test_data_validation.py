"""
Unit tests for ReviewLens Data Validation Module.
"""

import unittest
import pandas as pd
from modules.validation import validate_products_df, validate_reviews_df


class TestDataValidation(unittest.TestCase):

    def test_valid_products_df(self):
        df = pd.DataFrame([
            {
                "product_id": "test-prod-1",
                "product_name": "Test Phone",
                "brand": "TestBrand",
                "category": "Smartphones",
                "price": 49999.0,
                "rating": 4.5,
                "data_type": "prepared_demo"
            }
        ])
        res = validate_products_df(df)
        self.assertTrue(res.is_valid)
        self.assertEqual(len(res.errors), 0)
        self.assertEqual(res.stats["total_products"], 1)

    def test_missing_required_columns(self):
        df = pd.DataFrame([
            {"product_id": "p1", "price": 100}
        ])
        res = validate_products_df(df)
        self.assertFalse(res.is_valid)
        self.assertTrue(any("Missing required product columns" in e for e in res.errors))

    def test_negative_price_and_out_of_bounds_rating(self):
        df = pd.DataFrame([
            {
                "product_id": "p1",
                "product_name": "Name",
                "brand": "Brand",
                "category": "Cat",
                "price": -50.0,
                "rating": 6.5
            }
        ])
        res = validate_products_df(df)
        self.assertFalse(res.is_valid)
        self.assertTrue(any("negative price" in e for e in res.errors))
        self.assertTrue(any("ratings outside valid range" in e for e in res.errors))

    def test_duplicate_product_ids(self):
        df = pd.DataFrame([
            {"product_id": "p1", "product_name": "A", "brand": "B", "category": "C", "price": 10, "rating": 4},
            {"product_id": "p1", "product_name": "D", "brand": "B", "category": "C", "price": 20, "rating": 5}
        ])
        res = validate_products_df(df)
        self.assertFalse(res.is_valid)
        self.assertTrue(any("duplicate product_id" in e for e in res.errors))

    def test_valid_reviews_and_orphan_detection(self):
        products_df = pd.DataFrame([{"product_id": "p1"}])
        reviews_df = pd.DataFrame([
            {"review_id": "r1", "product_id": "p1", "review_text": "Great battery", "review_rating": 5.0},
            {"review_id": "r2", "product_id": "p_orphan", "review_text": "Good screen", "review_rating": 4.0}
        ])
        res = validate_reviews_df(reviews_df, products_df=products_df)
        # Strict validation treats orphan product_ids as validation errors
        self.assertFalse(res.is_valid)
        self.assertTrue(any("orphan" in e.lower() for e in res.errors))


if __name__ == "__main__":
    unittest.main()
