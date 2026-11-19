"""
Unit tests for ReviewLens Data Validation Module.
Verifies exhaustive edge cases:
  1. Valid products and reviews DataFrames
  2. Missing required columns
  3. Non-numeric and negative prices
  4. Out-of-bounds and NaN ratings
  5. Duplicate primary keys
  6. Empty / whitespace-only review texts
  7. Referential integrity & orphan review detection
  8. Canonical column mapping ('rating' vs 'review_rating', 'text' vs 'review_text')
  9. Complex delimiter injection (commas, quotes, newlines in fields)
"""

import unittest
import pandas as pd
import io
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

    def test_non_numeric_and_negative_price(self):
        # Non-numeric price
        df_str = pd.DataFrame([
            {"product_id": "p1", "product_name": "A", "brand": "B", "category": "C", "price": "Free", "rating": 4.0}
        ])
        res_str = validate_products_df(df_str)
        self.assertFalse(res_str.is_valid)
        self.assertTrue(any("non-numeric" in e.lower() for e in res_str.errors))

        # Negative price
        df_neg = pd.DataFrame([
            {"product_id": "p1", "product_name": "A", "brand": "B", "category": "C", "price": -150.0, "rating": 4.0}
        ])
        res_neg = validate_products_df(df_neg)
        self.assertFalse(res_neg.is_valid)
        self.assertTrue(any("negative price" in e.lower() for e in res_neg.errors))

    def test_out_of_bounds_and_nan_rating(self):
        # > 5.0
        df1 = pd.DataFrame([
            {"product_id": "p1", "product_name": "A", "brand": "B", "category": "C", "price": 100.0, "rating": 5.5}
        ])
        res1 = validate_products_df(df1)
        self.assertFalse(res1.is_valid)
        self.assertTrue(any("outside valid range" in e for e in res1.errors))

        # Non-numeric / NaN rating
        df2 = pd.DataFrame([
            {"product_id": "p1", "product_name": "A", "brand": "B", "category": "C", "price": 100.0, "rating": "Excellent"}
        ])
        res2 = validate_products_df(df2)
        self.assertFalse(res2.is_valid)
        self.assertTrue(any("non-numeric" in e.lower() for e in res2.errors))

    def test_duplicate_product_ids(self):
        df = pd.DataFrame([
            {"product_id": "p1", "product_name": "A", "brand": "B", "category": "C", "price": 10, "rating": 4},
            {"product_id": "p1", "product_name": "D", "brand": "B", "category": "C", "price": 20, "rating": 5}
        ])
        res = validate_products_df(df)
        self.assertFalse(res.is_valid)
        self.assertTrue(any("duplicate product_id" in e for e in res.errors))

    def test_valid_reviews_canonical_and_aliases(self):
        # Case A: canonical column names ('review_text' and 'rating')
        products_df = pd.DataFrame([{"product_id": "p1"}])
        rev_a = pd.DataFrame([
            {"review_id": "r1", "product_id": "p1", "review_text": "Battery is great.", "rating": 5.0}
        ])
        res_a = validate_reviews_df(rev_a, products_df=products_df)
        self.assertTrue(res_a.is_valid)

        # Case B: legacy column names ('text' and 'review_rating')
        rev_b = pd.DataFrame([
            {"review_id": "r1", "product_id": "p1", "text": "Screen is bright.", "review_rating": 4.5}
        ])
        res_b = validate_reviews_df(rev_b, products_df=products_df)
        self.assertTrue(res_b.is_valid)

    def test_orphan_reviews_detection(self):
        products_df = pd.DataFrame([{"product_id": "p1"}])
        reviews_df = pd.DataFrame([
            {"review_id": "r1", "product_id": "p1", "review_text": "Great battery", "rating": 5.0},
            {"review_id": "r2", "product_id": "p_unknown_orphan", "review_text": "Good screen", "rating": 4.0}
        ])
        res = validate_reviews_df(reviews_df, products_df=products_df)
        self.assertFalse(res.is_valid)
        self.assertTrue(any("orphan" in e.lower() for e in res.errors))

    def test_all_empty_reviews_fails(self):
        df = pd.DataFrame([
            {"review_id": "r1", "product_id": "p1", "review_text": "   ", "rating": 4.0},
            {"review_id": "r2", "product_id": "p1", "review_text": "", "rating": 5.0}
        ])
        res = validate_reviews_df(df)
        self.assertFalse(res.is_valid)
        self.assertTrue(any("empty" in e.lower() for e in res.errors))

    def test_delimiter_injection_csv_parsing(self):
        # CSV with embedded commas, quotes, and newlines
        csv_data = (
            'review_id,product_id,review_text,rating\n'
            'r1,p1,"Great display, but the battery; alas, it died.\nTakes 2 hours to charge.",4.0\n'
            'r2,p1,"Works as advertised, \\"premium\\" finish.",5.0\n'
        )
        parsed_df = pd.read_csv(io.StringIO(csv_data))
        res = validate_reviews_df(parsed_df)
        self.assertTrue(res.is_valid)
        self.assertEqual(len(parsed_df), 2)
        self.assertIn("Takes 2 hours", parsed_df.iloc[0]["review_text"])


if __name__ == "__main__":
    unittest.main()
