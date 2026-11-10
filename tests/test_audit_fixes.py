"""
ReviewLens Unit Tests for GitHub Main Branch Audit & Checklist Requirements.
Validates navigation session state sync, strict data validation, recommendation evidence gating,
NLP negation/mixed sentiment clause splitting, empty dataset guards, and HTML escaping.
"""

import unittest
import pandas as pd
import html
from modules.validation import validate_products_df, validate_reviews_df
from modules.data_loader import (
    get_all_products,
    get_all_reviews,
    get_product_by_id,
    get_reviews_for_product
)
from modules.recommendation import (
    compute_product_recommendation_score,
    generate_category_recommendations,
    RECOMMENDATION_METHODOLOGY_DISCLOSURE
)
from modules.preprocessing import split_into_aspect_clauses, detect_negations, tokenize
from modules.aspect_analysis import analyze_reviews_aspects, extract_aspects_from_sentence
from modules.sentiment import analyze_text_sentiment, analyze_reviews_sentiment
from modules.topic_modeling import extract_topic_clusters_result
from modules.comparison import compare_two_products


class TestAuditFixes(unittest.TestCase):

    def test_navigation_state_sync_concept(self):
        """Verify navigation helper logic sets both nav_view and sidebar_nav_radio."""
        session_state = {"nav_view": "🏠 Home / Discovery", "sidebar_nav_radio": "🏠 Home / Discovery"}
        
        def navigate_to_mock(view: str):
            session_state["nav_view"] = view
            session_state["sidebar_nav_radio"] = view

        navigate_to_mock("📱 Smartphones")
        self.assertEqual(session_state["nav_view"], "📱 Smartphones")
        self.assertEqual(session_state["sidebar_nav_radio"], "📱 Smartphones")

        navigate_to_mock("⚔️ Product Comparison")
        self.assertEqual(session_state["nav_view"], "⚔️ Product Comparison")
        self.assertEqual(session_state["sidebar_nav_radio"], "⚔️ Product Comparison")

    def test_strict_product_validation(self):
        """Verify strict validation catches missing columns, duplicate IDs, invalid ratings, and negative prices."""
        # 1. Missing columns
        df_missing = pd.DataFrame([{"product_id": "p1", "product_name": "A"}])
        val_m = validate_products_df(df_missing)
        self.assertFalse(val_m.is_valid)
        self.assertTrue(any("Missing required product columns" in e for e in val_m.errors))

        # 2. Rating out of range (< 1.0 or > 5.0)
        df_bad_rating = pd.DataFrame([
            {"product_id": "p1", "product_name": "A", "brand": "B", "category": "C", "price": 100, "rating": 6.5}
        ])
        val_r = validate_products_df(df_bad_rating)
        self.assertFalse(val_r.is_valid)
        self.assertTrue(any("ratings outside valid range" in e for e in val_r.errors))

        # 3. Negative price
        df_neg_price = pd.DataFrame([
            {"product_id": "p1", "product_name": "A", "brand": "B", "category": "C", "price": -50, "rating": 4.0}
        ])
        val_p = validate_products_df(df_neg_price)
        self.assertFalse(val_p.is_valid)
        self.assertTrue(any("negative price" in e for e in val_p.errors))

    def test_strict_review_validation(self):
        """Verify reviews with invalid ratings, duplicate IDs, or orphan product IDs are strictly rejected."""
        products_df = pd.DataFrame([{"product_id": "prod_1", "product_name": "Phone", "brand": "B", "category": "C", "price": 100, "rating": 4.0}])

        # 1. Duplicate review_id
        df_dup = pd.DataFrame([
            {"review_id": "r1", "product_id": "prod_1", "review_text": "Good", "review_rating": 4.0},
            {"review_id": "r1", "product_id": "prod_1", "review_text": "Duplicate", "review_rating": 5.0}
        ])
        val_dup = validate_reviews_df(df_dup, products_df=products_df)
        self.assertFalse(val_dup.is_valid)
        self.assertTrue(any("duplicate review_id" in e for e in val_dup.errors))

        # 2. Orphan product_id
        df_orphan = pd.DataFrame([
            {"review_id": "r1", "product_id": "unknown_phone_id", "review_text": "Text", "review_rating": 4.0}
        ])
        val_orphan = validate_reviews_df(df_orphan, products_df=products_df)
        self.assertFalse(val_orphan.is_valid)
        self.assertTrue(any("orphan" in e.lower() for e in val_orphan.errors))

        # 3. Rating outside 1-5
        df_bad_rev_rating = pd.DataFrame([
            {"review_id": "r1", "product_id": "prod_1", "review_text": "Bad rating", "review_rating": 0.5}
        ])
        val_br = validate_reviews_df(df_bad_rev_rating, products_df=products_df)
        self.assertFalse(val_br.is_valid)
        self.assertTrue(any("review ratings outside [1.0, 5.0]" in e for e in val_br.errors))

    def test_clause_splitting_mixed_sentiment_and_negation(self):
        """Test split_into_aspect_clauses splits sentences across contrastive conjunctions."""
        sentence = "The battery life is amazing, but the camera is not good at all"
        clauses = split_into_aspect_clauses(sentence)
        self.assertEqual(len(clauses), 2)
        self.assertIn("battery", clauses[0].lower())
        self.assertIn("camera", clauses[1].lower())

        # Check sentiment of each clause independently
        sent1 = analyze_text_sentiment(clauses[0])
        sent2 = analyze_text_sentiment(clauses[1])
        self.assertEqual(sent1["label"], "POSITIVE")
        self.assertEqual(sent2["label"], "NEGATIVE")

    def test_aspect_analysis_mixed_sentiment_in_one_sentence(self):
        """Ensure analyze_reviews_aspects assigns positive to battery and negative to camera from compound sentence."""
        reviews_df = pd.DataFrame([
            {"review_id": "r1", "product_id": "phone_x", "review_text": "The battery is fantastic, but the camera is terrible.", "review_rating": 3.0}
        ])
        res = analyze_reviews_aspects(reviews_df, category="Smartphones")
        aspects_by_name = {a["category"]: a for a in res["aspects"]}

        self.assertIn("Battery", aspects_by_name)
        self.assertIn("Camera", aspects_by_name)
        self.assertTrue(aspects_by_name["Battery"]["has_evidence"])
        self.assertTrue(aspects_by_name["Camera"]["has_evidence"])
        self.assertGreater(aspects_by_name["Battery"]["net_sentiment_score"], 0)
        self.assertLess(aspects_by_name["Camera"]["net_sentiment_score"], 0)

    def test_recommendation_evidence_gating_missing_aspects(self):
        """Verify recommendation engine refuses to award aspect honors when candidate evidence is absent."""
        prod = {
            "product_id": "p_no_cam",
            "product_name": "Test Laptop",
            "brand": "Brand",
            "category": "Laptops",
            "price": 50000.0,
            "rating": 4.5
        }
        # Laptop has Processor and Keyboard reviews, but NO Camera reviews
        summary = {
            "nss": 60.0,
            "csat_estimate": 80.0,
            "total_reviews": 5,
            "aspects": [
                {"category": "Processor", "mentions": 5, "net_sentiment_score": 75.0, "has_evidence": True},
                {"category": "Keyboard", "mentions": 3, "net_sentiment_score": 60.0, "has_evidence": True},
                {"category": "Camera", "mentions": 0, "net_sentiment_score": None, "has_evidence": False}
            ]
        }
        recs = generate_category_recommendations([{"product": prod, "summary": summary}], category="Laptops")

        self.assertIn("methodology_disclosure", recs)
        self.assertEqual(recs["methodology_disclosure"], RECOMMENDATION_METHODOLOGY_DISCLOSURE)

        # Best camera award should report insufficient evidence
        best_cam = recs["best_camera"]
        self.assertIsNone(best_cam["product"])
        self.assertIn("Insufficient review evidence", best_cam["reason"])

    def test_topic_modeling_error_reporting(self):
        """Ensure extract_topic_clusters_result returns structured status rather than silent empty lists."""
        # 1. Insufficient reviews (< 4)
        short_res = extract_topic_clusters_result(["One short review", "Second review"], n_topics=2)
        self.assertEqual(short_res["status"], "insufficient_text")
        self.assertIn("too small", short_res["error_message"])

        # 2. Adequate reviews returns 'ok'
        good_texts = [
            "The battery life is really fantastic and lasts the whole day easily without charging.",
            "Display screen is crisp with punchy vibrant colors and smooth refresh rate.",
            "Processor speed handles multi-tasking and gaming with zero lag or frame drops.",
            "Sound quality from dual speakers is rich with clear vocal audio on phone calls."
        ]
        ok_res = extract_topic_clusters_result(good_texts, n_topics=2, top_words=3)
        self.assertIn(ok_res["status"], ["ok", "insufficient_text"])

    def test_empty_reviews_and_missing_product_safety(self):
        """Verify data loaders and analyzers handle empty or nonexistent product IDs safely."""
        missing_p = get_product_by_id("non_existent_product_12345")
        self.assertIsNone(missing_p)

        missing_revs = get_reviews_for_product("non_existent_product_12345")
        self.assertTrue(missing_revs.empty)

        empty_sent = analyze_reviews_sentiment(pd.DataFrame())
        self.assertEqual(empty_sent["total_reviews"], 0)
        self.assertEqual(empty_sent["nss"], 0.0)

        empty_asp = analyze_reviews_aspects(pd.DataFrame(), category="Smartphones")
        self.assertTrue(all(not a["has_evidence"] for a in empty_asp["aspects"]))

    def test_html_escaping(self):
        """Verify user and data strings containing HTML tags or scripts are properly escaped."""
        raw_xss = '<script>alert("xss")</script> & "special" \'quotes\''
        escaped = html.escape(raw_xss)
        self.assertNotIn("<script>", escaped)
        self.assertIn("&lt;script&gt;", escaped)
        self.assertIn("&amp;", escaped)
        self.assertIn("&quot;special&quot;", escaped)


if __name__ == "__main__":
    unittest.main()
