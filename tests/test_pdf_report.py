"""
Unit tests for ReviewLens Executive PDF Report generator (modules/pdf_report.py).
Verifies:
  - Valid PDF header (%PDF).
  - Generation with complete data.
  - Generation with minimal/empty aspect and summary data.
  - Generation with spam audit summary included.
"""

import unittest
from modules.pdf_report import generate_product_pdf_report


class TestPDFReport(unittest.TestCase):

    def setUp(self):
        self.prod = {
            "product_id": "test-device-01",
            "product_name": "ReviewLens Ultra Pro Audio",
            "brand": "AcousticTech",
            "category": "Headphones",
            "price": 19999.0,
            "rating": 4.6
        }
        self.aspect_list = [
            {"category": "Audio Quality", "net_sentiment_score": 85.0, "mentions": 12, "has_evidence": True},
            {"category": "Battery Life", "net_sentiment_score": 60.0, "mentions": 8, "has_evidence": True},
            {"category": "Comfort", "net_sentiment_score": -20.0, "mentions": 4, "has_evidence": True}
        ]
        self.sentiment_res = {
            "csat_estimate": 88,
            "nss": 45.0,
            "total_reviews": 24
        }
        self.exec_summary = {
            "pros": [
                {"aspect": "Audio Quality", "nss": 85.0, "quote": "Crystal clear sound with tight bass response."}
            ],
            "cons": [
                {"aspect": "Comfort", "nss": -20.0, "quote": "Ear cushions become warm after two hours."}
            ]
        }

    def test_pdf_generation_valid_header(self):
        pdf_bytes = generate_product_pdf_report(
            self.prod,
            self.aspect_list,
            self.sentiment_res,
            self.exec_summary
        )
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 1000)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"), "Output must start with %PDF magic bytes")

    def test_pdf_generation_with_spam_audit(self):
        spam_summary = {
            "flagged_count": 2,
            "total_checked": 24,
            "burst_status": "No velocity spikes detected",
            "duplicate_status": "0 duplicate review text matches"
        }
        pdf_bytes = generate_product_pdf_report(
            self.prod,
            self.aspect_list,
            self.sentiment_res,
            self.exec_summary,
            spam_summary=spam_summary
        )
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        self.assertGreater(len(pdf_bytes), 1500)

    def test_pdf_generation_empty_pros_cons(self):
        pdf_bytes = generate_product_pdf_report(
            self.prod,
            [],
            {"csat_estimate": 0, "nss": 0.0, "total_reviews": 0},
            {"pros": [], "cons": []}
        )
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()
