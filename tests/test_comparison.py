"""Unit tests for competitor benchmarking and product comparison."""

import unittest
from reviewlens.pipeline import ReviewLensPipeline
from reviewlens.comparison import ProductComparator


class TestComparison(unittest.TestCase):

    def setUp(self):
        self.pipeline = ReviewLensPipeline()

    def test_product_comparison(self):
        reviews_a = [
            {"id": "1", "text": "Stunning 120Hz display with gorgeous brightness.", "rating": 5.0},
            {"id": "2", "text": "Phone gets dangerously hot and overheats while gaming.", "rating": 2.0},
        ]
        reviews_b = [
            {"id": "1", "text": "Stays super cool during heavy gaming without any heat.", "rating": 5.0},
            {"id": "2", "text": "Screen brightness is low and hard to see in sunlight.", "rating": 3.0},
        ]

        res_a = self.pipeline.analyze_dataset(reviews_a)
        res_b = self.pipeline.analyze_dataset(reviews_b)

        report = ProductComparator.compare(res_a, res_b, name_a="Phone A", name_b="Phone B")

        self.assertIsNotNone(report.overall_winner)
        self.assertIsNotNone(report.executive_verdict)
        self.assertIsNotNone(report.summary)
        self.assertEqual(report.summary, report.executive_verdict)
        self.assertTrue(len(report.aspect_comparisons) > 0)


if __name__ == "__main__":
    unittest.main()
