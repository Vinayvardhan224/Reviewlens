"""Unit tests for Amazon review scraper and ASIN extraction."""

import unittest
from scraper.amazon_scraper import AmazonReviewScraper


class TestScraper(unittest.TestCase):

    def setUp(self):
        self.scraper = AmazonReviewScraper()

    def test_extract_asin_from_raw_string(self):
        asin = self.scraper.extract_asin("B084DCJKSL")
        self.assertEqual(asin, "B084DCJKSL")

    def test_extract_asin_from_url(self):
        url = "https://www.amazon.com/All-New-Echo-Dot-4th-Gen/dp/B07XJ8C8F5/ref=sr_1_1"
        asin = self.scraper.extract_asin(url)
        self.assertEqual(asin, "B07XJ8C8F5")

    def test_generate_realistic_dataset_fallback(self):
        dataset = self.scraper.generate_realistic_product_dataset("B084DCJKSL")
        self.assertGreater(len(dataset), 5)
        first = dataset[0]
        self.assertIn("text", first)
        self.assertIn("rating", first)
        self.assertIn("verified_purchase", first)


if __name__ == "__main__":
    unittest.main()
