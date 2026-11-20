"""
Unit tests for ReviewLens Evaluation Benchmarks and Topic Modeling.
"""

import unittest
from modules.evaluation import evaluate_sentiment_model, evaluate_spam_model
from modules.topic_modeling import extract_topic_clusters
from reviewlens.visualization.charts import create_keyword_bar_chart


class TestEvaluationAndTopics(unittest.TestCase):

    def test_sentiment_evaluation_benchmark(self):
        res = evaluate_sentiment_model()
        self.assertEqual(res["total_test_samples"], 50)
        self.assertGreaterEqual(res["accuracy"], 0.75)
        self.assertIn("confusion_matrix", res)
        self.assertEqual(len(res["confusion_matrix"]), 3)
        self.assertEqual(len(res["confusion_matrix"][0]), 3)
        self.assertIn("f1_macro", res)
        self.assertIn("per_class", res)
        self.assertIn("POSITIVE", res["per_class"])
        self.assertIn("NEGATIVE", res["per_class"])
        self.assertIn("NEUTRAL", res["per_class"])

    def test_spam_evaluation_benchmark(self):
        res = evaluate_spam_model()
        self.assertEqual(res["total_test_samples"], 50)
        self.assertGreaterEqual(res["accuracy"], 0.85)
        self.assertIn("precision", res)
        self.assertIn("recall", res)
        self.assertIn("f1_score", res)
        self.assertIsInstance(res["false_positives"], list)
        self.assertIsInstance(res["false_negatives"], list)

    def test_benchmark_dataset_file_loading(self):
        """Verify dynamic loading from benchmark_sentiment.csv and baseline comparison."""
        from modules.evaluation import load_benchmark_dataset, evaluate_baseline_comparison, evaluate_curated_reviews_corpus
        data = load_benchmark_dataset()
        self.assertEqual(len(data), 50)
        self.assertEqual(len(data[0]), 3)
        res = evaluate_baseline_comparison()
        self.assertIn("baseline_accuracy", res)
        self.assertIn("vader_accuracy", res)
        self.assertGreaterEqual(res["sample_size"], 50)
        
        # Test reviews.csv corpus evaluation
        reviews_res = evaluate_curated_reviews_corpus()
        self.assertEqual(reviews_res["status"], "ok")
        self.assertEqual(reviews_res["sample_size"], 101)
        self.assertGreater(reviews_res["agreement_rate"], 0.6)

    def test_topic_modeling_lda(self):
        sample_texts = [
            "The battery life is exceptional and easily lasts through 2 full days of intensive screen time.",
            "Display color reproduction is vivid and crisp with dynamic 120Hz refresh rate and OLED.",
            "Battery endurance is reliable, fast charging reaches 80 percent within 30 minutes without heat.",
            "Screen brightness outdoors under direct sunlight is easily readable and sharp.",
            "Camera image quality produces vibrant low light shots with great dynamic contrast.",
            "Superb camera optics capture crisp portrait photos with natural bokeh blur."
        ]
        clusters = extract_topic_clusters(sample_texts, n_topics=2, top_words=4)
        self.assertIsInstance(clusters, list)
        self.assertGreaterEqual(len(clusters), 1)
        for c in clusters:
            self.assertIn("topic_id", c)
            self.assertIn("label", c)
            self.assertIn("keywords", c)
            self.assertIn("sample_reviews", c)

    def test_create_keyword_bar_chart_signatures(self):
        """Verify create_keyword_bar_chart supports both single-column and dual-column signatures without TypeError."""
        pos_kw = [{"term": "battery", "score": 0.8}, {"term": "screen", "score": 0.6}]
        neg_kw = [{"term": "overheating", "score": 0.7}, {"term": "slow", "score": 0.5}]

        # 1. Single column signature with title and color (used in Original Dashboard)
        fig_pos = create_keyword_bar_chart(pos_kw, "Top Positive Feedback Drivers", "#10B981")
        self.assertIsNotNone(fig_pos)
        self.assertEqual(len(fig_pos.data), 1)

        fig_neg = create_keyword_bar_chart(neg_kw, "Top Customer Pain Points", "#EF4444")
        self.assertIsNotNone(fig_neg)
        self.assertEqual(len(fig_neg.data), 1)

        # 2. Dual comparison signature
        fig_dual = create_keyword_bar_chart(pos_kw, neg_kw)
        self.assertIsNotNone(fig_dual)
        self.assertEqual(len(fig_dual.data), 2)

    def test_sst2_public_benchmark(self):
        """Verify empirical evaluation on the official Stanford SST-2 dataset (n=872)."""
        from modules.evaluation import evaluate_sst2_public_benchmark
        res = evaluate_sst2_public_benchmark()
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["sample_size"], 872)
        self.assertEqual(res["positive_count"], 444)
        self.assertEqual(res["negative_count"], 428)
        self.assertGreater(res["vader_accuracy_all"], 0.50)
        self.assertGreater(res["vader_accuracy_binary"], 0.60)
        self.assertGreaterEqual(res["baseline_accuracy"], 0.80)
        self.assertGreaterEqual(res["baseline_f1_macro"], 0.80)
        self.assertIn("data_provenance", res)

    def test_domain_lexicon_tech_collocations(self):
        """Verify tech review collocations ('no lag', 'snappy', etc.) are scored accurately."""
        from modules.sentiment import analyze_text_sentiment
        # Positive tech collocations
        pos1 = analyze_text_sentiment("Performance on the Exynos chip is snappy with no lag during everyday multitasking.")
        self.assertEqual(pos1["label"], "POSITIVE")
        self.assertGreater(pos1["compound"], 0.3)

        pos2 = analyze_text_sentiment("The screen is buttery smooth and phone never overheats.")
        self.assertEqual(pos2["label"], "POSITIVE")
        self.assertGreater(pos2["compound"], 0.3)

        # Negative tech collocations
        neg1 = analyze_text_sentiment("The chip lags and stutters while running basic navigation apps.")
        self.assertEqual(neg1["label"], "NEGATIVE")
        self.assertLess(neg1["compound"], -0.3)

        neg2 = analyze_text_sentiment("Device overheats severely when charging and battery drains completely in 4 hours.")
        self.assertEqual(neg2["label"], "NEGATIVE")
        self.assertLess(neg2["compound"], -0.3)

    def test_semeval_aspect_benchmark(self):
        """Verify SemEval-2014 Task 4 laptop aspect evaluation."""
        from modules.evaluation import evaluate_semeval_aspect_benchmark
        res = evaluate_semeval_aspect_benchmark(sample_limit=30)
        self.assertEqual(res["status"], "ok")
        self.assertGreaterEqual(res["recall"], 0.50)
        self.assertGreaterEqual(res["f1_score"], 0.50)
        self.assertIn("SemEval-2014", res["benchmark_name"])

    def test_ml_spam_classifier_benchmark(self):
        """Verify trained ML spam classifier metrics."""
        from modules.evaluation import evaluate_ml_spam_classifier
        res = evaluate_ml_spam_classifier()
        self.assertEqual(res["status"], "ok")
        self.assertGreaterEqual(res["accuracy"], 0.90)
        self.assertGreaterEqual(res["f1_score"], 0.90)
        self.assertEqual(len(res["feature_set"]), 9)

    def test_real_amazon_reviews_benchmark(self):
        """Verify genuine UCSD Amazon Electronics corpus agreement."""
        from modules.evaluation import evaluate_real_amazon_reviews
        res = evaluate_real_amazon_reviews()
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["sample_size"], 997)
        self.assertGreaterEqual(res["agreement_rate"], 0.70)
        self.assertEqual(res["unique_products"], 40)

    def test_error_analysis_structure(self):
        """Verify error analysis categorization."""
        from modules.evaluation import evaluate_error_analysis
        cases = evaluate_error_analysis()
        self.assertGreaterEqual(len(cases), 4)
        cats = [c["category"] for c in cases]
        self.assertIn("Sarcasm & Irony", cats)
        self.assertIn("Complex / Double Negation", cats)

    def test_executive_summary_pros_cons_accuracy(self):
        """Verify generate_executive_summary does not place positive sentiment in cons."""
        from modules.review_summary import generate_executive_summary
        mock_aspects = [
            {
                "category": "Display",
                "mentions": 5,
                "net_sentiment_score": 80.0,
                "positive_count": 5,
                "negative_count": 0,
                "positive_quotes": ["Gorgeous display with bright colors"],
                "negative_quotes": []
            },
            {
                "category": "Performance",
                "mentions": 4,
                "net_sentiment_score": 50.0,
                "positive_count": 3,
                "negative_count": 1,
                "positive_quotes": ["Snappy performance with no lag"],
                "negative_quotes": ["Minor frame drop in heavy 3D titles"]
            },
            {
                "category": "Charging",
                "mentions": 3,
                "net_sentiment_score": -66.7,
                "positive_count": 0,
                "negative_count": 2,
                "positive_quotes": [],
                "negative_quotes": ["Charging speed is terribly slow"]
            }
        ]
        res = generate_executive_summary("Test Phone", mock_aspects, 40.0, 70.0, 10)
        # Display and Performance have positive NSS and should be in pros
        pro_aspects = [p["aspect"] for p in res["pros"]]
        self.assertIn("Display", pro_aspects)
        self.assertIn("Performance", pro_aspects)

        # Performance (NSS +50) must NOT be in cons
        con_aspects = [c["aspect"] for c in res["cons"]]
        self.assertNotIn("Performance", con_aspects)
        self.assertIn("Charging", con_aspects)


if __name__ == "__main__":
    unittest.main()
