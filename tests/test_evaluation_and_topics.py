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
        self.assertEqual(reviews_res["sample_size"], 77)
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


if __name__ == "__main__":
    unittest.main()
