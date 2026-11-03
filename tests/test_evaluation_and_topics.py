"""
Unit tests for ReviewLens Evaluation Benchmarks and Topic Modeling.
"""

import unittest
from modules.evaluation import evaluate_sentiment_model, evaluate_spam_model
from modules.topic_modeling import extract_topic_clusters


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


if __name__ == "__main__":
    unittest.main()
