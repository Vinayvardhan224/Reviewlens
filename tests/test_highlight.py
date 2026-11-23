import unittest
from components.highlight import score_words, highlight_review_html, top_drivers

LEX = {"great": 3.1, "good": 1.9, "terrible": -2.1, "bad": -2.5, "battery": 0.0, "love": 3.2}


class HighlightTests(unittest.TestCase):
    def test_positive_and_negative_words_coloured(self):
        s = {t: v for t, v, _ in score_words("Great screen but terrible battery", LEX)}
        self.assertGreater(s["Great"], 0)
        self.assertLess(s["terrible"], 0)
        self.assertEqual(s["battery"], 0.0)

    def test_negation_flips_sign(self):
        s = {t: (v, n) for t, v, n in score_words("not good at all", LEX)}
        self.assertLess(s["good"][0], 0)
        self.assertTrue(s["good"][1])

    def test_negation_scope_is_limited(self):
        s = {t: v for t, v, _ in score_words("not that I think it is really good", LEX)}
        self.assertGreater(s["good"], 0)

    def test_contraction_negator(self):
        s = {t: v for t, v, _ in score_words("it isn't great", LEX)}
        self.assertLess(s["great"], 0)

    def test_html_is_escaped(self):
        out = highlight_review_html("<script>alert(1)</script> great", LEX)
        self.assertNotIn("<script>", out)
        self.assertIn("&lt;script&gt;", out)
        self.assertIn("great", out)

    def test_text_preserved(self):
        out = highlight_review_html("Love it, 5 stars!", LEX)
        for piece in ("Love", "it", "5", "stars", "!"):
            self.assertIn(piece, out)

    def test_empty_text(self):
        self.assertIn("<div", highlight_review_html("", LEX))
        self.assertEqual(top_drivers("", LEX), ([], []))

    def test_top_drivers_order(self):
        pos, neg = top_drivers("good great but bad and terrible", LEX)
        self.assertEqual(pos[0][0].lower(), "great")
        self.assertEqual(neg[0][0].lower(), "bad")


if __name__ == "__main__":
    unittest.main()
