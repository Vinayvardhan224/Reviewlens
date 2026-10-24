"""Unit tests for text cleaning and preprocessing."""

import unittest
from reviewlens.preprocessing.cleaner import clean_text, expand_contractions, convert_emojis, clean_html


class TestCleaner(unittest.TestCase):

    def test_expand_contractions(self):
        text = "I won't buy this again, it's terrible and didn't work."
        cleaned = expand_contractions(text)
        self.assertIn("will not", cleaned)
        self.assertIn("is", cleaned)
        self.assertIn("did not", cleaned)

    def test_clean_html(self):
        html_text = "<p>Great sound! <a href='https://example.com'>Read more</a></p>"
        cleaned = clean_html(html_text)
        self.assertNotIn("<p>", cleaned)
        self.assertNotIn("</a>", cleaned)
        self.assertIn("Great sound!", cleaned)

    def test_convert_emojis(self):
        emoji_text = "Love this product ❤️👍"
        cleaned = convert_emojis(emoji_text)
        self.assertIn("love", cleaned)
        self.assertIn("good", cleaned)

    def test_clean_text_full_pipeline(self):
        raw = "   <b>Don't buy</b> this https://spam.com/link! It's awful 👎   "
        result = clean_text(raw)
        self.assertNotIn("https://", result)
        self.assertNotIn("<b>", result)
        self.assertIn("Do not buy", result)
        self.assertIn("dislike", result)


if __name__ == "__main__":
    unittest.main()
