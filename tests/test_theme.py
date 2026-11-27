"""
Unit tests for ReviewLens Glassmorphism & Bento Shared Design System.
Verifies theme token generation, HTML escaping, semantic tone mapping,
and Plotly chart glass layout application.
"""

import unittest
import html
import plotly.graph_objects as go

from components.theme import (
    THEME_TOKENS,
    CHART_COLOURWAY,
    get_theme_css,
    glass_card,
    kpi_tile,
    apply_glass_layout
)


class TestThemeDesignSystem(unittest.TestCase):

    def test_theme_tokens_presence(self):
        """Verifies core CSS variables match specification."""
        self.assertEqual(THEME_TOKENS["--bg"], "#07051A")
        self.assertEqual(THEME_TOKENS["--primary"], "#7138F5")
        self.assertEqual(THEME_TOKENS["--primary-2"], "#9B5CFF")
        self.assertEqual(THEME_TOKENS["--accent"], "#22D3EE")
        self.assertEqual(THEME_TOKENS["--pos"], "#34D399")
        self.assertEqual(THEME_TOKENS["--neg"], "#F87171")
        self.assertEqual(THEME_TOKENS["--mixed"], "#FBBF24")

    def test_get_theme_css_reduce_effects(self):
        """Verifies reduce_effects removes blur and sets backdrop-filter to none."""
        css_full = get_theme_css(reduce_effects=False)
        self.assertIn("backdrop-filter: blur(var(--blur))", css_full)
        self.assertIn("@media (prefers-reduced-motion: reduce)", css_full)

        css_reduced = get_theme_css(reduce_effects=True)
        self.assertIn("backdrop-filter: none", css_reduced)
        self.assertIn("--blur: 0px", css_reduced)

    def test_glass_card_rendering(self):
        """Verifies glass_card outputs valid frosted container HTML."""
        inner = "<h4>Card Header</h4><p>Details</p>"
        card_html = glass_card(inner, span=2, extra_class="test-class")
        self.assertIn('class="rl-glass test-class"', card_html)
        self.assertIn("grid-column: span 2", card_html)
        self.assertIn(inner, card_html)

    def test_kpi_tile_tones(self):
        """Verifies semantic chip styling for positive, negative, mixed, and neutral tones."""
        # Positive
        pos_tile = kpi_tile("Sentiment NSS", "+68.4", delta="+12.5%", tone="pos")
        self.assertIn("rl-chip-pos", pos_tile)
        self.assertIn("+68.4", pos_tile)
        self.assertIn("Sentiment NSS", pos_tile)

        # Negative
        neg_tile = kpi_tile("Spam Detected", "14.2%", delta="+4.1%", tone="neg")
        self.assertIn("rl-chip-neg", neg_tile)

        # Mixed
        mixed_tile = kpi_tile("Mixed Aspect", "Neutral", delta="Notice", tone="mixed")
        self.assertIn("rl-chip-mixed", mixed_tile)

        # Neutral
        neut_tile = kpi_tile("Total Reviews", "1,248", delta="Volume", tone="neutral")
        self.assertIn("rl-chip-neutral", neut_tile)

    def test_kpi_tile_escaping(self):
        """Verifies user-provided strings in kpi_tile are escaped."""
        evil_label = '<script>alert("xss")</script>'
        tile = kpi_tile(evil_label, "100")
        self.assertNotIn("<script>", tile)
        self.assertIn(html.escape(evil_label), tile)

    def test_apply_glass_layout_plotly(self):
        """Verifies Plotly figures receive transparent background and ReviewLens colorway."""
        fig = go.Figure(data=[go.Bar(x=["A", "B"], y=[10, 20])])
        styled = apply_glass_layout(fig)
        self.assertEqual(styled.layout.paper_bgcolor, "rgba(0,0,0,0)")
        self.assertEqual(styled.layout.plot_bgcolor, "rgba(0,0,0,0)")
        self.assertEqual(list(styled.layout.colorway), CHART_COLOURWAY)
        self.assertEqual(styled.layout.font.color, "#F1F5F9")


if __name__ == "__main__":
    unittest.main()
