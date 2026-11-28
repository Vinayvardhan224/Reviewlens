"""
Unit tests for ReviewLens Install App page and installation navigation isolation.
Verifies:
  1. The navigation list contains the "📲 Install App" view in the "MORE" group.
  2. render_install_page properly escapes URLs and prevents HTML injection.
  3. The unauthenticated login page output never contains the words "Install" or "Download".
"""

import html
import unittest
from unittest.mock import MagicMock, patch
import streamlit as st

from modules.navigation import ALL_NAV_VIEWS, NAV_GROUPS, get_group_for_view
from components.install_page import render_install_page, get_pwa_shell_url


class TestInstallPage(unittest.TestCase):

    def setUp(self):
        st.session_state.clear()

    def test_navigation_contains_install_view(self):
        """Verifies that '📲 Install App' is defined in NAV_GROUPS['MORE'] and ALL_NAV_VIEWS."""
        self.assertIn("📲 Install App", ALL_NAV_VIEWS)
        self.assertIn("📲 Install App", NAV_GROUPS["MORE"])
        self.assertEqual(get_group_for_view("📲 Install App"), "MORE")

    def test_render_install_page_escapes_url(self):
        """Verifies that render_install_page strictly escapes URLs to prevent XSS/injection."""
        malicious_url = 'https://reviewlens.streamlit.app/?q="><script>alert("xss")</script>&param=<img src=x onerror=alert(1)>'
        captured_markdown = []

        mock_tab = MagicMock()
        mock_tab.__enter__ = MagicMock(return_value=mock_tab)
        mock_tab.__exit__ = MagicMock(return_value=None)

        mock_col = MagicMock()
        mock_col.__enter__ = MagicMock(return_value=mock_col)
        mock_col.__exit__ = MagicMock(return_value=None)

        with patch("streamlit.markdown", side_effect=lambda content, *args, **kwargs: captured_markdown.append(str(content))), \
             patch("streamlit.code") as mock_code, \
             patch("streamlit.link_button") as mock_link_button, \
             patch("streamlit.tabs", return_value=[mock_tab, mock_tab, mock_tab]), \
             patch("streamlit.columns", side_effect=lambda spec, *args, **kwargs: [mock_col] * (spec if isinstance(spec, int) else len(spec))):

            render_install_page(malicious_url)

            # Check that code was called with the raw URL safely for copying
            mock_code.assert_called_with(malicious_url, language="text")

            # Check link_button was called
            mock_link_button.assert_called_once()

            # Verify HTML escaping across all markdown chunks
            full_html = " ".join(captured_markdown)
            self.assertIn("Install ReviewLens", full_html)
            self.assertIn("Use ReviewLens like a normal app on your phone or computer. No app store needed.", full_html)

            # Raw unescaped dangerous tags must NOT be present
            self.assertNotIn('<script>alert("xss")</script>', full_html)
            self.assertNotIn('<img src=x onerror=alert(1)>', full_html)
            self.assertNotIn('"><script>', full_html)

            # Escaped entities must be present
            escaped_fragment = html.escape(malicious_url, quote=True)
            self.assertIn(escaped_fragment, full_html)

    def test_login_page_never_contains_install_or_download(self):
        """Verifies that the unauthenticated login page UI never outputs 'Install' or 'Download'."""
        st.session_state["user_profile"] = {"authenticated": False}

        captured_texts = []

        def capture_text(arg, *args, **kwargs):
            if isinstance(arg, str):
                captured_texts.append(arg)

        # Mock streamlit display calls and auth HTML renderer to collect all output text
        with patch("modules.auth._render_html", side_effect=capture_text), \
             patch("streamlit.markdown", side_effect=capture_text), \
             patch("streamlit.caption", side_effect=capture_text), \
             patch("streamlit.button", side_effect=lambda label, *args, **kwargs: captured_texts.append(str(label)) or False), \
             patch("streamlit.text_input", side_effect=lambda label, *args, **kwargs: captured_texts.append(str(label)) or ""):

            from modules.auth import render_auth_interface

            # 1. Test Login mode
            st.session_state["auth_mode"] = "login"
            render_auth_interface()

            # 2. Test Register mode
            st.session_state["auth_mode"] = "register"
            render_auth_interface()

        combined = " ".join(captured_texts).lower()
        self.assertGreater(len(combined), 100, "Should have captured UI text from the login interface")
        self.assertNotIn("install", combined, "Login page must not contain 'install'")
        self.assertNotIn("download", combined, "Login page must not contain 'download'")


if __name__ == "__main__":
    unittest.main()
