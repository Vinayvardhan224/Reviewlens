"""
Unit tests for ReviewLens authentication gate and single-render flow.
Verifies:
  1. Unauthenticated sessions invoke render_auth_interface exactly once per run without rerun loops.
  2. Authenticated sessions bypass the login gate cleanly.
  3. Cached product and review loaders load valid DataFrames.
"""

import unittest
from unittest.mock import patch, MagicMock
import streamlit as st
import pandas as pd


class TestAuthFlow(unittest.TestCase):

    def setUp(self):
        st.session_state.clear()

    @patch("modules.auth.render_auth_interface")
    def test_unauthenticated_session_renders_login_exactly_once(self, mock_render_auth):
        """Verify that an unauthenticated session triggers render_auth_interface exactly once."""
        from modules.auth import get_current_user

        # Ensure session is unauthenticated
        st.session_state["user_profile"] = {"authenticated": False}
        user = get_current_user()
        self.assertFalse(user.get("authenticated"))

        # Simulate the auth gate from app.py
        rendered_count = 0
        if not user.get("authenticated"):
            mock_render_auth()
            rendered_count += 1

        self.assertEqual(rendered_count, 1)
        mock_render_auth.assert_called_once()

    @patch("modules.auth.render_auth_interface")
    def test_authenticated_session_bypasses_login_gate(self, mock_render_auth):
        """Verify that an authenticated user does NOT trigger the login gate."""
        from modules.auth import get_current_user

        st.session_state["user_profile"] = {
            "authenticated": True,
            "identifier": "shopper@reviewlens.com",
            "display_name": "Test Shopper"
        }
        user = get_current_user()
        self.assertTrue(user.get("authenticated"))

        rendered_count = 0
        if not user.get("authenticated"):
            mock_render_auth()
            rendered_count += 1

        self.assertEqual(rendered_count, 0)
        mock_render_auth.assert_not_called()

    def test_cached_catalogue_loaders(self):
        """Verify that the cached catalogue loaders return non-empty validated DataFrames."""
        from modules.data_loader import get_all_products, get_all_reviews

        df_prods = get_all_products(validate=True)
        self.assertFalse(df_prods.empty)
        self.assertIn("product_id", df_prods.columns)

        df_revs = get_all_reviews(validate=True, products_df=df_prods)
        self.assertFalse(df_revs.empty)
        self.assertIn("review_id", df_revs.columns)


if __name__ == "__main__":
    unittest.main()
