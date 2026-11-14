"""
Unit tests for ReviewLens Authentication & Identity Module (modules/auth.py).
Verifies Google/Gmail sign-in, Phone SMS OTP generation, and verification flows.
"""

import unittest
from unittest.mock import patch
from modules.auth import (
    get_current_user,
    login_with_google,
    send_phone_otp,
    verify_phone_otp,
    logout_user
)


class TestAuthModule(unittest.TestCase):

    def setUp(self):
        # Reset mock session state before each test
        import streamlit as st
        st.session_state["user_profile"] = {
            "authenticated": False,
            "auth_method": None,
            "identifier": None,
            "display_name": "Guest Shopper",
            "avatar_letter": "G",
            "tier": "Guest Explorer",
            "tier_badge": "⚪ Guest",
            "member_since": "2026",
            "saved_comparisons": [],
            "custom_presets": {}
        }
        st.session_state.pop("pending_phone_auth", None)

    def test_default_guest_state(self):
        user = get_current_user()
        self.assertFalse(user["authenticated"])
        self.assertEqual(user["display_name"], "Guest Shopper")

    def test_google_login(self):
        profile = login_with_google("test.buyer@gmail.com", name="Test Buyer")
        self.assertTrue(profile["authenticated"])
        self.assertEqual(profile["auth_method"], "google")
        self.assertEqual(profile["identifier"], "test.buyer@gmail.com")
        self.assertEqual(profile["display_name"], "Test Buyer")
        self.assertEqual(profile["avatar_letter"], "T")

    def test_phone_otp_generation_and_verification(self):
        otp = send_phone_otp("+91", "9876543210")
        self.assertEqual(len(otp), 6)
        self.assertTrue(otp.isdigit())

        # Test verification with incorrect OTP
        failed = verify_phone_otp("000111")
        self.assertFalse(failed)

        # Test verification with correct OTP
        success = verify_phone_otp(otp)
        self.assertTrue(success)

        user = get_current_user()
        self.assertTrue(user["authenticated"])
        self.assertEqual(user["auth_method"], "phone")
        self.assertIn("+91", user["identifier"])

    def test_logout(self):
        login_with_google("user@gmail.com")
        self.assertTrue(get_current_user()["authenticated"])

        logout_user()
        self.assertFalse(get_current_user()["authenticated"])


if __name__ == "__main__":
    unittest.main()
