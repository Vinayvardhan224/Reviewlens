"""
Unit tests for ReviewLens Authentication & Identity Module (modules/auth.py).
Verifies:
  1. Default guest state
  2. Google/Gmail 1-click authentication
  3. Mobile SMS OTP phone verification
  4. Salted SHA-256 password hashing
  5. User registration & input validation
  6. Email & password authentication
  7. Logout session reset
"""

import unittest
from modules.auth import (
    get_current_user,
    login_with_google,
    send_phone_otp,
    verify_phone_otp,
    logout_user,
    hash_password,
    register_user,
    authenticate_user
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

    def test_hash_password(self):
        salt = "testsalt12345678"
        h1 = hash_password("MySecretPass123", salt)
        h2 = hash_password("MySecretPass123", salt)
        h3 = hash_password("DifferentPass123", salt)
        self.assertEqual(h1, h2)
        self.assertNotEqual(h1, h3)
        self.assertEqual(len(h1), 64)  # SHA-256 hex length

    def test_demo_account_authentication(self):
        res = authenticate_user("demo@reviewlens.com", "Password@123")
        self.assertTrue(res["success"])
        self.assertIn("Welcome back", res["message"])
        user = get_current_user()
        self.assertTrue(user["authenticated"])
        self.assertEqual(user["identifier"], "demo@reviewlens.com")

    def test_invalid_password_authentication(self):
        res = authenticate_user("demo@reviewlens.com", "WrongPassword!")
        self.assertFalse(res["success"])
        self.assertIn("Incorrect password", res["message"])
        self.assertFalse(get_current_user()["authenticated"])

    def test_nonexistent_email_authentication(self):
        res = authenticate_user("nonexistent.user.999@domain.com", "SomePassword123")
        self.assertFalse(res["success"])
        self.assertIn("No account found", res["message"])

    def test_register_user_success_and_validation(self):
        import secrets
        unique_email = f"user_{secrets.token_hex(4)}@reviewlens.com"
        res = register_user("Alice Researcher", unique_email, "SecurePass456", "SecurePass456")
        self.assertTrue(res["success"])
        self.assertEqual(res["user"]["display_name"], "Alice Researcher")
        self.assertTrue(get_current_user()["authenticated"])

        # Duplicate registration should fail
        dup = register_user("Alice Researcher", unique_email, "SecurePass456", "SecurePass456")
        self.assertFalse(dup["success"])
        self.assertIn("already exists", dup["message"])

    def test_register_validation_failures(self):
        # Mismatched passwords
        res1 = register_user("Bob", "bob@example.com", "Password123", "MismatchPass")
        self.assertFalse(res1["success"])
        self.assertIn("do not match", res1["message"])

        # Password too short
        res2 = register_user("Bob", "bob@example.com", "123", "123")
        self.assertFalse(res2["success"])
        self.assertIn("at least 6 characters", res2["message"])

        # Invalid email syntax
        res3 = register_user("Bob", "invalid-email-address", "Password123", "Password123")
        self.assertFalse(res3["success"])
        self.assertIn("valid email", res3["message"])

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
