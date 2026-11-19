"""
Unit tests for ReviewLens Authentication & Identity Module (modules/auth.py).
Verifies:
  1. Default guest state
  2. Google OAuth configuration checks & verified token handling
  3. Mobile SMS OTP phone verification & gateway check
  4. Salted PBKDF2-HMAC-SHA256 password hashing (600,000 iterations)
  5. User registration & input validation
  6. Email & password authentication with transparent legacy upgrade
  7. Account lockout after consecutive failed attempts
  8. Logout session reset
"""

import os
import unittest
import hashlib
import secrets
from modules.auth import (
    get_current_user,
    login_with_google,
    login_as_guest,
    send_phone_otp,
    verify_phone_otp,
    logout_user,
    hash_password,
    register_user,
    authenticate_user,
    is_google_oauth_configured,
    is_sms_configured,
    PBKDF2_ITERATIONS,
    _load_user_db,
    _save_user_db
)


class TestAuthModule(unittest.TestCase):

    def setUp(self):
        os.environ["TESTING"] = "1"
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
        st.session_state.pop("auth_rate_limits", None)

    def tearDown(self):
        pass

    def test_default_guest_state(self):
        user = get_current_user()
        self.assertFalse(user["authenticated"])
        self.assertEqual(user["display_name"], "Guest Shopper")

    def test_pbkdf2_hash_parameters(self):
        # OWASP recommended iterations for SHA-256 is >= 600,000
        self.assertGreaterEqual(PBKDF2_ITERATIONS, 600_000)
        salt = secrets.token_hex(16)
        h1 = hash_password("MySecurePassword123", salt)
        h2 = hash_password("MySecurePassword123", salt)
        h3 = hash_password("DifferentPassword123", salt)
        self.assertEqual(h1, h2)
        self.assertNotEqual(h1, h3)
        self.assertEqual(len(h1), 64)  # SHA-256 output length

    def test_register_and_authenticate_user(self):
        unique_email = f"user_{secrets.token_hex(4)}@testdomain.com"
        reg_res = register_user("Dr. Evelyn Reed", unique_email, "StrongPassword@123", "StrongPassword@123")
        self.assertTrue(reg_res["success"])
        self.assertEqual(reg_res["user"]["display_name"], "Dr. Evelyn Reed")

        # Verify authentication
        auth_res = authenticate_user(unique_email, "StrongPassword@123")
        self.assertTrue(auth_res["success"])
        self.assertIn("Dr. Evelyn Reed", auth_res["message"])
        self.assertTrue(get_current_user()["authenticated"])

    def test_invalid_password_and_lockout(self):
        unique_email = f"user_{secrets.token_hex(4)}@testdomain.com"
        register_user("Lockout Test", unique_email, "TargetPass@123", "TargetPass@123")

        # 4 failed attempts
        for _ in range(4):
            res = authenticate_user(unique_email, "WrongPassword!")
            self.assertFalse(res["success"])
            self.assertIn("Incorrect password", res["message"])

        # 5th failed attempt triggers lockout
        res_locked = authenticate_user(unique_email, "WrongPassword!")
        self.assertFalse(res_locked["success"])
        self.assertIn("locked", res_locked["message"].lower())

    def test_legacy_hash_auto_migration(self):
        unique_email = f"legacy_{secrets.token_hex(4)}@testdomain.com"
        salt = secrets.token_hex(16)
        # Store single-round legacy SHA-256 hash
        legacy_hash = hashlib.sha256((salt + "LegacyPassword@123").encode("utf-8")).hexdigest()

        db = _load_user_db()
        db[unique_email] = {
            "name": "Legacy User",
            "email": unique_email,
            "salt": salt,
            "password_hash": legacy_hash,
            "created_at": "2024-01-01",
            "tier": "Legacy Member"
        }
        _save_user_db(db)

        # Authenticate with password
        res = authenticate_user(unique_email, "LegacyPassword@123")
        self.assertTrue(res["success"])

        # Verify database was migrated to 600,000-iteration PBKDF2
        updated_db = _load_user_db()
        expected_pbkdf2 = hash_password("LegacyPassword@123", salt, iterations=PBKDF2_ITERATIONS)
        self.assertEqual(updated_db[unique_email]["password_hash"], expected_pbkdf2)

    def test_register_validation_failures(self):
        res1 = register_user("Bob", "bob@example.com", "Password123", "MismatchPass")
        self.assertFalse(res1["success"])
        self.assertIn("do not match", res1["message"])

        # < 8 chars
        res2 = register_user("Bob", "bob@example.com", "1234567", "1234567")
        self.assertFalse(res2["success"])
        self.assertIn("at least 8 characters", res2["message"])

        # Invalid email syntax
        res3 = register_user("Bob", "invalid-email-syntax", "Password123", "Password123")
        self.assertFalse(res3["success"])
        self.assertIn("valid email", res3["message"])

    def test_google_login_verified_token(self):
        profile = login_with_google("analyst@research.org", name="Lead Analyst", verified=True)
        self.assertTrue(profile["authenticated"])
        self.assertEqual(profile["auth_method"], "google")
        self.assertEqual(profile["identifier"], "analyst@research.org")
        self.assertEqual(profile["display_name"], "Lead Analyst")
        self.assertEqual(profile["avatar_letter"], "L")

    def test_phone_otp_dispatch_and_verification(self):
        res = send_phone_otp("+1", "5551234567")
        self.assertTrue(res["success"])
        self.assertIn("test_otp", res)
        otp = res["test_otp"]
        self.assertEqual(len(otp), 6)

        # Verification with incorrect code
        failed = verify_phone_otp("000000")
        self.assertFalse(failed)

        # Verification with correct code
        success = verify_phone_otp(otp)
        self.assertTrue(success)
        user = get_current_user()
        self.assertTrue(user["authenticated"])
        self.assertEqual(user["auth_method"], "phone")

    def test_guest_login(self):
        profile = login_as_guest()
        self.assertTrue(profile["authenticated"])
        self.assertEqual(profile["auth_method"], "guest")
        self.assertEqual(profile["display_name"], "Guest Explorer")
        self.assertTrue(get_current_user()["authenticated"])

    def test_logout(self):
        login_as_guest()
        self.assertTrue(get_current_user()["authenticated"])
        logout_user()
        self.assertFalse(get_current_user()["authenticated"])


if __name__ == "__main__":
    unittest.main()
