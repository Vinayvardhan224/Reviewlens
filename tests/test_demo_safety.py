"""
Unit tests for ReviewLens Demo Mode Safety guarantees.
Verifies that the mock OTP bypass ('123456') is strictly restricted:
- Works only when REVIEWLENS_DEMO_MODE=1 is set
- Works for designated DEMO_ACCOUNTS
- Works for new user sign-ups
- Strictly BLOCKED for password resets on existing real accounts
"""

import unittest
import os
import streamlit as st
from unittest.mock import patch

from modules.otp_manager import OTPManager, DEMO_ACCOUNTS
from modules.secrets_util import is_demo_mode_enabled


class TestDemoSafety(unittest.TestCase):

    def setUp(self):
        st.session_state.clear()
        if "REVIEWLENS_DEMO_MODE" in os.environ:
            del os.environ["REVIEWLENS_DEMO_MODE"]

    def tearDown(self):
        st.session_state.clear()
        if "REVIEWLENS_DEMO_MODE" in os.environ:
            del os.environ["REVIEWLENS_DEMO_MODE"]

    def test_demo_mode_disabled_by_default(self):
        """Production default: demo mode is strictly OFF."""
        self.assertFalse(is_demo_mode_enabled())

    def test_demo_code_rejected_when_demo_mode_off(self):
        """When demo mode is disabled, 123456 must NEVER succeed."""
        session_key = "test_otp_key"
        target = "demo@reviewlens.com"
        OTPManager.generate_and_store(
            session_key=session_key,
            target=target,
            channel="email",
            purpose="signup"
        )
        ok, msg = OTPManager.verify(session_key, target, entered_code="123456")
        self.assertFalse(ok)
        self.assertIn("Incorrect verification code", msg)

    def test_demo_code_allowed_for_demo_accounts_in_demo_mode(self):
        """When REVIEWLENS_DEMO_MODE=1, accounts in DEMO_ACCOUNTS can bypass with 123456."""
        os.environ["REVIEWLENS_DEMO_MODE"] = "1"
        self.assertTrue(is_demo_mode_enabled())

        session_key = "test_otp_key"
        target = DEMO_ACCOUNTS[0]
        OTPManager.generate_and_store(
            session_key=session_key,
            target=target,
            channel="email",
            purpose="forgot_password"
        )
        ok, msg = OTPManager.verify(session_key, target, entered_code="123456")
        self.assertTrue(ok)
        self.assertIn("successful", msg.lower())

    def test_demo_code_allowed_for_new_signups_in_demo_mode(self):
        """When REVIEWLENS_DEMO_MODE=1, brand-new registrations (purpose=signup) can use 123456."""
        os.environ["REVIEWLENS_DEMO_MODE"] = "1"

        session_key = "test_otp_key"
        target = "brandnewshopper999@example.com"
        OTPManager.generate_and_store(
            session_key=session_key,
            target=target,
            channel="email",
            purpose="signup"
        )
        ok, msg = OTPManager.verify(session_key, target, entered_code="123456")
        self.assertTrue(ok)
        self.assertIn("successful", msg.lower())

    def test_demo_code_BLOCKED_for_real_account_password_reset(self):
        """CRITICAL: Real existing accounts CANNOT have passwords reset via 123456 even in demo mode!"""
        os.environ["REVIEWLENS_DEMO_MODE"] = "1"

        session_key = "test_otp_key"
        real_account = "alice_real_user@gmail.com"
        self.assertNotIn(real_account.lower(), [d.lower() for d in DEMO_ACCOUNTS])

        OTPManager.generate_and_store(
            session_key=session_key,
            target=real_account,
            channel="email",
            purpose="forgot_password"
        )
        ok, msg = OTPManager.verify(session_key, target=real_account, entered_code="123456")
        self.assertFalse(ok)
        self.assertIn("Incorrect verification code", msg)


if __name__ == "__main__":
    unittest.main()
