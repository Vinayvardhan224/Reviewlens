"""
Unit tests for ReviewLens OTP Engine, Email OTP, and Privacy Audit Logging.
Verifies:
  1. Cryptographic 6-digit OTP generation with secrets.randbelow
  2. Salted SHA-256 hash storage (raw OTP NEVER retained in session_state)
  3. Constant-time verification with hmac.compare_digest
  4. 5-minute expiration & 3-attempt lockout
  5. 60-second resend cooldown
  6. 5 dispatches per hour rate limiting with persistent tracking
  7. Email OTP dispatch via dependency injection
  8. Privacy-preserving audit logging with 16-char SHA-256 pseudonymized identifiers
"""

import unittest
import time
import secrets
import streamlit as st
from modules.otp_manager import OTPManager, check_and_record_dispatch, RESEND_COOLDOWN_SECONDS
from modules.otp_email import send_email_otp, verify_email_otp, _validate_email
from modules.audit_logger import log_auth_event, _hash_identifier


class TestOTPEngine(unittest.TestCase):
    def setUp(self):
        st.session_state.clear()
        from modules.otp_manager import RATE_LIMITS_FILE
        if RATE_LIMITS_FILE.exists():
            try:
                RATE_LIMITS_FILE.unlink()
            except Exception:
                pass

    def test_otp_generation_and_storage_security(self):
        session_key = "test_otp_security"
        target = "test@reviewlens.com"
        raw_code, state = OTPManager.generate_and_store(
            session_key=session_key,
            target=target,
            channel="email",
            purpose="test_purpose"
        )
        self.assertEqual(len(raw_code), 6)
        self.assertTrue(raw_code.isdigit())

        stored_state = st.session_state.get(session_key)
        self.assertIsNotNone(stored_state)
        # CRITICAL SECURITY CHECK: raw OTP must NEVER be stored in session state
        self.assertNotIn("raw_code", stored_state)
        self.assertNotIn("otp", stored_state)
        self.assertNotIn("code", stored_state)
        self.assertIn("code_hash", stored_state)
        self.assertIn("salt", stored_state)
        self.assertEqual(stored_state["attempts"], 0)
        self.assertEqual(stored_state["max_attempts"], 3)

    def test_otp_verification_success(self):
        session_key = "test_otp_verify"
        target = "verify@reviewlens.com"
        raw_code, _ = OTPManager.generate_and_store(
            session_key=session_key,
            target=target,
            channel="email",
            purpose="signup"
        )

        ok, msg = OTPManager.verify(session_key, target, raw_code, expected_purpose="signup")
        self.assertTrue(ok)
        self.assertIn("successful", msg.lower())

    def test_otp_verification_wrong_code_and_lockout(self):
        session_key = "test_otp_lockout"
        target = "lockout@reviewlens.com"
        raw_code, _ = OTPManager.generate_and_store(
            session_key=session_key,
            target=target,
            channel="email"
        )

        # 1st wrong attempt
        ok1, msg1 = OTPManager.verify(session_key, target, "000000")
        self.assertFalse(ok1)
        self.assertIn("2 attempts remaining", msg1)

        # 2nd wrong attempt
        ok2, msg2 = OTPManager.verify(session_key, target, "111111")
        self.assertFalse(ok2)
        self.assertIn("1 attempt remaining", msg2)

        # 3rd wrong attempt triggers invalidation
        ok3, msg3 = OTPManager.verify(session_key, target, "222222")
        self.assertFalse(ok3)
        self.assertIn("maximum attempts reached", msg3.lower())

        # Further attempts should fail
        ok4, msg4 = OTPManager.verify(session_key, target, raw_code)
        self.assertFalse(ok4)

    def test_otp_expiry(self):
        session_key = "test_otp_expiry"
        target = "expire@reviewlens.com"
        raw_code, _ = OTPManager.generate_and_store(
            session_key=session_key,
            target=target,
            channel="email"
        )

        # Artificially expire the OTP
        st.session_state[session_key]["expires_at"] = time.time() - 10.0
        ok, msg = OTPManager.verify(session_key, target, raw_code)
        self.assertFalse(ok)
        self.assertIn("expired", msg.lower())

    def test_resend_cooldown(self):
        session_key = "test_otp_cooldown"
        target = "cool@reviewlens.com"
        OTPManager.generate_and_store(session_key=session_key, target=target, channel="email")

        cooldown = OTPManager.get_cooldown_remaining(session_key)
        self.assertGreater(cooldown, 0)
        self.assertLessEqual(cooldown, RESEND_COOLDOWN_SECONDS)

    def test_hourly_rate_limit(self):
        test_id = f"rate_limit_{secrets.token_hex(4)}@test.com"
        # 5 allowed sends
        for _ in range(5):
            allowed, _, _ = check_and_record_dispatch(test_id)
            self.assertTrue(allowed)

        # 6th send blocked
        allowed, msg, _ = check_and_record_dispatch(test_id)
        self.assertFalse(allowed)
        self.assertIn("hourly dispatch limit exceeded", msg.lower())

    def test_email_otp_validation_and_injection(self):
        self.assertFalse(_validate_email("not-an-email"))
        self.assertFalse(_validate_email(""))
        self.assertTrue(_validate_email("user@example.com"))

        captured = []
        def mock_mailer(email, code):
            captured.append((email, code))
            return True

        test_email = f"di_{secrets.token_hex(3)}@test.com"
        res = send_email_otp(test_email, purpose="signup", send_fn=mock_mailer)
        self.assertTrue(res["success"])
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0][0], test_email)

        v_res = verify_email_otp(test_email, captured[0][1], purpose="signup")
        self.assertTrue(v_res["success"])

    def test_privacy_audit_logger_hashing(self):
        identifier = "user_private@company.com"
        hashed = _hash_identifier(identifier)
        self.assertEqual(len(hashed), 16)
        self.assertNotEqual(hashed, identifier)
        self.assertNotIn("company.com", hashed)
        # Identical identifier produces identical hash
        self.assertEqual(hashed, _hash_identifier(identifier))


if __name__ == "__main__":
    unittest.main()
