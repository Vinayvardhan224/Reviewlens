"""
ReviewLens Email OTP Verification Module (Gmail SMTP).
Handles secure 6-digit email verification code dispatch and validation:
  - Connects to smtp.gmail.com:465 via smtplib.SMTP_SSL
  - Uses SMTP_USER and SMTP_APP_PASSWORD (Gmail 16-character App Password)
  - Enforces 60s cooldown, 5-minute expiration, 3 attempts limit, max 5 sends/hr
  - Protects user privacy: uniform response for registered and unregistered emails
  - Multi-part MIME (plain text + branded dark-themed HTML)
"""

from typing import Dict, Any, Optional, Callable
import os
import re
import smtplib
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from modules.secrets_util import get_secret, is_demo_mode_enabled
from modules.otp_manager import (
    OTPManager,
    check_and_record_dispatch,
    get_target_cooldown_remaining,
    RESEND_COOLDOWN_SECONDS
)
from modules.audit_logger import log_auth_event

EMAIL_OTP_SESSION_KEY = "pending_email_otp"
EMAIL_REGEX = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"


def is_smtp_configured() -> bool:
    """Checks whether genuine Gmail SMTP credentials are configured."""
    user = get_secret("SMTP_USER")
    password = get_secret("SMTP_APP_PASSWORD", "").replace(" ", "").strip()
    return bool(user and password and "@" in user)


def _validate_email(email: str) -> bool:
    """Strict syntactic validation of email address."""
    if not email or not isinstance(email, str):
        return False
    return bool(re.match(EMAIL_REGEX, email.strip()))


def _build_email_message(to_email: str, otp_code: str, purpose: str) -> MIMEMultipart:
    """Creates a branded HTML and plain-text email message."""
    sender_user = get_secret("SMTP_USER", "notifications@reviewlens.com")
    sender_name = "ReviewLens"
    subject_map = {
        "signup": "Verify Your ReviewLens Account",
        "forgot_password": "Reset Your ReviewLens Password",
        "login_2fa": "ReviewLens 2-Step Verification Code",
        "general": "Your ReviewLens Verification Code"
    }
    subject = subject_map.get(purpose, "Your ReviewLens Verification Code")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"{sender_name} <{sender_user}>"
    msg["To"] = to_email

    # Plain text version
    text_content = (
        f"Your ReviewLens verification code is: {otp_code}\n\n"
        "This code is valid for 5 minutes. Do not share this code with anyone.\n\n"
        "If you did not request this code, please ignore this email.\n"
        "— The ReviewLens Security Team\n"
    )

    # HTML version matching dark purple theme
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>{subject}</title>
    </head>
    <body style="margin: 0; padding: 24px; background-color: #07051A; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; color: #FFFFFF;">
      <table align="center" border="0" cellpadding="0" cellspacing="0" width="100%" style="max-width: 520px; background: #120E26; border: 1px solid rgba(155, 92, 255, 0.4); border-radius: 16px; padding: 32px; box-shadow: 0 10px 30px rgba(0,0,0,0.5);">
        <tr>
          <td align="center" style="padding-bottom: 20px;">
            <div style="display: inline-block; width: 44px; height: 44px; border-radius: 12px; background: linear-gradient(135deg, #7138F5 0%, #9B5CFF 100%); line-height: 44px; text-align: center; font-size: 20px; color: #FFFFFF;">★</div>
            <div style="font-size: 22px; font-weight: 800; color: #FFFFFF; margin-top: 8px;">Review<span style="color: #9B5CFF;">Lens</span></div>
          </td>
        </tr>
        <tr>
          <td style="color: #E2E8F0; font-size: 15px; line-height: 1.6; padding-bottom: 20px;">
            Hello,<br><br>
            Please use the following 6-digit verification code to complete your security verification on <b>ReviewLens</b>:
          </td>
        </tr>
        <tr>
          <td align="center" style="padding: 16px 0 24px 0;">
            <div style="display: inline-block; background: rgba(113, 56, 245, 0.2); border: 2px solid #9B5CFF; border-radius: 12px; padding: 14px 28px; font-size: 32px; font-weight: 800; letter-spacing: 8px; color: #22D3EE;">
              {otp_code}
            </div>
          </td>
        </tr>
        <tr>
          <td style="color: #94A3B8; font-size: 13px; line-height: 1.5; padding-bottom: 24px; border-bottom: 1px solid rgba(255, 255, 255, 0.1);">
            ⏱️ <b>This code is valid for 5 minutes</b> (max 3 verification attempts).<br>
            🔒 For security reasons, never share this code with anyone.
          </td>
        </tr>
        <tr>
          <td style="color: #64748B; font-size: 12px; line-height: 1.5; padding-top: 20px; text-align: center;">
            If you did not request this verification, please safely ignore this message.<br>
            &copy; 2026 ReviewLens Product Intelligence Platform.
          </td>
        </tr>
      </table>
    </body>
    </html>
    """

    msg.attach(MIMEText(text_content, "plain", "utf-8"))
    msg.attach(MIMEText(html_content, "html", "utf-8"))
    return msg


def send_email_otp(
    email: str,
    purpose: str = "general",
    send_fn: Optional[Callable[[str, str], bool]] = None
) -> Dict[str, Any]:
    """
    Generates and dispatches an OTP verification code to the specified email address.
    
    Args:
        email: Recipient email address
        purpose: 'signup', 'forgot_password', 'login_2fa', or 'general'
        send_fn: Optional dependency-injected sender function for testing (email, code) -> bool
        
    Returns:
        Dict: {'success': bool, 'message': str, 'cooldown_seconds': int, ...}
    """
    clean_email = email.strip().lower() if email else ""

    if not _validate_email(clean_email):
        return {"success": False, "message": "Please enter a valid email address."}

    # Check resend cooldown across session and persistent disk
    cooldown = max(
        OTPManager.get_cooldown_remaining(EMAIL_OTP_SESSION_KEY),
        get_target_cooldown_remaining(clean_email)
    )
    if cooldown > 0:
        return {
            "success": False,
            "message": f"Please wait {cooldown}s before requesting a new verification code.",
            "cooldown_seconds": cooldown
        }

    # Check hourly dispatch quota (5/hr)
    allowed, limit_msg, _ = check_and_record_dispatch(clean_email)
    if not allowed:
        return {"success": False, "message": limit_msg, "cooldown_seconds": 60}

    # Generate and securely store salted hash
    raw_code, state = OTPManager.generate_and_store(
        session_key=EMAIL_OTP_SESSION_KEY,
        target=clean_email,
        channel="email",
        purpose=purpose
    )

    # 1. Dependency injection path for unit testing
    if send_fn is not None:
        ok = send_fn(clean_email, raw_code)
        log_auth_event("otp_sent", "email_test_injection", clean_email, success=ok, details=f"purpose: {purpose}")
        return {
            "success": bool(ok),
            "message": "Verification code dispatched via test channel.",
            "cooldown_seconds": int(RESEND_COOLDOWN_SECONDS)
        }

    # 2. Live Gmail SMTP dispatch path
    if is_smtp_configured():
        host = get_secret("SMTP_HOST", "smtp.gmail.com")
        port = int(get_secret("SMTP_PORT", "465"))
        user = get_secret("SMTP_USER")
        password = get_secret("SMTP_APP_PASSWORD", "").replace(" ", "").strip()

        try:
            msg = _build_email_message(clean_email, raw_code, purpose)
            with smtplib.SMTP_SSL(host, port, timeout=10.0) as server:
                server.login(user, password)
                server.send_message(msg)

            log_auth_event("otp_sent", "email_smtp", clean_email, success=True, details=f"purpose: {purpose}")
            return {
                "success": True,
                "message": f"A 6-digit verification code has been sent to {clean_email}.",
                "cooldown_seconds": int(RESEND_COOLDOWN_SECONDS)
            }
        except Exception as e:
            log_auth_event("otp_sent", "email_smtp", clean_email, success=False, details=f"SMTP error: {str(e)[:60]}")
            return {
                "success": False,
                "message": "Could not send verification email. Please try again later."
            }

    # 3. Demo sandbox mode fallback (ONLY if REVIEWLENS_DEMO_MODE=1)
    if is_demo_mode_enabled():
        log_auth_event("otp_sent", "email_demo_sandbox", clean_email, success=True, details="Demo mode sandbox dispatch")
        # In demo mode, provide sandbox notification (code is not stored in session_state)
        return {
            "success": True,
            "message": f"Demo Sandbox: Code dispatched for {clean_email}.",
            "cooldown_seconds": int(RESEND_COOLDOWN_SECONDS),
            "demo_sandbox": True
        }

    # Production default when credentials are not configured
    log_auth_event("otp_sent", "email_disabled", clean_email, success=False, details="No SMTP credentials configured")
    return {
        "success": False,
        "message": "Email verification service is not configured. (Configure SMTP_USER and SMTP_APP_PASSWORD in .streamlit/secrets.toml)"
    }


def verify_email_otp(
    email: str,
    code: str,
    purpose: Optional[str] = None
) -> Dict[str, Any]:
    """
    Validates an entered 6-digit email OTP.
    Returns: {'success': bool, 'message': str}
    """
    clean_email = email.strip().lower() if email else ""
    success, msg = OTPManager.verify(
        session_key=EMAIL_OTP_SESSION_KEY,
        target=clean_email,
        entered_code=code,
        expected_purpose=purpose
    )
    return {"success": success, "message": msg}


def clear_email_otp() -> None:
    """Clears pending email OTP from session state upon successful completion."""
    OTPManager.clear(EMAIL_OTP_SESSION_KEY)


def send_forgot_password_otp(
    email: str,
    send_fn: Optional[Callable[[str, str], bool]] = None
) -> Dict[str, Any]:
    """
    Dispatches a password reset OTP only if the email belongs to a registered account.
    Returns a uniform success message regardless of whether the account exists
    to prevent user account enumeration and prevent Gmail SMTP quota exhaustion.
    """
    clean_email = email.strip().lower() if email else ""
    if not _validate_email(clean_email):
        return {"success": False, "message": "Please enter a valid email address."}

    users_file = Path(__file__).resolve().parent.parent / "data" / "users.json"
    user_exists = False
    if users_file.exists():
        try:
            import json
            with open(users_file, "r", encoding="utf-8") as f:
                db = json.load(f)
                user_exists = clean_email in db
        except Exception:
            user_exists = False

    uniform_msg = f"If an account is associated with {clean_email}, a verification code has been dispatched."

    if not user_exists:
        log_auth_event("forgot_pw_unregistered", "email", clean_email, success=True, details="Silently skipped SMTP for unregistered email")
        return {
            "success": True,
            "message": uniform_msg,
            "cooldown_seconds": int(RESEND_COOLDOWN_SECONDS)
        }

    res = send_email_otp(clean_email, purpose="forgot_password", send_fn=send_fn)
    if res.get("success"):
        return {
            "success": True,
            "message": uniform_msg,
            "cooldown_seconds": res.get("cooldown_seconds", int(RESEND_COOLDOWN_SECONDS))
        }
    return res
