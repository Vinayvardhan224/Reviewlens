"""
ReviewLens Authentication & Security Audit Logger.
Records privacy-preserving, cryptographically hashed security events
to a rotating/append-only JSONL log file (logs/auth_events.jsonl).
Never logs passwords, OTP codes, or unmasked personal identifiable information.
"""

from typing import Optional
import os
import time
import json
import hashlib
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
AUDIT_LOG_PATH = LOGS_DIR / "auth_events.jsonl"


def _hash_identifier(identifier: Optional[str]) -> str:
    """Produces a deterministic 16-character SHA-256 hash of an identifier."""
    if not identifier:
        return "anonymous"
    clean = identifier.strip().lower()
    return hashlib.sha256(clean.encode("utf-8")).hexdigest()[:16]


def log_auth_event(
    event_type: str,
    method: str,
    identifier: Optional[str] = None,
    success: bool = True,
    details: Optional[str] = None
) -> None:
    """
    Appends a security audit entry to logs/auth_events.jsonl.
    
    Args:
        event_type: Event category ('login_ok', 'login_fail', 'otp_sent', 'otp_verified', 'otp_failed', 'lockout', 'logout', 'register', 'password_reset')
        method: Authentication method ('email_password', 'email_otp', 'phone_otp', 'google', 'guest')
        identifier: Email or phone number (hashed before writing; never logged in plaintext)
        success: Whether the action succeeded
        details: Optional non-sensitive context message
    """
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": time.time(),
            "iso_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event_type": str(event_type),
            "method": str(method),
            "identifier_hash": _hash_identifier(identifier),
            "success": bool(success),
            "details": str(details) if details else ""
        }
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        # Audit logging failure should not crash user-facing auth
        pass
