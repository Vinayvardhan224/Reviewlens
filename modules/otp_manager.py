"""
ReviewLens Unified Cryptographic OTP & Rate-Limiting Manager.
Provides shared security primitives for both Email and SMS verification:
  1. Cryptographically secure 6-digit generation (secrets.randbelow)
  2. Salted SHA-256 OTP hashing (the raw code is never kept in session state or logs)
  3. Constant-time code comparison (hmac.compare_digest)
  4. 5-minute expiration window & 3-attempt brute-force limit
  5. 60-second resend cooldown
  6. Hourly dispatch throttling (max 5 sends/hour per recipient) with JSON persistence
"""

from typing import Dict, Any, Optional, Tuple, Callable
import time
import json
import secrets
import hashlib
import hmac
from pathlib import Path
import streamlit as st
from modules.secrets_util import is_demo_mode_enabled
from modules.audit_logger import log_auth_event

RATE_LIMITS_FILE = Path(__file__).resolve().parent.parent / "data" / "otp_rate_limits.json"
AUTH_LOCKOUTS_FILE = Path(__file__).resolve().parent.parent / "data" / "auth_lockouts.json"
OTP_EXPIRY_SECONDS = 300.0  # 5 minutes
RESEND_COOLDOWN_SECONDS = 60.0  # 60 seconds
MAX_ATTEMPTS = 3
MAX_SENDS_PER_HOUR = 5
GLOBAL_DAILY_LIMIT = 100  # Global cap across all users per 24 hours
GLOBAL_DAILY_KEY = "__global_daily__"
COOLDOWN_PREFIX = "__cooldown__"

LOCKOUT_DURATION_SECONDS = 600.0  # 10 minutes (after 5 failed password attempts)
MAX_FAILED_LOGIN_ATTEMPTS = 5

DEMO_ACCOUNTS = ["demo@reviewlens.com", "vinay@reviewlens.com", "evaluator@gmail.com"]


def _get_target_key(target: str) -> str:
    """Produces a normalized hash key for recipient rate limiting."""
    clean = target.strip().lower()
    return hashlib.sha256(clean.encode("utf-8")).hexdigest()[:24]


def _load_auth_lockouts() -> Dict[str, Dict[str, Any]]:
    """Loads persistent failed password attempts and lockout timestamps from disk."""
    if not AUTH_LOCKOUTS_FILE.exists():
        return {}
    try:
        with open(AUTH_LOCKOUTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_auth_lockouts(data: Dict[str, Dict[str, Any]]) -> None:
    """Saves persistent failed password attempts and lockout timestamps to disk."""
    try:
        AUTH_LOCKOUTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(AUTH_LOCKOUTS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass


def get_failed_attempts(identifier: str) -> Tuple[int, float]:
    """
    Retrieves current consecutive failed attempts and locked_until timestamp.
    Returns: (fails_count, locked_until_epoch)
    """
    key = _get_target_key(identifier)
    data = _load_auth_lockouts()
    record = data.get(key, {"fails": 0, "locked_until": 0.0})
    now = time.time()
    locked_until = record.get("locked_until", 0.0)
    # If lockout expired, reset
    if locked_until > 0.0 and now > locked_until:
        clear_failed_attempts(identifier)
        return 0, 0.0
    return record.get("fails", 0), locked_until


def record_failed_attempt(identifier: str) -> Tuple[int, float]:
    """
    Records a failed authentication attempt on disk.
    If fails >= 5, triggers a 10-minute persistent lockout.
    Returns: (new_fails_count, locked_until_epoch)
    """
    key = _get_target_key(identifier)
    data = _load_auth_lockouts()
    record = data.get(key, {"fails": 0, "locked_until": 0.0})
    now = time.time()

    fails = record.get("fails", 0) + 1
    locked_until = record.get("locked_until", 0.0)
    if fails >= MAX_FAILED_LOGIN_ATTEMPTS:
        locked_until = now + LOCKOUT_DURATION_SECONDS

    record["fails"] = fails
    record["locked_until"] = locked_until
    data[key] = record
    _save_auth_lockouts(data)
    return fails, locked_until


def clear_failed_attempts(identifier: str) -> None:
    """Clears failed attempts and lockouts upon successful login."""
    key = _get_target_key(identifier)
    data = _load_auth_lockouts()
    if key in data:
        data.pop(key, None)
        _save_auth_lockouts(data)


def _load_persistent_limits() -> Dict[str, Any]:
    """Loads recipient send timestamps from data/otp_rate_limits.json."""
    if not RATE_LIMITS_FILE.exists():
        return {}
    try:
        with open(RATE_LIMITS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_persistent_limits(data: Dict[str, Any]) -> None:
    """Saves recipient send timestamps to data/otp_rate_limits.json."""
    try:
        RATE_LIMITS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(RATE_LIMITS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass


def get_target_cooldown_remaining(target: str) -> int:
    """Returns remaining seconds for recipient 60s cooldown (disk-backed across sessions/tabs)."""
    key = _get_target_key(target)
    disk_data = _load_persistent_limits()
    last_sent = disk_data.get(f"{COOLDOWN_PREFIX}{key}", 0.0)
    now = time.time()
    rem = RESEND_COOLDOWN_SECONDS - (now - last_sent)
    return max(0, int(rem))


def check_and_record_dispatch(target: str, enforce_cooldown: bool = False) -> Tuple[bool, str, int]:
    """
    Validates that dispatch conforms to:
      1. Recipient 60s cooldown (if enforce_cooldown=True)
      2. Global daily limit (max 100/day across all recipients)
      3. Recipient hourly quota (max 5/hr)
    Returns: (is_allowed, error_message, sends_remaining_or_cooldown)
    """
    key = _get_target_key(target)
    now = time.time()
    cutoff_hour = now - 3600.0
    cutoff_day = now - 86400.0

    disk_data = _load_persistent_limits()

    # 1. Enforce persistent 60-second recipient cooldown if requested
    if enforce_cooldown:
        wait_s = get_target_cooldown_remaining(target)
        if wait_s > 0:
            return False, f"Please wait {wait_s}s before requesting a new verification code.", wait_s

    # 2. Enforce global daily dispatch quota (100 sends/24h)
    raw_global = disk_data.get(GLOBAL_DAILY_KEY, [])
    global_timestamps = [t for t in raw_global if isinstance(t, (int, float)) and t > cutoff_day]
    if len(global_timestamps) >= GLOBAL_DAILY_LIMIT:
        return False, "Daily verification dispatch quota reached. Please try again tomorrow.", 86400

    # 3. Enforce recipient hourly quota (5 sends/hr)
    raw_target = disk_data.get(key, [])
    target_timestamps = [t for t in raw_target if isinstance(t, (int, float)) and t > cutoff_hour]
    if len(target_timestamps) >= MAX_SENDS_PER_HOUR:
        oldest_active = target_timestamps[0]
        wait_seconds = int((oldest_active + 3600.0) - now)
        wait_minutes = max(1, (wait_seconds + 59) // 60)
        return False, f"Hourly dispatch limit exceeded (max {MAX_SENDS_PER_HOUR}/hr). Please wait {wait_minutes} minutes.", wait_seconds

    # Record dispatch across disk and in-memory caches
    global_timestamps.append(now)
    target_timestamps.append(now)
    disk_data[GLOBAL_DAILY_KEY] = global_timestamps
    disk_data[key] = target_timestamps
    disk_data[f"{COOLDOWN_PREFIX}{key}"] = now
    _save_persistent_limits(disk_data)

    remaining_sends = MAX_SENDS_PER_HOUR - len(target_timestamps)
    return True, "", remaining_sends


class OTPManager:
    """
    Unified cryptographically secure OTP state manager.
    Stores only salted hashes in session state.
    """

    @staticmethod
    def generate_and_store(
        session_key: str,
        target: str,
        channel: str,
        purpose: str = "general"
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generates a 6-digit random code, salts and hashes it, and stores the state.
        Returns: (raw_code, state_metadata)
        Note: The raw code is returned ONLY to caller dispatch function and is never
        stored in session state or persistent storage.
        """
        now = time.time()
        # Generate 6-digit integer code cryptographically
        code_int = secrets.randbelow(900000) + 100000
        raw_code = str(code_int)

        # Generate unique 16-byte cryptographic salt
        salt = secrets.token_hex(16)
        code_hash = hashlib.sha256((salt + raw_code).encode("utf-8")).hexdigest()

        otp_state = {
            "channel": str(channel),
            "target_key": _get_target_key(target),
            "salt": salt,
            "code_hash": code_hash,
            "created_at": now,
            "expires_at": now + OTP_EXPIRY_SECONDS,
            "cooldown_until": now + RESEND_COOLDOWN_SECONDS,
            "attempts": 0,
            "max_attempts": MAX_ATTEMPTS,
            "purpose": str(purpose),
            "verified": False
        }

        st.session_state[session_key] = otp_state
        return raw_code, otp_state

    @staticmethod
    def get_cooldown_remaining(session_key: str) -> int:
        """Returns seconds remaining before resend is allowed (0 if ready)."""
        state = st.session_state.get(session_key)
        if not state:
            return 0
        now = time.time()
        rem = state.get("cooldown_until", 0.0) - now
        return max(0, int(rem))

    @staticmethod
    def get_expiry_remaining(session_key: str) -> int:
        """Returns seconds remaining before OTP expires (0 if expired)."""
        state = st.session_state.get(session_key)
        if not state:
            return 0
        now = time.time()
        rem = state.get("expires_at", 0.0) - now
        return max(0, int(rem))

    @staticmethod
    def verify(
        session_key: str,
        target: str,
        entered_code: str,
        expected_purpose: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Verifies entered code against stored salted hash using constant-time comparison.
        Enforces 3-attempt limit and 5-minute expiration.
        """
        state = st.session_state.get(session_key)
        if not state:
            log_auth_event("otp_failed", "otp_manager", target, success=False, details="No pending OTP state found")
            return False, "No active verification request. Please request a new verification code."

        # Verify target match
        if state.get("target_key") != _get_target_key(target):
            log_auth_event("otp_failed", state.get("channel", "otp"), target, success=False, details="Target recipient mismatch")
            return False, "Verification code target mismatch. Please request a new code."

        # Verify purpose if specified
        if expected_purpose and state.get("purpose") != expected_purpose:
            log_auth_event("otp_failed", state.get("channel", "otp"), target, success=False, details="OTP purpose mismatch")
            return False, "Invalid verification context. Please restart this operation."

        now = time.time()
        if now > state.get("expires_at", 0.0):
            st.session_state.pop(session_key, None)
            log_auth_event("otp_failed", state.get("channel", "otp"), target, success=False, details="OTP expired")
            return False, "Verification code has expired (5-minute limit). Please request a new code."

        if state.get("attempts", 0) >= state.get("max_attempts", MAX_ATTEMPTS):
            st.session_state.pop(session_key, None)
            log_auth_event("otp_failed", state.get("channel", "otp"), target, success=False, details="Max attempts exceeded")
            return False, "Maximum verification attempts exceeded (3 attempts). Code invalidated."

        # Increment attempt counter
        state["attempts"] += 1
        clean_code = "".join(ch for ch in str(entered_code) if ch.isdigit())

        salt = state.get("salt", "")
        expected_hash = state.get("code_hash", "")
        candidate_hash = hashlib.sha256((salt + clean_code).encode("utf-8")).hexdigest()

        # Constant-time comparison
        is_match = hmac.compare_digest(candidate_hash, expected_hash)
        if not is_match and is_demo_mode_enabled():
            demo_hash = hashlib.sha256((salt + "123456").encode("utf-8")).hexdigest()
            if hmac.compare_digest(candidate_hash, demo_hash):
                # Restrict 123456 demo bypass:
                # Allowed ONLY for accounts in DEMO_ACCOUNTS and brand-new registrations (purpose == "signup").
                # NEVER allowed for password reset (purpose == "forgot_password") of non-demo accounts.
                clean_target = str(target).strip().lower()
                is_demo_target = clean_target in [acc.lower() for acc in DEMO_ACCOUNTS]
                is_signup = state.get("purpose") == "signup"
                if is_demo_target or is_signup:
                    is_match = True
                else:
                    log_auth_event(
                        "otp_bypass_blocked",
                        state.get("channel", "otp"),
                        target,
                        success=False,
                        details=f"Demo code 123456 blocked for real account password reset/login (purpose: {state.get('purpose')})"
                    )

        if is_match:
            state["verified"] = True
            log_auth_event("otp_verified", state.get("channel", "otp"), target, success=True, details=f"Verified for {state.get('purpose')}")
            return True, "Verification successful."

        remaining = max(0, state["max_attempts"] - state["attempts"])
        log_auth_event("otp_failed", state.get("channel", "otp"), target, success=False, details=f"Incorrect code, attempts left: {remaining}")
        if remaining == 0:
            st.session_state.pop(session_key, None)
            return False, "Incorrect verification code. Maximum attempts reached. Code invalidated."

        return False, f"Incorrect verification code. {remaining} attempt{'s' if remaining != 1 else ''} remaining."

    @staticmethod
    def is_verified(
        session_key: str,
        target: str,
        expected_purpose: Optional[str] = None
    ) -> bool:
        """
        Validates whether an OTP was successfully verified for this target and purpose,
        and has not expired.
        """
        state = st.session_state.get(session_key)
        if not state:
            return False
        if not state.get("verified", False):
            return False
        if state.get("target_key") != _get_target_key(target):
            return False
        if expected_purpose and state.get("purpose") != expected_purpose:
            return False
        if time.time() > state.get("expires_at", 0.0):
            return False
        return True

    @staticmethod
    def clear(session_key: str) -> None:
        """Removes the OTP state from session_state."""
        st.session_state.pop(session_key, None)
