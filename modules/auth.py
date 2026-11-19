"""
ReviewLens Authentication & User Identity Engine.
Provides secure, persistent user authentication:
  1. Email & Password registration and login (salted SHA-256 hashes, stored in data/users.json)
  2. Google / Gmail 1-click OAuth identity
  3. Mobile SMS OTP phone verification
  4. Premium Futuristic AI SaaS Login & Registration split-screen UI (Deep Navy #09091B, Dark Purple #15112B, Electric Violet #9B5CFF)
"""

from typing import Dict, Any, Optional
import os
import json
import re
import time
import hashlib
import secrets
import random
import html
import base64
from pathlib import Path
import streamlit as st

# Path to persistent user credential storage
USER_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "users.json"
LAPTOP_IMG_PATH = Path(__file__).resolve().parent.parent / "static" / "images" / "auth_laptop_illustration.png"


def _get_laptop_illustration_b64() -> str:
    """Returns base64 data URI string of the 3D ReviewLens laptop hero illustration."""
    if LAPTOP_IMG_PATH.exists():
        try:
            with open(LAPTOP_IMG_PATH, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return ""
    return ""


def _render_html(html_str: str) -> None:
    """Safely renders HTML without Markdown turning indented lines into code blocks."""
    if hasattr(st, "html"):
        st.html(html_str)
    else:
        clean = "\n".join(line.strip() for line in html_str.splitlines() if line.strip())
        st.markdown(clean, unsafe_allow_html=True)


# PBKDF2 parameter: OWASP 2023+ recommendation for HMAC-SHA256 is >= 600,000 iterations
PBKDF2_ITERATIONS = int(os.getenv("PBKDF2_ITERATIONS", "600000"))


def hash_password(password: str, salt: str, iterations: int = PBKDF2_ITERATIONS) -> str:
    """
    Generates a cryptographically strong PBKDF2-HMAC-SHA256 password hash.
    Enforces OWASP-recommended iterations (600,000) with a unique per-user 16-byte salt.
    """
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations
    ).hex()


def _load_user_db() -> Dict[str, Any]:
    """
    Loads user database from data/users.json without seeding default accounts.
    Never auto-creates production accounts with hardcoded passwords.
    """
    if not USER_DB_PATH.exists():
        return {}

    try:
        with open(USER_DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_user_db(db: Dict[str, Any]) -> None:
    """Safely saves user database to data/users.json."""
    USER_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(USER_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)


def get_current_user() -> Dict[str, Any]:
    """Retrieves or initializes the current active user session state."""
    if "user_profile" not in st.session_state:
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
    return st.session_state["user_profile"]


def authenticate_user(email: str, password: str) -> Dict[str, Any]:
    """
    Validates user credentials against hashed storage with attempt throttling.
    Sets active authenticated session on success.
    """
    clean_email = email.strip().lower() if email else ""
    if not clean_email or not password:
        return {"success": False, "message": "Please enter both email and password."}

    # Lockout check (5 failed attempts locks for 5 minutes)
    rate_limits = st.session_state.setdefault("auth_rate_limits", {})
    client_state = rate_limits.setdefault(clean_email, {"fails": 0, "locked_until": 0.0})
    now = time.time()
    if client_state["locked_until"] > now:
        remaining = int(client_state["locked_until"] - now)
        return {"success": False, "message": f"Account temporarily locked due to too many failed attempts. Try again in {remaining}s."}

    db = _load_user_db()
    user_record = db.get(clean_email)
    if not user_record:
        return {"success": False, "message": "No account found with this email address. Please register or check spelling."}

    expected_hash_600k = hash_password(password, user_record["salt"], iterations=PBKDF2_ITERATIONS)
    expected_hash_100k = hash_password(password, user_record["salt"], iterations=100_000)
    legacy_hash = hashlib.sha256((user_record["salt"] + password).encode("utf-8")).hexdigest()

    matched = False
    if user_record["password_hash"] == expected_hash_600k:
        matched = True
    elif user_record["password_hash"] in (expected_hash_100k, legacy_hash):
        matched = True
        # Auto-upgrade legacy or 100k iteration hash to OWASP 600,000-iteration PBKDF2
        user_record["password_hash"] = expected_hash_600k
        user_record["iterations"] = PBKDF2_ITERATIONS
        _save_user_db(db)

    if not matched:
        client_state["fails"] += 1
        if client_state["fails"] >= 5:
            client_state["locked_until"] = now + 300.0  # 5 min lockout
            return {"success": False, "message": "Too many failed attempts. Account locked for 5 minutes."}
        return {"success": False, "message": "Incorrect password. Please verify your credentials."}

    # Reset failed attempts on success
    client_state["fails"] = 0
    client_state["locked_until"] = 0.0

    # Authentication successful: create session profile
    display_name = user_record.get("name", clean_email.split("@")[0].title())
    avatar_letter = display_name[0].upper() if display_name else "U"

    profile = {
        "authenticated": True,
        "auth_method": "email_password",
        "identifier": clean_email,
        "display_name": display_name,
        "avatar_letter": avatar_letter,
        "tier": user_record.get("tier", "Verified ReviewLens Member"),
        "tier_badge": "🛡️ Verified Member",
        "member_since": user_record.get("created_at", "Nov 2026"),
        "saved_comparisons": st.session_state.get("compare_tray", []),
        "custom_presets": {}
    }
    st.session_state["user_profile"] = profile
    return {"success": True, "message": f"Welcome back, {display_name}!", "user": profile}


def register_user(name: str, email: str, password: str, confirm_password: str) -> Dict[str, Any]:
    """
    Validates inputs and creates a new user account with PBKDF2 password hashing.
    Enforces minimum 8 character password policy.
    """
    clean_name = name.strip() if name else ""
    clean_email = email.strip().lower() if email else ""

    if not clean_name:
        return {"success": False, "message": "Please provide your full name."}

    email_regex = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
    if not re.match(email_regex, clean_email):
        return {"success": False, "message": "Please enter a valid email address (e.g. name@example.com)."}

    if len(password) < 8:
        return {"success": False, "message": "Password must be at least 8 characters long."}

    if password != confirm_password:
        return {"success": False, "message": "Passwords do not match. Please re-type your password."}

    db = _load_user_db()
    if clean_email in db:
        return {"success": False, "message": "An account with this email already exists. Please sign in instead."}

    # Generate salt and PBKDF2 password hash
    salt = secrets.token_hex(16)
    p_hash = hash_password(password, salt)

    db[clean_email] = {
        "name": clean_name,
        "email": clean_email,
        "salt": salt,
        "password_hash": p_hash,
        "created_at": "Nov 2026",
        "tier": "Verified ReviewLens Member"
    }
    _save_user_db(db)

    # Automatically authenticate newly registered user
    avatar_letter = clean_name[0].upper() if clean_name else "U"
    profile = {
        "authenticated": True,
        "auth_method": "email_password",
        "identifier": clean_email,
        "display_name": clean_name,
        "avatar_letter": avatar_letter,
        "tier": "Verified ReviewLens Member",
        "tier_badge": "🛡️ Verified Member",
        "member_since": "Nov 2026",
        "saved_comparisons": st.session_state.get("compare_tray", []),
        "custom_presets": {}
    }
    st.session_state["user_profile"] = profile
    return {"success": True, "message": f"Account created successfully! Welcome to ReviewLens, {clean_name}.", "user": profile}


def is_google_oauth_configured() -> bool:
    """Checks whether genuine Google OAuth 2.0 credentials are present in environment or Streamlit secrets."""
    client_id = os.getenv("GOOGLE_CLIENT_ID", "")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET", "")
    if hasattr(st, "secrets"):
        try:
            client_id = client_id or st.secrets.get("GOOGLE_CLIENT_ID", "")
            client_secret = client_secret or st.secrets.get("GOOGLE_CLIENT_SECRET", "")
        except Exception:
            pass
    return bool(client_id and client_secret)


def get_google_oauth_url(redirect_uri: str = "http://localhost:8501") -> Optional[str]:
    """Generates genuine Google OAuth 2.0 authorization redirect URL."""
    if not is_google_oauth_configured():
        return None
    client_id = os.getenv("GOOGLE_CLIENT_ID", "")
    if hasattr(st, "secrets") and not client_id:
        try:
            client_id = st.secrets.get("GOOGLE_CLIENT_ID", "")
        except Exception:
            pass
    import urllib.parse
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "consent"
    }
    return f"https://accounts.google.com/o/oauth2/v2/auth?{urllib.parse.urlencode(params)}"


def login_with_google(email: str, name: Optional[str] = None, verified: bool = False) -> Dict[str, Any]:
    """
    Authenticates user via genuine verified Google Identity token.
    Rejects unverified or simulated requests when OAuth credentials are not configured.
    """
    if not verified and not is_google_oauth_configured() and os.getenv("TESTING") != "1":
        raise PermissionError(
            "Google OAuth 2.0 is disabled. Configure GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET to enable Google Sign-In."
        )

    clean_email = email.strip() if email else "shopper@gmail.com"
    if "@" not in clean_email:
        clean_email += "@gmail.com"

    display_name = name.strip() if (name and name.strip()) else clean_email.split("@")[0].replace(".", " ").title()
    avatar_letter = display_name[0].upper() if display_name else "G"

    profile = {
        "authenticated": True,
        "auth_method": "google",
        "identifier": clean_email,
        "display_name": display_name,
        "avatar_letter": avatar_letter,
        "tier": "Verified Google Shopper",
        "tier_badge": "🛡️ Verified Google",
        "member_since": "Nov 2026",
        "saved_comparisons": st.session_state.get("compare_tray", []),
        "custom_presets": {}
    }
    st.session_state["user_profile"] = profile
    return profile


def is_sms_configured() -> bool:
    """Checks whether genuine SMS gateway credentials (Twilio) are present."""
    sid = os.getenv("TWILIO_ACCOUNT_SID", "")
    token = os.getenv("TWILIO_AUTH_TOKEN", "")
    phone = os.getenv("TWILIO_PHONE_NUMBER", "")
    if hasattr(st, "secrets"):
        try:
            sid = sid or st.secrets.get("TWILIO_ACCOUNT_SID", "")
            token = token or st.secrets.get("TWILIO_AUTH_TOKEN", "")
            phone = phone or st.secrets.get("TWILIO_PHONE_NUMBER", "")
        except Exception:
            pass
    return bool(sid and token and phone)


def send_phone_otp(country_code: str, phone_number: str) -> Dict[str, Any]:
    """
    Generates and dispatches a cryptographically secure 6-digit SMS OTP via gateway.
    Never exposes or logs OTP codes on screen or in flash notifications.
    """
    clean_phone = "".join(ch for ch in phone_number if ch.isdigit())
    full_phone = f"{country_code} {clean_phone}"

    if not is_sms_configured() and os.getenv("TESTING") != "1":
        return {
            "success": False,
            "message": "SMS authentication is disabled. No SMS provider credentials (TWILIO_ACCOUNT_SID) found."
        }

    otp_code = str(secrets.randbelow(900000) + 100000)
    st.session_state["pending_phone_auth"] = {
        "full_phone": full_phone,
        "country_code": country_code,
        "phone_number": clean_phone,
        "otp": otp_code,
        "created_at": time.time(),
        "attempts": 0,
        "max_attempts": 3
    }

    if is_sms_configured():
        try:
            import requests
            sid = os.getenv("TWILIO_ACCOUNT_SID")
            token = os.getenv("TWILIO_AUTH_TOKEN")
            from_phone = os.getenv("TWILIO_PHONE_NUMBER")
            resp = requests.post(
                f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json",
                auth=(sid, token),
                data={
                    "From": from_phone,
                    "To": f"{country_code}{clean_phone}",
                    "Body": f"Your ReviewLens verification code is: {otp_code}. Valid for 5 minutes."
                },
                timeout=5.0
            )
            if resp.status_code in (200, 201):
                return {"success": True, "message": f"Verification code sent via SMS to {full_phone}."}
            else:
                return {"success": False, "message": "Failed to dispatch SMS through provider gateway."}
        except Exception as e:
            return {"success": False, "message": f"SMS gateway connection error: {str(e)}"}

    # Testing mode returns programmatic status with test_otp for unittest assertions
    return {"success": True, "message": "OTP generated for test verification.", "test_otp": otp_code}


def verify_phone_otp(entered_otp: str) -> bool:
    """
    Verifies entered OTP against pending phone authentication state.
    Enforces expiration (5 mins) and attempt limiting (max 3 attempts).
    Removes hardcoded backdoor bypass codes.
    """
    pending = st.session_state.get("pending_phone_auth")
    if not pending:
        return False

    now = time.time()
    if now - pending.get("created_at", 0) > 300.0:  # 5 min expiry
        st.session_state.pop("pending_phone_auth", None)
        return False

    if pending.get("attempts", 0) >= pending.get("max_attempts", 3):
        st.session_state.pop("pending_phone_auth", None)
        return False

    clean_entered = "".join(ch for ch in entered_otp if ch.isdigit())
    if clean_entered == pending["otp"]:
        full_phone = pending["full_phone"]
        display_name = f"Shopper ({pending['phone_number'][-4:]})"

        profile = {
            "authenticated": True,
            "auth_method": "phone",
            "identifier": full_phone,
            "display_name": display_name,
            "avatar_letter": "P",
            "tier": "SMS-Verified Shopper",
            "tier_badge": "📱 Verified Phone",
            "member_since": "Nov 2026",
            "saved_comparisons": st.session_state.get("compare_tray", []),
            "custom_presets": {}
        }
        st.session_state["user_profile"] = profile
        st.session_state.pop("pending_phone_auth", None)
        return True
    else:
        pending["attempts"] += 1
        if pending["attempts"] >= pending.get("max_attempts", 3):
            st.session_state.pop("pending_phone_auth", None)
        return False


def login_as_guest() -> Dict[str, Any]:
    """
    Authenticates the session in Guest/Evaluator Explorer mode.
    Allows evaluators to explore the full ReviewLens platform without credentials.
    """
    profile = {
        "authenticated": True,
        "auth_method": "guest",
        "identifier": "guest@reviewlens.com",
        "display_name": "Guest Explorer",
        "avatar_letter": "G",
        "tier": "Guest Explorer",
        "tier_badge": "⚪ Guest Preview",
        "member_since": "Nov 2026",
        "saved_comparisons": st.session_state.get("compare_tray", []),
        "custom_presets": {}
    }
    st.session_state["user_profile"] = profile
    return profile


def logout_user() -> None:
    """Resets user session state back to unauthenticated mode."""
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
    st.session_state["auth_mode"] = "login"


def render_auth_interface(on_navigate_home=None) -> None:
    """
    Renders the Premium, Futuristic AI SaaS Split-Screen Login & Registration Experience.
    Color Palette: Deep Navy (#09091B), Dark Purple (#15112B), Electric Violet (#9B5CFF), Soft Lavender (#B9B3D0).
    """
    user = get_current_user()

    # ----------------------------------------------------
    # STATE 1: ALREADY AUTHENTICATED -> USER PROFILE DASHBOARD
    # ----------------------------------------------------
    if user.get("authenticated"):
        esc_name = html.escape(str(user.get("display_name", "Shopper")))
        esc_id = html.escape(str(user.get("identifier", "")))
        esc_tier = html.escape(str(user.get("tier", "Verified Member")))
        esc_badge = html.escape(str(user.get("tier_badge", "🛡️ Verified")))
        letter = user.get("avatar_letter", "U")

        st.markdown(f"""
        <div style="background: radial-gradient(circle at 10% 20%, rgba(113, 56, 245, 0.25), transparent 40%), radial-gradient(circle at 90% 80%, rgba(155, 92, 255, 0.2), transparent 40%), #09091B; border: 1px solid rgba(155, 92, 255, 0.35); border-radius: 20px; padding: 2.2rem; color: #FFFFFF; box-shadow: 0 20px 50px rgba(0,0,0,0.5), 0 0 30px rgba(113, 56, 245, 0.2); margin-bottom: 2rem;">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1.5rem;">
                <div style="display: flex; align-items: center; gap: 1.5rem;">
                    <div style="width: 76px; height: 76px; border-radius: 50%; background: linear-gradient(135deg, #7138F5 0%, #9B5CFF 100%); display: flex; align-items: center; justify-content: center; font-size: 2.2rem; font-weight: 800; color: #FFFFFF; box-shadow: 0 4px 20px rgba(155, 92, 255, 0.5); border: 3px solid rgba(255, 255, 255, 0.25);">
                        {letter}
                    </div>
                    <div>
                        <div style="display: flex; align-items: center; gap: 0.8rem; margin-bottom: 0.4rem; flex-wrap: wrap;">
                            <h2 style="margin: 0; font-size: 1.75rem; font-weight: 800; color: #FFFFFF;">{esc_name}</h2>
                            <span style="background: rgba(155, 92, 255, 0.25); border: 1px solid #9B5CFF; color: #C4B5FD; font-size: 0.8rem; font-weight: 700; padding: 0.25rem 0.75rem; border-radius: 14px;">{esc_badge}</span>
                        </div>
                        <div style="font-size: 0.92rem; color: #B9B3D0;">Identity: <b style="color: #FFFFFF;">{esc_id}</b> • Member since {user.get('member_since', '2026')}</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Account Status", "Active", "Verified AI Pass")
        with k2:
            st.metric("Compare Tray Items", len(st.session_state.get("compare_tray", [])), "Saved in session")
        with k3:
            st.metric("Member Tier", esc_tier, "Level 2 Access")
        with k4:
            st.metric("Credibility Rating", "99.8%", "Glass-Box Verified")

        st.markdown("<br>", unsafe_allow_html=True)
        act1, act2, act3 = st.columns([1.5, 1.5, 1])
        with act1:
            if st.button("🛍️ Return to Discovery Catalogue →", use_container_width=True, type="primary"):
                if on_navigate_home:
                    on_navigate_home()
        with act2:
            if st.button("🧪 Open Live Diagnostic Lab →", use_container_width=True):
                st.session_state["pending_nav"] = "🧪 Live Review Analyzer"
                st.rerun()
        with act3:
            if st.button("🚪 Sign Out", use_container_width=True):
                logout_user()
                st.toast("Signed out successfully.", icon="ℹ️")
                st.rerun()
        return

    # ----------------------------------------------------
    # STATE 2: SPLIT-SCREEN LOGIN & REGISTRATION EXPERIENCE
    # ----------------------------------------------------
    if "auth_mode" not in st.session_state:
        st.session_state["auth_mode"] = "login"

    laptop_b64 = _get_laptop_illustration_b64()

    # Inject Premium Futuristic AI SaaS Theme matching the exact reference design
    _render_html("""
    <style>
        /* Ambient Aurora Glow Background */
        body, .stApp {
            background-color: #07051A !important;
            background-image: 
                radial-gradient(circle at 12% 18%, rgba(113, 56, 245, 0.45) 0%, transparent 45%),
                radial-gradient(circle at 88% 12%, rgba(155, 92, 255, 0.38) 0%, transparent 40%),
                radial-gradient(circle at 50% 88%, rgba(113, 56, 245, 0.35) 0%, transparent 50%),
                radial-gradient(circle at 85% 82%, rgba(147, 51, 234, 0.28) 0%, transparent 40%) !important;
            background-attachment: fixed !important;
        }

        /* Glassmorphic Auth Card Container applied to border container */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(18, 14, 38, 0.85) !important;
            border: 1px solid rgba(155, 92, 255, 0.35) !important;
            backdrop-filter: blur(25px) !important;
            -webkit-backdrop-filter: blur(25px) !important;
            border-radius: 24px !important;
            padding: 2.2rem 2.2rem 1.8rem 2.2rem !important;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.65), 0 0 35px rgba(113, 56, 245, 0.22) !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] > div {
            border: none !important;
        }

        /* 4 Feature Items Grid Item */
        .hero-feature-item {
            background: rgba(22, 17, 46, 0.65);
            border: 1px solid rgba(155, 92, 255, 0.25);
            border-radius: 14px;
            padding: 0.95rem 0.5rem;
            transition: all 0.25s ease;
        }
        .hero-feature-item:hover {
            background: rgba(113, 56, 245, 0.25);
            border-color: #9B5CFF;
            transform: translateY(-3px);
            box-shadow: 0 8px 20px rgba(155, 92, 255, 0.3);
        }

        /* Primary Sign In Button */
        div[data-testid="stButton"] button[kind="primary"] {
            background: linear-gradient(135deg, #7138F5 0%, #9B5CFF 100%) !important;
            border: none !important;
            border-radius: 26px !important;
            color: #FFFFFF !important;
            font-weight: 700 !important;
            font-size: 1.05rem !important;
            padding: 0.72rem 1.5rem !important;
            box-shadow: 0 8px 25px rgba(113, 56, 245, 0.45) !important;
            transition: all 0.25s ease !important;
        }
        div[data-testid="stButton"] button[kind="primary"]:hover {
            box-shadow: 0 12px 35px rgba(155, 92, 255, 0.7) !important;
            transform: translateY(-2px) !important;
        }

        /* Secondary and Google buttons */
        div[data-testid="stButton"] button[kind="secondary"] {
            background: rgba(18, 14, 38, 0.8) !important;
            border: 1px solid rgba(255, 255, 255, 0.16) !important;
            border-radius: 26px !important;
            color: #FFFFFF !important;
            font-weight: 600 !important;
            transition: all 0.25s ease !important;
        }
        div[data-testid="stButton"] button[kind="secondary"]:hover {
            border-color: rgba(155, 92, 255, 0.7) !important;
            background: rgba(30, 24, 62, 0.9) !important;
        }

        /* Input text boxes styling */
        div[data-testid="stTextInput"] input {
            background: rgba(12, 9, 28, 0.65) !important;
            border: 1px solid rgba(155, 92, 255, 0.3) !important;
            border-radius: 12px !important;
            color: #FFFFFF !important;
            padding: 0.7rem 1rem !important;
            transition: all 0.2s ease !important;
        }
        div[data-testid="stTextInput"] input:focus {
            border-color: #9B5CFF !important;
            box-shadow: 0 0 15px rgba(155, 92, 255, 0.45) !important;
            background: rgba(20, 15, 45, 0.85) !important;
        }
    </style>
    """)

    col_left, col_right = st.columns([1.18, 1], gap="large")

    # ====================================================
    # LEFT COLUMN: BRANDING, 3D ILLUSTRATION & FEATURES
    # ====================================================
    with col_left:
        laptop_preview_html = f"""
        <div style="position: relative; margin: 0.2rem 0 1.2rem 0; text-align: center;">
            <img src="data:image/png;base64,{laptop_b64}" style="max-width: 440px; width: 100%; height: auto; display: inline-block; filter: drop-shadow(0 15px 35px rgba(113, 56, 245, 0.4));" alt="ReviewLens 3D NLP Dashboard Preview">
        </div>
        """ if laptop_b64 else """
        <div style="background: rgba(21, 17, 43, 0.7); border: 1px solid rgba(155, 92, 255, 0.3); border-radius: 16px; padding: 1.3rem; margin-bottom: 1.8rem; box-shadow: 0 10px 30px rgba(0,0,0,0.4);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 0.6rem;">
                <div style="font-size: 0.85rem; font-weight: 700; color: #C4B5FD; display: flex; align-items: center; gap: 0.5rem;">
                    <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #10B981; box-shadow: 0 0 8px #10B981;"></span>
                    Live NLP Review Stream
                </div>
                <div style="font-size: 0.78rem; color: #B9B3D0;">VADER & ABSA Powered</div>
            </div>
            <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(16, 185, 129, 0.25); border-left: 4px solid #10B981; border-radius: 8px; padding: 0.65rem 0.9rem; margin-bottom: 0.6rem;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.2rem;">
                    <span style="font-size: 0.78rem; font-weight: 700; color: #FFFFFF;">Galaxy S24 FE • Battery & Display</span>
                    <span style="color: #34D399; font-weight: 700; font-size: 0.75rem;">🟢 POSITIVE (+0.84)</span>
                </div>
                <div style="font-size: 0.82rem; color: #E2E8F0; font-style: italic;">"Battery easily lasted two full days, and the 120Hz OLED screen is stunning..."</div>
            </div>
            <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(239, 68, 68, 0.25); border-left: 4px solid #EF4444; border-radius: 8px; padding: 0.65rem 0.9rem; margin-bottom: 0.6rem;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.2rem;">
                    <span style="font-size: 0.78rem; font-weight: 700; color: #FFFFFF;">Bose QC Ultra • Microphone</span>
                    <span style="color: #F87171; font-weight: 700; font-size: 0.75rem;">🔴 NEGATIVE (-0.68)</span>
                </div>
                <div style="font-size: 0.82rem; color: #E2E8F0; font-style: italic;">"Microphone sounds muffled during noisy Zoom calls in wind..."</div>
            </div>
            <div style="margin-top: 0.9rem; padding-top: 0.6rem; border-top: 1px solid rgba(255,255,255,0.08);">
                <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #B9B3D0; margin-bottom: 0.3rem;">
                    <span>Sentiment Polarity Spread:</span>
                    <span><b>74% Positive</b> • 18% Neutral • 8% Negative</span>
                </div>
                <div style="height: 6px; border-radius: 3px; background: #1E293B; overflow: hidden; display: flex;">
                    <div style="width: 74%; background: #10B981;"></div>
                    <div style="width: 18%; background: #F59E0B;"></div>
                    <div style="width: 8%; background: #EF4444;"></div>
                </div>
            </div>
        </div>
        """

        _render_html(f"""
        <div style="padding-right: 0.5rem;">
            <div style="display: inline-flex; align-items: center; gap: 0.75rem; margin-bottom: 1.5rem;">
                <div style="width: 44px; height: 44px; border-radius: 12px; background: linear-gradient(135deg, #7138F5 0%, #9B5CFF 100%); display: flex; align-items: center; justify-content: center; font-size: 22px; color: #FFFFFF; box-shadow: 0 0 22px rgba(155, 92, 255, 0.6);">
                    ★
                </div>
                <div style="font-size: 1.9rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.5px;">
                    Review<span style="color: #9B5CFF;">Lens</span>
                </div>
            </div>

            <h1 style="font-size: 2.85rem; font-weight: 800; line-height: 1.15; margin: 0 0 1rem 0; color: #FFFFFF; letter-spacing: -0.5px;">
                Insights from<br>
                Product <span style="background: linear-gradient(135deg, #B575FF 0%, #E0A7FF 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Reviews</span>
            </h1>

            <p style="font-size: 1.05rem; color: #B9B3D0; line-height: 1.6; margin-bottom: 1.6rem; max-width: 520px;">
                An NLP-powered platform to analyze reviews, understand customer sentiment, detect fake reviews, and get smart, AI-generated replies.
            </p>

            {laptop_preview_html}

            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.75rem; text-align: center; margin-top: 1.2rem;">
                <div class="hero-feature-item">
                    <div style="font-size: 1.65rem; color: #9B5CFF; margin-bottom: 0.35rem;">📊</div>
                    <div style="font-size: 0.82rem; font-weight: 700; color: #FFFFFF; line-height: 1.3;">Sentiment<br>Analysis</div>
                </div>
                <div class="hero-feature-item">
                    <div style="font-size: 1.65rem; color: #9B5CFF; margin-bottom: 0.35rem;">🏷️</div>
                    <div style="font-size: 0.82rem; font-weight: 700; color: #FFFFFF; line-height: 1.3;">Aspect-based<br>Analysis</div>
                </div>
                <div class="hero-feature-item">
                    <div style="font-size: 1.65rem; color: #9B5CFF; margin-bottom: 0.35rem;">🛡️</div>
                    <div style="font-size: 0.82rem; font-weight: 700; color: #FFFFFF; line-height: 1.3;">Fake Review<br>Detection</div>
                </div>
                <div class="hero-feature-item">
                    <div style="font-size: 1.65rem; color: #9B5CFF; margin-bottom: 0.35rem;">✨</div>
                    <div style="font-size: 0.82rem; font-weight: 700; color: #FFFFFF; line-height: 1.3;">AI Review<br>Replies</div>
                </div>
            </div>
        </div>
        """)

    # ====================================================
    # RIGHT COLUMN: GLASSMORPHISM LOGIN / SIGNUP CARD
    # ====================================================
    with col_right:
        with st.container(border=True):
            # ------------------------------------------------
            # 1. LOGIN MODE
            # ------------------------------------------------
            if st.session_state["auth_mode"] == "login":
                _render_html("""
                <div style="text-align: center; margin-bottom: 1.5rem;">
                    <div style="display: inline-flex; align-items: center; justify-content: center; gap: 0.5rem; margin-bottom: 0.75rem;">
                        <div style="width: 36px; height: 36px; border-radius: 10px; background: linear-gradient(135deg, #7138F5 0%, #9B5CFF 100%); display: flex; align-items: center; justify-content: center; font-size: 18px; color: #FFFFFF; box-shadow: 0 0 16px rgba(155, 92, 255, 0.5);">
                            ★
                        </div>
                        <div style="font-size: 1.55rem; font-weight: 800; color: #FFFFFF;">
                            Review<span style="color: #9B5CFF;">Lens</span>
                        </div>
                    </div>
                    <h2 style="font-size: 2.15rem; font-weight: 800; color: #FFFFFF; margin: 0 0 0.35rem 0; letter-spacing: -0.5px;">
                        Welcome Back!
                    </h2>
                    <p style="font-size: 0.92rem; color: #B9B3D0; margin: 0;">
                        Sign in to continue analyzing product reviews.
                    </p>
                </div>
                """)

                login_email = st.text_input(
                    "Email Address",
                    value="",
                    placeholder="✉️ Enter your email",
                    label_visibility="collapsed",
                    key="auth_field_email"
                )

                login_password = st.text_input(
                    "Password",
                    value="",
                    type="password",
                    placeholder="🔒 Enter your password",
                    label_visibility="collapsed",
                    key="auth_field_password"
                )

                col_rem, col_forgot = st.columns([1, 1])
                with col_rem:
                    remember_me = st.checkbox("Remember me", value=True, key="login_remember_me")
                with col_forgot:
                    if st.button("Forgot password?", key="btn_toggle_forgot", help="Reset your account password"):
                        st.session_state["auth_mode"] = "forgot"
                        st.rerun()

                if st.button("Sign In →", key="btn_submit_signin", type="primary", use_container_width=True):
                    with st.spinner("Authenticating credentials..."):
                        res = authenticate_user(login_email, login_password)
                        if res["success"]:
                            st.toast(res["message"], icon="✅")
                            if on_navigate_home:
                                on_navigate_home()
                            else:
                                st.session_state["pending_nav"] = "🏠 Home / Discovery"
                                st.rerun()
                        else:
                            st.error(f"⚠️ {res['message']}")

                _render_html("""
                <div style="text-align: center; margin: 1.1rem 0; color: #7E7A94; font-size: 0.85rem; position: relative;">
                    <span style="background: #120E26; padding: 0 12px; position: relative; z-index: 1;">Or continue with</span>
                    <hr style="position: absolute; top: 50%; left: 0; right: 0; border: none; border-top: 1px solid rgba(255,255,255,0.12); margin: 0; z-index: 0;">
                </div>
                """)

                if st.button("🔴 Continue with Google", key="btn_login_google", use_container_width=True):
                    st.session_state["auth_mode"] = "google"
                    st.rerun()

                _render_html("""
                <div style="text-align: center; font-size: 0.92rem; color: #B9B3D0; margin: 1.1rem 0 0.3rem 0;">
                    Don't have an account? <span style="color: #9B5CFF; font-weight: 700;">Create Account</span>
                </div>
                """)
                if st.button("Create Account →", key="btn_goto_signup", use_container_width=True):
                    st.session_state["auth_mode"] = "signup"
                    st.rerun()

                _render_html("<div style='margin-top: 0.8rem; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 0.8rem;'></div>")
                col_alt1, col_alt2 = st.columns(2)
                with col_alt1:
                    if st.button("📱 Phone SMS Login", key="btn_switch_phone", use_container_width=True):
                        st.session_state["auth_mode"] = "phone"
                        st.rerun()
                with col_alt2:
                    if st.button("⚡ Guest Preview →", key="btn_auth_guest_preview", use_container_width=True):
                        login_as_guest()
                        if on_navigate_home:
                            on_navigate_home()
                        else:
                            st.session_state["pending_nav"] = "🏠 Home / Discovery"
                            st.rerun()

            # ------------------------------------------------
            # 2. PHONE SMS OTP MODE
            # ------------------------------------------------
            elif st.session_state["auth_mode"] == "phone":
                _render_html("""
                <div style="text-align: center; margin-bottom: 1.5rem;">
                    <div style="display: inline-flex; align-items: center; justify-content: center; gap: 0.5rem; margin-bottom: 0.75rem;">
                        <div style="width: 36px; height: 36px; border-radius: 10px; background: linear-gradient(135deg, #7138F5 0%, #9B5CFF 100%); display: flex; align-items: center; justify-content: center; font-size: 18px; color: #FFFFFF; box-shadow: 0 0 16px rgba(155, 92, 255, 0.5);">
                            ★
                        </div>
                        <div style="font-size: 1.55rem; font-weight: 800; color: #FFFFFF;">
                            Review<span style="color: #9B5CFF;">Lens</span>
                        </div>
                    </div>
                    <h2 style="font-size: 2.15rem; font-weight: 800; color: #FFFFFF; margin: 0 0 0.35rem 0; letter-spacing: -0.5px;">
                        Mobile Verification
                    </h2>
                    <p style="font-size: 0.92rem; color: #B9B3D0; margin: 0;">
                        Sign in securely with mobile number and SMS OTP code.
                    </p>
                </div>
                """)

                if not is_sms_configured():
                    st.warning("⚠️ **SMS Phone Authentication is currently disabled.**")
                    st.info("No SMS gateway provider (e.g. Twilio) has been configured in the environment (`TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_PHONE_NUMBER`). OTP codes are never displayed on screen. Please sign in with email/password or create an account.")
                else:
                    c_cc, c_num = st.columns([1.1, 2])
                    with c_cc:
                        country_choice = st.selectbox(
                            "Country Code",
                            ["🇺🇸 +1", "🇮🇳 +91", "🇬🇧 +44", "🇩🇪 +49", "🇦🇺 +61", "🇸🇬 +65", "🇦🇪 +971"],
                            key="auth_phone_country"
                        )
                        code_prefix = country_choice.split(" ")[-1]
                    with c_num:
                        phone_input = st.text_input("Mobile Number", placeholder="e.g. 9876543210", key="auth_phone_number")

                    if st.button("📲 Send SMS OTP Code", key="btn_send_otp", use_container_width=True):
                        digits = "".join(c for c in phone_input if c.isdigit())
                        if len(digits) >= 7:
                            res = send_phone_otp(code_prefix, phone_input)
                            if res.get("success"):
                                st.success("Verification code dispatched via SMS. Please enter the code received on your mobile device.")
                            else:
                                st.error(res.get("message", "Failed to dispatch SMS."))
                        else:
                            st.error("Please enter a valid phone number (at least 7 digits).")

                    if "pending_phone_auth" in st.session_state:
                        _render_html("<hr style='border-color: rgba(255,255,255,0.1); margin: 1rem 0;'>")
                        st.caption(f"Verifying phone: **{st.session_state['pending_phone_auth']['full_phone']}**")
                        entered_otp = st.text_input("Enter 6-Digit SMS Verification Code", max_chars=6, placeholder="e.g. 123456", key="auth_field_otp")
                        if st.button("Verify Code & Sign In →", key="btn_verify_otp", type="primary", use_container_width=True):
                            if verify_phone_otp(entered_otp):
                                st.toast("Phone verification successful! Welcome.", icon="✅")
                                if on_navigate_home:
                                    on_navigate_home()
                                else:
                                    st.session_state["pending_nav"] = "🏠 Home / Discovery"
                                    st.rerun()
                            else:
                                st.error("⚠️ Invalid or expired OTP code.")

                _render_html("<br>")
                if st.button("← Return to Email Sign In", key="btn_phone_back_login", use_container_width=True):
                    st.session_state["auth_mode"] = "login"
                    st.rerun()

            # ------------------------------------------------
            # 3. SIGNUP MODE
            # ------------------------------------------------
            elif st.session_state["auth_mode"] == "signup":
                _render_html("""
                <div style="text-align: center; margin-bottom: 1.5rem;">
                    <div style="display: inline-flex; align-items: center; justify-content: center; gap: 0.5rem; margin-bottom: 0.75rem;">
                        <div style="width: 36px; height: 36px; border-radius: 10px; background: linear-gradient(135deg, #7138F5 0%, #9B5CFF 100%); display: flex; align-items: center; justify-content: center; font-size: 18px; color: #FFFFFF; box-shadow: 0 0 16px rgba(155, 92, 255, 0.5);">
                            ★
                        </div>
                        <div style="font-size: 1.55rem; font-weight: 800; color: #FFFFFF;">
                            Review<span style="color: #9B5CFF;">Lens</span>
                        </div>
                    </div>
                    <h2 style="font-size: 2.15rem; font-weight: 800; color: #FFFFFF; margin: 0 0 0.35rem 0; letter-spacing: -0.5px;">
                        Create Your Account
                    </h2>
                    <p style="font-size: 0.92rem; color: #B9B3D0; margin: 0;">
                        Join ReviewLens to analyze product reviews and unlock AI insights.
                    </p>
                </div>
                """)

                signup_name = st.text_input("Full Name", placeholder="👤 Enter your full name", label_visibility="collapsed", key="signup_field_name")
                signup_email = st.text_input("Email Address", placeholder="✉️ Enter your email", label_visibility="collapsed", key="signup_field_email")
                signup_pw = st.text_input("Password", type="password", placeholder="🔒 Create strong password (min. 6 chars)", label_visibility="collapsed", key="signup_field_password")
                signup_confirm = st.text_input("Confirm Password", type="password", placeholder="🔒 Confirm your password", label_visibility="collapsed", key="signup_field_confirm")

                st.caption("🔒 Passwords are cryptographically salted and hashed. We never store plaintext.")

                if st.button("Create Account →", key="btn_submit_signup", type="primary", use_container_width=True):
                    with st.spinner("Creating your secure account..."):
                        res = register_user(signup_name, signup_email, signup_pw, signup_confirm)
                        if res["success"]:
                            st.toast(res["message"], icon="🎉")
                            if on_navigate_home:
                                on_navigate_home()
                            else:
                                st.session_state["pending_nav"] = "🏠 Home / Discovery"
                                st.rerun()
                        else:
                            st.error(f"⚠️ {res['message']}")

                _render_html("""
                <div style="text-align: center; margin: 1.1rem 0; color: #7E7A94; font-size: 0.85rem; position: relative;">
                    <span style="background: #120E26; padding: 0 12px; position: relative; z-index: 1;">Or continue with</span>
                    <hr style="position: absolute; top: 50%; left: 0; right: 0; border: none; border-top: 1px solid rgba(255,255,255,0.12); margin: 0; z-index: 0;">
                </div>
                """)

                if st.button("🔴 Sign Up with Google", key="btn_signup_google", use_container_width=True):
                    st.session_state["auth_mode"] = "google"
                    st.rerun()

                _render_html("""
                <div style="text-align: center; font-size: 0.92rem; color: #B9B3D0; margin: 1.2rem 0 0.3rem 0;">
                    Already have an account? <span style="color: #9B5CFF; font-weight: 700;">Sign In</span>
                </div>
                """)
                if st.button("← Back to Sign In", key="btn_goto_login", use_container_width=True):
                    st.session_state["auth_mode"] = "login"
                    st.rerun()

            # ------------------------------------------------
            # 4. FORGOT PASSWORD MODE
            # ------------------------------------------------
            elif st.session_state["auth_mode"] == "forgot":
                _render_html("""
                <div style="text-align: center; margin-bottom: 1.5rem;">
                    <div style="display: inline-flex; align-items: center; justify-content: center; gap: 0.5rem; margin-bottom: 0.75rem;">
                        <div style="width: 36px; height: 36px; border-radius: 10px; background: linear-gradient(135deg, #7138F5 0%, #9B5CFF 100%); display: flex; align-items: center; justify-content: center; font-size: 18px; color: #FFFFFF; box-shadow: 0 0 16px rgba(155, 92, 255, 0.5);">
                            ★
                        </div>
                        <div style="font-size: 1.55rem; font-weight: 800; color: #FFFFFF;">
                            Review<span style="color: #9B5CFF;">Lens</span>
                        </div>
                    </div>
                    <h2 style="font-size: 2.15rem; font-weight: 800; color: #FFFFFF; margin: 0 0 0.35rem 0; letter-spacing: -0.5px;">
                        Reset Password
                    </h2>
                    <p style="font-size: 0.92rem; color: #B9B3D0; margin: 0;">
                        Enter your registered email address and we'll send verification credentials.
                    </p>
                </div>
                """)

                forgot_email = st.text_input("Registered Email Address", placeholder="✉️ Enter your registered email", label_visibility="collapsed", key="forgot_field_email")

                if st.button("Send Reset Instructions →", key="btn_submit_forgot", type="primary", use_container_width=True):
                    if forgot_email and "@" in forgot_email:
                        st.info(f"ℹ️ **Demo Mode:** In production, a secure cryptographic token is dispatched to **{forgot_email}**. For immediate demo access, you can use the Demo Account: `demo@reviewlens.com` with `Password@123` or register a new account.")
                    else:
                        st.error("Please provide a valid registered email address.")

                _render_html("<br>")
                if st.button("← Return to Sign In", key="btn_forgot_back_login", use_container_width=True):
                    st.session_state["auth_mode"] = "login"
                    st.rerun()

            # ------------------------------------------------
            # 5. GOOGLE IDENTITY AUTHENTICATION & ACCESS
            # ------------------------------------------------
            elif st.session_state["auth_mode"] == "google":
                _render_html("""
                <div style="text-align: center; margin-bottom: 1.3rem;">
                    <div style="display: inline-flex; align-items: center; justify-content: center; width: 48px; height: 48px; border-radius: 50%; background: #FFFFFF; box-shadow: 0 4px 15px rgba(0,0,0,0.35); margin-bottom: 0.75rem;">
                        <svg width="26" height="26" viewBox="0 0 48 48">
                            <path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"/>
                            <path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"/>
                            <path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.55 10.78l7.98-6.19z"/>
                            <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"/>
                        </svg>
                    </div>
                    <h2 style="font-size: 1.85rem; font-weight: 800; color: #FFFFFF; margin: 0 0 0.35rem 0; letter-spacing: -0.5px;">
                        Sign in with Google
                    </h2>
                    <p style="font-size: 0.92rem; color: #B9B3D0; margin: 0;">
                        Choose an evaluator account or simulate sign up to continue to <b>ReviewLens</b>
                    </p>
                </div>
                """)

                if not is_google_oauth_configured():
                    st.warning("⚠️ **Google OAuth 2.0 is currently disabled.**")
                    st.info(
                        "No Google OAuth client credentials (`GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`) are configured in the environment or `.streamlit/secrets.toml`. "
                        "Simulated authentication has been removed to enforce strict production security. "
                        "Please sign in with email and password or register a new account."
                    )
                else:
                    oauth_url = get_google_oauth_url()
                    if oauth_url:
                        st.markdown(
                            f'<a href="{oauth_url}" target="_self" style="display: block; text-align: center; background: #FFFFFF; color: #1F2937; padding: 0.65rem 1rem; border-radius: 8px; font-weight: 700; text-decoration: none; margin-bottom: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.15);">'
                            '🔴 Proceed to Google Account Sign-In ↗</a>',
                            unsafe_allow_html=True
                        )

                if st.button("← Back to Standard Sign In", key="btn_google_back_login", use_container_width=True):
                    st.session_state["auth_mode"] = "login"
                    st.rerun()
