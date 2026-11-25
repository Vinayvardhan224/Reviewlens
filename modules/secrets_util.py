"""
ReviewLens Unified Secrets & Configuration Access Utility.
Provides a single, safe interface for retrieving credentials and secrets
from environment variables or Streamlit secrets without throwing exceptions.
Loading precedence:
  1. os.environ
  2. st.secrets (top-level and nested dictionary access)
  3. Default fallback value
"""

import os
from typing import Any, Optional


def get_secret(name: str, default: str = "") -> str:
    """
    Safely retrieves a configuration key or secret value.
    Checks os.environ first, then st.secrets. Never raises exceptions.
    Supports dot-notation for nested st.secrets keys (e.g., 'auth.google.client_id').
    """
    # 1. Check environment variables
    env_val = os.getenv(name)
    if env_val is not None and env_val != "":
        return env_val

    # 2. Check Streamlit secrets if running within Streamlit context
    try:
        import streamlit as st
        if hasattr(st, "secrets") and st.secrets is not None:
            # Direct top-level match
            if name in st.secrets:
                val = st.secrets[name]
                if val is not None:
                    return str(val)

            # Dot-notation / hierarchy match (e.g. auth.google.client_id)
            if "." in name:
                parts = name.split(".")
                curr: Any = st.secrets
                for part in parts:
                    if isinstance(curr, dict) and part in curr:
                        curr = curr[part]
                    elif hasattr(curr, "__getitem__") and part in curr:
                        curr = curr[part]
                    else:
                        curr = None
                        break
                if curr is not None and curr != "":
                    return str(curr)

            # Also check uppercase variants in nested sections
            if name.startswith("GOOGLE_") and "auth" in st.secrets and "google" in st.secrets["auth"]:
                g_sec = st.secrets["auth"]["google"]
                k_map = {
                    "GOOGLE_CLIENT_ID": "client_id",
                    "GOOGLE_CLIENT_SECRET": "client_secret",
                    "GOOGLE_REDIRECT_URI": "redirect_uri"
                }
                sub_key = k_map.get(name)
                if sub_key and sub_key in g_sec:
                    return str(g_sec[sub_key])
    except Exception:
        pass

    return default


def is_demo_mode_enabled() -> bool:
    """
    Checks if ReviewLens Demo Mode is explicitly enabled via REVIEWLENS_DEMO_MODE=1.
    In production, demo mode is disabled by default.
    """
    val = get_secret("REVIEWLENS_DEMO_MODE", "0").strip().lower()
    return val in ("1", "true", "yes", "on")
