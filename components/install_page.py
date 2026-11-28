"""
ReviewLens Standalone Application Installation & PWA Guide.
Renders step-by-step instructions for installing ReviewLens on Android, iOS,
and Desktop platforms with zero app store download requirement.
"""

import html
import re
from pathlib import Path
from typing import Optional
import streamlit as st

from components.theme import glass_card, kpi_tile
from modules.secrets_util import get_secret


def get_pwa_shell_url(default_app_url: str = "") -> str:
    """
    Resolves the standalone PWA shell URL:
    1. Explicit REVIEWLENS_INSTALL_URL or REVIEWLENS_APP_URL from environment/secrets.
    2. PWA config in pwa/config.js.
    3. GitHub Pages deployment URL.
    4. Fallback to provided default_app_url.
    """
    # 1. Configured secret
    shell_url = get_secret("REVIEWLENS_INSTALL_URL") or get_secret("REVIEWLENS_APP_URL")
    if shell_url:
        return shell_url.strip()

    # 2. Check pwa/config.js
    try:
        pwa_config = Path(__file__).resolve().parent.parent / "pwa" / "config.js"
        if pwa_config.exists():
            content = pwa_config.read_text(encoding="utf-8")
            m = re.search(r'const\s+APP_URL\s*=\s*["\']([^"\']+)["\']', content)
            if m:
                target_url = m.group(1).strip()
                if target_url.startswith("https://"):
                    return target_url
    except Exception:
        pass

    # 3. Default to GitHub Pages or default_app_url
    return default_app_url or "https://vinayvardhan224.github.io/Reviewlens/"


def render_install_page(app_url: str, pwa_shell_url: Optional[str] = None) -> None:
    """
    Renders the dedicated Install App view for ReviewLens:
    - Title and introductory value proposition
    - Quick KPI tiles (footprint, cross-platform, PWA status)
    - Tabbed installation guides: Android, iPhone / iPad, Windows / Mac / Linux
    - Copyable app URL and optional QR code generator
    - Open installable version link button
    - Good to know advice tile
    """
    raw_app_url = str(app_url or "").strip()
    if not raw_app_url:
        raw_app_url = "https://reviewlens.streamlit.app"

    if pwa_shell_url is None:
        pwa_shell_url = get_pwa_shell_url(default_app_url=raw_app_url)

    safe_app_url = html.escape(raw_app_url, quote=True)
    safe_pwa_url = html.escape(str(pwa_shell_url).strip(), quote=True)

    # ------------------------------------------------------------------
    # 1. HERO TITLE & VALUE PROPOSITION
    # ------------------------------------------------------------------
    hero_html = f"""
    <div style="margin-bottom: 0.2rem;">
        <div class="rl-tile-title" style="color: var(--accent);">📲 PROGRESSIVE WEB APPLICATION (PWA)</div>
        <h1 style="font-size: 2.1rem; font-weight: 800; color: #FFFFFF; margin: 0 0 0.5rem 0; letter-spacing: -0.025em;">Install ReviewLens</h1>
        <p style="color: var(--text); font-size: 1.05rem; margin: 0; line-height: 1.55;">
            Use ReviewLens like a normal app on your phone or computer. No app store needed.
        </p>
    </div>
    """
    st.markdown(glass_card(hero_html), unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # 2. KPI TILES (FOOTPRINT, COMPATIBILITY, ARCHITECTURE)
    # ------------------------------------------------------------------
    kc1, kc2, kc3 = st.columns(3)
    with kc1:
        st.markdown(
            kpi_tile(
                label="App Store Footprint",
                value="0 MB",
                delta="No Download",
                tone="pos",
                sublabel="Direct web installation",
            ),
            unsafe_allow_html=True,
        )
    with kc2:
        st.markdown(
            kpi_tile(
                label="Supported Platforms",
                value="3 Systems",
                delta="Cross-Device",
                tone="pos",
                sublabel="Android • iOS • Desktop",
            ),
            unsafe_allow_html=True,
        )
    with kc3:
        st.markdown(
            kpi_tile(
                label="PWA Engine",
                value="PWA Ready",
                delta="Service Worker",
                tone="mixed",
                sublabel="Standalone window mode",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # 3. MAIN BENTO GRID: TABS (LEFT) & ACTIONS / GOOD TO KNOW (RIGHT)
    # ------------------------------------------------------------------
    col_guide, col_action = st.columns([1.35, 1], gap="medium")

    with col_guide:
        tab_android, tab_ios, tab_desktop = st.tabs(
            ["Android", "iPhone / iPad", "Windows / Mac / Linux"]
        )

        with tab_android:
            android_html = f"""
            <div>
                <div class="rl-tile-title" style="color: var(--pos);">🤖 Android (Google Chrome)</div>
                <ol style="margin: 0.5rem 0 0.8rem 1.2rem; padding: 0; color: var(--text); font-size: 0.92rem; line-height: 1.75;">
                    <li>Open the app link (<code style="background: rgba(255,255,255,0.08); padding: 2px 6px; border-radius: 4px; color: var(--accent);">{safe_app_url}</code>) in Google Chrome.</li>
                    <li>Tap the browser <strong>menu (three dots ⋮)</strong> in the top-right corner.</li>
                    <li>Select <strong>"Install app"</strong> or <strong>"Add to Home screen"</strong>.</li>
                    <li>Tap <strong>Install</strong> to add ReviewLens to your home screen and app drawer.</li>
                </ol>
                <div style="font-size: 0.8rem; color: var(--muted); margin-top: 0.4rem;">
                    ✓ Runs in a dedicated fullscreen container without browser URL bars or navigation clutter.
                </div>
            </div>
            """
            st.markdown(glass_card(android_html), unsafe_allow_html=True)

        with tab_ios:
            ios_html = f"""
            <div>
                <div class="rl-tile-title" style="color: var(--accent);">🍎 iPhone / iPad (Safari only)</div>
                <ol style="margin: 0.5rem 0 0.8rem 1.2rem; padding: 0; color: var(--text); font-size: 0.92rem; line-height: 1.75;">
                    <li>Open the app link (<code style="background: rgba(255,255,255,0.08); padding: 2px 6px; border-radius: 4px; color: var(--accent);">{safe_app_url}</code>) in Safari.</li>
                    <li>Tap the <strong>Share button</strong> (the square icon with an arrow pointing up at the bottom of the screen).</li>
                    <li>Scroll down and tap <strong>"Add to Home Screen"</strong>.</li>
                    <li>Tap <strong>Add</strong> in the top-right corner to place ReviewLens on your iOS Home Screen.</li>
                </ol>
                <div style="font-size: 0.8rem; color: var(--muted); margin-top: 0.4rem;">
                    ℹ️ Apple iOS requires Safari to pin Progressive Web Apps to your home screen.
                </div>
            </div>
            """
            st.markdown(glass_card(ios_html), unsafe_allow_html=True)

        with tab_desktop:
            desktop_html = f"""
            <div>
                <div class="rl-tile-title" style="color: var(--primary-2);">💻 Windows / Mac / Linux (Chrome, Edge, Brave)</div>
                <ol style="margin: 0.5rem 0 0.8rem 1.2rem; padding: 0; color: var(--text); font-size: 0.92rem; line-height: 1.75;">
                    <li>Open the app link in Google Chrome, Microsoft Edge, or Brave.</li>
                    <li>Click the <strong>Install icon</strong> (computer monitor with a down arrow) in the address bar, or open the browser menu (⋮).</li>
                    <li>Click <strong>"Install ReviewLens"</strong> when prompted.</li>
                    <li>ReviewLens will launch in its own standalone desktop window and create desktop/taskbar shortcuts.</li>
                </ol>
                <div style="font-size: 0.8rem; color: var(--muted); margin-top: 0.4rem;">
                    ✓ Enjoy native-like taskbar integration, quick switching (Alt+Tab), and focused workflow.
                </div>
            </div>
            """
            st.markdown(glass_card(desktop_html), unsafe_allow_html=True)

    with col_action:
        # 1. Copy app link box
        st.markdown(
            """
            <div class="rl-tile-title" style="color: var(--text); margin-bottom: 0.35rem;">📋 Copy app link</div>
            """,
            unsafe_allow_html=True,
        )
        st.code(raw_app_url, language="text")

        # 2. QR code generator (if qrcode package installed, otherwise silent skip)
        try:
            import qrcode
            import io

            qr = qrcode.QRCode(
                version=1,
                box_size=5,
                border=2,
            )
            qr.add_data(raw_app_url)
            qr.make(fit=True)
            img = qr.make_image(fill_color="#FFFFFF", back_color="#0D0A24")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            st.image(buf.getvalue(), caption="Scan to open on phone", width=170)
        except Exception:
            pass

        # 3. Open installable version link button
        st.link_button(
            "🚀 Open installable version",
            pwa_shell_url,
            use_container_width=True,
            type="primary",
        )

        st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)

        # 4. Good to know advice tile
        good_to_know_html = """
        <div>
            <div class="rl-tile-title" style="color: var(--mixed);">💡 Good to know</div>
            <ul style="margin: 0.45rem 0 0 1.15rem; padding: 0; color: var(--text); font-size: 0.86rem; line-height: 1.65;">
                <li><strong>Needs an internet connection</strong>: ReviewLens runs live NLP sentiment, aspect decomposition, and spam heuristics that require backend network access.</li>
                <li><strong>Free hosting wake-up</strong>: If the cloud container has been idle, it may take about 30 seconds to wake up on the first connection.</li>
                <li><strong>Google sign-in</strong>: Google OAuth may require opening in the standard browser version rather than an in-app webview depending on your device security policies.</li>
            </ul>
        </div>
        """
        st.markdown(glass_card(good_to_know_html), unsafe_allow_html=True)
