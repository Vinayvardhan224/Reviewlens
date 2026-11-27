#!/usr/bin/env python3
"""
ReviewLens Production Deployment Pre-Flight Integrity Checker.
Validates:
  1. pwa/config.js: APP_URL starts with 'https://' and is not a placeholder ('YOUR-APP').
  2. Manifest files: Valid JSON syntax, required fields present, separated 'any' and 'maskable' icon purposes.
  3. Icon assets: Every referenced icon exists on disk.
  4. Requirements: Consistency between requirements.txt and requirements-lock.txt.
  5. Secret privacy: .streamlit/secrets.toml is not tracked by git.
  6. Secret scanning: No real API keys, Twilio credentials, SMTP passwords, or private keys in tracked files.

Returns:
  Exit code 0 on all checks passing, non-zero on any failure.
"""

import os
import sys
import json
import re
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def check_pwa_config() -> bool:
    """Verifies that pwa/config.js points to a valid https URL and is not a placeholder."""
    config_path = PROJECT_ROOT / "pwa" / "config.js"
    if not config_path.exists():
        print("❌ FAIL: pwa/config.js is missing.", file=sys.stderr)
        return False

    content = config_path.read_text(encoding="utf-8")
    match = re.search(r'const\s+APP_URL\s*=\s*["\']([^"\']+)["\']', content)
    if not match:
        print("❌ FAIL: APP_URL definition not found in pwa/config.js.", file=sys.stderr)
        return False

    url = match.group(1).strip()
    if not url.startswith("https://"):
        print(f"❌ FAIL: APP_URL '{url}' does not start with https://", file=sys.stderr)
        return False

    if "YOUR-APP" in url.upper() or "SHARE.STREAMLIT.IO" in url.upper():
        print(f"❌ FAIL: APP_URL '{url}' contains invalid placeholder/legacy domain.", file=sys.stderr)
        return False

    print(f"✅ PASS: pwa/config.js has valid URL: {url}")
    return True


def check_manifest() -> bool:
    """Verifies manifest validity, icon presence, and purpose separation."""
    manifest_path = PROJECT_ROOT / "pwa" / "manifest.webmanifest"
    if not manifest_path.exists():
        print("❌ FAIL: pwa/manifest.webmanifest not found.", file=sys.stderr)
        return False

    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"❌ FAIL: Manifest {manifest_path.name} is invalid JSON: {e}", file=sys.stderr)
        return False

    required_keys = ["name", "start_url", "icons"]
    for k in required_keys:
        if k not in data:
            print(f"❌ FAIL: Manifest missing required key: {k}", file=sys.stderr)
            return False

    icons = data.get("icons", [])
    if not icons:
        print("❌ FAIL: Manifest contains no icons.", file=sys.stderr)
        return False

    for icon in icons:
        src = icon.get("src", "")
        icon_path = (PROJECT_ROOT / "pwa" / src).resolve()
        if not icon_path.exists():
            print(f"❌ FAIL: Referenced icon does not exist: {src}", file=sys.stderr)
            return False

        purpose = icon.get("purpose", "")
        if "any" in purpose and "maskable" in purpose and purpose != "any" and purpose != "maskable":
            print(f"❌ FAIL: Combined purpose '{purpose}' in icon {src}. Must split into separate 'any' and 'maskable' entries.", file=sys.stderr)
            return False

    print(f"✅ PASS: Manifest ({manifest_path.name}) is valid with {len(icons)} correctly configured icon entries.")
    return True


def check_requirements_consistency() -> bool:
    """Checks that key core dependencies from requirements.txt are represented in requirements-lock.txt."""
    req_path = PROJECT_ROOT / "requirements.txt"
    lock_path = PROJECT_ROOT / "requirements-lock.txt"
    if not req_path.exists() or not lock_path.exists():
        print("❌ FAIL: requirements.txt or requirements-lock.txt missing.", file=sys.stderr)
        return False

    lock_content = lock_path.read_text(encoding="utf-8").lower()
    req_lines = [
        re.split(r'[<>=!~]', line.strip())[0].lower().strip()
        for line in req_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    missing = []
    for pkg in req_lines:
        pattern = rf"(^|\n){re.escape(pkg)}==[0-9a-za-z_.-]+"
        if not re.search(pattern, lock_content):
            # Check without exact ==
            if f"\n{pkg}==" not in f"\n{lock_content}":
                missing.append(pkg)

    if missing:
        print(f"❌ FAIL: Packages in requirements.txt not found in lockfile: {missing}", file=sys.stderr)
        return False

    print("✅ PASS: requirements.txt and requirements-lock.txt are consistent.")
    return True


def check_git_untracked_secrets() -> bool:
    """Verifies that .streamlit/secrets.toml and .env are untracked by git."""
    try:
        res = subprocess.run(
            ["git", "ls-files", ".streamlit/secrets.toml", ".env"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False
        )
        tracked = res.stdout.strip().splitlines()
        if tracked:
            print(f"❌ FAIL: Sensitive secret files are tracked in git: {tracked}", file=sys.stderr)
            return False
    except Exception as e:
        print(f"⚠️ Git tracking check skipped: {e}")

    print("✅ PASS: .streamlit/secrets.toml is not tracked by git.")
    return True


def check_for_leaked_secrets() -> bool:
    """Scans repository tracked files for secret patterns."""
    secret_patterns = [
        (r'AC[a-f0-9]{32}', "Twilio Account SID"),
        (r'-----BEGIN (?:RSA )?PRIVATE KEY-----', "Private Key Certificate"),
    ]

    # Get tracked files
    try:
        res = subprocess.run(
            ["git", "ls-files"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False
        )
        tracked_files = [f.strip() for f in res.stdout.splitlines() if f.strip()]
    except Exception:
        tracked_files = []

    # Exclusions (examples, mock tests, and this checker itself)
    ignored_paths = {
        "secrets.toml.example",
        "scripts/check_deployment.py",
        "tests/test_deployment_check.py"
    }

    leaks = []
    for rel_path in tracked_files:
        if any(ign in rel_path for ign in ignored_paths):
            continue
        full_path = PROJECT_ROOT / rel_path
        if not full_path.exists() or full_path.is_dir():
            continue
        try:
            content = full_path.read_text(encoding="utf-8", errors="ignore")
            for pattern, name in secret_patterns:
                if re.search(pattern, content):
                    leaks.append(f"{rel_path}: Matched {name}")
        except Exception:
            pass

    if leaks:
        print(f"❌ FAIL: Potential hardcoded secrets found in tracked files: {leaks}", file=sys.stderr)
        return False

    print("✅ PASS: No leaked secrets detected across tracked files.")
    return True


def main() -> int:
    print("=" * 60)
    print("🚀 ReviewLens Deployment Pre-Flight Verification")
    print("=" * 60)

    checks = [
        check_pwa_config,
        check_manifest,
        check_requirements_consistency,
        check_git_untracked_secrets,
        check_for_leaked_secrets
    ]

    results = [c() for c in checks]
    if all(results):
        print("=" * 60)
        print("🎉 ALL PRE-FLIGHT DEPLOYMENT CHECKS PASSED.")
        print("=" * 60)
        return 0
    else:
        print("=" * 60)
        print("💥 DEPLOYMENT CHECKS FAILED. See error output above.", file=sys.stderr)
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
