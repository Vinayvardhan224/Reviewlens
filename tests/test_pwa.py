"""
Unit tests for ReviewLens PWA Web App Manifest, Service Worker, and Configuration Scripts.
Verifies:
  1. pwa/manifest.webmanifest is valid JSON and contains required PWA fields
  2. All referenced icon files exist on disk with valid dimensions
  3. scripts/set_app_url.py validates HTTPS scheme and updates pwa/config.js correctly
"""

import unittest
import json
from pathlib import Path
from PIL import Image
from scripts.set_app_url import validate_and_format_url, set_app_url, CONFIG_PATH

PWA_DIR = Path(__file__).resolve().parent.parent / "pwa"
MANIFEST_PATH = PWA_DIR / "manifest.webmanifest"


class TestPWAShell(unittest.TestCase):
    def test_manifest_webmanifest_validity(self):
        self.assertTrue(MANIFEST_PATH.exists(), "pwa/manifest.webmanifest must exist")
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("name", data)
        self.assertIn("short_name", data)
        self.assertIn("start_url", data)
        self.assertIn("display", data)
        self.assertEqual(data["display"], "standalone")
        self.assertEqual(data["background_color"], "#07051A")
        self.assertIn("icons", data)
        self.assertGreaterEqual(len(data["icons"]), 2)

        # Check discrete purpose entries
        purposes = [icon.get("purpose") for icon in data.get("icons", [])]
        self.assertIn("any", purposes)
        self.assertIn("maskable", purposes)
        self.assertNotIn("any maskable", purposes)

    def test_manifest_icons_exist_and_valid(self):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        for icon in data.get("icons", []):
            src = icon["src"]
            icon_file = PWA_DIR / src
            self.assertTrue(icon_file.exists(), f"Icon {src} must exist at {icon_file}")
            with Image.open(icon_file) as img:
                self.assertEqual(img.format, "PNG")
                sizes = icon["sizes"].split("x")
                self.assertEqual(img.width, int(sizes[0]))
                self.assertEqual(img.height, int(sizes[1]))

    def test_set_app_url_https_validation(self):
        # Valid HTTPS URLs
        self.assertEqual(
            validate_and_format_url("https://reviewlens.streamlit.app"),
            "https://reviewlens.streamlit.app"
        )
        self.assertEqual(
            validate_and_format_url("https://share.streamlit.io/user/repo/app.py"),
            "https://share.streamlit.io/user/repo/app.py"
        )

        # Invalid URLs (HTTP rejected, bad formats rejected)
        with self.assertRaises(ValueError):
            validate_and_format_url("http://insecure-site.com")

        with self.assertRaises(ValueError):
            validate_and_format_url("ftp://not-web.com")

        with self.assertRaises(ValueError):
            validate_and_format_url("")

        with self.assertRaises(ValueError):
            validate_and_format_url("not-a-url")

    def test_set_app_url_file_writing(self):
        original = None
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                original = f.read()

        try:
            test_target = "https://custom-test-deploy.streamlit.app"
            set_app_url(test_target)
            self.assertTrue(CONFIG_PATH.exists())
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn(test_target, content)
        finally:
            if original is not None:
                with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                    f.write(original)


if __name__ == "__main__":
    unittest.main()
